# NCCL Bootstrap 在各场景下做的事

> 接 [`concepts.md`](concepts.md) 里 bootstrap 的总览，本文按"实际部署形态 × GPU/网络拓扑"展开各场景的细节。

## 0. 不变量：所有场景共享的 bootstrap 框架

不管你在哪种部署形态、什么拓扑，bootstrap **总要完成**这 6 件事：

```
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. 选 OOB 接口        — 找一个能让所有 rank 互通的 IP iface   │
   │ 2. rank 0 起 listener — listening socket + accept 后台线程    │
   │ 3. UID 分发           — 用户把 ncclUniqueId 送到所有 rank     │
   │ 4. rendezvous         — 各 rank connect 到 rank 0, 形成 star  │
   │ 5. 元数据 allgather   — ncclPeerInfo / 拓扑 / handle 交换    │
   │ 6. ring 形成 + barrier — rank N 连 rank (N+1)%N, 备用通道    │
   └──────────────────────────────────────────────────────────────┘
```

不同场景的差别在于 "**OOB 接口选哪个**" 和 "**rank 之间是同主机/跨主机/同 fabric**" —— 这决定了 bootstrap socket 走 lo / 内网 / 公网，以及 step 5 里交换的 handle 类型不同。

下面按由简入繁的顺序拆 5 个典型场景。

---

## 题外话：用户与 NCCL 的职责边界

用户调一行 `ncclCommInitRank(comm, nRanks, uid, myRank)`，哪些事 NCCL 干、哪些事用户必须自己处理？很多人卡在这条边界上，所以先单独拎出来。

```
   用户负责                                        NCCL 负责
   ────────                                        ────────
   1. 在 rank 0 调 ncclGetUniqueId(&uid)           生成 128B UID
                                                   (内含 rank 0 的 IP:port + magic)
   ────────────────────────────────────────────────────────────────────────────
   2. 把这 128B UID 送到 *所有* rank               ★ NCCL 完全不管这一步 ★
      自选机制:
        - 单进程多 rank: 内存指针 / argv
        - MPI:           MPI_Bcast(uid, 128B, ...)
        - PyTorch:       store.set/get("nccl_uid", ...)
        - 共享文件:      写到 NFS / 本地盘 (你的 all_reduce_2proc 用这个)
        - env / argv:    父进程 fork 时透传
   ────────────────────────────────────────────────────────────────────────────
   3. 每个 rank 各自调                             NCCL 内部把剩下全做完:
      ncclCommInitRank(                            - rank 0 起 listener
        comm, nRanks, uid, myRank)                 - accept (nRanks-1) 次 (阻塞!)
                                                   - 其他 rank 用 uid 里的 ip:port connect
       ↑                                           - 形成 star → 闭成 ring
       └── nRanks 是用户告诉 NCCL 的!              - 各 rank 本地 fillInfo (80B)
           NCCL 由此知道要等多少 peer              - bootstrapAllGather peerInfo
                                                   - 探拓扑 / 算图 / 建 transport
                                                   - 返回用户态
```

**rank 0 会等所有 rank 到齐**：因为 `nRanks` 是构造函数参数，rank 0 的 accept 线程会一直 accept 到收齐 (nRanks-1) 条连接才往下走；任何一个 rank 没及时进 `ncclCommInitRank`，整个 init 就 hang 在 rank 0 的 accept 或某个 `bootstrapAllGather` 上。这就是诊断表里 "ncclCommInitRank 卡死" 的根因。

### rendezvous 和 peerInfo 的先后

一个反直觉的点：**peerInfo 不是 rendezvous 的输入，而是 rendezvous 之后的产物**。各 rank 在 connect rank 0 之前，相互完全不认识；rendezvous 用的是 UID 里编码的 rank 0 IP:port。

```
   时间 →
   ──────────────────────────────────────────────────────────────────────────
   t0  rank 0: ncclGetUniqueId()                            ✓ 有 UID
   t1  用户: 把 UID 送到其他 rank (MPI/store/file/...)      ✓ 各 rank 拿到 UID
   t2  各 rank 进 ncclCommInitRank(comm, nRanks, uid, ...)    (只知道自己 + UID)
   t3  rank 0 listen / 其他 rank connect 到 UID 里的地址    ✓ rendezvous (step 4)
                                                              (拿到 socket fd, 但
                                                               不知对端 hostHash/
                                                               pidHash/busId/...)
   t4  各 rank 本地 fillInfo() → 自己那 80B peerInfo         ✓ 自己那一格
   t5  bootstrapAllGather(peerInfo, 80B)                    ✓ peerInfo[nRanks]
                                                              每 rank 都有全集
   ──────────────────────────────────────────────────────────────────────────
       ↑                       ↑                                  ↑
   只有 UID                rendezvous 完成                    每 rank 有 peer 全集
   (rank → ip:port 都        (拿到 socket 通路, 但              ("自我介绍"完成,
    还不知道)                  还没"自我介绍")                   后续选 transport
                                                                  全靠这张表)
```

所以"用户在自己上下文中拿到 peerInfo 去 rendezvous"这个直觉是反的——peerInfo **是 rendezvous 完成后的第一笔信息交换** (即 Phase A allgather)。rendezvous 用 UID（预先共享的约定）建通路，peerInfo 用这条通路完成"自我介绍"。

---

## 题外话：Step 1 — 选 OOB 接口（`bootstrapNetInit`）

不变量 6 步里的**第一步**——"选一个能让所有 rank 互通的 IP iface"。源码在 `bootstrap.cc:107` 的 `bootstrapNetInit`。后续整个 OOB 通路（UID 里的 IP、rank 0 listen、所有 bootstrap socket、RAS 网络）都基于它选出来的这一个 iface。

### 调用时机：每**进程**一次（不是每 rank 一次）

```c
// init.cc:145-156
static void initOnceFunc() {
  NCCLCHECKGOTO(ncclOsInitialize(), initResult, exit);
  initGdrCopy();
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);    // ← 选 iface
  initNvtxRegisteredEnums();
exit:;
}

static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);             // ← 进程内只跑一次
  return initResult;
}
```

每个 `ncclCommInitRank` 都进 `ncclInit()`，但被 `std::call_once` 兜住——一个进程里只真正执行一次：

| 场景 | bootstrapNetInit 实际执行次数 |
|---|---|
| 单进程 `ncclCommInitAll`（1 进程多 rank） | **1 次** |
| 单机多进程（每进程 1 rank） | 每进程各 1 次 |
| 一个进程同时起多个 comm | **1 次** (`call_once` 兜底) |

### 它到底选了什么

源码 `bootstrap.cc:100-101`：

```c
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];   // 比如 "ens4f0np0"
static union ncclSocketAddress bootstrapNetIfAddr;    // 比如 192.168.80.101
```

两个**进程级 static** 变量。后续三处吃它：

| 位置 | 用途 |
|---|---|
| `bootstrap.cc:451` | `memcpy(&handle->addr, &bootstrapNetIfAddr, ...)` —— rank 0 的 IP 编码进 `ncclUniqueId`（用户分发的 128B）|
| `bootstrap.cc:527` | 所有 bootstrap socket 的 `bind`/`connect` 地址 |
| `bootstrap.cc:824` | RAS（Reliability/Serviceability）网络复用 |

**Step 1 选错 → UID 里编码错 IP → 其他 rank connect 不上 → init hang**。

### 选 iface 的优先级（`socket.cc:169-205` `ncclFindInterfaces`）

```
   ┌─ A. NCCL_SOCKET_IFNAME 显式指定 ──────────────────────────┐
   │   只在它指定的 iface 里找                                 │
   │     "eth0"        → 白名单                                │
   │     "^docker,lo"  → 黑名单 (^ 开头, 逗号分隔)             │
   │   找不到 → 报错退出, 不 fallback                          │
   └──────────────────────────────────────────────────────────┘
                                ↓ (没设)
   ┌─ B. 自动选, 按优先级 ───────────────────────────────────┐
   │   1. 名字以 "ib" 开头的 iface     ← IPoIB 优先!          │
   │        (ib0, ib1, ...)                                   │
   │   2. 若 NCCL_COMM_ID=<ip:port> 设了:                     │
   │        找跟它同子网的 iface                              │
   │   3. 排除 docker / lo / virbr 后剩下第一个               │
   │        (eth0, ens4f0, enp..., wlan0 都行)                │
   │   4. docker                                              │
   │   5. lo                                                  │
   │   6. virbr                                               │
   └──────────────────────────────────────────────────────────┘
```

注意：**iface 是按 sysfs 列表顺序拿第一个匹配的**，没有任何吞吐/延迟评估。

特殊路径：`bootstrapNetInit` 自己里（`bootstrap.cc:111-124`），如果 `NCCL_COMM_ID` 已经设了，**直接走 `ncclFindInterfaceMatchSubnet`**，完全跳过上面 6 级优先——理由是 UID 由用户外部指定了，iface 必须跟那个 IP 同子网。

### 标准和目的

这步要选出的 IP 必须满足三个**硬要求**：

1. **所有 rank 互相可达** —— TCP connect 不被防火墙拦、子网路由通
2. **每个 host 上能稳定 listen** —— rank 0 要 bind 起 listener
3. **iface name 在所有 host 上能解析** —— 多机时每台机器都要找得到同名 iface

**不**关心：

- 带宽/延迟 —— bootstrap 一次 init 总流量约 20 KB（2 rank）到几百 KB（16 rank），是后续 GB/s 量级通信的零头
- 数据路径用什么 transport —— OOB 只是控制平面，跟 NVLink/IB/PCIe 选择完全无关

**为什么 IPoIB 排第一**：IB 集群里 ib0 几乎必然同子网、可路由、没乱七八糟的过滤。是大概率正确的猜测。

### 不同拓扑下的权衡

| 拓扑 | 自动选什么 | 风险 / 取舍 |
|---|---|---|
| 单进程多卡 | 第一个非 lo iface | 走 loopback fast-path，谁都行 |
| 单机多进程 | 同上 | 内核 loopback 路径，仍 OK |
| 裸金属单 NIC | eth0 / ens... | 没坑 |
| **裸金属多 NIC**（管理 + 数据） | 列表第一个非 docker/lo 的 ← 可能是慢管理网 | 管理网通常 1G/限速；bootstrap 流量小本身 OK，但和数据流抢同一 NIC 会有干扰 → 显式 `NCCL_SOCKET_IFNAME=` 选数据网更稳 |
| IB 集群 | ib0 (IPoIB) | 优点：肯定通；缺点：和数据 IB 共享 HCA 资源（流量小可忽略）|
| 以太 + IB 混合 | **IPoIB 优先** | 想 OOB 完全独立（断 IB 也能诊断）就显式选 eth0 |
| 容器 / k8s | `^docker,lo,virbr` 已默认排除 | CNI 网（cali..., flannel.1）不在排除内 → 显式给 |
| 跨子网 / 跨 AZ | 自动选大概率挑错 | `NCCL_COMM_ID=<rank0 ip:port>` + `NCCL_SOCKET_IFNAME=` 双显式 |
| MNNVL | 跟跨机一样选 IP iface | NVLink fabric **不**做 OOB（chicken-and-egg：要先 OOB 才能发现 fabric）|

**生产铁律**：**别让 NCCL 猜**——显式设 `NCCL_SOCKET_IFNAME`。"测试机 work、部署机 hang" 的常见根因是网卡命名变了（`eno1` / `eth0` / `ens4f0` / `enp...`），自动选挑错。

### `bootstrapNetInitDone` 标记位

```c
// bootstrap.cc:102, 107-140
static int bootstrapNetInitDone = 0;
static std::mutex bootstrapNetMutex;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {                  // 快路径: 不取锁先看
    std::lock_guard<std::mutex> lock(...);
    if (bootstrapNetInitDone == 0) {                // 取锁后 double-check
      // ... 选 iface, 写 bootstrapNetIfName/Addr ...
      bootstrapNetInitDone = 1;                     // ← 标记完成
    }
  }
  return ncclSuccess;
}
```

含义和性质：

| 性质 | 说明 |
|---|---|
| **进程级**（不是 rank 级） | 同进程所有 rank 共享同一个 IfName/IfAddr |
| 写在赋值之后 | 一旦 == 1，外面读 IfName/IfAddr 一定有效 |
| **永不复位** | comm destroy 也不重置；想换 iface 只能重启进程 |
| double-check locking | thread-safe 单例；外层 `call_once` 是双保险 |

### 一张图：调用 + 数据流

```
   每个进程第一次进 ncclCommInitRank
                ↓
   ncclInit() ── std::call_once ──→ initOnceFunc()
                                          ↓
                                    bootstrapNetInit()
                                          ↓
                       ┌── NCCL_SOCKET_IFNAME 设了? ──┐
                       │ 是                            │ 否
                       ▼                               ▼
                  按它找/失败                    自动 6 级优先:
                                                 ib* → COMM_ID 同子网
                                                 → 非 docker/lo/virbr
                                                 → docker → lo → virbr
                       └─────────────┬──────────────┘
                                     ▼
              bootstrapNetIfName / bootstrapNetIfAddr 写好
                                     ▼
                       bootstrapNetInitDone = 1
                                     ▼
              ┌─────────┬─────────────┬──────────────┐
              ▼         ▼             ▼              ▼
          UID 编码   bootstrap     bootstrap        RAS
          (rank 0)    listen      connect/send
          451 行     527 行         527 行          824 行
```

确认实际选了啥：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=BOOTSTRAP ./run.sh
# 看 "Bootstrap: Using ens4f0np0:192.168.80.101<0>"  ← bootstrap.cc:135 这行
```

---

## 题外话：rendezvous 是什么

上面 step 4 写的是 "rendezvous（各 rank connect 到 rank 0）"，这个词在 NCCL 文档里反复出现，先把它澄清下。

**字面**：法语 `rendez-vous`，由 `rendez`（去呈现自己 / 报到）+ `vous`（你/你们）构成 → "你来报到" → "约会、约定见面、集合点"。日常法语就是 "见面"、"预约"。18 世纪起被英语借走，泛指**"预先约定的会面地点 / 集结点"**。军事里很常见，"rendezvous point" 即"集合点"。

**到分布式系统**：被技术圈征用来描述**"互不认识的进程在一个预先约定的位置找到彼此"**。本质就是"集合点"在网络里的对应物。

```
   现实世界                    分布式系统
   ────────                    ──────────
   "周六下午 3 点                "我们都去连 192.168.1.10:23456,
    咖啡馆 A 见!"                 这个 socket 就是我们的集合点"
        │                              │
   你不需要知道每个朋友          每个 rank 不需要事先知道
   长什么样、住哪里,             其他 rank 的 hostname/PID,
   到了咖啡馆 A 自然碰上          连上那个 socket 自然碰上
```

**为什么需要 rendezvous**：N 个进程**互相不认识**（没有 hostname 列表、不知道 PID、不知道 GPU），但它们都拿到了**同一个"约定"**（128 字节的 `ncclUniqueId`，里头藏着 rank 0 的 IP:port）。所有 rank 拿着这个约定去同一个地方"报到"，在那里第一次彼此见面 —— 这就是 rendezvous。

**rendezvous ≠ 自我介绍**：rendezvous 只负责"到了集合点"，不包含 step 5 的 `ncclPeerInfo` allgather（"互相通报身份"）。两者是相邻但独立的步骤。

**其他系统也用这个词**：

| 系统 | rendezvous 指什么 |
|---|---|
| **MPI** | `MPI_Init` 内部的进程发现阶段（PMI rendezvous） |
| **PyTorch** | `torch.distributed.init_process_group` 底层就叫 `c10d::Store rendezvous`，机制类似 NCCL bootstrap |
| **etcd / ZooKeeper** | 典型用法之一：所有节点去读同一个 key 当 rendezvous 点 |
| **TCP 协议** | "TCP rendezvous" 指两端**同时** SYN，简单建连（罕见但术语存在） |
| **NVLink Fabric Manager** | 跨节点 NVLink 也有 fabric rendezvous 这个内部术语 |

**和 "discovery" 的细微区别**：`discovery` 强调"发现"（事先不知道谁会冒出来），`rendezvous` 强调"按约定碰头"（事先有共享的约定）。NCCL 用 rendezvous 而不用 discovery，恰恰是因为有 `ncclUniqueId` 这个预先共享的约定。

---

## 题外话：Step 5 元数据交换详解

上面 step 5 写的是"元数据 allgather"。其实**它不是一次 allgather，是一组多次的混合交换**。把它拆开看：

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Phase A   AllGather1: ncclPeerInfo                  ←每 rank 80B 左右   │
   │            "我是谁、我的 GPU/PID/host/能力"                              │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  Phase B   拓扑探测: 各 rank 只探"自己那部分" (自己的 GPU + 自己看到的   │
   │            NIC), 同 node ranks 通过 bootstrapIntraNodeAllGather 融合,    │
   │            合成完整 ncclTopoSystem (6 类节点的图)                        │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  Phase C   本地算法图搜索 (不走 bootstrap, 各 rank 跑相同算法)           │
   │            得到 Ring/Tree/CollNet/NVLS 4 套 ncclTopoGraph                │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  Phase D   AllGather3: graphInfo[4] + topoRanks      ←每 rank ~200B      │
   │            "我算出来的环/树是这样的、我在每个 channel 上的邻居是谁"      │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  Phase E   AllGather (CollNet only, 仅当 collnet 支持时)                 │
   │            collnetShareInfo, dense rank 映射, 兼容性矩阵                 │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  Phase F   逐 peer × 逐 channel 的点对点 handle 交换 (Send/Recv, 非 AG)  │
   │            ncclConnect (256B/channel/peer)                               │
   │            内容由 transport 决定: P2P 用 IPC handle, IB 用 QP/GID,       │
   │            SHM 用 shm fd 等                                              │
   └──────────────────────────────────────────────────────────────────────────┘
```

下面每个 phase 单独拆。

### Phase A: AllGather1 — `ncclPeerInfo`

源码 `nccl/src/include/transport.h:37`：

```c
struct ncclPeerInfo {
    int       rank;            // 我的 rank
    int       cudaDev;          // 我用哪张卡 (CUDA device index)
    int       nvmlDev;          // 我用哪张卡 (NVML device index, 可能 ≠ cudaDev)
    int       gdrSupport;       // 我的 GPU 支不支持 GPUDirect RDMA
    uint64_t  hostHash;         // 我的 hostname 哈希 (同机 rank 此值相同)
    uint64_t  pidHash;          // 我的 PID 哈希 (同进程 rank 此值相同)
    dev_t     shmDev;           // 我的 /dev/shm 所在设备号 (判断 SHM 是否可用)
    int64_t   busId;            // 我的 GPU PCIe bus ID
    ncclComm *comm;             // 同进程 peer 才用得上 (跨进程是无效指针)
    int       cudaCompCap;      // 我的 GPU 算力 (sm_80 → 80)
    nvmlGpuFabricInfoV_t fabricInfo;  // ★ MNNVL: clusterUuid + cliqueId + state
    int       cuMemSupport;     // 我支持 CUDA Virtual Memory Mgmt API 吗
    int       version;          // 我的 NCCL 版本
};
```

每条 80 字节左右。**这是后续所有选择的输入**：

| 字段 | 拿来判断什么 |
|---|---|
| `hostHash` | 跟我同 host 的 rank → 走 P2P/SHM；不同 host → 走 NET |
| `pidHash` | 同 host 同进程 → 不用 CUDA IPC（地址空间同）；同 host 不同进程 → 要 IPC |
| `busId` + `cudaDev` | 算 NVLink 连接、PCIe 距离 |
| `gdrSupport` | 跨机 NET transport 能不能上 GPUDirect RDMA |
| `cudaCompCap` | 选 algo / kernel 变体 |
| `fabricInfo.cliqueId` | 跨主机但同 NVLink fabric → 仍可走 P2P/NVLS（MNNVL） |
| `cuMemSupport` | 选 CUMEM 后端 vs 老 cudaIpc |
| `version` | mismatch 时 WARN |

```
   Phase A 在 bootstrap 上的流向:

   rank 0           rank 1           rank 2           rank 3
     │                │                │                │
     │ fillInfo()     │ fillInfo()     │ fillInfo()     │ fillInfo()
     │ → 本地 80B     │ → 本地 80B     │ → 本地 80B     │ → 本地 80B
     ▼                ▼                ▼                ▼
   ┌──────────────────────────────────────────────────────┐
   │   bootstrapAllGather() over TCP ring                 │
   │   每 rank 出 80B → 每 rank 收齐 N×80B = ncclPeerInfo[]│
   └──────────────────────────────────────────────────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
   完整 peer 表 (本地一份, 内容相同)
```

#### Phase A 在代码里 ("每 rank 都拿到完整 peerInfo[]" 的证据)

```c
// nccl/src/init.cc:998-1018  initTransportsRank()

// AllGather1 - begin
NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1),                 // [L999]  分配 nranks+1 slot
              ret, fail);                                            //         (+1 给 CollNet root 占位)
NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo + rank, comm->commHash), // [L1000] 只填自己那一格
              ret, fail);                                            //         (从本进程取 hostHash/
                                                                     //          pidHash/cudaDev/busId)
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap,                    // [L1001] 整张表 allgather
              comm->peerInfo, sizeof(struct ncclPeerInfo)),          //         回来后 slot i = rank i
              ret, fail);
COMPILER_ATOMIC_STORE(&comm->peerInfoValid, true, ...);              // [L1002] 标记可用

for (int i = 0; i < nranks; i++) {                                   // [L1005] 遍历全集
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) // ← 这里用 i 索引
        nNodes++;                                                    //   = 任意 peer 的字段
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;     //   都本地可读
    ...
}
// AllGather1 - end
```

| init.cc 行 | 关键点 |
|---|---|
| 999 | `comm->peerInfo` 是 `struct ncclPeerInfo*`，长度 `nranks+1` |
| 1000 | `fillInfo` 只写 `comm->peerInfo + rank` 这一格；全部字段从本进程取 (`getHostHash`、`getPidHash`、`cudaGetDeviceProperties`、`stat("/dev/shm")`、`comm->busId`)。**不需要事先知道任何 peer** |
| 1001 | `bootstrapAllGather` —— 每 rank 出 80B、收 `nranks*80B`。这是 rendezvous 完成后的**第一笔**数据交换 |
| 1005-1018 | 全 rank 遍历 `peerInfo[i]`，导出 `nNodes`、`cuMemSupport`、`globalCrossNicSupport` 等全局属性 |

`comm->peerInfo` 在 communicator 整个生命周期里都在内存里，直到 `ncclCommDestroy` 才在 `init.cc:311` `free(comm->peerInfo)`。后续 Phase C/D/F 选 transport (P2P/IPC/CUMEM/IB/NVLS) 全部基于这张表查 hostHash/pidHash/busId/fabricInfo 做决策。

> **本仓最小复刻**：`my_nccl/src/main.cc:96-124` 同样的 4 步——calloc + fillInfo + bootstrapAllGather + 遍历打印。跑 `all_reduce_2proc` 时两个 rank 都打印同样的 2 行 peerInfo (hostHash 相同、pidHash 不同)，就是"每 rank 都拿到全集"的直接眼见证据。

```
   每 rank 进程本地           bootstrap AllGather              每 rank 进程本地
   ───────────────            ─────────────────                ───────────────
   peerInfo[rank]      ──→    通过 star → ring TCP    ──→    peerInfo[0..nranks-1]
   = fillInfo()                收齐 nranks × 80B                (完整一份,
   (只有自己 80B)                                                内容各 rank 相同)
```

后面 Phase D 的 `allGather3Data` (init.cc:1243)、Phase F 的 `ncclConnect` 256B blob (transport.cc:100) 也是同样模式——每 rank 出自己那部分 → bootstrapAllGather/Send/Recv → 每 rank 拿到全集。

### Phase B: 拓扑探测（有去重机制，不是各 rank 全干一遍！）

> **⚠ 直觉陷阱**：如果说"每个 rank 都本地读 /sys 探拓扑"，那同 node 8 个 rank 不就把同样的 /sys 树读 8 遍？是不是浪费？
>
> 答：**NCCL 不是这么做的**。它有一套"各 rank 只探自己那部分 + intra-node 合并"的去重设计。

#### 真实流程（源码 `nccl/src/graph/topo.cc:728` `ncclTopoGetSystem`）

```
   ┌─ Step 1: 看有没有 NCCL_TOPO_FILE / 默认 XML 缓存 ─────────────────┐
   │   if (NCCL_TOPO_FILE) → 直接读 → 跳过下面所有探测              │
   │   else 尝试 /var/run/nvidia-topologyd/virtualTopology.xml        │
   │   都没有 → 各 rank 自己探                                        │
   └────────────────────────────────────────────────────────────────┘
                              ↓ (没缓存时)
   ┌─ Step 2: 本地探, 但只探"自己那一份" ───────────────────────────┐
   │   ncclTopoFillGpu(busId_of_my_rank)                            │
   │     ↳ 只 fill 当前 rank 管的那张 GPU 进 XML                    │
   │   ncclNet->devices(...) + getProperties() for each NIC         │
   │     ↳ 自己 process 看到的 NIC                                  │
   │                                                                 │
   │   注释原文 (topo.cc:755):                                       │
   │     "Detect only the GPU managed by this process.               │
   │      We'll get any others through XML fusion."                  │
   └────────────────────────────────────────────────────────────────┘
                              ↓
   ┌─ Step 3: 同 node ranks 互换自己的 XML ────────────────────────┐
   │   bootstrapIntraNodeAllGather(local rank, local XML, ...)     │
   │                                                                │
   │   ★ 用的是 IntraNode 版本! 不是普通 bootstrapAllGather:       │
   │     - 只在 hostHash 相同的 rank 之间交换                       │
   │     - 走 UDS (Unix Domain Socket) 不走 TCP, 更快               │
   │     - 跨 node 不交换 (跨 node 的部分留给 IB/NET 后面去发现)    │
   └────────────────────────────────────────────────────────────────┘
                              ↓
   ┌─ Step 4: 融合所有本地 rank 的 XML ─────────────────────────────┐
   │   for each peer_xml in local_ranks:                            │
   │     ncclTopoFuseXml(my_xml, peer_xml)                          │
   │                                                                 │
   │   融合后, 每个本地 rank 都拿到包含本 node 所有 GPU+NIC 的       │
   │   完整 XML (内容相同)                                          │
   └────────────────────────────────────────────────────────────────┘
                              ↓
   ┌─ Step 5: (可选) 转 XML 落盘 ──────────────────────────────────┐
   │   if (NCCL_TOPO_DUMP_FILE)                                     │
   │     ncclTopoDumpXmlToFile(...)  ←下次跑直接读                  │
   └────────────────────────────────────────────────────────────────┘
                              ↓
   ┌─ Step 6: XML → 内存图结构 ────────────────────────────────────┐
   │   ncclTopoGetSystemFromXml(xml, &ncclTopoSystem, hostHash)     │
   │   → 得到内存里的 ncclTopoSystem 图                             │
   └────────────────────────────────────────────────────────────────┘
```

**所以同 node 两个 rank 会不会重复**：
- ❌ 不会重复全部 /sys 扫描——每个 rank 只 fill 自己管的 GPU + 自己看到的 NIC
- ✅ 但 NCCL **乐于让 PCIe 父节点树等公共部分被各 rank 各自走一遍**——XML fusion 会去重，多探一遍开销也只 ms 级。读 nvml 也是各 rank 自己读（每张 GPU 一个 handle）
- ✅ 真正去重的关键是：**每个 rank 出的 XML 只包含自己 GPU + 自己 NIC 的子图**，靠 bootstrapIntraNodeAllGather 拼起来

#### 一张图：8 卡同 node 的拓扑发现

```
      Node aig-a100 (hostHash=0xabcd, 8 ranks)
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │  rank 0   rank 1   rank 2   rank 3  ...  rank 7                     │
   │  ┌────┐  ┌────┐  ┌────┐  ┌────┐       ┌────┐                       │
   │  │探  │  │探  │  │探  │  │探  │       │探  │                       │
   │  │GPU0│  │GPU1│  │GPU2│  │GPU3│       │GPU7│  ←本地探, 各探一张    │
   │  │+NIC│  │+NIC│  │+NIC│  │+NIC│       │+NIC│   (PCIe 父链可能重复) │
   │  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘       └─┬──┘                       │
   │    │       │       │       │             │                          │
   │    └───────┴───────┴─bootstrapIntraNodeAllGather────────────────┐   │
   │             (UDS, 不走 TCP)                                     │   │
   │                                                                  │   │
   │    rank 0 收到 [GPU0,GPU1,GPU2,GPU3,...,GPU7] 8 份 XML 子图     │   │
   │    rank 1 收到 [GPU0,GPU1,GPU2,GPU3,...,GPU7] 8 份 XML 子图     │   │
   │    ...    (各 rank 都收同样的 8 份)                              │   │
   │                                                                  │   │
   │    本地 ncclTopoFuseXml() 合并 → 完整 ncclTopoSystem (各 rank   │   │
   │    内存里都是一份相同的图)                                       │   │
   │                                                                  │   │
   └─────────────────────────────────────────────────────────────────────┘

   跨 node 怎么办?
   ─────────────
   bootstrapIntraNodeAllGather 只在同 hostHash 的 rank 间做.
   跨 node 的拓扑不在 Phase B 处理 — 各 node 的 NCCL 后面用网络
   transport (IB) 实际连通时再"发现"对端.
   每个 node 自己保存自己的拓扑视图, NCCL 用 hostHash 字段在
   ncclTopoSystem 里区分跨 node 的"节点".
```

#### `ncclTopoSystem` 的样子（最终的拓扑表示）

源码 `nccl/src/graph/topo.h:155`：

```c
struct ncclTopoSystem {
    int systemId;
    uint64_t hostHashes[NCCL_TOPO_MAX_NODES];   // 每个 node 的 hostHash
    int nHosts;
    struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];  // 6 大类节点
    float maxBw, totalBw;
};
```

6 类节点：

| Type | 含义 | 例子 |
|---|---|---|
| `GPU = 0` | NVIDIA GPU | A100 / H100 |
| `PCI = 1` | PCIe 桥 / switch | PLX / PCIe root complex |
| `NVS = 2` | NVSwitch | DGX A100 上的 6 个 NVSwitch |
| `CPU = 3` | NUMA 域（不是物理 CPU socket）| socket 0 / socket 1 |
| `NIC = 4` | 网卡硬件 | bnxt_re0, mlx5_0 |
| `NET = 5` | NCCL 看到的网络端点 | port 0, port 1 |

节点之间用 `ncclTopoLink` 连边，link 类型：

| Link type | 数值 | 含义 |
|---|---|---|
| `LINK_LOC` | 0 | 自己到自己 |
| `LINK_NVL` | 1 | NVLink |
| `LINK_PCI` | 3 | PCIe |
| `LINK_SYS` | 7 | 跨 NUMA / QPI / UPI |
| `LINK_NET` | 8 | 网络 |

**算路径时**还多出一组 "PATH 类型"（多跳的）：

| Path | 含义 |
|---|---|
| `PATH_LOC = 0` | 本地 |
| `PATH_NVL = 1` | 一跳 NVLink |
| `PATH_NVB = 2` | NVLink 经一个中间 GPU 中转 |
| `PATH_PIX = 3` | 一个 PCIe 桥 |
| `PATH_PXB = 4` | 多个 PCIe 桥（不过 host bridge）|
| `PATH_PXN = 5` | GPU 经过另一个本地 GPU 转发到 NIC（rail-local）|
| `PATH_PHB = 6` | 经过 PCIe host bridge（即过 CPU 根桥）|
| `PATH_SYS = 7` | 跨 NUMA 域 |
| `PATH_NET = 8` | 走网络 |
| `PATH_DIS = 9` | 不通 |

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH` 看到的 `(1/240.0/NVL)`、`(2/24.0/PHB)` 等就是 (跳数 / 带宽 / 路径类型) 的缩写。

#### 本机实测的拓扑 XML (NCCL_TOPO_DUMP_FILE 真实产物)

> 真实文件在 [`logs/aig-a100-topo.xml`](../my_nccl_test/all_reduce_2proc/logs/aig-a100-topo.xml)
> （25 行，1.7 KB，rank 0 在 init 时 dump 出来的最终融合 XML）

简化后结构：

```xml
<system version="1">
  <cpu numaid="0" host_hash="0x40c83cfe216cc077" affinity=... arch="x86_64" vendor="GenuineIntel">
    <pci busid="0000:04:00.0" link_speed="32.0 GT/s PCIe" link_width="16">   <!-- PCIe root -->
      <pci busid="0000:06:00.0" link_speed="5.0 GT/s" link_width="16">       <!-- PLX switch -->
        <pci busid="0000:08:00.0" link_speed="16.0 GT/s PCIe">               <!-- GPU 0 endpoint -->
          <gpu dev="0" sm="80" rank="0" gdr="1">
            <nvlink target="0000:09:00.0" count="12" tclass="0x030200"/>     <!-- 12 条 NVLink to GPU 1 -->
          </gpu>
        </pci>
        <pci busid="0000:09:00.0" link_speed="16.0 GT/s PCIe">               <!-- GPU 1 endpoint -->
          <gpu dev="1" sm="80" rank="1" gdr="1">
            <nvlink target="0000:08:00.0" count="12"/>                        <!-- 反向 12 条 NVLink -->
          </gpu>
        </pci>
      </pci>
    </pci>
  </cpu>
  <cpu numaid="1" host_hash="0x40c83cfe216cc077">
    <pci busid="0000:d4:00.0" link_speed="8.0 GT/s PCIe" link_width="8">
      <nic>
        <net name="bnxt_re0" dev="0" speed="2500" gdr="0"/>                  <!-- RoCE 网卡, gdr=0 -->
      </nic>
    </pci>
  </cpu>
</system>
```

可以注意几件事：
- **两个 `<cpu>` 节点** = 两个 NUMA 域（看 `affinity` 位图能反推 socket 拓扑）
- **GPU 0/1 挂在 NUMA 0 下**，通过两层 PCIe switch (`0x1000:c030` = LSI/PLX 桥) 连到 root
- **`<nvlink target=... count="12">`** —— A100 SXM4 之间 12 条 NVLink link，每条 25 GB/s ≈ 总 300 GB/s（理论），NCCL 算的 maxBw 240 GB/s 是实测有效值
- **NIC 在 NUMA 1**，跟 GPU 不同 socket —— 如果走 IB 跨机数据通路，需要跨 SYS link（QPI/UPI），所以 GDR 默认 disable
- **`host_hash="0x40c83cfe216cc077"`** —— 这就是 `ncclPeerInfo.hostHash` 那个值，整 system 范围内唯一

用 xq 玩一玩：

```bash
xq '.' aig-a100-topo.xml              # JSON 化后看树
xq '..|.gpu?|select(.!=null)' aig-a100-topo.xml   # 列所有 GPU 节点
xq '..|.["@busid"]?|select(.!=null)' aig-a100-topo.xml   # 列所有 PCIe busid
xq '.system.cpu | length' aig-a100-topo.xml            # 几个 NUMA 域
```

下次再起 NCCL 直接：

```bash
NCCL_TOPO_FILE=aig-a100-topo.xml ./run.sh    # 跳过 Phase B 探测
```

#### 本机的内存表示 ncclTopoSystem (从 trace_all.log 反推, 跟 XML 一一对应)

```
   ncclTopoSystem (aig-a100, 2 卡 A100 + 1 IB)
   ════════════════════════════════════════════════

   hostHashes = [0xabcd...]   (本 host 一个)
   nHosts     = 1

   nodes[CPU]  count=2:
                ├─ CPU/0-0  (NUMA 0)  ←─┐
                └─ CPU/0-1  (NUMA 1) ←─┐│
                          │            │└──SYS 10 GB/s
                          └────────────┘  (跨 socket QPI)

   nodes[PCI]  count=1:
                └─ PCI/0-4000  (PCIe root, NUMA 0 下)
                          │
                       ┌──┴──┐
                  PCI[24]  PCI[24]     ← PCIe Gen4 x16, 24 GB/s
                       │     │
   nodes[GPU]  count=2:│     │
                ├─ GPU/0-8000 (A100, GPU 0)
                │             │
                │             └─NVL[240]─┐
                │                        │  ← NVLink, 240 GB/s
                └─ GPU/0-9000 (A100, GPU 1)
                              │
                              └─NVL[240]─┘

   nodes[NIC]  count=1:
                └─ NIC/0-d4000 (Broadcom RoCE, NUMA 1 下)

   nodes[NET]  count=1:
                └─ NET (port 0 on bnxt_re0)

   nodes[NVS]  count=0   (A100 SXM4 板没装 NVSwitch, 直连)

   maxBw   = 240.0     (NVLink)
   totalBw = 240.0
```

把 trace_all.log 第 20-32 行那段树状打印对回去，每一行都能在上面这张图里定位到。

#### 缓存机制：复用拓扑的几种方式

| 方法 | 用法 | 适用场景 |
|---|---|---|
| `NCCL_TOPO_FILE=path/to/topo.xml` | 启动前指向预先 dump 好的 XML | CI/生产固定机型，加速 init |
| `/var/run/nvidia-topologyd/virtualTopology.xml` | 默认查询路径（无需任何 env）| 有 `nvidia-topologyd` 守护进程的机器 |
| `NCCL_TOPO_DUMP_FILE=path` | 当前 run **导出**所有 rank 融合后的 XML | 第一次跑后给后续用 |
| `NCCL_TOPO_DUMP_FILE_RANK=N` | 指定哪个 rank 来 dump（默认 0） | 多 rank 时避免大家都写 |

工作流：

```bash
# 第一次跑, 让 rank 0 把拓扑导出
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml ./run.sh

# 之后所有跑用这份缓存
NCCL_TOPO_FILE=/tmp/nccl_topo.xml ./run.sh
```

缓存后 Phase B **整段跳过**——直接读 XML、不动 /sys、不发 bootstrap intra-node allgather。生产里频繁起 NCCL（PyTorch DDP 重启）能省几十 ms。

### Phase C: 算法图搜索（完全本地）

各 rank **完全本地**完成（这一段真的不走 bootstrap）：

```
   ┌─ 在 Phase B 的 ncclTopoSystem 上, 跑 4 套算法 ─────────────┐
   │   ncclTopoCompute(topo, ringGraph)    Pattern=RING         │
   │   ncclTopoCompute(topo, treeGraph)    Pattern=TREE         │
   │   ncclTopoCompute(topo, collNet...)   Pattern=...          │
   │   ncclTopoCompute(topo, nvlsGraph)    Pattern=NVLS         │
   │   ↓                                                         │
   │   每个 ncclTopoGraph 包含: nChannels, bwIntra/Inter,       │
   │   typeIntra/Inter (NVL/PIX/PHB/SYS), pattern, sameChannels │
   └─────────────────────────────────────────────────────────────┘
```

因为每个 rank 都用同样的算法 + Phase A/B 后大家拿到的 ncclTopoSystem 相同，**理论上各 rank 算出来的图一致**。

但为了**实测确认一致并对齐元数据**，仍需要下一步 Phase D。

### Phase D: AllGather3 — 把每 rank 算的图汇总

源码 `init.cc:749`：

```c
struct graphInfo {                   // 8 个 int, 32B
    int pattern;                     // RING / TREE / NVLS ...
    int nChannels;                   // 这个 algorithm 用几个 channel
    int sameChannels;                // 各 channel 是否相同结构
    float bwIntra, bwInter;          // 节点内 / 节点间带宽
    int typeIntra, typeInter;        // NVL/PIX/PHB/SYS
    int crossNic;                    // 是否跨网卡
};

struct ncclTopoRanks {               // 每 channel 一条
    int ringRecv[MAXCHANNELS];       // 我在 channel c 的环里, 从谁收
    int ringSend[MAXCHANNELS];       // 我在 channel c 的环里, 给谁发
    int ringPrev[MAXCHANNELS];       // 环的上游邻居
    int ringNext[MAXCHANNELS];       // 环的下游邻居
    int treeToParent[MAXCHANNELS];   // 树的父节点 rank
    int treeToChild0[MAXCHANNELS];   // 子节点 0
    int treeToChild1[MAXCHANNELS];   // 子节点 1
    int nvlsHeads[MAXCHANNELS];      // NVLS 多播头节点
    int nvlsHeadNum;
};

struct allGatherInfo {
    struct graphInfo  graphInfo[NCCL_NUM_ALGORITHMS];  // 4 套 algo × 32B = 128B
    struct ncclTopoRanks topoRanks;                    // ~3.6 KB (MAXCHANNELS=32)
    int cpuArch;                                       // x86 / arm / ppc
    int cpuVendor;                                     // intel / amd / hygon
};
```

每 rank ~3.8 KB。汇总后大家：
- 知道全局 nNodes、rankToNode 映射（从 hostHash 算出来的）
- 检查 CPU 异构（不同架构会 WARN）
- 形成全局的 ring/tree 拓扑（每个 rank 的邻居都对齐）

### Phase E: CollNet (可选，仅当支持 SHARP/NVLS Multicast)

```
collnet 路径下还有 3 次 bootstrap allgather (coll_net.cc:1242, 1306, 1386):
  - userToDenseRank 映射 (int per rank, 4B)
  - collnetShareInfo (跨 NCCL communicator 共享 collnet head 的信息)
  - collnet 兼容性矩阵
```

普通用户场景一般跑不到。

### Phase F: 逐 peer 点对点 handle 交换 (`ncclConnect`)

**这是 step 5 里数据量最大、最讲究"按拓扑分情况"的一段**。源码在 `transport.cc:100` `ncclTransportP2pSetup`：

```c
#define CONNECT_SIZE 256
struct ncclConnect {
    char data[CONNECT_SIZE];   // 不透明 256B blob, 内容由具体 transport 填
};
```

每对 (myRank, peerRank) × 每个 channel × 每个方向（send/recv），都要塞一个 256B 的 ncclConnect。**不再是 allgather**——是按"轮"做的 send/recv：

```c
for (int i=1; i<comm->nRanks; i++) {
    recvPeer = (rank - i + nRanks) % nRanks;
    sendPeer = (rank + i) % nRanks;
    bootstrapSend(comm->bootstrap, recvPeer, tag, data, len);
    bootstrapSend(comm->bootstrap, sendPeer, tag, data, len);
    bootstrapRecv(comm->bootstrap, recvPeer, tag, data, len);
    bootstrapRecv(comm->bootstrap, sendPeer, tag, data, len);
}
```

每"轮"`i` 跟距离为 `i` 的两个邻居（左右环）交换；ring 越大，轮数越多。

**256 字节里填什么 → 看选的是哪种 transport**：

| transport | 填的内容 (`p2pConnectInfo` / `shmConnectInfo` / `ncclIbConnectionMetadata`) |
|---|---|
| **P2P (CUDA IPC / CUMEM)** | `rank, read, p2pBuff (ncclP2pBuff: handle + size + offset), desc` —— 对方拿这个 IPC handle 直接 mmap 自己的显存 |
| **SHM** | `ncclShmIpcDesc_t (shm fd + size), shmBuffInfo` —— 通过 UDS 传 fd，建一段同机共享内存 |
| **IB / RoCE** | `ncclIbQpInfo[N] (qpn, ece) + ncclIbDevInfo[N] (lid, gid, mtu, link_layer, fifoRkey, remoteGid) + fifoAddr + devName + ndevs` —— 完整的 IB QP 信息，对端拿来做 `ibv_modify_qp` 进 RTR/RTS state |
| **NET sockets (fallback)** | 仅 IP+port，简单 |

```
   Phase F: 256B 包里填啥, 完全看 transport 选了谁

   ┌─────────────── ncclConnect (256B) ───────────────┐
   │                                                  │
   │  case P2P (单机 NVLink/PCIe):                    │
   │    ┌─ p2pConnectInfo ──────────────────────┐    │
   │    │ rank, read                            │    │
   │    │ p2pBuff { IPC handle, size, offset }  │    │
   │    │ desc { extra IPC desc }               │    │
   │    └───────────────────────────────────────┘    │
   │                                                  │
   │  case SHM (同机 fallback):                       │
   │    ┌─ shmConnectInfo ──────────────────────┐    │
   │    │ ncclShmIpcDesc { fd, size }           │    │
   │    │ shmBuffInfo { offsets }               │    │
   │    └───────────────────────────────────────┘    │
   │                                                  │
   │  case IB (跨机):                                  │
   │    ┌─ ncclIbConnectionMetadata ────────────┐    │
   │    │ qpInfo[N] { qpn, ece }                │    │
   │    │ devs[N]   { lid, gid, mtu, rkey,...}  │    │
   │    │ fifoAddr, devName, ndevs              │    │
   │    └───────────────────────────────────────┘    │
   │                                                  │
   └──────────────────────────────────────────────────┘
```

### 一张表：Step 5 在 6 种场景下的具体差异

| 维度 | 单进程多卡 | 单机 PCIe | 单机 NVLink | 跨机 IB | MNNVL | 多 comm split |
|---|---|---|---|---|---|---|
| **Phase A** (peerInfo) | 跑（同进程内）| 跑 | 跑 | 跑 | 跑 | 父 comm 已有, 仅子集 |
| `hostHash` 关系 | 全 rank 相同 | 全 rank 相同 | 全 rank 相同 | 跨机 rank 不同 | 跨机 rank 不同 | 取决于子组 |
| `pidHash` 关系 | 全 rank 相同 | 全 rank 不同 | 全 rank 不同 | 跨机/同机均不同 | 跨机/同机均不同 | 取决于子组 |
| `fabricInfo.cliqueId` | 同 (但没用) | 同 (但没用) | 同 (但没用) | 不同 (但没 MNNVL) | **同 → 触发 MNNVL P2P** | 取决于父 comm |
| **Phase B** (本地拓扑) | 跑 | 跑 | 跑 | 跑 | 跑 + fabric | 复用父 |
| **Phase C** (算图) | Ring only | Ring | Ring + Tree | Ring + Tree + CollNet | + NVLS | 复用父 |
| **Phase D** (graphInfo) | 跑（结果平凡）| 跑 | 跑 | 跑（节点感知）| 跑（clique 感知）| 仅子集 |
| **Phase E** (CollNet) | 否 | 否 | 否 | 否 (一般) | 是 (NVLS) | 看父 |
| **Phase F** ncclConnect 内容 | p2pBuff (进程内指针) | p2pConnectInfo (CUDA IPC handle) | p2pConnectInfo (CUMEM handle) | 大头：ncclIbConnectionMetadata | P2P/NVLS handle (跨主机！) | 复用父连接, 仅 reconfig |
| 单次 Phase F 大小 | 256B (用一点点) | 256B (满载 IPC handle) | 256B (满载 CUMEM handle) | 256B (满载 IB QP+GID) | 256B (NVLS 多播 handle) | 256B (描述子集) |
| Phase F 总轮数 | (N-1) | (N-1) | (N-1) | (N-1) | (N-1) | 复用 |

### 一张图：4 种部署形态下 Phase F 真实交换路径

```
   场景 1: 单进程多卡 (N=2)
   ───────────────────────
   process
   ┌────────────────────┐
   │ rank 0 ─┐          │   ncclConnect 256B 在进程内
   │         │  ←━━━━→  │   只复制结构体, 没真"传"
   │ rank 1 ─┘          │
   └────────────────────┘


   场景 2/3: 单机多进程 (N=2, 同机不同进程)
   ───────────────────────
   process 0       process 1
   ┌─────────┐    ┌─────────┐
   │ rank 0  │    │ rank 1  │
   │ GPU 0   │    │ GPU 1   │
   └────┬────┘    └────┬────┘
        │              │
        └──TCP 同机────┘
           bootstrap     ← Phase F 经 bootstrap socket
           交换 IPC handle (256B)
                          ↓
   建好后 → GPU 直接通过 IPC mmap 对方显存
            GPU 0 ═══ NVLink/PCIe ═══ GPU 1


   场景 4: 跨机 IB (N=2, 两机各一卡)
   ───────────────────────────
   Node A                          Node B
   ┌─────────┐                     ┌─────────┐
   │ rank 0  │                     │ rank 1  │
   │ GPU 0   │                     │ GPU 0   │
   └────┬────┘                     └────┬────┘
        │                               │
        └─── TCP 跨机 (OOB) ────────────┘
              bootstrap socket
              交换 IB QP/GID/MR (256B)
                                  ↓
        建好后 → IB Verbs 建 RDMA 连接
                GPU → IB HCA  ═══════ IB Switch ═══════ IB HCA → GPU
                (GPUDirect RDMA, 如果支持)


   场景 5: MNNVL (N=2, 跨 tray 但同 NVLink fabric)
   ────────────────────────────────────────
   Tray A                          Tray B
   ┌─────────┐                     ┌─────────┐
   │ rank 0  │                     │ rank 1  │
   │ GPU 0   │                     │ GPU 0   │
   │ clique=42                     │ clique=42 ← 同 cliqueId!
   └────┬────┘                     └────┬────┘
        │                               │
        └─── TCP 跨机 (OOB) ────────────┘
              bootstrap socket
              交换 NVLS 多播 handle (256B)
                                  ↓
        建好后 → 通过跨节点 NVLink 直接 P2P / NVLS multicast
                GPU 0 ═══ NVLink fabric (跨 tray!) ═══ GPU 0
                (240 GB/s 级, 不走 IB)
```

### Step 5 一次 init 的总字节预算 (2 rank 估算)

| Phase | 内容 | 数据量 |
|---|---|---|
| A (ncclPeerInfo allgather) | 2 × 80B = 160B | <1 KB |
| D (allGatherInfo allgather) | 2 × ~3.8 KB ≈ 7.6 KB | ~8 KB |
| F (per-peer ncclConnect) | 1 轮 × 2 方向 × 24 channel × 256B = 12 KB | ~12 KB |
| **总计** | | **~20 KB** |

跨机 IB 8 卡 × 2 机 (N=16) 的话：
| Phase | 数据量 |
|---|---|
| A | 16 × 80B = 1.3 KB |
| D | 16 × 3.8 KB ≈ 60 KB |
| F | 15 轮 × 2 × 24 × 256B ≈ 180 KB（per rank，每个 rank 都做） |
| **总计 per rank** | **~240 KB** |

这就是 bootstrap socket 上跑的全部数据。**比起后续每秒几十 GB 的实际通信，是几乎可以忽略的开销**——但所有 fast path 都建立在这 240 KB 元数据正确传完的基础上。

### 调 step 5 怎么 debug

```bash
# 只看元数据相关子系统:
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,BOOTSTRAP ./run.sh

# 看 GRAPH 子系统能确认 Phase B/C/D 的结果
# 看 NET/IB 子系统能确认 Phase F 的 ncclIbConnectionMetadata 交换
```

源码定位表（想自己读源码）：

| Phase | 文件 : 行 |
|---|---|
| A: ncclPeerInfo allgather | `nccl/src/init.cc:770` |
| A: ncclPeerInfo struct | `nccl/src/include/transport.h:37` |
| B: 拓扑探测入口 | `nccl/src/graph/topo.cc:728` `ncclTopoGetSystem` |
| B: 同 node fusion 关键调用 | `nccl/src/graph/topo.cc` `bootstrapIntraNodeAllGather` + `ncclTopoFuseXml` |
| B: ncclTopoSystem 定义 | `nccl/src/graph/topo.h:155` |
| B: ncclTopoNode 定义 | `nccl/src/graph/topo.h:111` |
| B: 节点类型 + Link/Path 类型枚举 | `nccl/src/graph/topo.h:32-82` |
| C: 算法图搜索 | `nccl/src/init.cc:891-944` |
| D: allGatherInfo allgather | `nccl/src/init.cc:971` |
| D: 结构体定义 | `nccl/src/init.cc:737-754` |
| F: 逐 peer 交换主循环 | `nccl/src/transport.cc:100` `ncclTransportP2pSetup` |
| F: 各 transport 填 256B | `nccl/src/transport/p2p.cc`, `shm.cc`, `net_ib.cc` |
| F: P2P handle 结构 | `nccl/src/transport/p2p.cc:29` `p2pConnectInfo` |
| F: IB handle 结构 | `nccl/src/transport/net_ib.cc:785-818` `ncclIbQpInfo` + `ncclIbConnectionMetadata` |

---

## 场景 1：单进程多卡（`ncclCommInitAll`）

例：`my_nccl_test/all_reduce_2gpu/` —— 1 个进程，绑 2 张同机 GPU。

```
              进程 (PID = 1234)
   ┌──────────────────────────────────────────┐
   │  ncclCommInitAll(comms[2], 2, {0,1})     │
   │  ┌──── 内部展开 ────┐                    │
   │  │ ncclGetUniqueId  │                    │
   │  │       ↓          │                    │
   │  │ Group{           │                    │
   │  │   ncclCommInit   │  ───▶ comm[0] (GPU 0)
   │  │   RankDev(0)     │                    │
   │  │   ncclCommInit   │  ───▶ comm[1] (GPU 1)
   │  │   RankDev(1)     │                    │
   │  │ }                │                    │
   │  └──────────────────┘                    │
   └──────────────────────────────────────────┘
```

**bootstrap 仍然走完整流程**——`ncclCommInitAll` 只是 `ncclCommInitRank` 的方便封装（`init.cc:1838-1846`）。但因为两 rank 都在同进程：

| step | 实际怎么发生 |
|---|---|
| 1. 选 OOB iface | 走 `findInterfaces` 选第一个非 lo/docker 的 iface（本机日志看到 `ens4f0np0:192.168.80.101`） |
| 2. rank 0 起 listener | 起在本机某个端口 |
| 3. UID 分发 | 直接传指针，**不用走文件 / MPI** |
| 4. rendezvous | TCP connect 走 localhost / 网卡 loopback 路径 |
| 5. 元数据 allgather | `ncclPeerInfo` 里 `hostHash` 相同 → NCCL 后续直接走 **CUDA IPC** transport |
| 6. ring 形成 | 同进程里建几条 TCP 也照建（开销 ~ms 级，不痛） |

**特殊点**：因为同进程，所有 GPU 都在一个 CUDA context 下，**CUDA IPC handle 都不需要走 fd-passing**——直接拿到对方的指针即可。bootstrap 阶段不用做 IPC handle 交换的繁琐步骤。

**用户层 API 简化**：你只要给 `ncclCommInitAll` 的 device list，UID 自动管完。

---

## 场景 2：单机多进程，PCIe-only（无 NVLink）

例：旧的工作站，2 张 RTX 30xx 通过主板 PCIe 互联，没 NVSwitch。

```
        Host (1 个 OS, 1 个 hostname, 1 个 ifaddr)
   ┌─────────────────────────────────────────────────┐
   │                                                 │
   │   Process 0 (PID 100)        Process 1 (PID 101)│
   │   ┌────────────┐             ┌────────────┐    │
   │   │  rank 0    │             │  rank 1    │    │
   │   │  GPU 0     │             │  GPU 1     │    │
   │   └────┬───────┘             └────┬───────┘    │
   │        │                          │            │
   │        │  bootstrap TCP           │            │
   │        │  via lo / eth0           │            │
   │        └───────────┬──────────────┘            │
   │                    │                            │
   │   ─────────────────┴────────────────────        │
   │     CPU / PCIe Switch                            │
   │          GPU 0  ━━━ PCIe Gen4 ━━━ GPU 1         │
   │              (P2P via PCIe BAR1)                 │
   └─────────────────────────────────────────────────┘
```

bootstrap 流程：
- step 1 (OOB iface)：通常自动选 eth0（或本地用户的网卡）。**两边 IP 不一样但同子网**。
- step 2-4 (rendezvous)：TCP 经内核 loopback fast-path（内核认得 dst 是本机 IP，不走网卡 PHY，但仍走 socket 协议栈）。
- step 5 (allgather)：**关键**——交换 `ncclPeerInfo`，包括：
  - `hostHash`：基于 hostname 算的 64 位哈希。两 rank 相同 → 同机。
  - `pidHash`：基于 PID。两 rank 不同 → 不同进程。
  - `busId`：PCIe bus ID。
  - `cudaDev`：GPU 序号。
- step 5 续 (transport handle 交换)：因为同机不同进程，NCCL 选 **P2P transport (CUDA IPC)**——rank 0 拿到自己 GPU buffer 的 `cudaIpcGetMemHandle`，通过 bootstrap TCP 把 64 字节 handle 发给 rank 1，rank 1 用 `cudaIpcOpenMemHandle` 把对端 buffer 映射到自己地址空间。
- step 6：bootstrap ring 闭合。

**P2P 走不走 NVLink 是后面 channel 建立时决定的**——本场景因为没 NVLink，最终走 PCIe BAR1，但 bootstrap 不关心，bootstrap 只把 handle 送到。

```
   bootstrap 期 (TCP)          init 完成后 (NCCL channel)
   ─────────────────           ─────────────────────────
   rank 0 ─ TCP ─ rank 1       GPU 0 ═══ PCIe P2P ═══ GPU 1
                                          (BAR1)
                                       慢 ~16 GB/s
```

---

## 场景 3：单机多进程，NVLink + NVSwitch（最典型）

例：你这台 `aig-a100`，2 × A100-SXM4 via NVSwitch。也是 `all_reduce_2proc` 实测的场景。

```
        Host aig-a100
   ┌────────────────────────────────────────────────────┐
   │  Process 0 (rank 0, GPU 0)    Process 1 (rank 1, GPU 1)
   │           │                          │
   │           │   bootstrap TCP          │
   │           └──── ens4f0np0 ───────────┘
   │                                                    │
   │      ─────────────────────────────                 │
   │       NVSwitch                                     │
   │         ║                ║                         │
   │         ║  240 GB/s      ║                         │
   │       GPU 0  ═══════════ GPU 1                     │
   └────────────────────────────────────────────────────┘
```

bootstrap 做的事跟场景 2 一样，但 step 5 里 ncclPeerInfo 多出来的字段决定了 transport 不一样：
- `nvmlDev`：NVML 设备 ID，用来查 NVLink 拓扑
- 通过 bootstrap allgather 后，每 rank 都能算出全局 NVLink 连通矩阵

```
   bootstrap 期 (TCP)              init 完成后 (NCCL channel)
   ─────────────────              ────────────────────────────
   rank 0 ─ TCP ─ rank 1          GPU 0 ═══ NVLink/NVSwitch ═══ GPU 1
                                          (CUMEM 映射)
                                       快 ~240 GB/s
```

`NCCL_DEBUG=INFO` 里 `via P2P/CUMEM/read` 那行的 CUMEM 后端 vs 场景 2 的 CUDA IPC，**底层都靠 bootstrap 阶段交换 64 字节 handle**——只是后端选择不同（CUMEM 比老的 cudaIpc API 高效，A100+ 默认走 CUMEM）。

---

## 场景 4：跨机多进程，IP + IB/RoCE（生产典型）

例：2 台机器，每机 8 卡，共 16 rank。

```
   Node A (rank 0..7)                        Node B (rank 8..15)
   ┌─────────────────────────┐               ┌─────────────────────────┐
   │ p0 p1 p2 p3 p4 p5 p6 p7 │               │ p8 ... p15              │
   │  │  │   ........        │               │  │      ........        │
   │  └──┴──── bootstrap ────┐               ┌─────┘                    │
   │  (主机内 lo / eth0)     │               │                          │
   │                         │               │                          │
   │  GPU0─NVL─...─GPU7      │               │  GPU0─NVL─...─GPU7      │
   │           ║             │               │           ║              │
   │           IB HCA ═══════╪═══ IB switch ═╪═══════ IB HCA            │
   └─────────────────────────┘               └─────────────────────────┘
                            ▲                                ▲
                            │                                │
                  bootstrap 跨机也走它?                       │
                  ───────────────────────┴────────────────────┘
                            通常走 IB 的 IPoIB / 或独立的 mgmt 网, 不走数据 IB
```

bootstrap 的关键变化：

### A. OOB iface 选择变讲究

源码 `socket.cc:324` 的优先级：

```
1. NCCL_SOCKET_IFNAME 显式指定        ← 生产建议
2. ib*                              ← IPoIB
3. NCCL_COMM_ID 隐含的 subnet
4. 任何 non-docker non-lo iface
5. docker
6. lo
```

跨机时两机的 OOB iface IP 必须**同子网可路由**，否则 rank 8 connect 不到 rank 0。`NCCL_SOCKET_IFNAME=eth0` / `=ib0` 是常见配置。

### B. UID 分发不能再用文件

跨机文件系统不共享（或共享但慢/不可靠）。生产里几乎一定走：

```
方式 A: MPI                          方式 B: PyTorch
─────────                            ───────────────
                                     
rank 0:                              rank 0:
  ncclGetUniqueId(&id)                 ncclGetUniqueId(&id)
  MPI_Bcast(&id, 128, MPI_BYTE,        store.set("nccl_uid", id_bytes)
            0, MPI_COMM_WORLD)         
                                     rank >0:
rank >0:                               id_bytes = store.get("nccl_uid")
  (id 已经在 MPI_Bcast buffer 里)     
                                     共享 store: TCPStore / FileStore / etcd
```

PyTorch 的 `torch.distributed.init_process_group` 底下就是这套——`init_method=tcp://addr:port` 就是 TCPStore；`init_method=file://path` 是 FileStore（要 NFS）。

### C. step 5 (元数据 allgather) 内容大幅扩展

跨机要交换的多出来：
- **IB GID / LID / QP info**：IB Queue Pair 信息，用来跟对端建 RDMA 连接。每对 (rank, peer) 都要交换。
- **网络后端的 transport handle**：比 CUDA IPC 多一倍，因为跨机 P2P 用 GDR（GPU Direct RDMA）需要在 IB 注册 GPU 显存的 MR (Memory Region)。
- **`hostHash` 不同的 rank** → NCCL 选 **NET transport (IB/RoCE)** 而不是 P2P。同机的 rank 仍然走 P2P。

### D. bootstrap ring 跨机闭合

```
   Node A                                              Node B
   ┌──────────────────────────────┐                   ┌────────────────────────────────┐
   │ r0 → r1 → r2 → ... → r7 ────────────────────────→ r8 → r9 → ... → r15 ────┐       │
   │ ▲                            │   (跨机 TCP)      │                          │       │
   │ └──────────────────────────────────────────────────────────────────────────┘       │
   └──────────────────────────────┘                   └────────────────────────────────┘
```

bootstrap ring 是逻辑环（每个 rank 连下一个），所以跨机时其中**至少一条 TCP 跨主机**（实际不只一条，n 跨机连接数取决于 NCCL 的 ring 排布）。

### E. NCCL channel 形成时的不同 transport

```
                                            数据通路
                                            ─────────────
   bootstrap 期间 (TCP via OOB)             同机 P2P:    GPU ━━ NVLink ━━ GPU (CUMEM)
   ──────────────────────────────           跨机 NET:    GPU → IB HCA → switch
   rank 0 ←─ TCP ─→ rank N (任意)                       → IB HCA → GPU (GDR)
   交换 IB QP info / IPC handle             
                                            网络后端: NET/IB plugin
                                            ────────────────────
                                            写 IB Verbs API:
                                              ibv_post_send/recv
                                              对端的 QP RTR/RTS state
```

---

## 场景 5：多机 NVLink fabric（MNNVL：DGX H100 NVL / GB200 NVL72）

例：GB200 NVL72 一柜 72 卡，跨 18 台机箱通过 NVLink Switch 互联。

```
    NVLink Switch Tray (18 个 GH200)
     ║   ║   ║   ║   ║   ║   ║   ║
   ──╫───╫───╫───╫───╫───╫───╫───╫── NVLink fabric (跨节点!)
     ║   ║   ║   ║   ║   ║   ║   ║
    Tray 1  Tray 2  ...   Tray N
    GPU x 4 ........ (72 卡总共)
```

bootstrap 的扩展：

### A. ncclPeerInfo 多出 fabric 字段

源码 `init.cc:562-575` 看到的：
```c
// MNNVL: Request the fabric UUID and partition info
INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x ...",
     busId, fabricInfo.uuidLow, fabricInfo.uuidHigh, fabricInfo.cliqueId, ...);
```

bootstrap 阶段每个 rank 上报自己所在的 **fabric UUID** + **clique ID**。同一个 cliqueId 的 rank 之间是 NVLink 可达的（哪怕在不同主机）。

`comm.h:457`：
```c
int MNNVL;                  // true when MNNVL is available
struct cliqueInfo clique;   // Our MNNVL clique information
int cliqueRank;             // Our rank within the MNNVL clique
```

### B. transport 选择多一档

```
   优先级 (高 → 低):
   1. P2P/NVLS  同 fabric clique + 支持 NVLink Switch Multicast → 走它 (H100 NVL+)
   2. P2P/CUMEM  同主机 + NVLink                                → 走它 (DGX A100/H100)
   3. P2P/IPC    同主机 + PCIe                                  → 走它
   4. SHM        同主机 fallback                                → 走它
   5. NET        跨主机或跨 fabric clique                       → 走 IB/RoCE
```

NCCL_DEBUG=INFO 日志的 `via P2P/CUMEM/read` 的中间字段 (`CUMEM`) 就是这套选择的结果。

### C. bootstrap socket 仍然走 IP

**这是个反直觉但重要的点**：哪怕你有跨节点 NVLink，bootstrap 仍走传统 TCP/IP。原因：
- NVLink fabric 的发现本身需要 bootstrap 先告诉每 rank "对方在哪个 clique"——这是个 chicken-and-egg。bootstrap 必须用最普适的 IP 协议起步。
- NVLink fabric 的协议栈不暴露 socket-like API；不适合做控制平面。

---

## 场景 6：同机多卡多 comm（用户起多个通信组）

例：一个进程内同时跑 DDP 和 模型并行 (TP)，需要两个独立通信组。

```
   Process (pid 1234)
   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   │  comm_dp  (data-parallel,    8 rank, ncclUniqueId A) │
   │  comm_tp  (tensor-parallel,  2 rank, ncclUniqueId B) │
   │                                                      │
   │  每个 comm 自己一份 bootstrap socket / ring          │
   │                                                      │
   └──────────────────────────────────────────────────────┘
```

`bootstrap.cc:729` 的 `bootstrapSplit` 提供了**派生子组**的快捷方式（MPI_Comm_split 等价物）——这时**子组 comm 不重新建 OOB socket**，复用父 comm 的连接打个新 session。

但如果用户**完全独立**地起两个 comm（各自 ncclGetUniqueId）：
- 各自一套 bootstrap socket
- 各自一套 ring
- 互不知道对方存在
- 内存/线程开销翻倍（多一组 Proxy Service 线程）

所以 PyTorch 的 process group 派生子组通常用 split 模式而不是独立 init。

---

## 一张总览表：场景 → bootstrap 细节差异

| 维度 | 场景 1<br>单进程多卡 | 场景 2<br>单机多进程<br>PCIe | 场景 3<br>单机多进程<br>NVLink | 场景 4<br>跨机 IB | 场景 5<br>MNNVL | 场景 6<br>多 comm |
|---|---|---|---|---|---|---|
| OOB iface | 任意 | eth0/lo | eth0/lo | **IB/eth0 跨子网** | 同跨机 | 复用父 |
| UID 分发 | 进程内传指针 | 进程内传 / 文件 | 进程内传 / 文件 | **MPI/TCPStore/etcd** | 同跨机 | bootstrapSplit |
| listener 在哪 | 进程主线程 | rank 0 进程 | rank 0 进程 | rank 0 进程（跨机） | 跨机 | 复用 |
| Step 5 交换内容 | 几乎不用 | hostHash, busId, IPC handle | + NVLink 拓扑 | + IB GID/QP, MR | + fabric UUID, cliqueId | 子组成员列表 |
| transport 选择 | CUDA IPC (同进程) | P2P/IPC (PCIe BAR1) | P2P/CUMEM (NVLink) | NET/IB + GDR (跨机) | P2P/NVLS (跨机 NVLink) | 继承父 |
| ring 形成 | 名义上的 | 同主机 TCP loopback | 同主机 TCP loopback | **跨机 TCP** | 跨机 TCP | 复用父 |
| 复杂度（init 时延） | 极低 | ~ms | ~ms | ~10s ms（网络握手） | ~10s ms | <<父 init |

---

## 实战诊断思路

| 症状 | 根因常在 bootstrap 哪一步 | 怎么查 |
|---|---|---|
| `ncclCommInitRank` 卡死 | step 4 rendezvous 失败：rank 没到、UID 没分发到 | `NCCL_DEBUG=INFO` 看哪个 rank 没打 "Init START" |
| `ncclCommInitRank` 返回 error | step 1 OOB iface 选错（不可达） | `NCCL_DEBUG=INFO` 看 `Bootstrap : Using xxx`，确认 IP 路由通 |
| 跨机能 init 但跑得慢 | step 5 选错 transport (没走 IB 走 sockets) | `NCCL_DEBUG=INFO` 看 `Using network` 行，应该是 `IB` 不是 `Socket` |
| 多机 init 时不时报 connection refused | bootstrap 端口被防火墙拦 | 临时关防火墙试，或用 `NCCL_PORT_RANGE` 指定开放端口段 |
| NCCL_DEBUG 看到选了 lo 网卡 | 自动探测被 lo 抓走 | `NCCL_SOCKET_IFNAME=eth0`（白名单）或 `=^lo,docker`（黑名单） |
| 跨机但 transport 选了 NET/Socket | 没 IB 库 / IB plugin 没装 | 装 OFED + `libnccl-net.so` 或 HPC-X |

## 实战调参手册

```bash
# 强制指定 OOB iface (生产强烈建议显式给)
export NCCL_SOCKET_IFNAME=eth0          # 白名单
export NCCL_SOCKET_IFNAME=^docker,lo    # 黑名单
export NCCL_SOCKET_IFNAME=ib0           # 用 IPoIB

# 强制走 IPv4 / IPv6
export NCCL_SOCKET_FAMILY=AF_INET       # 或 AF_INET6

# bootstrap 端口固定 (防火墙友好)
export NCCL_COMM_ID=192.168.1.10:23456  # rank 0 监听这个端口

# 多 IB HCA 选哪张
export NCCL_IB_HCA=mlx5_0,mlx5_2        # 指定 HCA 列表

# 看 bootstrap 整个过程
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET
```

---

## 引用

- 实测呼应：[`run_info.log`](../my_nccl_test/all_reduce_2proc/logs/run_info.log)、[`nccl_debug_info_walkthrough.md`](nccl_debug_info_walkthrough.md)
- 概念基础：[`concepts.md`](concepts.md)
- 源码：
  - `nccl/src/bootstrap.cc` 全文：bootstrap 14 个 API
  - `nccl/src/init.cc:1791` `ncclCommInitAll` 实现（场景 1）
  - `nccl/src/init.cc:1838-1846` Group{InitRank, InitRank, ...} 展开
  - `nccl/src/misc/socket.cc:324` OOB iface 选择（场景 4 关键）
  - `nccl/src/init.cc:536-575` MNNVL fabric/clique 信息（场景 5）
  - `nccl/src/include/comm.h:454-460` MNNVL 字段
