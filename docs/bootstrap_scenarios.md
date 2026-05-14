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
   │  Phase B   本地拓扑探测 (不走 bootstrap, 各 rank 各自读 /sys)            │
   │            生成本地 ncclTopoSystem (PCIe/NVLink/IB/NUMA 图)              │
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

### Phase B & C: 拓扑探测 + 算法图搜索（不走 bootstrap）

各 rank **本地**完成的两件事：

```
   ┌─ Phase B: 探测本地物理拓扑 ─────────────────────────┐
   │   读 /sys/class/pci/, /sys/class/nvidia*/...        │
   │   读 nvml: NVLink connectivity, fabric info         │
   │   读 IB devices: ibv_get_device_list                │
   │   ↓                                                  │
   │   生成 struct ncclTopoSystem (PCIe + NVLink + NIC   │
   │   + CPU NUMA 的图结构)                              │
   └──────────────────────────────────────────────────────┘

   ┌─ Phase C: 在拓扑上算 4 套算法图 ────────────────────┐
   │   ncclTopoCompute(topo, ringGraph)    Pattern=RING  │
   │   ncclTopoCompute(topo, treeGraph)    Pattern=TREE  │
   │   ncclTopoCompute(topo, collNet...)   Pattern=...   │
   │   ncclTopoCompute(topo, nvlsGraph)    Pattern=NVLS  │
   │   ↓                                                  │
   │   每个 ncclTopoGraph 包含: nChannels, bwIntra/Inter,│
   │   typeIntra/Inter (NVL/PIX/PHB/SYS),  pattern...    │
   └──────────────────────────────────────────────────────┘
```

这些步骤**完全本地**，不需要 bootstrap socket。但**因为每个 rank 都用同样的算法**，加上 Phase A 已经把 `ncclPeerInfo` 同步过，所以各 rank 算出来的图**理论上一致**——理论上。

为了**实测确认一致并对齐元数据**，需要 Phase D。

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
| B: 拓扑探测入口 | `nccl/src/graph/topo.cc` `ncclTopoGetSystem` |
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
