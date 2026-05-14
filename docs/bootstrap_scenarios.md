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
