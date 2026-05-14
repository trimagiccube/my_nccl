# NCCL_DEBUG=INFO 实战走读

> 把 `my_nccl_test/all_reduce_2proc` 跑出来的 `NCCL_DEBUG=INFO` 日志逐行拆开看。
> Log 原文：[`my_nccl_test/all_reduce_2proc/logs/run_info.log`](../my_nccl_test/all_reduce_2proc/logs/run_info.log)

## 测试环境

| | |
|---|---|
| 机器 | aig-a100 |
| GPU | 2 × NVIDIA A100-SXM4-40GB (busId 0x8000, 0x9000) |
| 拓扑 | 单机双卡，NVLink + PCIe |
| 网卡 | `ens4f0np0`（普通 IP，用作 bootstrap OOB） + `bnxt_re0`（RoCE，发现但未真用，因为本机数据走 NVLink P2P） |
| NCCL | 2.23.4+cuda12.9 |
| CUDA Driver | 13000 |
| 测试 | 2 进程，每进程一张卡，`ncclAllReduce(sum)`，1M float |

## 日志结构总览

NCCL_DEBUG=INFO 信息量非常大，但可以切成清晰的 10 个 phase：

```
┌────────────────────────────────────────────────────────────────┐
│ Phase 0: 进程启动 + cudaSetDevice           (你的 printf)       │
│ Phase 1: Bootstrap 网络选 OOB 接口          (line 7)            │
│ Phase 2: Plugin 探测 (net / profiler / tuner)  (line 10-12)     │
│ Phase 3: 选 transport 网络                  (line 11-13)        │
│ Phase 4: ncclCommInitRank START + commId    (line 14)           │
│ Phase 5: Bootstrap allgather + 拓扑下发     (line 15-17)        │
│ Phase 6: 24 个 Channel 的 Ring + Tree 拓扑   (line 18-42)        │
│ Phase 7: Proxy Service 线程上线             (line 44-45)        │
│ Phase 8: Init COMPLETE + 阶段耗时分解        (line 50-51)        │
│ Phase 9: P2P 连接实际建立 (Run AllReduce 时) (line 72-95)        │
│ Phase 10: Destroy                           (line 98, 127)       │
└────────────────────────────────────────────────────────────────┘
```

下面逐 phase 拆。

---

## Phase 0 — 进程启动

```
[rank 0] pid=1278747 bound to GPU 0
[rank 1] pid=1278748 bound to GPU 1
```

这两行是你 `.cu` 里的 `printf`，不是 NCCL 输出。验证 `cudaSetDevice(rank)` 已生效。

## Phase 1 — Bootstrap OOB 接口选择

```
aig-a100:1278747:1278747 [0] NCCL INFO Bootstrap : Using ens4f0np0:192.168.80.101<0>
```

**含义**：NCCL 扫了一遍可用网卡，选 `ens4f0np0`（192.168.80.101）当 **bootstrap OOB（out-of-band）** 通道。这个 socket 是 [concepts.md](concepts.md) 里说的"rank 0 的 listening socket"——用于 init 阶段交换拓扑元数据，**不传后续的实际数据**。

**前缀解码** `aig-a100:1278747:1278747 [0]`：
```
hostname : 主进程PID : 当前线程TID  [cudaDev]
aig-a100 : 1278747   : 1278747       [0]
```
线程 TID = 主 PID 时表示主线程在打。后面会看到子线程的 TID（proxy service 线程）。

## Phase 2 — Plugin 探测

```
NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal network plugin.
NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so libnccl-net.so. Using internal tuner plugin.
```

NCCL 支持三种外部插件：
- `libnccl-net.so` — 自定义网络后端（厂商的 IB 加速、SHARP 等）
- `libnccl-profiler.so` — 性能 profiler
- `libnccl-tuner.so` — 算法/buffer size 选择 tuner

都没找到 → 用内置版本。生产里 NVIDIA HPC SDK 或 Mellanox HPC-X 包会提供 net plugin。

## Phase 3 — 网络后端选择

```
NCCL INFO NET/IB : Using [0]bnxt_re0:1/RoCE [RO]; OOB ens4f0np0:192.168.80.101<0>
NCCL INFO Using network IB
```

**关键**：本机有张 Broadcom RoCE 卡 `bnxt_re0`，NCCL 就把"网络后端"选成了 IB 模式（RoCE 在 NCCL 里走 IB 协议栈）。

但是！请注意——**这只是"如果需要走网络的话用哪个"**。实际上 2 张同机 GPU 之间会走 NVLink P2P，根本不会走 IB。后面 Phase 9 会看到。

NCCL 的 transport 选择是**按 peer**做的：每对 (myRank, peerRank) 单独选最快路径。优先级大致是：
```
P2P (NVLink / PCIe BAR1)  >  SHM (共享内存)  >  NET (IB/RoCE/sockets)
```

## Phase 4 — ncclCommInitRank START

```
NCCL INFO ncclCommInitRank comm 0x5ce61e459ee0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 8000 commId 0xf90ad11ce037dc5b - Init START
                                ^^^^^^^^^^^^^^                                                       ^^^^^^^^^^^^^^^^^^^^
                                rank 0 的 comm 指针 (heap)                                          commId (commHash)
NCCL INFO ncclCommInitRank comm 0x61476c9e1c90 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 9000 commId 0xf90ad11ce037dc5b - Init START
                                ^^^^^^^^^^^^^^                                                       ^^^^^^^^^^^^^^^^^^^^
                                rank 1 的 comm 指针 (heap, 不同进程)                                同一个 commId！
```

**这就是上一段 [concepts.md](concepts.md) 里说的那个**：两个 rank 的 `ncclComm_t` **指针不同**（不同进程的堆地址），但 **`commId` 相同**——`0xf90ad11ce037dc5b` 就是通信组的"身份证号"，bootstrap 阶段从 `<magic, addr>` hash 出来的。

## Phase 5 — Bootstrap 计时 + 拓扑摘要

```
Bootstrap timings total 0.001790 (create 0.000033, send 0.000168, recv 0.001293, ring 0.000026, delay 0.000000)
Setting affinity for GPU 0 to ffffffff,00000000,ffffffff
comm 0x5ce61e459ee0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
```

- **Bootstrap timings**：bootstrap socket 上完成 allgather 总共 1.8 ms。绝大部分时间花在 `recv`。
- **CPU affinity**：根据 GPU 的 NUMA 亲和性把进程绑到对应 CPU core mask。`ffffffff,00000000,ffffffff` 是位图，表示 NUMA node 0 的 CPU。
- **拓扑摘要**：
  - `nNodes=1` 单机
  - `localRanks=2` 本机有 2 个 rank
  - `MNNVL=0` 没用 Multi-Node NVLink（仅相关 NVL72 等机型）

## Phase 6 — 24 个 Channel 的 Ring + Tree

```
Channel 00/24 : 0 1
Channel 01/24 : 0 1
...
Channel 23/24 : 0 1
```

**Ring 含义**：每个 channel 是一个独立的 ring。这里 `0 1` 表示 rank 顺序：rank 0 → rank 1 → (回到 rank 0)。24 个 channel 都是同样的 ring，因为 2 个 rank 只有一种 ring 拓扑。

```
   Channel N (×24 个并行 channel):

         rank 0 (GPU 0)
            ▲    │
            │    │
            └────┘
            │    │
            ▼    │
         rank 1 (GPU 1)
```

**为什么 24 个**：NCCL 把数据切成多份在多个 channel 上并发跑，能充分用 NVLink 多条带宽 + 隐藏延迟。channel 数由拓扑算出来（A100 NVSwitch 互联给的就是 24）。

**Trees 行（rank 0）**：
```
Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 ... [6] -1/-1/-1->0->1 ...
```

格式：`child0/child1/child2 -> me -> parent`。

| Channel | rank 0 的树 | rank 1 的树 | 形态 |
|---|---|---|---|
| 0-5, 12-17 | `1/-1/-1 -> 0 -> -1` (root, child=1) | `-1/-1/-1 -> 1 -> 0` (leaf) | `0 ── 1` (0 是根) |
| 6-11, 18-23 | `-1/-1/-1 -> 0 -> 1` (leaf) | `0/-1/-1 -> 1 -> -1` (root, child=0) | `1 ── 0` (1 是根) |

**为什么一半正、一半反**：NCCL 对 AllReduce 有 **double-tree** 算法——同时跑两棵方向相反的树，让两个 rank 都既是 root 又是 leaf，降低 root 节点的瓶颈。Ring vs Tree 的选择是按 size + 拓扑动态决定的。

```
  正向树 (channel 0-5, 12-17):    反向树 (channel 6-11, 18-23):

      rank 0 (root)                    rank 1 (root)
         │                                 │
         ▼                                 ▼
      rank 1 (leaf)                     rank 0 (leaf)
```

```
P2P Chunksize set to 524288
```
P2P 一次搬运的块大小：512 KB。

## Phase 7 — Proxy Service 线程上线

```
aig-a100:1278747:1278767 [0] NCCL INFO [Proxy Service] Device 0 CPU core 66
aig-a100:1278747:1278768 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 3
                  ^^^^^^^
                  注意这两个 TID ≠ 主线程 PID, 是新起的辅助线程
```

每个 comm 起 2 个辅助线程，绑到指定 CPU core：
- **Proxy Service** (core 66)：处理网络异步收发的 proxy 操作（IB/sockets 后端用，本例没真用上但起来 standby）
- **Proxy Service UDS** (core 3)：通过 Unix Domain Socket 跟其他进程的 proxy 通信（多进程同机的协调）

进程/线程关系一图：

```
  Process: rank 0 (pid 1278747)              Process: rank 1 (pid 1278748)
  ┌──────────────────────────────┐           ┌──────────────────────────────┐
  │ Main thread (tid 1278747)    │           │ Main thread (tid 1278748)    │
  │  ↳ ncclCommInitRank          │           │  ↳ ncclCommInitRank          │
  │  ↳ ncclAllReduce             │           │  ↳ ncclAllReduce             │
  ├──────────────────────────────┤           ├──────────────────────────────┤
  │ Proxy Service     (tid 67)   │  ◀─UDS─▶  │ Proxy Service     (tid 65)   │
  │ Proxy Service UDS (tid 68)   │           │ Proxy Service UDS (tid 66)   │
  ├──────────────────────────────┤           ├──────────────────────────────┤
  │ P2P connection  (tid 69, 出  │  ◀NVLink▶ │ P2P connection  (tid 70, 出  │
  │                  现在 AllReduce 第一次调用时)
  └──────────────────────────────┘           └──────────────────────────────┘
                                  │
                                  └─ bootstrap socket (init 阶段) → 用完关掉
```

## Phase 8 — Init COMPLETE + 阶段耗时

```
threadThresholds 8/8/64 | 16/8/64 | 512 | 512
24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
CC Off, Multi-GPU CC Off, workFifoBytes 1048576
```

- `threadThresholds`：每个 kernel 用多少线程的阈值（Ring/Tree/CollNet/NVLS 各一组）
- channel 清单：24 个集合通信 channel，0 个 NVLS（本机 A100 SXM4 不支持 NVLink Switch Multicast，H100 NVL 才有）
- `CC Off`：Collective Compression 关（实验功能）
- `workFifoBytes`：内核 work queue 大小

```
ncclCommInitRank ... - Init COMPLETE
Init timings - ncclCommInitRank: rank 0 nranks 2 total 0.20
   (kernels 0.15, alloc 0.03, bootstrap 0.00, allgathers 0.00,
    topo 0.01, graphs 0.00, connections 0.01, rest 0.01)
```

**0.20 秒里**：
| 阶段 | 时间 | 干了啥 |
|---|---|---|
| **kernels** | 0.15s | 加载 NCCL CUDA kernel 到 GPU（最大头，第一次最慢） |
| **alloc** | 0.03s | 显存/host 内存分配 |
| **bootstrap** | <0.01s | 已经在 Phase 5 看到的 1.8ms |
| **allgathers** | <0.01s | 所有 rank 拓扑元数据 allgather |
| **topo** | 0.01s | 解析 PCIe/NVLink 拓扑（读 `/sys/class/...`） |
| **graphs** | <0.01s | 算 ring/tree 图 |
| **connections** | 0.01s | 建立 P2P 描述符（实际数据通道在 AllReduce 第一次调用时才上） |

下次进程内再起一个 comm 会快很多，kernels 已经加载过。

## Phase 9 — P2P 真正连上来（AllReduce 触发）

```
aig-a100:1278748:1278770 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/CUMEM/read
                                                       ^^^^^^^^^^         ^^^^^^^^^^^^^^^^^
                                                       rank1→rank0         transport=P2P, 后端=CUMEM, 模式=read
... (24 个 channel, 每个一条)
NCCL INFO Connected all rings
```

```
aig-a100:1278747:1278769 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
... (24 个 channel)
NCCL INFO Connected all rings
```

**这是整份 log 最关键的一段**——确认了**实际数据通道是 P2P/CUMEM/read**：

| 字段 | 含义 |
|---|---|
| **P2P** | GPU 到 GPU 直接访问（NVLink 或 PCIe BAR1），不经过 host 内存 |
| **CUMEM** | 用 CUDA Virtual Memory Management API（`cuMemMap` 等），相比老的 `cudaIpc*` 更高效，省一次 mmap |
| **read** | 接收方主动从对方显存"读"（vs `write` = 发送方写到对方显存）。read 在 A100 NVLink 上一般更快 |

24 个 channel 各开一条 P2P 连接，所以你看到 24 行 `Channel XX/0 : ... via P2P/CUMEM/read`。

```
   Channel N 在 NVLink 上的连接 (×24 条并发):

         rank 0 (GPU 0)
                  │
              ┌───┴───┐
              │ NVLink│  ← P2P/CUMEM/read (直接 GPU↔GPU, 不经 CPU/RAM)
              │       │
              └───┬───┘
                  │
         rank 1 (GPU 1)
```

为什么 P2P 真正建立在 Phase 9（AllReduce 触发时）而不是 Phase 6（Init 阶段）？因为 NCCL 用 **lazy connect**——建得太早占资源，第一次 collective 触发时才按需建。所以你看到 P2P 行在 rank 1 的 `OK` 之后才出现（rank 1 已经在跑 AllReduce 了，rank 0 的 init 还在打完 timing 那行）。

## Phase 10 — Destroy

```
NCCL INFO comm 0x61476c9e1c90 rank 1 nranks 2 cudaDev 1 busId 9000 - Destroy COMPLETE
NCCL INFO comm 0x5ce61e459ee0 rank 0 nranks 2 cudaDev 0 busId 8000 - Destroy COMPLETE
```

`ncclCommDestroy` 收尾：关闭 proxy 线程、释放 P2P 描述符、清显存。

## 全流程时序图

```
 时间 →
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ rank 0                                                                      │
 │ ┌────┐ ┌──────┐ ┌────────┐ ┌─────────┐ ┌──────┐ ┌───────┐ ┌──────────┐ ┌──┐│
 │ │启动│→│bootstrap│→│plugin│→│CommInit │→│channel│→│proxy  │→│AllReduce │→│销│
 │ │    │ │socket │ │探测  │ │START    │ │/tree │ │线程上 │ │P2P真连接 │ │毁││
 │ └────┘ └──────┘ └──────┘ └────┬────┘ └──────┘ └───────┘ └────┬─────┘ └──┘│
 │                               │                               │           │
 │                              barrier (等 rank 1 也到)         │           │
 │                               │                               │           │
 │ rank 1                        │                               │           │
 │ ┌────┐ ┌──────┐ ┌────────┐ ┌──┴──────┐ ┌──────┐ ┌───────┐ ┌──┴───────┐ ┌──┐│
 │ │启动│→│bootstrap│→│plugin│→│CommInit │→│channel│→│proxy  │→│AllReduce │→│销│
 │ │    │ │connect│ │探测  │ │START    │ │/tree │ │线程上 │ │P2P真连接 │ │毁││
 │ └────┘ └──────┘ └──────┘ └─────────┘ └──────┘ └───────┘ └──────────┘ └──┘│
 │                                                                             │
 │      ▲                                                  ▲                   │
 │      └─ bootstrap 用 OOB 网卡                            └─ 这里实际数据走 │
 │         (ens4f0np0)                                         NVLink P2P     │
 └─────────────────────────────────────────────────────────────────────────────┘
```

## 几个隐藏的小坑

### 1. 日志会交错

log 是两个进程并发写到同一个 stdout 的，所以会撞行。最明显的：

```
line 51: ... allgathers 0.00, topo 0.01, graphs 0.00, con[rank 1] pid=1278748 bound to GPU 1
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                          rank 1 的 printf 插进了 rank 0 的输出中间

line 100: nections 0.01, rest 0.01)
          ^^^^^^^^^
          (这是 line 51 被截断的下半截)
```

教学代码无所谓，生产里要清晰拆分日志可以：
- 每个进程独立写文件：`./prog 0 > rank0.log` / `./prog 1 > rank1.log`
- 或者用 stderr 的 line-buffered 特性（NCCL 输出走 stderr，更不容易撞）

### 2. RoCE 网卡被发现但没真用

`NET/IB : Using [0]bnxt_re0:1/RoCE` 和 `Using network IB` 让人以为数据走 IB——其实只是 NCCL 的网络后端被选好待命，单机 2 卡数据走 NVLink P2P（Phase 9 的 `P2P/CUMEM/read`）。要跨机才真正用到 IB。

### 3. 24 channel 全是一样的 ring

因为 2 个 rank 排不出更多花样。如果是 8 卡，你会看到 channel 是不同的 ring 排列（比如 `0 1 3 2 6 7 5 4`），NCCL 利用 NVSwitch 多路径并发。

## 如何继续挖

| 想看什么 | 怎么开 |
|---|---|
| 更详细的 NCCL 内部状态 | `NCCL_DEBUG=TRACE`（量级翻 10 倍） |
| 只看某个子系统 | `NCCL_DEBUG_SUBSYS=INIT` / `COLL` / `P2P` / `NET` 等 |
| 强制不用 NVLink | `NCCL_P2P_DISABLE=1`（会看到 SHM 或 NET 接管） |
| 强制走网络 | `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`（极端慢，但能看 IB 路径） |
| 强制 ring 不用 tree | `NCCL_ALGO=Ring`（debug 算法切换问题时有用） |
| 信息倒到文件 | `NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log` （`%h`=host, `%p`=pid，每进程一份，不会撞） |

---

## 引用

- Log: `my_nccl_test/all_reduce_2proc/logs/run_info.log`
- 代码: `my_nccl_test/all_reduce_2proc/all_reduce_2proc.cu`
- 配套概念: [concepts.md](concepts.md)
