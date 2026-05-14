# NCCL_DEBUG=TRACE 走读（含一个常见误区）

> 接 [`nccl_debug_info_walkthrough.md`](nccl_debug_info_walkthrough.md)：同样跑 `my_nccl_test/all_reduce_2proc`，把 `NCCL_DEBUG` 从 `INFO` 升到 `TRACE` 会看到啥。

## 实测三份日志

| 命令 | 文件 | 行数 | 大小 |
|---|---|---|---|
| `NCCL_DEBUG=INFO` | [`run_info.log`](../my_nccl_test/all_reduce_2proc/logs/run_info.log) | 129 | 11 KB |
| `NCCL_DEBUG=TRACE` | [`run_trace.log`](../my_nccl_test/all_reduce_2proc/logs/run_trace.log) | **129** | **11 KB**（和 INFO 一样！） |
| `NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL` | [`run_trace_all.log`](../my_nccl_test/all_reduce_2proc/logs/run_trace_all.log) | **2297** | **246 KB**（×18） |

`NCCL_DEBUG=TRACE` 单设——跟 INFO **没差别**。这是常见误区。

## 为什么 TRACE 单设没用

看源码 `nccl/src/debug.cc:22`：

```c
static uint64_t ncclDebugMask = NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV;
//                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                              默认只开这三个子系统的输出
```

NCCL 的日志由**两个旋钮**联合过滤，源码 `debug.cc:186`：

```c
if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) return;
//   ▲                          ▲
//   level 过滤                  subsys mask 过滤
```

| 旋钮 | 控制 | 默认 |
|---|---|---|
| `NCCL_DEBUG` (level) | 输出**严重度**：VERSION < WARN < INFO < TRACE | `WARN` |
| `NCCL_DEBUG_SUBSYS` (mask) | 输出**哪些子系统**的消息 | `INIT \| BOOTSTRAP \| ENV` |

两个 **AND 关系**——任何一道不通过都不输出。所以：
- 把 level 升到 TRACE 但 mask 不变 → 还是只 `INIT|BOOTSTRAP|ENV` 三个子系统的消息，只不过这三个子系统里如果有 TRACE() 调用现在会显示。问题是这三个子系统里几乎所有调用都用 `INFO()` 不用 `TRACE()`，所以肉眼看不到差异。
- 把 mask 开到 `ALL` 但 level 还是 INFO → 也能解锁大量新输出，但比 TRACE+ALL 略少（少了那一点 TRACE() 行）。

```
                            ┌─ level    ┐
                            │  default  │
              ┌─ NCCL_DEBUG → INFO────────→ ┌─ subsys mask ┐
              │             │           │   │  default     │
   你的环境变量              │ TRACE     │   │ INIT|BOOTSTRAP|ENV
              │             └───────────┘   │              │
              └─ NCCL_DEBUG_SUBSYS  ────────→│ ALL          │
                                            │ ^P2P (除 P2P)│
                                            │ COLL,NET     │
                                            └──────────────┘

      只升 level → 透出更多 severity, 但子系统不变 → 没差别
      只开 mask → 透出更多子系统, level 还是 INFO  → 涨很多
      两个都开 → 全开                              → 最详细
```

## 完整子系统清单

源码 `debug.cc:61-91` 解析 `NCCL_DEBUG_SUBSYS` 时支持的：

| 子系统 | 默认开 | 内容 |
|---|---|---|
| `INIT` | ✅ | comm 初始化关键事件（已经在 INFO log 看过） |
| `BOOTSTRAP` | ✅ | bootstrap socket 操作 |
| `ENV` | ✅ | 读取的环境变量 |
| `COLL` | ❌ | 每次集合通信（AllReduce 等）触发的事件 |
| `P2P` | ❌ | GPU-to-GPU 点对点通信细节 |
| `SHM` | ❌ | 共享内存 transport |
| `NET` | ❌ | 网络 transport（IB/sockets）细节 |
| `GRAPH` | ❌ | 算法图搜索（ring/tree 怎么算出来的） |
| `TUNING` | ❌ | tuner 选 algo / proto / buffer 大小 |
| `ALLOC` | ❌ | 内存分配 |
| `CALL` | ❌ | 所有 NCCL API 调用入口/出口（极冗长） |
| `PROXY` | ❌ | proxy 线程的网络代理操作 |
| `NVLS` | ❌ | NVLink Switch（H100 / GH200 才用） |
| `REG` | ❌ | 用户 buffer 注册 |
| `PROFILE` | ❌ | profile 事件 |

特殊语法：
- `NCCL_DEBUG_SUBSYS=ALL` — 全开
- `NCCL_DEBUG_SUBSYS=COLL,P2P` — 只开这两个
- `NCCL_DEBUG_SUBSYS=^TUNING,CALL` — `^` 前缀 = 反选，开启除 TUNING / CALL 外的所有

## SUBSYS=ALL 多出来的是什么

129 → 2297 行多出 ~2168 行，按类别：

### 1. 详细系统拓扑（`NCCL_GRAPH`）

```
=== System : maxBw 240.0 totalBw 240.0 ===
CPU/0-0 (1/1/2)
+ PCI[48.0] - PCI/0-4000 (1000c0301000ffff)
              + PCI[24.0] - GPU/0-8000 (0)
                            + NVL[240.0] - GPU/0-9000
              + PCI[24.0] - GPU/0-9000 (1)
                            + NVL[240.0] - GPU/0-8000
+ SYS[10.0] - CPU/0-1
CPU/0-1 (1/1/2)
+ SYS[10.0] - CPU/0-0
+ PCI[6.0] - NIC/0-d4000
==========================================
```

**翻译这棵树**：
```
              CPU socket 0  ━━ SYS[10] ━━ CPU socket 1
                  │                            │
              ┌───┴───┐                        │
              │ PCIe  │                       NIC (RoCE)
              │ root  │
              │       │
       ┌──────┘       └──────┐
   PCI[24.0]              PCI[24.0]
   GPU/0-8000  ━━ NVL[240] ━━  GPU/0-9000
   (GPU 0)                     (GPU 1)
```

bw 单位是 GB/s：NVLink 240 GB/s（A100），PCIe Gen4 x16 双向 ~24 GB/s，跨 socket 10 GB/s（QPI/UPI 估算值）。

### 2. 每对 GPU 的最快路径矩阵（`NCCL_GRAPH`）

```
GPU/0-8000 :GPU/0-8000 (0/5000.0/LOC)  GPU/0-9000 (1/240.0/NVL)  CPU/0-0 (2/24.0/PHB)  CPU/0-1 (3/10.0/SYS)
GPU/0-9000 :GPU/0-8000 (1/240.0/NVL)   GPU/0-9000 (0/5000.0/LOC) CPU/0-0 (2/24.0/PHB)  CPU/0-1 (3/10.0/SYS)
                       ▲       ▲
                       hop数   带宽
```

读 `GPU/0-8000 → GPU/0-9000 (1/240.0/NVL)`：从 GPU 0 到 GPU 1 走 1 跳，240 GB/s，类型 NVL（NVLink）。这就是 NCCL 选 P2P/CUMEM 时背后的判断依据。

### 3. 算法图搜索 (`NCCL_GRAPH`)

```
Pattern 4, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX, sameChannels 1
 0 : GPU/0 GPU/1
 1 : GPU/0 GPU/1
 ...
Pattern 1, crossNic 0, nChannels 12, bw 40.000000/40.000000, type NVL/PIX, sameChannels 0
 0 : GPU/0 GPU/1
 ...
 6 : GPU/1 GPU/0     ← 注意反转
```

Pattern N 是 NCCL 内部的算法/拓扑组合枚举：
- **Pattern 1** = Ring（balanced），12 channel，bw 40 GB/s
- **Pattern 4** = Tree（balanced），12 channel，bw 20 GB/s

两组加起来正好 24 channel——印证了 INFO log 里"24 coll channels"和"Trees 一半正一半反"的来源。

### 4. 内存分配 (`NCCL_ALLOC`)

```
init.cc:1714 Cuda Host Alloc Size 4 pointer 0x7f92d7200000
```

每次 NCCL 内部 malloc/cudaMalloc/cudaMallocHost 都打一行。可以用来查内存泄漏或者 init 占用。

### 5. GPUDirect / NET 细节（`NCCL_NET`）

```
NET/IB : GPU Direct RDMA Disabled for HCA 0 'bnxt_re0'
```

GDR 这里没用上（驱动没装/没开 `nv_peer_mem`）。跨机训练场景下要确认这行说 Enabled。

## 实际操作旋钮

```bash
# 最有用的常见组合 ↓

# 1. 默认 INFO + 关键子系统 — 平衡视图
NCCL_DEBUG=INFO ./run.sh

# 2. INFO + 全子系统 — 看 NCCL 选了啥拓扑/算法 (不打 TRACE 级)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL ./run.sh

# 3. TRACE + 全子系统 — 最详细
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL ./run.sh

# 4. 只关心某一类 (比如调网络问题)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET ./run.sh

# 5. 排除掉最吵的 CALL 子系统
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=^CALL ./run.sh

# 6. 输出落到独立文件 (避免和 stdout 混)
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL \
NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log \
    ./run.sh
# 会得到 /tmp/nccl-aig-a100-1234.log /tmp/nccl-aig-a100-1235.log
```

## 一个推论

如果你想知道 NCCL 在某次跑里选了 ring 还是 tree、走了 NVLink 还是 IB——`NCCL_DEBUG=INFO` 就够（已经能看到 `via P2P/CUMEM/read`、Channel ring 序列）。

如果你想知道 NCCL **为什么**这么选——上 `NCCL_DEBUG_SUBSYS=ALL`，看 System 拓扑树 + Pattern 搜索 + 路径矩阵。

如果你在 debug 集合通信的运行时行为（卡死、CRC mismatch）——`NCCL_DEBUG_SUBSYS=COLL,P2P,PROXY` 一般够。

如果连这都不够——只剩重编 NCCL 加自己的 TRACE 行了，因为现成的 `TRACE()` 调用极少。

## 引用

- Log 三份：
  - [`run_info.log`](../my_nccl_test/all_reduce_2proc/logs/run_info.log) — `NCCL_DEBUG=INFO` 基线
  - [`run_trace.log`](../my_nccl_test/all_reduce_2proc/logs/run_trace.log) — `NCCL_DEBUG=TRACE` 反例（跟 INFO 同输出）
  - [`run_trace_all.log`](../my_nccl_test/all_reduce_2proc/logs/run_trace_all.log) — `TRACE + SUBSYS=ALL` 完整版
- 源码：`nccl/src/debug.cc:22` 默认 mask，`debug.cc:53-105` SUBSYS 解析
- 配套：[`nccl_debug_info_walkthrough.md`](nccl_debug_info_walkthrough.md)
