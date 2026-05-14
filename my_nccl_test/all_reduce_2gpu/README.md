# my_nccl_test — 单进程 2 卡 all_reduce 最小样例

> 目标：用最少的代码把 NCCL `ncclAllReduce` 跑起来，能编、能过校验、能改着学。

## 一句话总结

> GPU 0 输入全 1.0，GPU 1 输入全 2.0；做一次 `ncclAllReduce(sum)`，两张卡都拿到 3.0。

```
              ─── 通信前 ───                    ─── 通信后 ───

   GPU 0:  sendbuf = [1.0 1.0 ... 1.0]      recvbuf = [3.0 3.0 ... 3.0]
                                  ╲       ╱
                                   ╲ NCCL╱     op = Sum
                                   ╱AllRed╲    1.0 + 2.0 = 3.0
                                  ╱       ╲    (广播回每张卡)
   GPU 1:  sendbuf = [2.0 2.0 ... 2.0]      recvbuf = [3.0 3.0 ... 3.0]
```

这就是 all-reduce 的语义：**把所有 rank 的输入按算子归约，结果发回所有 rank**。
对比一下兄弟集合通信：

```
  reduce        → 结果只到一个 root rank
  all-reduce    → 结果发回所有 rank      ← 本例
  reduce-scatter→ 归约后每个 rank 拿一段
  all-gather    → 不归约, 拼接后所有 rank 都拿全量
  broadcast     → 一个 root 的数据发给所有 rank
```

## 文件清单

```
my_nccl_test/all_reduce_2gpu/
├── all_reduce_2gpu.cu   # 主程序 (~70 行, 含错误检查宏)
├── Makefile             # 一条规则编 + clean + help
├── run.sh               # 包好 LD_LIBRARY_PATH 的运行脚本
└── README.md            # 本文档
```

## 编译 + 运行

依赖前置：`../../nccl/build/lib/libnccl.so` 必须存在（由 `nccl-bench/build_nccl.sh` 产出）。

```bash
cd ~/nccl-repo/my_nccl_test/all_reduce_2gpu

make            # 编译, 产出 ./all_reduce_2gpu
./run.sh        # 运行 (会自动 export LD_LIBRARY_PATH)

# 实际输出:
# === Running: .../all_reduce_2gpu ===
#     NCCL lib:   .../nccl/build/lib
#     NCCL_DEBUG: WARN
#
# NCCL version 2.23.4+cuda12.9
# [GPU 0] recvbuf[0] = 3.0  (expected 3.0)
# [GPU 1] recvbuf[0] = 3.0  (expected 3.0)
# OK
```

想看 NCCL 选了什么 algo / proto / transport：
```bash
NCCL_DEBUG=INFO ./run.sh
```

## 代码 5 步流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. 每张卡 cudaMalloc sendbuf+recvbuf, 填初值, cudaStreamCreate       │
│                                                                      │
│       GPU 0           GPU 1                                          │
│     ┌─────┐         ┌─────┐                                          │
│     │send │ ← 1.0   │send │ ← 2.0      (各自独立 stream)             │
│     │recv │         │recv │                                          │
│     └─────┘         └─────┘                                          │
├──────────────────────────────────────────────────────────────────────┤
│ 2. ncclCommInitAll(comms, 2, {0,1})                                  │
│    单进程多卡: 一次出 nGpus 个 comm; 每个 comm 绑一张卡。            │
│    内部会握手, 选出 P2P / SHM / NVLink 路径 (本机有 NVLink 走它)。   │
├──────────────────────────────────────────────────────────────────────┤
│ 3. ncclGroupStart()                                                  │
│       for each device i:                                             │
│           ncclAllReduce(send[i], recv[i], N, Float, Sum,             │
│                         comms[i], streams[i])                        │
│    ncclGroupEnd()                                                    │
│                                                                      │
│    ⚠ 单进程多卡场景必须包在 Group 里, 否则第一个 ncclAllReduce      │
│      调用会阻塞等对端 → 死锁。Group 让 NCCL 一次拿到所有卡的请求,    │
│      内部排出一个 schedule, 再启动 kernel。                          │
├──────────────────────────────────────────────────────────────────────┤
│ 4. cudaStreamSynchronize(streams[i]) → cudaMemcpy 拷回 host 校验     │
│    检 recvbuf[0] == 3.0 (即 1.0 + 2.0)                               │
├──────────────────────────────────────────────────────────────────────┤
│ 5. 释放: cudaFree / cudaStreamDestroy / ncclCommDestroy              │
└──────────────────────────────────────────────────────────────────────┘
```

## 关键 NCCL 概念

| 概念 | 一句话 | 本样例对应的代码 |
|---|---|---|
| **rank** | 通信组里某个参与者的编号；2 卡时 rank=0/1 | 隐式：`devs[i]` 的下标 i 即 rank |
| **communicator (`ncclComm_t`)** | 通信组句柄，每个 rank 一份；同组的 comm 互相握手过 | `comms[2]` |
| **`ncclCommInitAll`** | 单进程多卡的便捷初始化（多进程要用 `ncclCommInitRank` + `ncclUniqueId`） | `ncclCommInitAll(comms, 2, devs)` |
| **stream** | NCCL 在 CUDA stream 上排 kernel；同 stream 的事件天然串行 | `streams[2]` |
| **group call** | 把多卡上的 NCCL 调用合成一次提交，避免互等 | `ncclGroupStart/End` 夹住的循环 |
| **datatype / op** | NCCL 自己的枚举：`ncclFloat`、`ncclSum` 等 | `ncclAllReduce(..., ncclFloat, ncclSum, ...)` |

## 单进程多卡 vs 多进程多卡

本样例是**单进程 2 卡**（最简模型）。生产里更常见的是多进程，因为 PyTorch / Horovod 都按一卡一进程跑。两者初始化方式不一样：

```
单进程多卡 (本样例):              多进程多卡 (生产典型):
                                
  主进程                            进程 0          进程 1
    │                                 │                │
    ├─ ncclCommInitAll                ├─ ncclUniqueId  │
    │    ↓ 一次给出 N 个 comm         │   ↙ 通过环境/MPI/TCP 共享
    │                                 │  ┌────────┐    │
   comm[0]  comm[1]                   │  │ id     │ ───┤
    │         │                       │  └────────┘    │
   GPU 0    GPU 1                     ├─ CommInitRank  ├─ CommInitRank
                                      │   (id, 2, 0)   │   (id, 2, 1)
                                     GPU 0           GPU 1

  ✓ 简单, 单机                     ✓ 适合一卡一进程 (PyTorch DDP)
  ✗ 单 host 进程, 跨机不行         ✓ 可跨机 (需要 IB/RoCE/sockets)
                                   ✗ 启动更复杂 (要协调 ncclUniqueId 广播)
```

## 输入数据的意义

* GPU 0 全 `1.0`，GPU 1 全 `2.0` 是**人为挑的"好认"数字**：归约结果 3.0 不可能与任何一个输入重合，校验更可靠。
* 单元素验证（只看 `recvbuf[0]`）是为了代码短。要全量校验也容易：把 `cudaMemcpy` 整段拉回来，逐位比 3.0。
* 数据量取 1M float（4 MB）：足够 NCCL 走"正常"通路而不是退化到 in-place fast path，又不会让本机 NVLink 跑成显存压力测试。

## 改着学的几个方向

| 想试的事 | 改哪里 |
|---|---|
| 改成 4 卡 | `nGpus = 4`，`devs = {0,1,2,3}`，stack 数组改成 `vector` |
| 换 op | `ncclSum` → `ncclMax` / `ncclMin` / `ncclProd` / `ncclAvg` |
| 换 dtype | `ncclFloat` → `ncclHalf` / `ncclInt32` / `ncclBfloat16`（输入也要改） |
| 看通信走哪条路 | 跑前 `export NCCL_DEBUG=INFO`，看 `Channel 00 : 0 -> 1 ...` 那段 |
| 强制走环 / 树 | `NCCL_ALGO=Ring` / `NCCL_ALGO=Tree` |
| 关掉 NVLink 看 PCIe | `NCCL_P2P_DISABLE=1`，再跑一次对比带宽 |
| 改成多进程版 | 把 `CommInitAll` 换成 `ncclGetUniqueId` + 进程间共享 id + `ncclCommInitRank`，进程数=GPU 数 |

## 常见坑

**Q: 链接报 `cannot find -lnccl`**
A: `make NCCL_HOME=/path/to/nccl/build`，或确认 `../../nccl/build/lib/libnccl.so` 存在。

**Q: 运行报 `libnccl.so.2: cannot open shared object file`**
A: 没 export `LD_LIBRARY_PATH`，用 `run.sh` 包好的；手动：
```bash
export LD_LIBRARY_PATH="$HOME/nccl-repo/nccl/build/lib:$LD_LIBRARY_PATH"
```

**Q: 程序卡住不返回**
A: 99% 是忘了把多卡上的 NCCL 调用包在 `ncclGroupStart/End` 里。单进程多卡的每个 `ncclAllReduce` 都是阻塞语义，第一个调用会卡着等其他 rank 进入，没 Group 就互等死锁。

**Q: 输出 nan / 0**
A: 经常是 stream 没同步就读 host，或者 sendbuf 没初始化好。检查 `cudaStreamSynchronize` 与 `cudaMemcpy` 顺序。

**Q: 想看带宽，不只是正确性**
A: 用 `nccl-bench/run_test.sh allreduce`，那是 nccl-tests 官方 perf 程序，会扫一组 size 并打 busbw。
