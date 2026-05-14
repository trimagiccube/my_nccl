# my_nccl_test/all_reduce_2proc — 多进程 2 卡 all_reduce

> 目标：用最少代码演示**一卡一进程**的 NCCL 通信组初始化（PyTorch DDP / Horovod 的标准模型）。
> 不依赖 MPI，靠一个临时文件分发 `ncclUniqueId`。

## 和兄弟样例的区别

| | `../all_reduce_2gpu/` | **`./all_reduce_2proc/` (本目录)** |
|---|---|---|
| 进程数 | 1 | 2 |
| 每进程 GPU | 2 | 1 |
| 通信组初始化 | `ncclCommInitAll` | `ncclGetUniqueId` + `ncclCommInitRank` |
| 多卡 NCCL 调用 | 必须包 `ncclGroupStart/End` | **不需要** Group |
| 跨机能力 | 不可 | 可（把进程拉到不同机器即可） |
| 像谁 | NCCL 文档单进程示例 | PyTorch DDP / Horovod |

## 一句话总结

> 两个进程各管一张卡，rank 0 上 1.0、rank 1 上 2.0；`ncclAllReduce(sum)` 后两进程都拿到 3.0。

```
        ─── 通信前 ───                          ─── 通信后 ───

  Process 0 / GPU 0                          Process 0 / GPU 0
    sendbuf = [1.0 1.0 ... 1.0]                recvbuf = [3.0 3.0 ... 3.0]
                                ╲           ╱
                                 ╲ NCCL    ╱     op = Sum
                                 ╱AllReduce╲     1.0 + 2.0 = 3.0
                                ╱           ╲    每个 rank 都有完整结果
  Process 1 / GPU 1                          Process 1 / GPU 1
    sendbuf = [2.0 2.0 ... 2.0]                recvbuf = [3.0 3.0 ... 3.0]
```

## 文件清单

```
my_nccl_test/all_reduce_2proc/
├── all_reduce_2proc.cu   # 主程序 (~100 行, 单一二进制吃 rank/nranks)
├── Makefile              # make / clean / help
├── run.sh                # 拉起 NRANKS 个子进程的 launcher
└── README.md             # 本文档
```

## 编译 + 运行

依赖前置：`../../nccl/build/lib/libnccl.so` 必须存在（由 `nccl-bench/build_nccl.sh` 产出）。

```bash
cd ~/nccl-repo/my_nccl_test/all_reduce_2proc

make            # 编译, 产出 ./all_reduce_2proc
./run.sh        # launcher: 起 2 个进程, wait 收集退出码

# 实际输出 (顺序不固定, 两进程并发):
# === Running: .../all_reduce_2proc, NRANKS=2 ===
#     NCCL lib:      .../nccl/build/lib
#     NCCL_DEBUG:    WARN
#     NCCL_UID_FILE: /tmp/nccl_uid_938655.bin
#
# [rank 1] pid=938674 bound to GPU 1
# [rank 1] recvbuf[0] = 3.0  (expected 3.0)
# [rank 1] OK
# [rank 0] pid=938673 bound to GPU 0
# NCCL version 2.23.4+cuda12.9
# [rank 0] recvbuf[0] = 3.0  (expected 3.0)
# [rank 0] OK
# ALL RANKS OK
```

跑 4 进程（如果机器有 ≥4 卡）：
```bash
NRANKS=4 ./run.sh
```

看 NCCL 握手细节：
```bash
NCCL_DEBUG=INFO ./run.sh
```

## ncclUniqueId 分发：bootstrap 流程

这是多进程 NCCL 唯一“反直觉”的一步。NCCL 内部需要一个 rendezvous 点把所有 rank 撮合到一起，这个点是个 128 字节的 `ncclUniqueId`，**必须由 rank 0 生成并发到所有其它 rank**。NCCL 不规定怎么发——你随意。本样例用的是文件：

```
         rank 0 (GPU 0)                           rank 1 (GPU 1)
  ──────────────────────────                ──────────────────────────
   ncclGetUniqueId(&id)
        │                                        │
        ▼                                        │ 轮询 stat(path) 直到
   fopen("tmp"); fwrite(id);                     │ st_size == 128
   rename("tmp", path)  ◀── 原子可见 ─────────▶  │
        │                                        ▼
        │                                   fopen(path); fread(&id)
        │                                        │
        ▼                                        ▼
   ncclCommInitRank(&comm, 2, id, 0) ━━━━━ barrier ━━━━ ncclCommInitRank(&comm, 2, id, 1)
                              （两边都进来才返回，本身就是同步点）
        │                                        │
        ▼                                        ▼
   ncclAllReduce(...)                       ncclAllReduce(...)
```

要点：
* **生成只能在 rank 0 一次**——同一个 id 用一次后失效；下一次通信组要新的。
* `ncclCommInitRank` 内部会阻塞，等齐 `nranks` 个 rank 都 join 才返回。所以你不需要额外的 barrier。
* 文件是“最弱依赖”的方式。生产里常见的替代：
  * **MPI**：用 `MPI_Bcast` 把 id 从 rank 0 广播到所有 rank。
  * **socket / etcd / Redis**：跨机时如果文件系统不共享。
  * **PyTorch**：torch.distributed 起的时候自动帮你做（底下走 TCPStore / FileStore / MPI）。

## 代码 5 步流程

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. cudaSetDevice(rank)      一进程一卡, rank → device 一对一映射     │
├──────────────────────────────────────────────────────────────────────┤
│ 2. share_unique_id():                                                │
│      rank 0 → ncclGetUniqueId → 写文件                               │
│      rank>0 → 轮询读文件                                             │
├──────────────────────────────────────────────────────────────────────┤
│ 3. ncclCommInitRank(&comm, nranks, id, rank)                         │
│      隐式 barrier: 所有 rank 都进入才返回。                          │
│      内部协商 transport (NVLink / PCIe / IB / sockets)、构建环/树。  │
├──────────────────────────────────────────────────────────────────────┤
│ 4. ncclAllReduce(sendbuf, recvbuf, N, Float, Sum, comm, stream)      │
│      ⚠ 单进程单 comm: 不需要 ncclGroupStart/End。                    │
│      (Group 只是为了把"多卡 NCCL 调用"合成一次提交避免互等;          │
│       单 comm 进程只有一个调用, 没人可等。)                          │
├──────────────────────────────────────────────────────────────────────┤
│ 5. cudaStreamSynchronize → 校验 → cudaFree / ncclCommDestroy         │
│    rank 0 顺手 unlink uid 文件 (一次性, 用完即弃)                    │
└──────────────────────────────────────────────────────────────────────┘
```

## launcher (`run.sh`) 干了什么

```
shell ($$ = 938655)
  ├─ 生成 UID_FILE=/tmp/nccl_uid_938655.bin    (用 shell PID 防多次运行撞名)
  ├─ trap EXIT 清理临时文件
  ├─ fork: NCCL_UID_FILE=… ./all_reduce_2proc 0 2 &   PID0
  ├─ fork: NCCL_UID_FILE=… ./all_reduce_2proc 1 2 &   PID1
  ├─ wait PID0 / wait PID1   分别收退出码
  └─ 任一 rank 非 0 → 整体 exit 1
```

`NCCL_UID_FILE` 用 shell PID 命名是为了防止两次并发的 `./run.sh` 撞到同一个文件互相覆盖。

## 关键 NCCL 概念（与单进程版的对比）

| 概念 | 单进程 2 卡 (`all_reduce_2gpu`) | 多进程 2 卡 (本样例) |
|---|---|---|
| 通信组初始化 | `ncclCommInitAll(comms, n, devs)`，主程序一次出 N 个 comm | 每个进程 `ncclCommInitRank` 一次，出自己的 1 个 comm |
| rank 是谁 | 进程内的 i = 0..N-1 | 进程级身份，由命令行 / MPI rank / 环境变量传入 |
| 同步靠什么 | `ncclGroupStart/End` 把多卡调用合并 | `ncclCommInitRank` 本身是 barrier；之后每进程自己跑 |
| ncclUniqueId | 不需要，`CommInitAll` 内部搞定 | **必须**显式生成并发到所有 rank |
| 多卡调用模式 | 必须 Group | 单 comm 不需要 Group |

## 改着学的几个方向

| 想试的事 | 改哪里 |
|---|---|
| 改成 4 进程 4 卡 | `NRANKS=4 ./run.sh`（需要机器有 ≥4 卡） |
| 换 op / dtype | `.cu` 里 `ncclSum` / `ncclFloat` 改一下，输入相应改 |
| 看通信走哪条路 | `NCCL_DEBUG=INFO ./run.sh`，看 `Channel 00 : 0[…] -> 1[…]` |
| 强制 PCIe（屏蔽 NVLink）| `NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO ./run.sh` |
| 跨机 2 卡 2 进程 | 把可执行拷到另一台机器，UID 文件改成 NFS / scp 同步过去；或上 MPI |
| 换成 socket 分发 id | 改 `share_unique_id()`：rank 0 listen on a port, rank>0 connect 拿 id |
| 看 ncclUniqueId 长啥样 | id 是 128 字节裸 bytes，`hexdump /tmp/nccl_uid_*.bin` 看一眼 |

## 常见坑

**Q: 程序卡在 `ncclCommInitRank` 不返回**
A: 99% 是有 rank 没起来（崩了 / OOM / `cudaSetDevice` 失败 / 拿不到 UID 文件）。`ncclCommInitRank` 等齐所有 rank 才返回，少一个就死等。先单跑 `./all_reduce_2proc 0 1`（自己一个组）看能不能起来。

**Q: 多次跑 `./run.sh` 会不会互相影响**
A: 不会。`NCCL_UID_FILE=/tmp/nccl_uid_$$.bin` 带了 shell PID，每次唯一。

**Q: `share_unique_id` rank>0 轮询的 30s 超时合理吗**
A: 教学代码取的保守值（300×100ms）。生产里 rank 0 应该几乎立刻就写好；如果 30s 还没写好，多半是 rank 0 起不来。

**Q: 改成 `cudaSetDevice(rank)` 之外的映射？比如 4 卡机只用卡 2 和卡 3**
A: 启动时 `CUDA_VISIBLE_DEVICES=2,3 NRANKS=2 ./run.sh` 即可——CUDA 把 2,3 重映射成 0,1，代码不用改。

**Q: 为什么不用 MPI**
A: 用 MPI 当然更标准（`MPI_Bcast` 发 id 就一行），但需要装 `libopenmpi-dev` 并 `mpirun -np 2` 启动。本样例是教学起点，零外部依赖更容易上手。学完想升级到 MPI 版很简单——把 `share_unique_id` 替换成 `MPI_Bcast(&id, 128, MPI_BYTE, 0, MPI_COMM_WORLD)`，再用 `mpirun` 启即可。

**Q: 想知道两边带宽**
A: 这个样例只校验正确性。要打带宽用 `nccl-bench/run_test.sh allreduce`（那是 nccl-tests 的官方 perf 程序，扫一组 size 并打 busbw）。
