# nccl-bench — 用户态构建并运行 NCCL + nccl-tests

> 目标：从 `nccl-repo/nccl` 编出 `libnccl.so`，再用它链接编 `nccl-repo/nccl-tests`，最后跑性能测试。
> **不动系统目录**（不 sudo / 不 apt / 不 ldconfig / 不修改 `/usr/local`）。所有产物落在两个源码目录的 `build/` 子目录里。

## 目录布局

脚本通过 `dirname $0/..` 自动定位同级目录，所以只要按如下布局放置就能开箱即用：

```
nccl-repo/
├── nccl/            # NCCL 源码 (https://github.com/NVIDIA/nccl)
├── nccl-tests/      # NCCL tests 源码 (https://github.com/NVIDIA/nccl-tests)
└── nccl-bench/      # ← 本目录
    ├── build_nccl.sh
    ├── run_test.sh
    └── README.md
```

> 路径不一样也行，用环境变量覆盖：
> `NCCL_DIR=/path/to/nccl NCCL_TESTS_DIR=/path/to/nccl-tests ./build_nccl.sh`

## 当前机器环境

| 项 | 值 |
|---|---|
| 平台 | Ubuntu 22.04 / x86_64 / 128 核 |
| CUDA 选用 | **`/usr/local/cuda-12.9`** （脚本自动选；见下方“CUDA 版本”） |
| 系统 symlink | `/usr/local/cuda` → `cuda-13.0`（nvcc 13.0.48，**这份编不过 NCCL**） |
| GPU | 2 × NVIDIA A100-SXM4-40GB (compute_80) |
| Driver | 580.82.07 |
| 源码 | `nccl-repo/nccl`、`nccl-repo/nccl-tests` |

### CUDA 版本注意（重要）

本机 `/usr/local/cuda` 默认指向 **CUDA 13.0**，但 CUDA 13 **删除了** `PFN_cuMulticastCreate` 这类非版本号别名（只保留带 `_v12010` 后缀的版本），而 NCCL master 仍依赖旧名 → 编译会报：

```
../include/cudawrap.h:110: error: identifier "PFN_cuMulticastCreate" is undefined
```

`build_nccl.sh` 已加自动探测：默认会从 `12.9 → 12.8 → 12.6 → /usr/local/cuda` 中挑第一个**含非版本号别名**的版本。需要强制指定也可以：

```bash
CUDA_HOME=/usr/local/cuda-12.9 ./build_nccl.sh
```

## 一键执行

```bash
cd ~/nccl-repo/nccl-bench
./build_nccl.sh           # 完整流程: clean + build + smoke test
./build_nccl.sh --help    # 看完整选项
```

脚本流程：

```
预检 (CUDA/路径) → 清理 NCCL build → 编 NCCL → 校验产物
                → 清理 tests build → 编 nccl-tests → 校验 ldd
                → 跑 all_reduce_perf smoke test (8B-64M)
```

成功后产物：

```
nccl/build/lib/libnccl.so.2.x.y         # 编出来的 NCCL 库
nccl/build/include/nccl.h               # 对应头文件
nccl-tests/build/all_reduce_perf        # 链接到上面那份 NCCL
nccl-tests/build/all_gather_perf
nccl-tests/build/...                    # 共 10 个集合通信测试
```

## 跑测试

```bash
cd ~/nccl-repo/nccl-bench

# 默认 all_reduce, 8B → 1GB, 2x 步进, 用全部 GPU
./run_test.sh

# 指定测试 + 参数
./run_test.sh allreduce -b 1M -e 1G -f 2
./run_test.sh allgather -b 8  -e 128M -f 2
./run_test.sh sendrecv  -b 8  -e 16M  -f 2

# 列出所有可用测试
./run_test.sh --list

# 看握手过程
NCCL_DEBUG=INFO ./run_test.sh allreduce

# 显示帮助
./run_test.sh --help
```

### 简写映射

| 简写 | 完整 | 简写 | 完整 |
|---|---|---|---|
| `ar` / `allreduce` | all_reduce_perf | `ag` / `allgather` | all_gather_perf |
| `rs` / `reducescatter` | reduce_scatter_perf | `bcast` / `broadcast` | broadcast_perf |
| `a2a` / `alltoall` | alltoall_perf | `sr` / `sendrecv` | sendrecv_perf |
| `reduce` | reduce_perf | `gather` / `scatter` / `hypercube` | *_perf |

## 手动跑（不用 run_test.sh）

记得用 `LD_LIBRARY_PATH` 前缀指到用户构建的 NCCL，而**不是**系统的：

```bash
export LD_LIBRARY_PATH="$HOME/nccl-repo/nccl/build/lib:$LD_LIBRARY_PATH"
cd ~/nccl-repo/nccl-tests/build
./all_reduce_perf -b 8 -e 1G -f 2 -g 2
```

`ldd ./all_reduce_perf | grep nccl` 用来确认链接到的是你编的那份。

## nccl-tests 参数说明（最常用）

| 参数 | 含义 |
|---|---|
| `-b <size>` | 起始消息大小（如 `8`、`1K`、`1M`） |
| `-e <size>` | 终止消息大小 |
| `-f <factor>` | 每步乘的因子（2 表示翻倍递增） |
| `-g <ngpus>` | 单进程内使用的 GPU 数（同进程多卡场景） |
| `-n <iters>` | 每个 size 跑的迭代数（默认 20） |
| `-w <iters>` | 预热迭代数（默认 5） |
| `-c <0/1>` | 是否校验结果（1 = 开，默认开） |
| `-o <op>` | 归约算子（sum/prod/max/min/avg） |
| `-d <dtype>` | 数据类型（int32/float32/half/...） |

输出列：`size`、`count`、`type`、`redop`、`root`、`time(us)`、`algbw`（算法带宽）、`busbw`（总线带宽）、`#wrong`。**`busbw` 是关键指标** —— 把环上多跳的开销折算掉后的有效带宽。

## 自定义构建参数

`build_nccl.sh` 全部接环境变量覆盖：

```bash
# 多卡架构 (A100 + H100 共用一份二进制)
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
              -gencode=arch=compute_90,code=sm_90" \
./build_nccl.sh

# 用不同的 CUDA
CUDA_HOME=/usr/local/cuda-12.8 ./build_nccl.sh

# 不同的源码位置 (默认是脚本同级目录下的 ../nccl, ../nccl-tests)
NCCL_DIR=/opt/my_nccl NCCL_TESTS_DIR=/opt/my_tests ./build_nccl.sh

# 只重编 nccl-tests (NCCL 没动)
./build_nccl.sh --only tests

# 增量编 (不 clean)
./build_nccl.sh --no-clean

# 编完不跑 smoke
./build_nccl.sh --no-test
```

## 不动系统的关键点

| 风险点 | 这套流程怎么避开 |
|---|---|
| 全局 `libnccl.so` 被改 | 产物只落在 `nccl/build/lib`，不 `make install`、不 `ldconfig` |
| `apt install libnccl-dev` 污染 | 不调 apt，不需要 sudo |
| `LD_LIBRARY_PATH` 永久污染 | `run_test.sh` 每次只在子进程 export，不写 `~/.bashrc` |
| `/usr/local/cuda` 切换 | 只**读**这个 symlink，不改它指向 |
| PATH 污染 | 完全不动 |

可以随时 `rm -rf nccl/build nccl-tests/build` 清干净，机器完全回到原样。

## 常见问题

**Q: 跑测试报 `libnccl.so.2: cannot open shared object file`**
A: 忘记 export `LD_LIBRARY_PATH`。用 `run_test.sh` 包装好；或手动：
```bash
export LD_LIBRARY_PATH="$HOME/nccl-repo/nccl/build/lib:$LD_LIBRARY_PATH"
```

**Q: 链接到了系统 NCCL 而不是我编的**
A: `ldd ~/nccl-repo/nccl-tests/build/all_reduce_perf | grep nccl`，看路径。如果指错了，确认 `LD_LIBRARY_PATH` 把 `nccl/build/lib` 放在最前面。

**Q: 我想看 NCCL 选了哪种 algo / proto / transport**
A: `NCCL_DEBUG=INFO` 看握手；`NCCL_DEBUG=TRACE` 看更细节；`NCCL_DEBUG_SUBSYS=ALL` 全开。

**Q: 单机 2 卡跑 sendrecv 跨机的逻辑能测吗**
A: 同进程内 2 卡只能测同机段（P2P/SHM）。要测跨机段，得 `mpirun` 启两机两进程，且 nccl-tests 编译时打开 `MPI=1`：
```bash
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi NCCL_HOME=~/nccl-repo/nccl/build
mpirun -np 2 -H hostA:1,hostB:1 ./all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

**Q: 编译失败说找不到 `cuda_runtime.h`**
A: `CUDA_HOME` 没指对。`ls $CUDA_HOME/bin/nvcc` 应该存在，`ls $CUDA_HOME/include/cuda_runtime.h` 也应该存在。

**Q: A100 之外的卡怎么编**
A: 改 `NVCC_GENCODE`。常见对应：
- V100: `sm_70`
- T4: `sm_75`
- A100 / A30: `sm_80`
- A40 / RTX 30xx: `sm_86`
- H100 / H200: `sm_90`
- B100: `sm_100`

## 文件清单

```
nccl-bench/
├── build_nccl.sh   # 构建脚本（顶部有完整 --help）
├── run_test.sh     # 测试封装脚本（顶部有完整 --help）
└── README.md       # 本文档
```
