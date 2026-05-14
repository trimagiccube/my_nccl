# nccl-repo

> 个人用的 NCCL 工作区：上游源码（submodule）+ 用户态构建脚本 + 自己写的最小学习样例。

## 布局

```
nccl-repo/
├── nccl/              # submodule → github.com:NVIDIA/nccl
├── nccl-tests/        # submodule → github.com:NVIDIA/nccl-tests
├── nccl-bench/        # 构建 + 跑 nccl-tests 的封装脚本
│   ├── build_nccl.sh
│   ├── run_test.sh
│   └── README.md
└── my_nccl_test/      # 自己写的最小学习样例
    ├── all_reduce_2gpu/    # 单进程 2 卡 (ncclCommInitAll + Group)
    └── all_reduce_2proc/   # 多进程 2 卡 (ncclGetUniqueId + ncclCommInitRank)
```

## 克隆

```bash
git clone --recurse-submodules <repo-url>
# 或者已经 clone 完了:
git submodule update --init --recursive
```

## 上手

```bash
# 1. 编 NCCL + nccl-tests (用户态, 不动系统)
cd nccl-bench && ./build_nccl.sh

# 2. 跑官方 perf 测试
./run_test.sh allreduce

# 3. 跑自己的最小样例
cd ../my_nccl_test/all_reduce_2gpu && ./run.sh
cd ../all_reduce_2proc          && ./run.sh
```

每个子目录都有自己的 `README.md` 和 `--help`，细节看那里。

## 不在 git 里的东西

`.gitignore` 排除：
- 各级 `build/` 目录（编译产物，几百 MB）
- `my_nccl_test/*/all_reduce_{2gpu,2proc}` 二进制
- 编辑器临时文件

`nccl/` 和 `nccl-tests/` 是 submodule，本仓只记录 HEAD SHA；它们的 `build/` 和工作区内的文件类型 / 权限变化（filesystem 副作用）都不会进父仓。
