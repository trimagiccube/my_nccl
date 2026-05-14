#!/usr/bin/env bash
# ============================================================================
# run_test.sh — 用本地构建的 NCCL 跑 nccl-tests
# ----------------------------------------------------------------------------
# 用法:
#   ./run_test.sh                       # all_reduce 默认参数 (-b 8 -e 1G -f 2)
#   ./run_test.sh allgather             # all_gather, 用默认参数
#   ./run_test.sh allreduce -b 1M -e 1G -f 2
#   ./run_test.sh sendrecv -b 8 -e 16M -f 2
#   ./run_test.sh --list                # 列出可用 perf 程序
#   ./run_test.sh -h | --help           # 显示这段帮助
#
# 简写映射:
#   ar  | allreduce      → all_reduce_perf
#   ag  | allgather      → all_gather_perf
#   rs  | reducescatter  → reduce_scatter_perf
#   bcast | broadcast    → broadcast_perf
#   reduce               → reduce_perf
#   a2a | alltoall       → alltoall_perf
#   sr  | sendrecv       → sendrecv_perf
#   gather / scatter / hypercube → *_perf
#
# 关键环境变量 (脚本会自动设置, 外部可覆盖):
#   NCCL_DIR        默认: 脚本同级的 ../nccl
#   NCCL_TESTS_DIR  默认: 脚本同级的 ../nccl-tests
#   NGPUS           默认: nvidia-smi -L 数量
#   NCCL_DEBUG      默认: WARN  (可设 INFO / TRACE 看握手过程)
#
# 常用 nccl-tests 参数透传 (写在测试名后面):
#   -b <size>   起始消息大小 (8 / 1K / 1M / 1G)
#   -e <size>   终止消息大小
#   -f <factor> 步进因子 (2 = 翻倍)
#   -n <iters>  每个 size 跑的迭代数 (默认 20)
#   -w <iters>  预热迭代数         (默认 5)
#   -c <0|1>    是否校验结果        (默认 1)
#   -o <op>     归约算子 (sum/prod/max/min/avg)
#   -d <dtype>  数据类型 (int32/float32/half/...)
# ============================================================================

set -euo pipefail

usage() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

# ---- 先处理 -h/--help, 避免任何副作用 ----
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage; exit 0
fi

# ---- 路径自动探测 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NCCL_DIR="${NCCL_DIR:-$REPO_ROOT/nccl}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$REPO_ROOT/nccl-tests}"
NGPUS="${NGPUS:-$(nvidia-smi -L | wc -l)}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

BUILD="$NCCL_TESTS_DIR/build"
[ -d "$BUILD" ] || { echo "nccl-tests not built at $BUILD. Run build_nccl.sh first." >&2; exit 1; }

list_progs() {
  echo "Available perf programs in $BUILD:"
  ls "$BUILD" | grep '_perf$' | sed 's/_perf$//' | column -c 80
}

if [ "${1:-}" = "--list" ]; then list_progs; exit 0; fi

# ---- 默认测试: all_reduce ----
TEST="${1:-all_reduce}"
if [ $# -gt 0 ]; then shift; fi

# ---- 简写映射 ----
case "$TEST" in
  ar|allreduce|all_reduce)         EXE="all_reduce_perf" ;;
  ag|allgather|all_gather)         EXE="all_gather_perf" ;;
  rs|reducescatter|reduce_scatter) EXE="reduce_scatter_perf" ;;
  bcast|broadcast)                 EXE="broadcast_perf" ;;
  reduce)                          EXE="reduce_perf" ;;
  a2a|alltoall)                    EXE="alltoall_perf" ;;
  sr|sendrecv)                     EXE="sendrecv_perf" ;;
  gather)                          EXE="gather_perf" ;;
  scatter)                         EXE="scatter_perf" ;;
  hypercube)                       EXE="hypercube_perf" ;;
  *) EXE="${TEST}_perf" ;;
esac

[ -x "$BUILD/$EXE" ] || { echo "No such test: $BUILD/$EXE" >&2; list_progs; exit 1; }

# 用户没传额外参数时给一组通用默认值: 8B → 1GB, 2x 步进
DEFAULT_ARGS=(-b 8 -e 1G -f 2 -g "$NGPUS")
if [ $# -eq 0 ]; then ARGS=("${DEFAULT_ARGS[@]}"); else ARGS=("$@" -g "$NGPUS"); fi

echo "=== Running: $EXE ${ARGS[*]} ==="
echo "    NCCL lib:   $NCCL_DIR/build/lib"
echo "    NCCL_DEBUG: $NCCL_DEBUG"
echo

LD_LIBRARY_PATH="$NCCL_DIR/build/lib:${LD_LIBRARY_PATH:-}" \
NCCL_DEBUG="$NCCL_DEBUG" \
  "$BUILD/$EXE" "${ARGS[@]}"
