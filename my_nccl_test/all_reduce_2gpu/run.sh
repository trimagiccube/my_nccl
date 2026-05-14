#!/usr/bin/env bash
# ============================================================================
# run.sh — 运行 my_nccl_test/all_reduce_2gpu/all_reduce_2gpu
# ----------------------------------------------------------------------------
# 用法:
#   ./run.sh                # 跑测试 (会自动 make 一次)
#   ./run.sh -h | --help    # 显示这段帮助
#
# 环境变量:
#   NCCL_DIR    用户态 NCCL 路径 (默认: ../../nccl, 即 build_nccl.sh 的产物)
#   NCCL_DEBUG  默认 WARN, 想看握手过程可设 INFO / TRACE
#
# 做了什么:
#   1) 若 ./all_reduce_2gpu 不存在则 make
#   2) export LD_LIBRARY_PATH 指向用户构建的 libnccl.so
#   3) 跑可执行, 退出码透传出来 (0 = 通过)
# ============================================================================

set -euo pipefail

usage() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage; exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NCCL_DIR="${NCCL_DIR:-$REPO_ROOT/nccl}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

EXE="$SCRIPT_DIR/all_reduce_2gpu"
if [ ! -x "$EXE" ]; then
  echo "[info] $EXE 不存在, 先 make 一次"
  make -C "$SCRIPT_DIR"
fi

[ -f "$NCCL_DIR/build/lib/libnccl.so" ] \
  || { echo "[err] 找不到 $NCCL_DIR/build/lib/libnccl.so, 先跑 nccl-bench/build_nccl.sh" >&2; exit 1; }

echo "=== Running: $EXE ==="
echo "    NCCL lib:   $NCCL_DIR/build/lib"
echo "    NCCL_DEBUG: $NCCL_DEBUG"
echo

LD_LIBRARY_PATH="$NCCL_DIR/build/lib:${LD_LIBRARY_PATH:-}" \
NCCL_DEBUG="$NCCL_DEBUG" \
  "$EXE"
