#!/usr/bin/env bash
# ============================================================================
# run.sh — 启动 2 个进程, 每个进程一张卡, 做 all_reduce
# ----------------------------------------------------------------------------
# 用法:
#   ./run.sh                # 跑测试 (会自动 make 一次)
#   ./run.sh -h | --help    # 显示这段帮助
#
# 环境变量:
#   NCCL_DIR    用户态 NCCL 路径 (默认: ../../nccl, 即 build_nccl.sh 的产物)
#   NRANKS      进程/GPU 数, 默认 2
#   NCCL_DEBUG  默认 WARN, 想看握手过程可设 INFO / TRACE
#
# 做了什么:
#   1) 若 ./all_reduce_2proc 不存在则 make
#   2) 用一个临时文件做 ncclUniqueId 分发: NCCL_UID_FILE=/tmp/nccl_uid_$$.bin
#   3) 并发拉起 NRANKS 个子进程, 每个传 (rank, nranks) 参数
#   4) wait 收集所有 rank 的退出码, 任何一个非 0 即整体失败
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
NRANKS="${NRANKS:-2}"

EXE="$SCRIPT_DIR/all_reduce_2proc"
if [ ! -x "$EXE" ]; then
  echo "[info] $EXE 不存在, 先 make 一次"
  make -C "$SCRIPT_DIR"
fi

[ -f "$NCCL_DIR/build/lib/libnccl.so" ] \
  || { echo "[err] 找不到 $NCCL_DIR/build/lib/libnccl.so, 先跑 nccl-bench/build_nccl.sh" >&2; exit 1; }

NGPUS=$(nvidia-smi -L | wc -l)
if [ "$NRANKS" -gt "$NGPUS" ]; then
  echo "[err] NRANKS=$NRANKS 超过本机 GPU 数 ($NGPUS)" >&2
  exit 1
fi

UID_FILE="/tmp/nccl_uid_$$.bin"
trap 'rm -f "$UID_FILE" "$UID_FILE".tmp.*' EXIT

echo "=== Running: $EXE, NRANKS=$NRANKS ==="
echo "    NCCL lib:      $NCCL_DIR/build/lib"
echo "    NCCL_DEBUG:    $NCCL_DEBUG"
echo "    NCCL_UID_FILE: $UID_FILE"
echo

PIDS=()
for ((r=0; r<NRANKS; ++r)); do
  LD_LIBRARY_PATH="$NCCL_DIR/build/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG="$NCCL_DEBUG" \
  NCCL_UID_FILE="$UID_FILE" \
    "$EXE" "$r" "$NRANKS" &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [ "$FAIL" -eq 0 ]; then echo "ALL RANKS OK"; else echo "SOME RANKS FAILED" >&2; fi
exit "$FAIL"
