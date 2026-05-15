#!/usr/bin/env bash
# ============================================================================
# run.sh — 拉起 NRANKS 个 my_nccl_bootstrap 进程做 bootstrap rendezvous + Phase A
# ----------------------------------------------------------------------------
# 用法:
#   ./run.sh                # 跑测试 (默认 NRANKS=2, 会自动 make)
#   ./run.sh -h | --help    # 显示这段帮助
#
# 环境变量:
#   NRANKS      进程数, 默认 2
#
# 做了什么:
#   1) 若 ./my_nccl_bootstrap 不存在则 make
#   2) 用 /tmp/my_nccl_uid_$$.bin 作为 UID 分发文件 (shell PID 防撞)
#   3) 并发拉起 NRANKS 个子进程, 每个传 (rank, nranks, uid_file)
#   4) wait 收集所有 rank 的退出码, 任何一个非 0 即整体失败
# ============================================================================

set -euo pipefail

usage() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage; exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NRANKS="${NRANKS:-2}"

EXE="$SCRIPT_DIR/my_nccl_bootstrap"
if [ ! -x "$EXE" ]; then
  echo "[info] $EXE 不存在, 先 make 一次"
  make -C "$SCRIPT_DIR"
fi

UID_FILE="/tmp/my_nccl_uid_$$.bin"
trap 'rm -f "$UID_FILE" "$UID_FILE".tmp.*' EXIT

echo "=== Running: $EXE, NRANKS=$NRANKS ==="
echo "    UID_FILE: $UID_FILE"
echo

PIDS=()
for ((r=0; r<NRANKS; ++r)); do
  "$EXE" "$r" "$NRANKS" "$UID_FILE" &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

echo
if [ "$FAIL" -eq 0 ]; then
  echo "===== ALL RANKS OK ====="
else
  echo "===== SOME RANKS FAILED =====" >&2
fi
exit "$FAIL"
