#!/usr/bin/env bash
# ============================================================================
# build_nccl.sh — 用户态编译 NCCL + nccl-tests，零系统侧改动
# ----------------------------------------------------------------------------
# 用法:
#   ./build_nccl.sh                 # 完整流程: clean + build + smoke test
#   ./build_nccl.sh --no-clean      # 不 clean, 增量编
#   ./build_nccl.sh --no-test       # 不跑 smoke test
#   ./build_nccl.sh --only nccl     # 只编 NCCL
#   ./build_nccl.sh --only tests    # 只编 nccl-tests
#   ./build_nccl.sh -h | --help     # 显示这段帮助
#
# 关键环境变量 (外部可覆盖):
#   NCCL_DIR        NCCL 源码目录       (默认: 脚本同级的 ../nccl)
#   NCCL_TESTS_DIR  nccl-tests 源码目录 (默认: 脚本同级的 ../nccl-tests)
#   CUDA_HOME       CUDA 工具链路径     (默认: 自动挑能编 NCCL 的 12.x)
#   NVCC_GENCODE    目标架构            (默认: sm_80, A100)
#                   多卡架构示例:
#                   "-gencode=arch=compute_80,code=sm_80 \
#                    -gencode=arch=compute_90,code=sm_90"
#   JOBS            并行编译任务数      (默认: nproc)
#
# 产物:
#   $NCCL_DIR/build/lib/libnccl.so.*    # 用户态 NCCL
#   $NCCL_DIR/build/include/nccl.h
#   $NCCL_TESTS_DIR/build/*_perf        # 链接到上面的 nccl-tests 可执行
#
# 不动系统目录:
#   不 make install / 不 ldconfig / 不改 /usr/local。清理只需:
#     rm -rf $NCCL_DIR/build $NCCL_TESTS_DIR/build
# ============================================================================

set -euo pipefail

usage() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

# ---- 解析参数 (先处理 -h/--help, 避免之后任何副作用) ----
DO_CLEAN=1
DO_TEST=1
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-clean) DO_CLEAN=0; shift;;
    --no-test)  DO_TEST=0;  shift;;
    --only)     ONLY="${2:-}"; shift 2;;
    -h|--help)  usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

# ---- 路径自动探测 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NCCL_DIR="${NCCL_DIR:-$REPO_ROOT/nccl}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$REPO_ROOT/nccl-tests}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_80,code=sm_80}"
JOBS="${JOBS:-$(nproc)}"

# 自动选 CUDA: 用户显式给了就用; 否则优先挑能编 NCCL 的 12.x
# (CUDA 13.0 删了 PFN_cuMulticastCreate 这类非版本号别名, 会让 NCCL master 编不过)
pick_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ]; then echo "$CUDA_HOME"; return; fi
  for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
    [ -x "$c/bin/nvcc" ] || continue
    if grep -q 'define PFN_cuMulticastCreate\b' "$c/include/cudaTypedefs.h" 2>/dev/null \
       || ! grep -q 'PFN_cuMulticastCreate_v' "$c/include/cudaTypedefs.h" 2>/dev/null; then
      echo "$c"; return
    fi
  done
  echo /usr/local/cuda
}
CUDA_HOME="$(pick_cuda_home)"

# ---- 颜色输出 ----
C_RESET=$'\e[0m'; C_BLUE=$'\e[34m'; C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'
log()  { echo "${C_BLUE}[$(date +%H:%M:%S)]${C_RESET} $*"; }
warn() { echo "${C_YELLOW}[warn]${C_RESET} $*" >&2; }
err()  { echo "${C_RED}[err]${C_RESET}  $*" >&2; exit 1; }
ok()   { echo "${C_GREEN}[ok]${C_RESET}   $*"; }

# ---- 预检 ----
log "Preflight checks"
[ -d "$NCCL_DIR" ]        || err "NCCL_DIR not found: $NCCL_DIR  (set NCCL_DIR=... to override)"
[ -d "$NCCL_TESTS_DIR" ]  || err "NCCL_TESTS_DIR not found: $NCCL_TESTS_DIR"
[ -x "$CUDA_HOME/bin/nvcc" ] || err "nvcc not found under $CUDA_HOME/bin"
"$CUDA_HOME/bin/nvcc" --version | tail -2 | sed 's/^/        /'

# 检查 PFN_cuMulticast 非版本号别名是否存在 (CUDA 13 删了, NCCL master 仍依赖)
if [ -f "$CUDA_HOME/include/cudaTypedefs.h" ] \
   && grep -q "PFN_cuMulticastCreate_v" "$CUDA_HOME/include/cudaTypedefs.h" \
   && ! grep -q "define PFN_cuMulticastCreate\b" "$CUDA_HOME/include/cudaTypedefs.h"; then
  warn "$CUDA_HOME 缺少 PFN_cuMulticastCreate 非版本号别名"
  warn "(CUDA 13.0+ 移除了这些别名)。NCCL 编译预计失败。"
  warn "建议改用 CUDA 12.6 / 12.8 / 12.9: CUDA_HOME=/usr/local/cuda-12.9 $0"
fi
echo "        CUDA_HOME      = $CUDA_HOME"
echo "        NCCL_DIR       = $NCCL_DIR"
echo "        NCCL_TESTS_DIR = $NCCL_TESTS_DIR"
echo "        NVCC_GENCODE   = $NVCC_GENCODE"
echo "        JOBS           = $JOBS"

# ---- 步骤 1: 编 NCCL ----
build_nccl() {
  log "Building NCCL"
  cd "$NCCL_DIR"
  if [ "$DO_CLEAN" = "1" ]; then
    log "  make clean (NCCL)"
    make clean >/dev/null 2>&1 || true
  fi
  # NCCL Makefile 内部使用 CUDA_HOME / NVCC_GENCODE; BUILDDIR 默认就是 ./build
  log "  make -j${JOBS} src.build  (~30s ~ 几分钟)"
  time make -j"${JOBS}" src.build \
    CUDA_HOME="$CUDA_HOME" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    2>&1 | tail -20

  local lib="$NCCL_DIR/build/lib/libnccl.so"
  [ -f "$lib" ] || err "Expected $lib not produced"
  local realname
  realname=$(readlink -f "$lib")
  ok "NCCL built: $realname"
  echo "        nccl version: $(strings "$realname" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\+' | head -1 || echo unknown)"
}

# ---- 步骤 2: 编 nccl-tests ----
build_tests() {
  log "Building nccl-tests"
  cd "$NCCL_TESTS_DIR"
  if [ "$DO_CLEAN" = "1" ]; then
    log "  make clean (nccl-tests)"
    make clean >/dev/null 2>&1 || true
  fi
  log "  make -j${JOBS} (link to user-built NCCL)"
  time make -j"${JOBS}" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_DIR/build" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    2>&1 | tail -20

  local exe="$NCCL_TESTS_DIR/build/all_reduce_perf"
  [ -x "$exe" ] || err "Expected $exe not produced"
  ok "nccl-tests built"
  echo "        Linkage to user NCCL:"
  LD_LIBRARY_PATH="$NCCL_DIR/build/lib:${LD_LIBRARY_PATH:-}" \
    ldd "$exe" | grep -E "libnccl|libcudart" | sed 's/^/          /'
}

# ---- 步骤 3: smoke test ----
smoke_test() {
  local nGpus
  nGpus=$(nvidia-smi -L | wc -l)
  log "Smoke test: all_reduce_perf on $nGpus GPU(s)"
  cd "$NCCL_TESTS_DIR/build"
  LD_LIBRARY_PATH="$NCCL_DIR/build/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=WARN \
    ./all_reduce_perf -b 8 -e 64M -f 2 -g "$nGpus"
  ok "Smoke test done"
}

# ---- 执行 ----
case "$ONLY" in
  nccl)  build_nccl ;;
  tests) build_tests ;;
  "")    build_nccl; build_tests ;;
  *)     err "--only must be one of: nccl | tests" ;;
esac

if [ "$DO_TEST" = "1" ] && [ -z "$ONLY" -o "$ONLY" = "tests" ]; then
  smoke_test
fi

echo
ok "All done."
cat <<EOF

下次手动跑测试, 先 export 这一行 (或直接用 run_test.sh):

  export LD_LIBRARY_PATH="$NCCL_DIR/build/lib:\$LD_LIBRARY_PATH"

然后比如:

  $NCCL_TESTS_DIR/build/all_reduce_perf -b 8 -e 1G -f 2 -g $(nvidia-smi -L | wc -l)
EOF
