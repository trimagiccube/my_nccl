# Vendored files

> 所有文件复制自 `../../nccl/` submodule, 基线 commit **8fb057cda230f6c8f6ed50dfcc92e40e1d88772c** (v2.23.4-1-5-g8fb057c).
> 改动遵循 NCCL 的 BSD license (LICENSE.txt 在 ../../nccl/).

## Phase 1 (本次)

| my_nccl 路径 | 来源 nccl 路径 | 是否已改 |
|---|---|---|
| `include/nccl.h` | `build/include/nccl.h` (构建产物, 已展开 .h.in 模板) | ☐ 未改 |
| `include/socket.h` | `src/include/socket.h` | ☐ 未改 |
| `include/bootstrap.h` | `src/include/bootstrap.h` | ☐ 未改 |
| `include/utils.h` | `src/include/utils.h` | ☐ 未改 |
| `include/debug.h` | `src/include/debug.h` | ☐ 未改 |
| `include/transport.h` | `src/include/transport.h` | ☐ 未改 |
| `include/comm.h` | `src/include/comm.h` | ☐ 未改 (Phase 2 会 stub 字段) |
| `include/core.h` | `src/include/core.h` | ☐ 未改 |
| `include/param.h` | `src/include/param.h` | ☐ 未改 |
| `src/vendor/socket.cc` | `src/misc/socket.cc` | ☐ 未改 |
| `src/vendor/bootstrap.cc` | `src/bootstrap.cc` | ☐ 未改 |
| `src/vendor/utils.cc` | `src/misc/utils.cc` | ☐ 未改 |
| `src/vendor/debug.cc` | `src/debug.cc` | ☐ 未改 |
| `src/vendor/param.cc` | `src/misc/param.cc` | ☐ 未改 |

每次 `Phase X` 中有修改就在表里标记 ✓，并在下面单独列改动点。

## 改动清单

### Phase 2

- **`include/nvtx.h`** — **整体替换为空 stub**。CUDA 12.9 自带的 nvtx3 头里没有 `nvtxPayloadSchemaEntry_t` / `nvtxPayloadSchemaAttr_t`（NCCL 主线用的 payload_v2 API），编不过。my_nccl 不做 profiler tracing，所以把所有 `NVTX3_FUNC_*` 宏 no-op 化。如果将来恢复 profiler 把这文件还原 + 装新 nvtx3 即可。
- 其余 13 个 vendored 文件保持未改 ✓
