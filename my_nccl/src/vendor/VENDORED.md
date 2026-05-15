# Vendored files

> 所有文件复制自 `../../nccl/` submodule, 基线 commit **8fb057cda230f6c8f6ed50dfcc92e40e1d88772c** (v2.23.4-1-5-g8fb057c).
> 改动遵循 NCCL 的 BSD license (LICENSE.txt 在 ../../nccl/).

## 当前 vendor 状态

`include/` 下保留 **31 个** header（Phase 1+2 实际需要的最小集；用 `g++ -M` 算出闭包后裁剪）。
`src/vendor/` 下保留 **5 个** .cc。

### include/ (31 个)

```
alloc.h       bitops.h         bootstrap.h    checks.h       collectives.h
comm.h        core.h           cudawrap.h     debug.h        device.h
graph.h       info.h           ipcsocket.h    nccl_common.h  nccl.h
nccl_net.h    nccl_profiler.h  nccl_tuner.h   net_device.h   net.h
nvmlwrap.h    nvtx.h           p2p.h          param.h        profiler.h
proxy.h       register.h       shmutils.h     socket.h       strongstream.h
utils.h
```

所有 header 来源 `nccl/src/include/<同名>.h`（基线 8fb057c），其中 **`nccl.h` 例外**——拷自 `nccl/build/include/nccl.h`（构建产物，已展开 .h.in 模板）。

### src/vendor/ (5 个)

| my_nccl 路径 | 来源 nccl 路径 |
|---|---|
| `src/vendor/socket.cc` | `src/misc/socket.cc` |
| `src/vendor/bootstrap.cc` | `src/bootstrap.cc` |
| `src/vendor/utils.cc` | `src/misc/utils.cc` |
| `src/vendor/debug.cc` | `src/debug.cc` |
| `src/vendor/param.cc` | `src/misc/param.cc` |

### Phase 3+ 预计要补回来的 header

由后续 vendor `graph/topo.cc` + `graph/xml.cc` 触发再拷回：可能涉及 `transport.h`、`coll_net.h`、`gdrwrap.h`、`ibv*.h`、`channel.h`、`cpuset.h` 等。**按需补**，不预先全拷。

## 改动清单

### Phase 2

- **`include/nvtx.h`** — **整体替换为空 stub**。CUDA 12.9 自带的 nvtx3 头里没有 `nvtxPayloadSchemaEntry_t` / `nvtxPayloadSchemaAttr_t`（NCCL 主线用的 payload_v2 API），编不过。my_nccl 不做 profiler tracing，所以把所有 `NVTX3_FUNC_*` 宏 no-op 化。如果将来恢复 profiler 把这文件还原 + 装新 nvtx3 即可。
- 其余 13 个 vendored 文件保持未改 ✓
