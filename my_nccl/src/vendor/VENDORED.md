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

### Phase 3a

- **`include/comm.h`** — **裁掉 ncclComm 几乎所有字段**。原 ncclComm 是 line 399-603 共 205 行 ~80 字段，my_nccl 只用到 **8 个**：
  - `rank`, `nRanks` — 我自己设
  - `cudaDev`, `magic`, `bootstrap` — bootstrapInit 内部读/写
  - `abortFlag` — bootstrap.cc 多处读, 我们传一个指向 0 的 uint32_t
  - `ncclNet` — 只在 OOB_NET_ENABLE=1 分支读, 默认为 0 不会走到
  - `topParentRanks` — 只在 bootstrapSplit 用, 我们不调
  其余字段（intraBarrierGate / intraComm0 / sharedRes / channels / proxyState / collNetSupport / nvlsSupport / ...）全部去掉。同时去掉了文件末尾几个引用裁掉字段的 inline helper（ncclCommPollCallbacks / IntraBarrierIn/Out / UserRedOpMangle / 两个 forward decl），它们一个都没人调。
  **结果**: comm.h 从 707 行降到 428 行 (-279 行)。
- **`src/vendor/bootstrap.cc`** — **bootstrapSplit (line 729-806) 用 `#if 0/#endif` 包住**。它访问 `comm->config.splitShare`，而 config 字段被我们裁掉了。整个 split 子组功能 my_nccl 用不到，**禁用整段函数**比"为它再加 stub 字段"干净。注意 `socketConnect` 这个小 helper 被原本顺手定义在 bootstrapSplit 后面，挪到 `#endif` 之外保留（其他函数还要用）。
- **`src/stubs/proxy_stub.cc`** — Phase 2 那个 no-op stub 现在改成"接管/释放"实现：bootstrap.cc:1118 注释明说 "proxy things are free'd elsewhere"，真 ncclProxyInit 会把 sock/peerAddresses/peerAddressesUDS 三块 alloc 接管。我们 stub 不起 proxy 线程，但必须 free 这三块，否则 ASan 报 leak。
- **`include/transport.h`** — 上 Phase 1 时被一并裁掉（因为还没用），Phase 3a 用 ncclPeerInfo 又拷回来。
