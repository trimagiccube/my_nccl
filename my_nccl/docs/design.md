# my_nccl 设计说明

> 这份文档说明实现路径 **B** 的具体策略：从 NCCL vendor 哪些文件、要 stub 什么、wire format 怎么对齐。

## 总体策略

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ NCCL 完整源码 ~50k 行                                            │
   │                                                                  │
   │   ┌───── 我们要的 ─────┐    ┌──── 不要的 (替换成 stub) ────┐    │
   │   │                   │     │                               │    │
   │   │ socket.cc         │     │ transport/*  (P2P/IB/SHM/Net) │    │
   │   │ bootstrap.cc      │     │ proxy.cc                      │    │
   │   │ graph/topo.cc     │     │ enqueue.cc / collectives.cu   │    │
   │   │ graph/xml.cc      │     │ device/*.cu (NCCL kernels)    │    │
   │   │ debug.cc          │     │ init.cc (1800 行, 我们只要其 │    │
   │   │ utils.cc          │     │   中调 Phase A+B 的那一段)    │    │
   │   │ + 各自的 .h       │     │ misc/{net,cudawrap,...}       │    │
   │   │                   │     │                               │    │
   │   └───────────────────┘     └───────────────────────────────┘    │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

   最终 my_nccl 二进制 ~ 5-8k 行 (含 NCCL vendor 的部分)
   + 我们自己写的 main.cc ~ 100-200 行
   + stubs ~ 100 行
```

## 关键约束（已锁定）

1. **不改 NCCL 子模组本身**。所有改动落在 `my_nccl/src/vendor/`。`../nccl/` 仍是干净的上游 submodule。
2. **保留 NCCL 的 BSD license header**。每个 vendored 文件顶部加一行 `// Vendored from nccl/src/<path>:<commit>` 标注来源。
3. **基线 commit**：`8fb057c (v2.23.4-1-5)`，跟 `../nccl/` submodule 一致。
4. **`nccl.h` 直接拷构建产物** `../nccl/build/include/nccl.h`（已展开模板，省事），不再处理 `.h.in` 模板变量。
5. **vendored 文件保留原名** —— `bootstrap.cc` / `topo.cc` 等不加前缀，用 `src/vendor/` 目录区分。方便跟上游 `git blame` / `diff` 对照。
6. **`comm.h` 完整 vendor + stub 字段**（不另写 `mini_comm.h`）—— 这样能直接用 NCCL 函数签名，少改动。
7. **默认编译开 ASan + UBSan** —— `-fsanitize=address,undefined -fno-omit-frame-pointer`。教学项目，bug 早发现。Release 模式留个 Makefile target 关掉。

## 必需的 NCCL 源文件清单（初步）

下面这张表是 Phase 1-3 的"采购清单"。实际开搬时按这个顺序，每次拷 1-2 个再编译看缺啥。

| 路径 | 大小 | 必要性 | 备注 |
|---|---|---|---|
| `src/include/nccl.h.in` | ~10 KB | 必须 | 拷 `nccl/build/include/nccl.h`（已展开模板）更省事 |
| `src/include/socket.h` | ~2 KB | 必须 | TCP 原语 |
| `src/misc/socket.cc` | ~30 KB | 必须 | TCP 原语实现 |
| `src/include/bootstrap.h` | ~2 KB | 必须 | bootstrap API |
| `src/bootstrap.cc` | ~40 KB | 必须 | bootstrap 实现 |
| `src/include/transport.h` | ~10 KB | 必须 | `ncclPeerInfo` 结构定义 |
| `src/include/comm.h` | ~25 KB | 必须 | `ncclComm` 结构（要 stub 大量字段） |
| `src/include/utils.h` | ~5 KB | 必须 | `ncclCalloc` 等 |
| `src/utils.cc` | ~10 KB | 必须 | 同上 |
| `src/include/debug.h` | ~3 KB | 必须 | INFO/WARN 宏 |
| `src/debug.cc` | ~10 KB | 必须 | 日志实现 |
| `src/graph/topo.h` | ~8 KB | 必须 | `ncclTopoSystem` 结构 |
| `src/graph/topo.cc` | ~30 KB | 必须 | `ncclTopoGetSystem` 等 |
| `src/graph/xml.h` | ~5 KB | 必须 | XML 结构 |
| `src/graph/xml.cc` | ~30 KB | 必须 | XML 读写 |
| `src/graph/paths.cc` | ~20 KB | 不确定 | Phase B 只到生成 ncclTopoSystem。如果 `topo.cc` 里有调 `ncclTopoComputePaths` 才需要 |
| `src/include/graph.h` | ~5 KB | 必须 | 函数声明 |

## 必须 stub 掉的依赖

NCCL 的 `comm` 结构互相牵连，bootstrap/topo 代码里会读它的字段。哪些可以塞默认值、哪些必须实现：

| 字段 / 函数 | bootstrap/topo 是否真用 | 处理 |
|---|---|---|
| `comm->bootstrap` | ✅ 是，存的就是 bootstrap socket state | 必须保留 |
| `comm->rank, nRanks` | ✅ 是 | 必须保留 |
| `comm->peerInfo` | ✅ 是 | 必须保留 |
| `comm->topo` | ✅ 是（输出） | 必须保留 |
| `comm->commHash, magic` | ✅ 是 | 必须保留 |
| `comm->channels[]` | ❌ Phase C 之后才用 | stub 成空数组 |
| `comm->proxyState` | ❌ Phase F 才用 | stub 成 NULL |
| `comm->sharedRes` | ❌ 同上 | stub |
| `comm->ncclNet` | ⚠️ topo.cc 探测 NIC 时会用 | 提供一个返回 0 NIC 的 stub plugin |
| `ncclTransportP2pSetup` | ❌ Phase F | 写个空实现 |
| `ncclProxyConnect` | ❌ Phase F | 写个空实现 |
| `ncclNvlsInit` | ❌ Phase C | 直接返回 ncclSuccess |
| `collNetSupport` | ❌ Phase C | 返回 0 |

## 怎么不被 init.cc 拖累

NCCL 的入口 `ncclCommInitRank` 在 `init.cc` 里，1800+ 行，混了 Phase A-F 所有逻辑。我们**不 vendor init.cc**，直接自己写 main.cc，按这个流程：

```c
// 我们的 main.cc 流程伪代码:

int main(int argc, char **argv) {
    int rank = atoi(argv[1]);
    int nranks = atoi(argv[2]);

    // 跟 NCCL 大致一样的初始化
    ncclComm *comm = my_calloc_comm(rank, nranks);

    // === Step 1-4: rendezvous ===
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
        write_uid_file(id, argv[3]);
    } else {
        read_uid_file(&id, argv[3]);
    }
    bootstrapNetInit();
    bootstrapInit(1, &id, comm);   // 各 rank join 进通信组

    // === Step 5 Phase A: ncclPeerInfo allgather ===
    comm->peerInfo = calloc(nranks, sizeof(ncclPeerInfo));
    fillInfo(comm, &comm->peerInfo[rank], comm->commHash);   // 复用 NCCL 的
    bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(ncclPeerInfo));

    // === Step 5 Phase B: 拓扑探测 ===
    ncclTopoGetSystem(comm, &comm->topo);   // 复用 NCCL 完整调用

    // 输出 XML
    if (rank == 0) {
        ncclTopoDumpXmlToFile(argv[4], /* xml 从 topo 反算出来 */);
    }

    // 干净退出
    bootstrapClose(comm->bootstrap);
    return 0;
}
```

注意 `ncclTopoGetSystem` 里头会用到 `comm->peerInfo` (要先做 Phase A) 和 `comm->bootstrap` (要先做 rendezvous)，所以**顺序必须严格**。

## wire format 兼容性

**当前范围**：用户场景是 my_nccl 内部 N 个进程互通。**不要求**跟真实 NCCL 进程互操作。

如果以后想做"my_nccl 进程能跟真 NCCL 进程组队"（路径 C 的扩展），需要：
1. `ncclUniqueId` 的 128 字节 layout 跟 NCCL 完全一致 (这个 vendor `bootstrap.cc` 自动保证)
2. bootstrap socket 上的消息格式跟 NCCL 一致 (同上)
3. `ncclPeerInfo` 结构 layout 一致 (vendor `transport.h` 自动保证)
4. `version` 字段填一致的 NCCL_VERSION_CODE

由于全部代码 vendor 自 NCCL 自己，**理论上 wire format 天生兼容**。可以在 Phase 5 后做个 mini-experiment：起 1 个 my_nccl + 1 个真 NCCL 进程，看能不能互相 rendezvous（预期能，因为 wire format 同源）。

## 构建系统

简单 Makefile：

```makefile
NCCL_VENDOR = src/vendor
CUDA_HOME ?= /usr/local/cuda-12.9

CXX     = $(CUDA_HOME)/bin/nvcc
# 教学项目: 默认开 ASan + UBSan + 完整调试符号
SANITIZE ?= -fsanitize=address,undefined -fno-omit-frame-pointer
CXXFLAGS = -O0 -g -std=c++17 $(SANITIZE) -Iinclude -I$(CUDA_HOME)/include
# Release 模式: make SANITIZE= my_nccl_bootstrap
LDFLAGS  = -L$(CUDA_HOME)/lib64 -lcudart -lnvidia-ml -lpthread $(SANITIZE)

SRCS = src/main.cc \
       $(NCCL_VENDOR)/socket.cc \
       $(NCCL_VENDOR)/bootstrap.cc \
       $(NCCL_VENDOR)/utils.cc \
       $(NCCL_VENDOR)/debug.cc \
       $(NCCL_VENDOR)/topo.cc \
       $(NCCL_VENDOR)/xml.cc \
       src/stubs/transport_stub.cc \
       src/stubs/proxy_stub.cc

my_nccl_bootstrap: $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
```

依赖：`libcudart` (`cudaSetDevice` 等)、`libnvidia-ml` (`nvmlDeviceGet*`)、`pthread`。

不依赖 `libnccl.so` —— 这是关键，我们是 NCCL 的"切片"，不是 wrapper。

## 验收：拓扑 XML 一致性

参考 `aig-a100-topo.xml` (NCCL 真实 dump)，目标是 `diff` 为空或只剩白名单字段。

可能的合理差异：
- NCCL 内部用的临时 `<keep="1">` 等中间属性可能被 trim 掉（应该一样）
- xml namespace / DOCTYPE 等 (NCCL 不用)
- 元素顺序 (XML 语义无关)

如果差异过大，说明 vendor 出来的 topo.cc 哪里被 stub 错了 —— 回去补。
