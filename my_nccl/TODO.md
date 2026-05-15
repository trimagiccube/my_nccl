# TODO

> 每个 Phase 结束的标志是"能编译 / 能跑出预期"，然后 commit。Phase 0 完成（骨架就位）。

---

## Phase 1 — Vendor 基础源文件

目标：把 bootstrap 协议层需要的最少 NCCL 源文件搬过来，stub 掉无关字段，**先不动 topo**。

- [ ] 1.1 拷 `nccl/build/include/nccl.h`（构建产物, 已展开模板）→ `include/nccl.h`
- [ ] 1.2 拷 `nccl/src/include/socket.h`, `nccl/src/misc/socket.cc` → `include/`, `src/vendor/`
- [ ] 1.3 拷 `nccl/src/include/bootstrap.h`, `nccl/src/bootstrap.cc`
- [ ] 1.4 拷 `nccl/src/include/utils.h`, `nccl/src/utils.cc`
- [ ] 1.5 拷 `nccl/src/include/debug.h`, `nccl/src/debug.cc`
- [ ] 1.6 拷 `nccl/src/include/transport.h`（只为 `ncclPeerInfo` 结构）
- [ ] 1.7 拷 `nccl/src/include/comm.h`，**stub** 掉所有 channel/proxy/kernel/CUDA stream 相关字段
- [ ] 1.8 整理 `#include` 链，把所有未满足依赖列成表（决定下一步是 stub 还是再 vendor）

**完成条件**：上面 7 个文件能单独 `g++ -c` 编过（可能允许 main 链接错误，但 .o 都出来）。

---

## Phase 2 — Stub transport/proxy/kernel

目标：让 vendored 代码能链接成功，**不要求功能正确**，只要符号都解析得了。

- [ ] 2.1 列出 `bootstrap.cc` 里所有访问 `comm->xxx` 的字段，确认哪些 stub 字段被读、需要"看起来合理"的默认值
- [ ] 2.2 写 `src/stubs/transport_stub.cc`，桩掉 `ncclTransportP2pSetup`, `selectTransport<>`, proxy 启动等 `ncclTopoGetSystem` 调用链里碰到的符号
- [ ] 2.3 写 `src/stubs/proxy_stub.cc`，桩掉 `ncclProxyConnect` 等
- [ ] 2.4 写 `src/stubs/kernel_stub.cc`，桩掉 CUDA kernel 加载相关（如果链接路径上有的话）
- [ ] 2.5 写最小 `main.cc`：只调 `ncclGetUniqueId` 并打印；目标是验证链接通过

**完成条件**：`./my_nccl_minimal` 单进程能跑起来，调出 ncclUniqueId 并打印。

---

## Phase 3 — Vendor 拓扑层

目标：把 Phase B 需要的 topo + xml 代码搬过来。

- [ ] 3.1 拷 `nccl/src/graph/topo.h`, `topo.cc`
- [ ] 3.2 拷 `nccl/src/graph/xml.h`, `xml.cc`
- [ ] 3.3 看 `topo.cc` 里有没有引用 `paths.cc` 的内容（如 `ncclTopoComputePaths`）。**Phase B 只到生成 ncclTopoSystem，不算路径**，所以 `paths.cc` 大概率不需要
- [ ] 3.4 stub `collNetSupport()` / `collNetDevices()` / `ncclNvlsInit()` → 直接返回 0
- [ ] 3.5 stub IB 部分（`ncclNet->devices()` 等）→ 返回 0 个设备；或者保留 net plugin 加载但跳过 IB
- [ ] 3.6 跟前面 stub 一起编译，让 `ncclTopoGetSystem` 这个函数能被引用并链接成功

**完成条件**：链接通过，未跑。

---

## Phase 4 — 写 main.cc + 构建系统

目标：可以单进程跑通"调 bootstrapGetUniqueId → 调拓扑探测 → dump XML"。

- [ ] 4.1 `src/main.cc`：
    - 解析 argv: `<rank> <nranks> <uid_file_path>`
    - 把 NCCL 大段 init 逻辑里**只有 Phase A + Phase B 用到的那段**复制过来到我的 main
    - rank 0: `ncclGetUniqueId(&id)`, 写文件
    - rank > 0: 轮询读 uid 文件
    - 所有 rank: 用裸 socket 跑 bootstrapInit
    - 用裸 socket 跑一次 `bootstrapAllGather` 同步 `ncclPeerInfo`
    - 调 `ncclTopoGetSystem` 走完 Phase B 全套
    - rank 0 调 `ncclTopoDumpXmlToFile` 把融合后的 XML 落盘
    - 各 rank 调 `bootstrapClose` 干净退出
- [ ] 4.2 `Makefile`：链接 libcudart + libnvidia-ml；指向 `include/`；编出 `my_nccl_bootstrap`
- [ ] 4.3 `run.sh`：起 NRANKS 个进程，传 rank/nranks/uid_file，等所有进程返回
- [ ] 4.4 第一次成功的单进程 + 双进程 run

**完成条件**：`./run.sh` 输出 `/tmp/my_topo.xml`，文件存在且语法是 well-formed XML。

---

## Phase 5 — 验收

目标：跟真实 NCCL 的 XML 一致。

- [ ] 5.1 `tests/test_topo_match.sh`：
    ```bash
    NCCL_TOPO_DUMP_FILE=/tmp/nccl_real_topo.xml ../my_nccl_test/all_reduce_2proc/run.sh
    ./run.sh   # 出 /tmp/my_topo.xml
    diff /tmp/nccl_real_topo.xml /tmp/my_topo.xml
    ```
- [ ] 5.2 检查 diff。可能差异：
    - host_hash 因 hostname 一致应该一样
    - GPU 顺序 / busid / link_speed / nvlink count 必须一样
    - NIC 部分应该一样
    - 如果 NCCL 多了 `<keep>` 之类的内部 attribute，记下来允许
- [ ] 5.3 修正 main.cc / Makefile / stubs 直到 XML 完全一致

**完成条件**：`diff` 输出为空，或者剩下的都是已知白名单字段。

---

## Phase 6 — 文档 + 教学

- [ ] 6.1 在 `docs/design.md` 里记录最终的 vendor 文件清单 + 每个文件改了哪里
- [ ] 6.2 写 `docs/walkthrough.md` 把 main.cc 逐段拆开讲，对照 `bootstrap_scenarios.md`
- [ ] 6.3 README.md 状态改成 🟢

---

## Future / Out-of-scope

> 这些**不在当前项目范围内**，仅记录概念，不实现。

- [ ] (TODO) **Step 6: bootstrap ring 闭合**。文档里 step 6 在 0-412 行外但确实属于完整 bootstrap 流程。本项目当前只用 rank 0 当 star 中心做 allgather；真 NCCL 还会让 rank N connect 到 (N+1)%N 形成 ring，用来在 init 后的"持续控制平面"上发小消息。要做的话：
    - 每 rank 也起一个 listening socket（rank 0 自然已经起了）
    - rank 0 把所有 rank 的 listening addr broadcast 下去
    - 每 rank connect 到 (rank+1)%N 的 addr
    - 改 `bootstrapAllGather` 走 ring 而不是 star
- [ ] **跨主机模式**：本项目纯单 host，不做。需要的话要加 OOB iface 选择、跨 host 的 `bootstrapAllGather`、可能要替换 UID 分发方式（不再用文件）
- [ ] **真正的 NCCL wire format 兼容**（路径 C）：让 my_nccl 进程能跟真 NCCL 进程组队。需要严格对齐 byte layout
- [ ] **Multi-comm / split / abort** 等高级 bootstrap API
- [ ] **bootstrap socket 上的应用层 barrier/broadcast 暴露成 API** （类似 PyTorch FileStore / TCPStore）
