# TODO

> 每个 Phase 结束的标志是"能编译 / 能跑出预期"，然后 commit。
> ✅ Phase 0 (骨架) ✅ Phase 1 (vendor 基础源) ✅ Phase 2 (链接通 + 单进程出 ncclUniqueId)
> ✅ Phase 3a (2 进程 rendezvous + Phase A allgather)

---

## Phase 1 — Vendor 基础源文件 ✅

目标：把 bootstrap 协议层需要的最少 NCCL 源文件搬过来。

- [x] 1.1 拷 `nccl/build/include/nccl.h`（构建产物, 已展开模板）→ `include/nccl.h`
- [x] 1.2 拷 `socket.{h,cc}`, `bootstrap.{h,cc}`, `utils.{h,cc}`, `debug.{h,cc}`, `param.{h,cc}`
- [x] 1.3 拷 `transport.h`, `comm.h`, `core.h`
- [x] 1.4 整个 `nccl/src/include/*.h` 都拷了（46 个 header）省得边编边补
- [x] 1.5 学到一个事实：`nvtx.h` 在 CUDA 12.9 上编不过（nvtx3 版本不对应）→ **替换成 stub**（详见 VENDORED.md）

**实际结果**：5 个 vendor .cc 全部单独 `g++ -c` 通过，零警告。

---

## Phase 2 — Stub + 最小 main ✅

目标：让 vendored 代码能链接成功 + 跑出 ncclUniqueId。

- [x] 2.1 写最小 `src/main.cc`：调 `bootstrapNetInit` + `bootstrapGetUniqueId`，打印 128B id 的 hex
- [x] 2.2 写 `Makefile`：vendor + stubs + main，含 ASan/UBSan，`make help` 支持
- [x] 2.3 首次链接失败：仅缺 1 个 `ncclProxyInit`（bootstrap 末尾起 proxy 线程时调）→ 写 `src/stubs/proxy_stub.cc` no-op
- [x] 2.4 链接通过 → `./my_nccl_bootstrap` 跑出真 ncclUniqueId

**实际产物**：
- `magic=0xc02b5815d3aedd4f`（真随机）
- 字节解码：IPv4 = 192.168.80.101 port 24542（socket addr）
- 跟之前 `aig-a100` 实测 NCCL 的 OOB iface 完全对得上

**没用到的 stub（推迟）**：
- transport_stub.cc / kernel_stub.cc：当前链接路径上没碰到
- 只 stub 了 `ncclProxyInit`，已足够单进程跑 bootstrap rendezvous

---

## Phase 3a — 2 进程 rendezvous + Phase A allgather ✅

- [x] 3a.1 把 `transport.h` vendor 回来 (Phase A 需要 ncclPeerInfo)
- [x] 3a.2 重写 src/main.cc: 吃 `<rank> <nranks> <uid_file>`, 跑完整 rendezvous + bootstrapInit + bootstrapAllGather + 输出 peerInfo
- [x] 3a.3 写 run.sh launcher (跟 my_nccl_test/all_reduce_2proc 的风格一致)
- [x] 3a.4 修 ASan leak: proxy_stub 接管/释放 bootstrap.cc:697-701 那 3 个分配 (注释明确说 "proxy things are free'd elsewhere")
- [x] 3a.5 跑 NRANKS=2 通过, 校验:
    - 双方 magic 一致 → 进入同一通信组
    - hostHash 一致 → 同机识别
    - pidHash 不同 → 不同进程识别
    - allgather 完整双向可见

---

## Phase 3b — Vendor 拓扑层 (Phase B)

目标：把 Phase B 需要的 topo + xml 代码搬过来, 真探测出 ncclTopoSystem。

- [ ] 3b.1 拷 `nccl/src/graph/topo.h`, `topo.cc`, `xml.h`, `xml.cc`
- [ ] 3b.2 看 `topo.cc` 里有没有引用 `paths.cc` 的内容（如 `ncclTopoComputePaths`）。**Phase B 只到生成 ncclTopoSystem，不算路径**，所以 `paths.cc` 大概率不需要
- [ ] 3b.3 stub `collNetSupport()` / `collNetDevices()` / `ncclNvlsInit()` → 直接返回 0
- [ ] 3b.4 stub IB 部分（`ncclNet->devices()` 等）→ 返回 0 个设备
- [ ] 3b.5 把 main.cc 接上 `ncclTopoGetSystem`, dump XML

**完成条件**：链接通过 + 跑出来一份 XML。

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
