# my_nccl — NCCL 的 bootstrap 最小子集

> **一句话**：从 NCCL 砍出最小可工作子集，只跑 bootstrap_scenarios.md 第 0-412 行描述的部分功能：
> rendezvous + Phase A (ncclPeerInfo allgather) + Phase B (拓扑探测 + intra-node fusion + ncclTopoSystem)。
> **不做**真正的集合通信（AllReduce 等都不在范围内）。

## 范围对照表

参考 [`../docs/bootstrap_scenarios.md`](../docs/bootstrap_scenarios.md) 的 6-step 框架：

| Step | 名称 | 本项目 |
|---|---|---|
| 1 | 选 OOB iface | ✅ 做 |
| 2 | rank 0 起 listener | ✅ 做 |
| 3 | UID 分发 (文件) | ✅ 做（沿用 my_nccl_test 模式） |
| 4 | rendezvous | ✅ 做 |
| 5 — Phase A | ncclPeerInfo allgather | ✅ 做 |
| 5 — Phase B | 本地探测 + IntraNode fusion | ✅ 做（核心目标） |
| 5 — Phase C | 算法图搜索 (Ring/Tree/CollNet/NVLS) | ❌ 不做 |
| 5 — Phase D | graphInfo allgather | ❌ 不做 |
| 5 — Phase E | CollNet 协商 | ❌ 不做 |
| 5 — Phase F | ncclConnect handle 交换 | ❌ 不做 |
| 6 | bootstrap ring 闭合 | 📝 TODO (`TODO.md`) |
| AllReduce/Bcast 等 | 集合通信 | ❌ 完全不做 |

## 验收标准

跑完后导出一份 XML 拓扑文件，**跟真 NCCL `NCCL_TOPO_DUMP_FILE` 的产物 `diff` 应当字节级一致**（或可解释差异）：

```bash
./run.sh                                                    # 我的输出: my_topo.xml
diff my_topo.xml ../my_nccl_test/all_reduce_2proc/logs/aig-a100-topo.xml
```

## 实现路径

**路径 B**（从 NCCL 砍）：vendor 必要的源文件到 `src/vendor/`，stub 掉 transport/proxy/kernel 相关字段，写一个 ~100 行的 `main.cc` 当 entry point，在 Phase C 之前 early return。

详见 [`docs/design.md`](docs/design.md)。

## 状态

🟡 **Phase 0 完成**（骨架）。下一步进 Phase 1 vendor 源文件 —— 参看 [`TODO.md`](TODO.md)。

## 目录布局

```
my_nccl/
├── README.md                ← 本文件
├── TODO.md                  ← 分阶段任务清单
├── docs/
│   └── design.md            ← Vendoring 策略 + stub 点位 + wire format
├── src/
│   ├── main.cc              ← 我的 entry point (Phase 3 写)
│   ├── vendor/              ← 从 nccl/src 拷贝/改动的文件
│   └── stubs/               ← 替换掉的 transport/proxy/kernel 桩
├── include/                 ← vendor 过来的头文件
├── tests/
│   └── test_topo_match.sh   ← diff 输出 XML 跟 NCCL 的
├── Makefile                 ← Phase 4 写
└── run.sh                   ← launcher, Phase 4 写
```

## 基线版本

固定追跟 NCCL submodule 的 HEAD：`8fb057c (v2.23.4-1-5)`。换上游版本时所有 vendored 文件都得重新审视。
