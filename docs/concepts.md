# NCCL 概念笔记：同步 & ncclUniqueId

> 走读自己 repo 里 `nccl/` 子模组的源码总结出来的，行号对应 `8fb057c` (v2.23.4-1-5)。

## TL;DR

1. **NCCL 不提供"找到对方"那一步的同步**——`ncclUniqueId` 怎么从 rank 0 到其他 rank，你得自己想办法（文件 / MPI_Bcast / socket / PyTorch TCPStore）。
2. **一旦所有 rank 拿到 UID 进入 `ncclCommInitRank`，NCCL 接管同步**——这个函数本身就是 barrier。
3. **`ncclUniqueId` 标记的是一个通信组 (`ncclComm_t`)，不是一次集合通信**。同一个 comm 可以跑任意多次 AllReduce / Broadcast。
4. UID 底下其实是 `<magic, rank 0 的 socket 地址>` 二元组。

---

## 1. NCCL 提供哪些同步

| 类型 | NCCL 管不管 | 说明 |
|---|---|---|
| **Bootstrap 同步** (一开始大家怎么找到对方) | ❌ | 你必须自己把 `ncclUniqueId` 从 rank 0 送到所有 rank |
| **Init 同步** (join 通信组) | ✅ | `ncclCommInitRank` 本身就是 barrier，等齐所有 rank 才返回 |
| **集合通信同步** (AllReduce 等执行时的 rank 间协调) | ✅ | NCCL kernel 自己在 stream 上排，网络层做同步 |
| **应用层 barrier** ("大家都到这一步了再继续") | ❌ | 没有 `ncclBarrier()`。要么 `ncclAllReduce` 一个 1 字节 buffer 当 barrier，要么走应用自己的同步 |
| **Stream 同步** (CPU 等 GPU 完成) | ❌ | NCCL 不管，`cudaStreamSynchronize` 由你调 |

**关键洞察**：所谓"NCCL 不管 bootstrap"指的就是上面第一行。`my_nccl_test/all_reduce_2proc` 里那段"文件分发 UID + 轮询"是在**补 NCCL 不管的那一段**。一旦 UID 到位、所有 rank 进入 `ncclCommInitRank`，后面的同步 NCCL 全包了。

### 内部其实有一套 bootstrap 原语

虽然用户层面没有 `ncclBarrier`，但 NCCL 内部为了 init/destroy 自己用，有一套：

```c
// nccl/src/include/bootstrap.h
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapBarrier (void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapSend    (void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv    (void* commState, int peer, int tag, void* data, int size);
```

这些都跑在 bootstrap 的 TCP socket 上（小消息，慢但通用），init 阶段交换拓扑、IB GID、PCIe 路径之类。

---

## 2. ncclUniqueId 标记什么

**一对一对应一个通信组，不是一次 op。**

```
ncclUniqueId  ──1对1──▶  ncclComm_t  ──1对多──▶  AllReduce / Broadcast / Send / ... 任意多次
   (128 字节)              (通信组句柄)             (这个 comm 上的所有 op)
```

具体：
* 一个 UID 喂给一组 rank 的 `ncclCommInitRank`，**生成一个 `ncclComm_t`**；
* 这个 comm 之后能跑任意多次集合通信，不需要再换 UID；
* 想要**新的通信组**（不同 rank 子集 / 销毁后重建），就要**新生成一个 UID**。

---

## 3. UID 内部到底是什么

它是 128 字节的 opaque blob，但底下就两个字段：

```c
// nccl/src/include/bootstrap.h:14
struct ncclBootstrapHandle {
    uint64_t magic;                  // 随机 nonce, 防错配 / 防 UID 被滥用
    union ncclSocketAddress addr;    // rank 0 的 bootstrap listening socket 地址
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId),
              "Bootstrap handle is too large to fit inside NCCL unique ID");
```

`ncclGetUniqueId` 的实现就是把这个 handle `memcpy` 进 128 字节填充：

```c
// nccl/src/init.cc:101
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
    NCCLCHECK(ncclInit());
    NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
    struct ncclBootstrapHandle handle;
    NCCLCHECK(bootstrapGetUniqueId(&handle));  // ←这里面起 socket、填 addr/magic
    memset(out, 0, sizeof(*out));
    memcpy(out, &handle, sizeof(handle));
    ...
}
```

所以"标记"靠的是 `<magic, rank 0 的 socket 地址>` 二元组。任何拿到这两个字段的 rank 都能：
1. 解析出 rank 0 的 IP:port
2. TCP connect 过去
3. 用 magic 做握手校验（防止跑错 ID 串组）

### 为什么是 rank 0 的地址，不是某个第三方"协调员"

NCCL 走的是"rank 0 当 bootstrap root"模型。`ncclGetUniqueId` 本质上做了三件事：

```c
// nccl/src/bootstrap.cc:372  bootstrapCreateRoot()
1. ncclSocketInit(listenSock, addr, magic, ncclSocketTypeBootstrap, ...)
2. ncclSocketListen(listenSock)            // 系统分配一个空闲端口
3. ncclSocketGetAddr(listenSock, addr)     // 把实际监听地址回填到 handle->addr
4. pthread_create(bootstrapRoot, args)     // 起后台线程, 准备 accept 来的 rank
```

第 4 步那个 `bootstrapRoot` 线程是后台 detach 的——一直跑着等其他 rank 连过来，accept 完做完拓扑交换就退出。

### 那 ID 是怎么"用一次就废"的

```
rank 0 调 ncclGetUniqueId          rank 1..N-1
  ├─ 起 socket + 拿空闲 port
  ├─ pthread_create bootstrapRoot   ┊
  ├─ 填 magic + addr 到 handle      ┊
  └─ 返回 UID 给用户                ┊
                                     ┊
   ↓ (你的代码: 把 UID 发到 rank 1..N-1, 例如写文件)
                                     ┊
rank 0 调 ncclCommInitRank          rank 1..N-1 调 ncclCommInitRank(UID, rank)
  ├─ bootstrapRoot 收到所有 N 个     │  ├─ 解 UID 拿到 <magic, addr>
  │  rank 的连接                    │  ├─ TCP connect 到那个 addr
  ├─ 交换每个 rank 的 GPU 拓扑 / IB │  ├─ magic 握手
  │  GID / PCIe 路径                │  ├─ 上报自己的拓扑给 rank 0
  ├─ 建出 ring / tree                │  ├─ 接收 rank 0 算出的 ring/tree 拓扑
  ├─ 选定 transport                  │  ├─ 跟邻居建立 transport 连接
  │  (NVLink/IB/sockets/SHM)         │  │  (NVLink/IB/sockets/SHM)
  └─ bootstrapRoot 线程 exit         │  └─ 返回 comm 给用户

之后所有集合通信走 transport, 不再用 bootstrap socket
```

`bootstrapRoot` 线程做完一次组就退出。同一个 UID 想再用？rank 0 的 listening socket 已经关了，`magic` 也仅此一次有效——必然失败。**所以 UID 是一次性的。**

要再建一个通信组：rank 0 重新 `ncclGetUniqueId` 生成新的 UID，重新分发，重新 `ncclCommInitRank`。

---

## 4. 应用层 barrier 怎么写

NCCL 没有 `ncclBarrier`，但有两个常见替代：

```c
// 替代 1: AllReduce 一个 1 字节 buffer 当 barrier (推荐)
char tag = 0;
ncclAllReduce(&tag, &tag, 1, ncclChar, ncclSum, comm, stream);
cudaStreamSynchronize(stream);  // 同步 host

// 替代 2: ncclSend/ncclRecv 跟相邻 rank 互相握手 (更轻, 但需要写更多)
```

替代 1 的开销在毫秒级，足够大多数训练场景。

---

## 5. 源码引用表

按你想看的顺序：

| 文件 : 行 | 内容 |
|---|---|
| `nccl/src/nccl.h.in` | 公开 API 声明：`ncclGetUniqueId` / `ncclCommInitRank` / `ncclAllReduce` 等 |
| `nccl/src/include/bootstrap.h:14-17` | `ncclBootstrapHandle` 结构定义 |
| `nccl/src/include/bootstrap.h` | bootstrap 内部 API（barrier、allgather、broadcast...） |
| `nccl/src/init.cc:101-113` | `ncclGetUniqueId` 实现，看 memcpy 怎么填 128 字节 |
| `nccl/src/bootstrap.cc:372-396` | `bootstrapCreateRoot` — rank 0 起 listener + 后台线程 |
| `nccl/src/bootstrap.cc:397` | `bootstrapGetUniqueId` |
| `nccl/src/bootstrap.cc:601` | `bootstrapInit` — 其他 rank 怎么连过来、怎么交换拓扑 |
| `nccl/src/transport/` | 各种 transport 实现：`p2p.cc` / `shm.cc` / `net.cc` 等 |

跑 `NCCL_DEBUG=INFO` 时输出的 `Channel 00 : 0 -> 1 ...` 就是上面 "建出 ring/tree" 那一步的结果。

---

## 6. 一些常见疑惑

**Q: rank 0 监听端口被防火墙挡住了怎么办**
A: bootstrap 走 TCP，端口是系统随机分配的。可以 `export NCCL_SOCKET_IFNAME=eth0` 选网卡；端口范围可用 `NCCL_SOCKET_FAMILY` / Linux 的 `ip_local_port_range` 限制。

**Q: 如果 rank 0 还没起来，其他 rank 就调 `ncclCommInitRank` 会怎样**
A: 它会等 UID 文件 / MPI_Bcast / 等待源就绪——这部分是你的 bootstrap 实现的事；NCCL 拿到 UID 才会去 connect。

**Q: 同一进程里能有多个 ncclComm_t 吗**
A: 能。`ncclCommInitAll(comms[N], N, devs[N])` 就给单进程出 N 个 comm。每个对应一张卡，互相协作时要 `ncclGroupStart/End` 包起来（见 `my_nccl_test/all_reduce_2gpu/`）。

**Q: 一个 ncclComm_t 上跑 AllReduce 的同时能跑 Broadcast 吗**
A: 在同一个 stream 上是顺序的；不同 stream 上可以并行，但 NCCL 内部会用 channel 串行化。教学样例里别这么玩，先把单 op 跑明白。

**Q: 进程崩了 UID 文件没清理怎么办**
A: 没事，UID 用完即废。下次 `./run.sh` 用新 PID 命名一个新文件。`my_nccl_test/all_reduce_2proc/run.sh` 里有 `trap '... rm -f' EXIT` 顺手清。

**Q: NCCL 内部的 bootstrapBarrier 我能调到吗**
A: 不能，它没出现在 `nccl.h` 里，只能修 NCCL 源码再编。但你不需要——`ncclAllReduce` 一个小 buffer 已经足够当 barrier 用。
