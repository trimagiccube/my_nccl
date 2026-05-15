// Replaces nccl/src/include/nvtx.h (8fb057c) — gutted stub.
//
// my_nccl 不做 profiler tracing, NVTX 完全不需要. 原 nvtx.h 依赖的
// nvtx3 payload_v2 API 在 CUDA 12.9 自带的 nvtx3 里不存在, 所以原文件
// 编不过. 用本空 stub 顶上, 把 NCCL 源里出现的所有 NVTX 宏都定义成 no-op.
//
// 如果将来要恢复 profiler, 把这文件换回原版 + 装新版 nvtx3 即可.

#ifndef NCCL_NVTX_H_
#define NCCL_NVTX_H_

// NCCL 代码里出现过的 NVTX 宏都 no-op 化:
#define NVTX3_FUNC_WITH_PARAMS(ID, S, P)         do {} while (0)
#define NVTX3_FUNC_RANGE_IN(domain)              do {} while (0)
#define NVTX3_FUNC_RANGE()                       do {} while (0)

// Schema 静态 ID (核心代码会引用, 给个虚拟值)
#define NVTX_SID_CommInitRank        0
#define NVTX_SID_CommInitAll         1
#define NVTX_SID_CommAbort           2
#define NVTX_SID_CommDestroy         3
#define NVTX_SID_CommFinalize        4
#define NVTX_SID_CommGetAsyncError   5
#define NVTX_SID_AllGather           6
#define NVTX_SID_AllReduce           7
#define NVTX_SID_Broadcast           8
#define NVTX_SID_ReduceScatter       9
#define NVTX_SID_Reduce              10
#define NVTX_SID_Send                11
#define NVTX_SID_Recv                12
#define NVTX_SID_CommInitRankConfig  13
#define NVTX_SID_CommInitRankScalable 14
#define NVTX_SID_CommSplit           15
#define NVTX_SID_GroupEnd            16

#endif
