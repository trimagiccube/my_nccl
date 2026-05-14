// all_reduce_2gpu.cu — 单进程 2 卡 all_reduce 最小样例
//   GPU 0 输入全 1.0, GPU 1 输入全 2.0, all_reduce(sum) 后两张卡都拿到 3.0。
//
// 编译: make
// 运行: ./run.sh   或   LD_LIBRARY_PATH=<...>/nccl/build/lib ./all_reduce_2gpu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL_CHECK(call) do { ncclResult_t e = (call); \
    if (e != ncclSuccess) { fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)

int main() {
    const int N = 1 << 20;                  // 每卡 1M 个 float = 4MB
    const int nGpus = 2;
    int devs[2] = {0, 1};

    float *sendbuf[2], *recvbuf[2];
    cudaStream_t streams[2];

    // 1) 每张卡上分配 send/recv buffer, 灌输入数据, 创建 stream
    for (int i = 0; i < nGpus; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaMalloc(&sendbuf[i], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recvbuf[i], N * sizeof(float)));

        float fill = (float)(i + 1);        // GPU 0 -> 1.0, GPU 1 -> 2.0
        float *host = (float*)malloc(N * sizeof(float));
        for (int j = 0; j < N; ++j) host[j] = fill;
        CUDA_CHECK(cudaMemcpy(sendbuf[i], host, N * sizeof(float), cudaMemcpyHostToDevice));
        free(host);

        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 2) 单进程多卡: 一次 init 出 nGpus 个 comm
    ncclComm_t comms[2];
    NCCL_CHECK(ncclCommInitAll(comms, nGpus, devs));

    // 3) all_reduce: 单进程多卡场景必须包在 GroupStart/End 里, 否则会死锁
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nGpus; ++i) {
        NCCL_CHECK(ncclAllReduce(sendbuf[i], recvbuf[i], N,
                                 ncclFloat, ncclSum,
                                 comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // 4) 同步各卡 stream, 把第一个元素拷回 host 校验
    for (int i = 0; i < nGpus; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        float first;
        CUDA_CHECK(cudaMemcpy(&first, recvbuf[i], sizeof(float), cudaMemcpyDeviceToHost));
        printf("[GPU %d] recvbuf[0] = %.1f  (expected 3.0)\n", devs[i], first);
        if (first != 3.0f) { fprintf(stderr, "FAIL on GPU %d\n", devs[i]); return 1; }
    }

    // 5) 释放
    for (int i = 0; i < nGpus; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaFree(sendbuf[i]));
        CUDA_CHECK(cudaFree(recvbuf[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }

    printf("OK\n");
    return 0;
}
