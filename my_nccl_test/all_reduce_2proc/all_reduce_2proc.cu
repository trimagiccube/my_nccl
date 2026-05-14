// all_reduce_2proc.cu — 多进程 2 卡 all_reduce 最小样例 (一卡一进程)
//
// 用法 (一般通过 run.sh 启): ./all_reduce_2proc <rank> <nranks>
//   每个进程绑一张卡: device = rank
//   rank 0 调 ncclGetUniqueId, 写入 $NCCL_UID_FILE (默认 /tmp/nccl_uid.bin),
//   rank > 0 轮询读这个文件, 然后大家一起 ncclCommInitRank.
//
// 校验: GPU 0 全 1.0, GPU 1 全 2.0, AllReduce(Sum) 后两边都拿到 3.0。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "[rank %d] CUDA %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL_CHECK(call) do { ncclResult_t e = (call); \
    if (e != ncclSuccess) { fprintf(stderr, "[rank %d] NCCL %s:%d: %s\n", rank, __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)

// ncclUniqueId 在 rank 0 上生成, 通过文件分发给其它 rank。
// 写: 先写 .tmp 再 rename, 避免对端读到半个文件。
// 读: 轮询 (最多 ~30s) 直到 size == sizeof(ncclUniqueId)。
static void share_unique_id(int rank, ncclUniqueId *id, const char *path) {
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(id));
        char tmp[256];
        snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
        FILE *f = fopen(tmp, "wb");
        if (!f) { perror(tmp); exit(1); }
        fwrite(id, sizeof *id, 1, f);
        fclose(f);
        if (rename(tmp, path) != 0) { perror("rename"); exit(1); }
    } else {
        for (int i = 0; i < 300; ++i) {
            struct stat st;
            if (stat(path, &st) == 0 && st.st_size == (off_t)sizeof(*id)) {
                FILE *f = fopen(path, "rb");
                if (f && fread(id, sizeof *id, 1, f) == 1) { fclose(f); return; }
                if (f) fclose(f);
            }
            usleep(100000);  // 100ms
        }
        fprintf(stderr, "[rank %d] timed out waiting for %s\n", rank, path);
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <rank> <nranks>\n", argv[0]);
        return 1;
    }
    int rank   = atoi(argv[1]);
    int nranks = atoi(argv[2]);
    const char *uid_path = getenv("NCCL_UID_FILE");
    if (!uid_path) uid_path = "/tmp/nccl_uid.bin";

    // 1) 绑卡: 一进程一卡
    CUDA_CHECK(cudaSetDevice(rank));
    printf("[rank %d] pid=%d bound to GPU %d\n", rank, (int)getpid(), rank);

    // 2) 分发 ncclUniqueId
    ncclUniqueId id;
    share_unique_id(rank, &id, uid_path);

    // 3) 加入通信组 (本身就是 barrier: 等所有 rank 都进来才返回)
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    // 4) 分配 buffer, 灌输入, 跑 all_reduce
    const int N = 1 << 20;  // 1M float
    float *sendbuf, *recvbuf;
    CUDA_CHECK(cudaMalloc(&sendbuf, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvbuf, N * sizeof(float)));

    float fill = (float)(rank + 1);  // rank 0 -> 1.0, rank 1 -> 2.0
    float *host = (float*)malloc(N * sizeof(float));
    for (int j = 0; j < N; ++j) host[j] = fill;
    CUDA_CHECK(cudaMemcpy(sendbuf, host, N * sizeof(float), cudaMemcpyHostToDevice));
    free(host);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 单进程单 comm: 不需要 ncclGroupStart/End
    NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, N, ncclFloat, ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5) 校验
    float first;
    CUDA_CHECK(cudaMemcpy(&first, recvbuf, sizeof(float), cudaMemcpyDeviceToHost));
    float expected = (float)(nranks * (nranks + 1) / 2);  // 1+2+...+nranks
    printf("[rank %d] recvbuf[0] = %.1f  (expected %.1f)\n", rank, first, expected);
    int rc = (first == expected) ? 0 : 1;

    // 6) 释放
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    // rank 0 顺手清理掉 uid 文件 (一次性, 用完即弃)
    if (rank == 0) unlink(uid_path);

    if (rc == 0) printf("[rank %d] OK\n", rank);
    else         fprintf(stderr, "[rank %d] FAIL\n", rank);
    return rc;
}
