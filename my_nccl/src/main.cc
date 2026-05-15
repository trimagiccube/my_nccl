// my_nccl/main.cc — 2 进程 bootstrap rendezvous + Phase A (peerInfo allgather)
//
// 用法 (通过 run.sh 启动): ./my_nccl_bootstrap <rank> <nranks> <uid_file>
//
// 各 rank 干的事:
//   1. bootstrapNetInit                    选 OOB iface
//   2. rank 0: bootstrapGetUniqueId        生成 UID
//      rank > 0: 轮询 UID 文件
//   3. 全员 bootstrapInit                  TCP rendezvous, 形成通信组
//   4. fillInfo + bootstrapAllGather       Phase A: 交换 ncclPeerInfo
//   5. 校验 hostHash/pidHash 关系          同机不同进程
//   6. bootstrapClose                      干净收尾

#include "nccl.h"
#include "bootstrap.h"
#include "comm.h"
#include "transport.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

static int share_unique_id(int rank, ncclBootstrapHandle* handle, const char* path) {
  if (rank == 0) {
    char tmp[256];
    snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
    FILE* f = fopen(tmp, "wb");
    if (!f) { perror(tmp); return -1; }
    fwrite(handle, sizeof(*handle), 1, f);
    fclose(f);
    if (rename(tmp, path) != 0) { perror("rename"); return -1; }
    return 0;
  }
  // rank > 0: poll
  for (int i = 0; i < 300; ++i) {
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size == (off_t)sizeof(*handle)) {
      FILE* f = fopen(path, "rb");
      if (f && fread(handle, sizeof(*handle), 1, f) == 1) { fclose(f); return 0; }
      if (f) fclose(f);
    }
    usleep(100000);
  }
  fprintf(stderr, "[rank %d] timed out waiting for %s\n", rank, path);
  return -1;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <rank> <nranks> <uid_file>\n", argv[0]);
    return 1;
  }
  const int rank   = atoi(argv[1]);
  const int nranks = atoi(argv[2]);
  const char* uid_path = argv[3];

  printf("[rank %d] pid=%d starting (nranks=%d)\n", rank, (int)getpid(), nranks);

  // 1. bootstrapNetInit
  if (bootstrapNetInit() != ncclSuccess) {
    fprintf(stderr, "[rank %d] bootstrapNetInit failed\n", rank);
    return 1;
  }

  // 2. rendezvous via UID file
  ncclBootstrapHandle handle;
  if (rank == 0) {
    if (bootstrapGetUniqueId(&handle) != ncclSuccess) {
      fprintf(stderr, "[rank 0] bootstrapGetUniqueId failed\n");
      return 1;
    }
    printf("[rank 0] generated UID magic=0x%lx\n", (unsigned long)handle.magic);
  }
  if (share_unique_id(rank, &handle, uid_path) != 0) return 1;
  printf("[rank %d] got UID magic=0x%lx\n", rank, (unsigned long)handle.magic);

  // 3. bootstrapInit: 所有 rank 在 rank 0 的 socket 上 rendezvous
  ncclComm comm = {};
  uint32_t abortFlag = 0;
  comm.rank      = rank;
  comm.nRanks    = nranks;
  comm.abortFlag = &abortFlag;
  // comm.magic 由 bootstrapInit 内部设置

  if (bootstrapInit(1, &handle, &comm) != ncclSuccess) {
    fprintf(stderr, "[rank %d] bootstrapInit failed\n", rank);
    return 1;
  }
  printf("[rank %d] bootstrapInit OK, comm.magic=0x%lx (commHash)\n",
         rank, (unsigned long)comm.magic);

  // 4. Phase A: 填 myInfo + bootstrapAllGather
  ncclPeerInfo* peerInfo = (ncclPeerInfo*)calloc(nranks + 1, sizeof(ncclPeerInfo));
  ncclPeerInfo* myInfo = &peerInfo[rank];

  myInfo->rank         = rank;
  myInfo->cudaDev      = rank;        // 一进程一卡, rank == device idx
  myInfo->nvmlDev      = rank;
  myInfo->version      = NCCL_VERSION_CODE;
  myInfo->hostHash     = getHostHash() + comm.magic;
  myInfo->pidHash      = getPidHash()  + comm.magic;
  myInfo->cuMemSupport = 0;
  struct stat sb;
  if (stat("/dev/shm", &sb) == 0) myInfo->shmDev = sb.st_dev;
  myInfo->busId        = 0x8000 + (int64_t)rank * 0x1000;  // 占位, 真值 Phase B 才查
  myInfo->gdrSupport   = 0;
  myInfo->comm         = &comm;
  myInfo->cudaCompCap  = 80;

  if (bootstrapAllGather(comm.bootstrap, peerInfo, sizeof(ncclPeerInfo)) != ncclSuccess) {
    fprintf(stderr, "[rank %d] bootstrapAllGather failed\n", rank);
    return 1;
  }
  printf("[rank %d] === peerInfo after AllGather ===\n", rank);
  for (int i = 0; i < nranks; ++i) {
    printf("[rank %d]   peer[%d]: rank=%d cudaDev=%d hostHash=0x%016lx pidHash=0x%016lx busId=0x%lx\n",
           rank, i,
           peerInfo[i].rank, peerInfo[i].cudaDev,
           (unsigned long)peerInfo[i].hostHash,
           (unsigned long)peerInfo[i].pidHash,
           (unsigned long)peerInfo[i].busId);
  }

  // 5. 校验语义: 同机 → hostHash 全相同; 不同进程 → pidHash 不同
  int ok = 1;
  for (int i = 0; i < nranks; ++i) {
    if (peerInfo[i].hostHash != myInfo->hostHash) {
      printf("[rank %d] FAIL: peer[%d] hostHash mismatch (expect same host)\n", rank, i);
      ok = 0;
    }
    if (i != rank && peerInfo[i].pidHash == myInfo->pidHash) {
      printf("[rank %d] FAIL: peer[%d] pidHash equals mine (expect different process)\n", rank, i);
      ok = 0;
    }
  }
  if (ok) printf("[rank %d] OK: hostHash 全部相同 + pidHash 各异\n", rank);

  // 6. cleanup
  bootstrapClose(comm.bootstrap);
  free(peerInfo);
  if (rank == 0) unlink(uid_path);

  return ok ? 0 : 1;
}
