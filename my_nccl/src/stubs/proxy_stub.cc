// stubs/proxy_stub.cc — Phase 2 stub for NCCL proxy thread.
//
// my_nccl 不做 transport (Phase F), 所以 proxy 线程完全不需要.
// bootstrap.cc:715 / bootstrap.cc:792 (ncclTransportP2pSetup 那条线)
// 会调 ncclProxyInit. 我们 stub 成 no-op: 不分配 proxyState, 不起线程.

#include "nccl.h"
#include "comm.h"
#include "socket.h"

// 签名跟 nccl/src/include/proxy.h:353 严格一致 (C++ linkage)
ncclResult_t ncclProxyInit(
    struct ncclComm* /*comm*/,
    struct ncclSocket* /*sock*/,
    union ncclSocketAddress* /*peerAddresses*/,
    uint64_t* /*peerAddressesUDS*/) {
  return ncclSuccess;
}
