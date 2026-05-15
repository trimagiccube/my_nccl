// stubs/proxy_stub.cc — Phase 2 stub for NCCL proxy thread.
//
// my_nccl 不做 transport (Phase F), 所以 proxy 线程完全不需要.
// bootstrap.cc:715 / bootstrap.cc:792 (ncclTransportP2pSetup 那条线)
// 会调 ncclProxyInit. 我们 stub 成 no-op: 不分配 proxyState, 不起线程.

#include "nccl.h"
#include "comm.h"
#include "socket.h"

// 签名跟 nccl/src/include/proxy.h:353 严格一致 (C++ linkage)
//
// 真 NCCL 的 ncclProxyInit 会 take ownership of (sock, peerAddresses, peerAddressesUDS),
// 把它们存进 comm->sharedRes->proxyState, 由 ncclProxyDestroy 释放.
// bootstrap.cc:1118 注释明确说 "proxy things are free'd elsewhere".
//
// 我们 stub 不起 proxy 线程, 但必须 free 掉这三个分配, 否则 ASan 报 leak.
ncclResult_t ncclProxyInit(
    struct ncclComm* /*comm*/,
    struct ncclSocket* sock,
    union ncclSocketAddress* peerAddresses,
    uint64_t* peerAddressesUDS) {
  // 关闭 listening socket (createListenSocket 已经 listen() 了一个 fd)
  if (sock) {
    ncclSocketClose(sock);
    free(sock);
  }
  free(peerAddresses);
  free(peerAddressesUDS);
  return ncclSuccess;
}
