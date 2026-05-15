// my_nccl/main.cc — bootstrap 最小子集 entry point
//
// 当前阶段 (Phase 2): 只验证链接 + 调出 ncclUniqueId
//   ./my_nccl_bootstrap
//   -> 调 bootstrapGetUniqueId, 打印 128B 唯一 id 的 hex.
//
// 后续 Phase 3-4 会在这里串起完整的 rendezvous + Phase A/B.

#include "nccl.h"
#include "bootstrap.h"
#include "core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int /*argc*/, char** /*argv*/) {
  // 复用 NCCL init.cc:101 的逻辑, 但绕过它的 ncclInit():
  ncclBootstrapHandle handle;
  if (bootstrapNetInit() != ncclSuccess) {
    fprintf(stderr, "bootstrapNetInit failed\n");
    return 1;
  }
  if (bootstrapGetUniqueId(&handle) != ncclSuccess) {
    fprintf(stderr, "bootstrapGetUniqueId failed\n");
    return 1;
  }

  ncclUniqueId id;
  memset(&id, 0, sizeof(id));
  memcpy(&id, &handle, sizeof(handle));

  printf("ncclUniqueId (%zu bytes):\n", sizeof(id));
  const unsigned char* p = reinterpret_cast<const unsigned char*>(&id);
  for (size_t i = 0; i < sizeof(id); ++i) {
    printf("%02x", p[i]);
    if ((i + 1) % 32 == 0) printf("\n");
    else if ((i + 1) % 4 == 0) printf(" ");
  }
  printf("\nmagic=0x%lx\n", (unsigned long)handle.magic);
  return 0;
}
