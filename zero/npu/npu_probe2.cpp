// probe NPU driver: extension list + wider external-memmap stype sweep.
#include "zero/ze_min.h"
#include <cstring>
#include <sys/mman.h>
#include <vector>

struct ze_driver_extension_properties_t { char name[256]; uint32_t version; };
typedef ze_result_t (*PFN_extp)(ze_driver_handle_t, uint32_t*, ze_driver_extension_properties_t*);

static uint32_t probe(ze_context_handle_t ctx, const char* who, uint8_t* mem, size_t bytes) {
  for (uint32_t st = 0x00020000u; st <= 0x00020400u; ++st) {
    struct { uint32_t stype; const void* pNext; const void* pSystemMemory; uint64_t size; }
        ed{st, nullptr, mem, bytes};
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, &ed, 0};
    void* p = nullptr;
    if (zeMemAllocHost(ctx, &hd, bytes, 4096, &p) != 0 || !p) continue;
    memset(mem, 0, 64);
    ((volatile uint8_t*)p)[7] = 0x5a;
    const bool same = mem[7] == 0x5a;
    if (same) { printf("%s: stype 0x%08x maps the memory (p=%p mem=%p)\n", who, st, p, (void*)mem); return st; }
    zeMemFree(ctx, p);
  }
  printf("%s: nothing in 0x20000..0x20400 maps the memory\n", who);
  return 0;
}

int main() {
  if (!ze_load() || zeInit(0) != ZE_RESULT_SUCCESS) { zlog::eprintf("L0 init failed\n"); return 1; }
  void* zl = plat::dl_open("ze_loader.dll", "libze_loader.so.1");
  PFN_extp extp = zl ? (PFN_extp)plat::dl_sym(zl, "zeDriverGetExtensionProperties") : nullptr;
  uint32_t nd = 0;
  zeDriverGet(&nd, nullptr);
  std::vector<ze_driver_handle_t> drv(nd);
  zeDriverGet(&nd, drv.data());
  for (auto d : drv) {
    uint32_t n = 1; ze_device_handle_t x = nullptr;
    zeDeviceGet(d, &n, &x);
    ze_device_properties_t p{}; p.stype = ZE_STYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(x, &p);
    printf("driver %p: device type %d (%s)\n", (void*)d, (int)p.type, p.name);
    if (extp) {
      uint32_t ne = 0;
      extp(d, &ne, nullptr);
      std::vector<ze_driver_extension_properties_t> exts(ne);
      extp(d, &ne, exts.data());
      for (auto& e : exts)
        if (strstr(e.name, "memmap") || strstr(e.name, "virtual") || strstr(e.name, "external") ||
            strstr(e.name, "dma") || strstr(e.name, "ipc"))
          printf("   ext: %s v%u\n", e.name, e.version);
    }
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t ctx = nullptr;
    if (zeContextCreate(d, &cd, &ctx) != 0) continue;
    const size_t bytes = 4u << 20;
    uint8_t* mem = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    probe(ctx, (int)p.type == 1 ? "GPU" : (int)p.type == 5 ? "NPU" : "OTHER", mem, bytes);
    zeContextDestroy(ctx);
  }
  return 0;
}
