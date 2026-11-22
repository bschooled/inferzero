// Microbenchmark for P1 (alignment / stride / padding audit of decode GEMV streams):
//
// 1. Raw memory read bandwidth across:
//    - 16-byte per-lane vector loads (K_READ)
//    - Sub-group block reads (K_READ_B, 2 KB in flight per sub-group)
// 2. Synthetic decode-GEMV row-stream benchmark at realistic matrix shapes (e.g., 17408 x 5120,
//    10240 x 5120, 5120 x 17408, 248320 x 5120) for Q4_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, Q6_K, F16:
//    (a) GGUF unpadded / unaligned stride (e.g. Q6_K 210 B/sb, IQ2_XXS 66 B/sb, IQ2_XS 74 B/sb)
//    (b) inferzero native GPU raw stride (8-B / 16-B aligned superblocks, regrouped d-last)
//    (c) +64 B / +128 B / +256 B / +512 B padded row stride (Gufo LPDDR channel-aliasing probe)
//    (d) 16-B-aligned repacked superblock / planar layout
//    Measured for both 16-byte per-lane vector loads (gemv_v pattern) and sub-group block reads (gemv pattern).
// 3. Real decode kernel benchmark (decode.cl gemv / gemv_v / gemv_mt / gemv_mt_v) with -DROW_PAD=0 vs
//    +64 / +512 B row padding and 16B-aligned superblocks over a full >512 MiB weight footprint.
//
// Usage:
//   ./bin/align_probe [--seconds 0.5]
#include "../zero/clrt.h"
#include "../src/quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static const char* kProbeCl = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 1. Raw contiguous 16-byte per-lane loads (8 in flight per lane)
__kernel void bw_vec16(__global const uint4* x, int n4, __global uint* out) {
  uint4 a[8];
  for (int u = 0; u < 8; ++u) a[u] = (uint4)(0);
  const int g = get_global_size(0);
  for (int i = get_global_id(0); i + 7 * g < n4; i += 8 * g) {
#pragma unroll
    for (int u = 0; u < 8; ++u) a[u] += x[i + u * g];
  }
  for (int u = 1; u < 8; ++u) a[0] += a[u];
  if ((a[0].x ^ a[0].y ^ a[0].z ^ a[0].w) == 0x9e3779b9u) out[0] = 1;
}

// 2. Raw contiguous sub-group block reads (2 KB in flight per sub-group)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void bw_block(__global const uint* x, int n, __global uint* out) {
  const size_t sg = get_global_id(0) / 16, nsg = get_global_size(0) / 16;
  uint8 a = (uint8)(0), b = (uint8)(0), c = (uint8)(0), d = (uint8)(0);
  for (size_t i = sg * 512; i + 512 <= (size_t)n; i += nsg * 512) {
    __global const uint* p = x + i;
    a ^= intel_sub_group_block_read8(p);
    b ^= intel_sub_group_block_read8(p + 128);
    c ^= intel_sub_group_block_read8(p + 256);
    d ^= intel_sub_group_block_read8(p + 384);
  }
  a ^= b ^ c ^ d;
  if ((a.s0 ^ a.s1 ^ a.s2 ^ a.s3 ^ a.s4 ^ a.s5 ^ a.s6 ^ a.s7) == 0x9e3779b9u) out[0] = 1;
}

// 3. Decode-GEMV row stream using sub-group block reads (inferzero's `gemv` access pattern):
//    Each 16-lane sub-group owns GROWS rows. Row r starts at r * row_stride_bytes.
//    Within a row, the sub-group steps by win_bytes (2 superblocks = 2 * sb_bytes), issuing
//    win_calls = ceil(win_bytes / 64) sub-group block reads per step.
#ifndef GROWS
#define GROWS 4
#endif
#ifndef WIN_CALLS
#define WIN_CALLS 5
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void row_stream_block(__global const uchar* raw, int nrows, int nsteps, int win_bytes,
                               int row_stride, __global uint* out) {
  const int sg = get_global_id(0) / 16;
  const int r0 = sg * GROWS;
  if (r0 >= nrows) return;
  uint acc = 0;
  for (int s = 0; s < nsteps; ++s) {
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int row = min(r0 + r, nrows - 1);
      __global const uint* src = (__global const uint*)(raw + (size_t)row * row_stride + (size_t)s * win_bytes);
#pragma unroll
      for (int c = 0; c < WIN_CALLS; ++c) {
        acc ^= intel_sub_group_block_read(src + c * 16);
      }
    }
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

// 4. Decode-GEMV row stream using per-lane vector loads (inferzero's `gemv_v` access pattern):
//    Lane l in [0..15] owns 32-block (l % 8) of superblock (sp + l / 8).
#ifndef VMODE
#define VMODE 0
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void row_stream_vec(__global const uchar* raw, int nrows, int nsb, int sb_bytes,
                             int row_stride, __global uint* out) {
  const int lane = get_sub_group_local_id();
  const int sg = get_global_id(0) / 16;
  const int r0 = sg * GROWS;
  if (r0 >= nrows) return;
  const int ib = lane % 8, hs = lane / 8;
  uint acc = 0;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int s = (sp + hs < nsb) ? (sp + hs) : sp;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int row = min(r0 + r, nrows - 1);
      __global const uchar* p = raw + (size_t)row * row_stride + (size_t)s * sb_bytes;
#if VMODE == 0
      // 16-byte aligned uint4 per lane (16 B per 32-block + 16 B scales/high-bits)
      __global const uint* pu = (__global const uint*)p;
      const uint4 q0 = vload4(0, pu + 4 * ib);
      const uint4 q1 = vload4(0, pu + 32 + 2 * ib);
      acc ^= q0.x ^ q0.y ^ q0.z ^ q0.w ^ q1.x ^ q1.y;
#elif VMODE == 1
      // inferzero regrouped IQ2/IQ3: 8-byte aligned vload2(ib, pu) + scale word + fp16 d at end
      __global const uint* pu = (__global const uint*)p;
      const uint2 q = vload2(ib, pu);
      const uint sc = pu[16 + ib / 4];
      const uint d = ((__global const ushort*)p)[(sb_bytes - 2) / 2];
      acc ^= q.x ^ q.y ^ sc ^ d;
#elif VMODE == 2
      // Unregrouped GGUF (2-byte misaligned: fp16 d at offset 0, qs at offset 2)
      const uint d = (uint)(*(__global const ushort*)p);
      __global const ushort* ps = (__global const ushort*)(p + 2 + 8 * ib);
      const ushort4 q = vload4(0, ps);
      const uint sc = (uint)(*(__global const ushort*)(p + 66 + (ib & ~1)));
      acc += d + ((uint)q.x | ((uint)q.y << 16)) + ((uint)q.z | ((uint)q.w << 16)) + sc;
#elif VMODE == 3
      // Q6_K vblock pattern: 2x vload8 (uint8 = 32B L + 32B H, matching decode.cl:1317-1318) + ushort scale + ushort d
      const int nn = ib / 4, quarter = ib % 4;
      const uint8 L = vload8(0, (__global const uint*)(p + 64 * nn + ((quarter & 1) ? 32 : 0)));
      const uint8 H = vload8(0, (__global const uint*)(p + 128 + 32 * nn));
      const uint sw = (uint)(*(__global const ushort*)(p + 192 + 2 * ib));
      const uint d = (uint)(*(__global const ushort*)(p + 208));
      acc += L.s0 ^ L.s1 ^ L.s2 ^ L.s3 ^ L.s4 ^ L.s5 ^ L.s6 ^ L.s7 ^ H.s0 ^ H.s1 ^ H.s2 ^ H.s3 ^ H.s4 ^ H.s5 ^ H.s6 ^ H.s7 ^ sw ^ d;
#elif VMODE == 4
      // F16 row stream: each lane reads 16 halves (32 bytes = uint8) per 256-element superblock
      __global const uint* pu = (__global const uint*)p;
      const uint8 v0 = vload8(lane, pu);
      const uint8 v1 = vload8(lane, pu + 128);
      acc ^= v0.s0 ^ v0.s4 ^ v1.s0 ^ v1.s4;
#elif VMODE == 5
      // IQ2_S GGUF unregrouped (82B, 2B-aligned): d@0(2B), qs@2(32B), sgn@34(32B), qh@66(8B), sc@74(8B)
      const uint d = (uint)(*(__global const ushort*)p);
      const ushort2 qv = vload2(0, (__global const ushort*)(p + 2 + 4 * ib));
      const ushort2 sv = vload2(0, (__global const ushort*)(p + 34 + 4 * ib));
      const uint qh = (uint)(*(__global const ushort*)(p + 66 + (ib & ~1)));
      const uint scb = (uint)(*(__global const ushort*)(p + 74 + (ib & ~1)));
      acc += d + ((uint)qv.x | ((uint)qv.y << 16)) + ((uint)sv.x | ((uint)sv.y << 16)) + qh + scb;
#elif VMODE == 6
      // IQ2_S inferzero regrouped (88B or 96B): qs@0(32B), sgn@32(32B), qh@64(8B), sc@72(8B), d@80(2B)
      __global const uint* pu = (__global const uint*)p;
      const uint qs4 = pu[ib], sg4 = pu[8 + ib];
      const uint h = (pu[16 + ib / 4] >> (8 * (ib % 4))) & 0xFF, scb = (pu[18 + ib / 4] >> (8 * (ib % 4))) & 0xFF;
      const uint d = (uint)((__global const ushort*)p)[40];
      acc += d + qs4 + sg4 + h + scb;
#endif
    }
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
)CLC";

static void fill_random(ClRt& cl, cl_mem db, size_t bytes) {
  const size_t step = (size_t)64 << 20;
  std::vector<uint32_t> h(step / 4);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (size_t off = 0; off < bytes; off += step) {
    const size_t n = std::min(step, bytes - off);
    for (size_t i = 0; i < n / 4; ++i) {
      s ^= s << 13, s ^= s >> 7, s ^= s << 17;
      h[i] = (uint32_t)(s >> 16);
    }
    CL_CHECK(clEnqueueWriteBuffer(cl.queue, db, CL_TRUE, off, n, h.data(), 0, nullptr, nullptr));
  }
  clFinish(cl.queue);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  double seconds = 0.4;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
  }

  ClRt cl;
  if (!cl.init()) return 1;

  char name[256] = {}, drv[128] = {};
  cl_uint eus = 0, mhz = 0, thr = 0;
  cl_ulong mem = 0;
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x1002, sizeof(eus), &eus, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x100C, sizeof(mhz), &mhz, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x101F, sizeof(mem), &mem, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x4255, sizeof(thr), &thr, nullptr);

  printf("=== P1 ALIGNMENT & STRIDE MICROBENCHMARK ===\n");
  printf("GPU: %s | %u EUs | %u threads/EU | max %u MHz | mem %.2f GiB | driver %s\n\n",
         name, eus, thr, mhz, mem / 1073741824.0, drv);

  const size_t buf_bytes = (size_t)768 << 20;  // 768 MiB working set (> all caches)
  cl_mem db = cl.buffer(CL_MEM_READ_WRITE, buf_bytes + 65536);
  cl_mem bo = cl.buffer(CL_MEM_READ_WRITE, 4096);
  fill_random(cl, db, buf_bytes + 65536);

  auto time_kernel = [&](cl_kernel k, uint32_t gws, uint32_t lws) {
    ZE_CHECK(zeKernelSetGroupSize(k->k, lws, 1, 1));
    ze_group_count_t gc{gws / lws, 1, 1};
    auto launch = [&] {
      const auto t0 = clk::now();
      ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
      zrt::sync();
      return std::chrono::duration<double>(clk::now() - t0).count();
    };
    launch();  // warmup
    double best = 1e30, spent = 0;
    int reps = 0;
    while (spent < seconds || reps < 4) {
      const double t = launch();
      best = std::min(best, t);
      spent += t;
      ++reps;
    }
    return best;
  };

  // --- Section 1: Raw Ceiling ---
  {
    cl_program p = cl.build(kProbeCl, "-cl-std=CL3.0");
    cl_kernel kv = cl.kernel(p, "bw_vec16");
    cl_kernel kb = cl.kernel(p, "bw_block");
    int n4 = (int)(buf_bytes / 16), n1 = (int)(buf_bytes / 4);
    set_args(kv, 0, db, n4, bo);
    const double tv2k = time_kernel(kv, 2048 * 256, 256);
    const double tv8k = time_kernel(kv, 8192 * 256, 256);
    const uint32_t thr_all = std::max<uint32_t>(1024, eus * std::max<cl_uint>(thr, 8));
    const uint32_t gb = std::max<uint32_t>(256, thr_all * 4 / 16);
    set_args(kb, 0, db, n1, bo);
    const double tb = time_kernel(kb, gb * 256, 256);
    printf("--- 1. Raw Memory Read Ceiling (%.0f MiB random buffer) ---\n", buf_bytes / 1048576.0);
    printf("  16-byte per-lane loads (2048 WGs) : %6.2f GB/s\n", buf_bytes / tv2k / 1e9);
    printf("  16-byte per-lane loads (8192 WGs) : %6.2f GB/s\n", buf_bytes / tv8k / 1e9);
    printf("  Sub-group block reads  (2 KB/sg)  : %6.2f GB/s\n\n", buf_bytes / tb / 1e9);
    clReleaseKernel(kv);
    clReleaseKernel(kb);
    clReleaseProgram(p);
  }

  // --- Section 2: Block-Read Row Stream (`gemv` pattern) vs Row Stride Padding ---
  {
    printf("--- 2. Sub-Group Block-Read Row Stream (`gemv` pattern, GROWS=4) Across Strides ---\n");
    printf("  %-26s | %7s | %9s | %9s | %9s | %9s | %9s | %9s\n",
           "Quant / Matrix Shape", "SB (B)", "NativeRow", "Pad +0 B", "Pad +64B", "Pad+128B", "Pad+256B", "Pad+512B");
    printf("  ---------------------------+---------+-----------+-----------+-----------+-----------+-----------+----------\n");

    struct Case {
      const char* label;
      int K;
      int sb_bytes;
      int grows;
    };
    const Case cases[] = {
        {"Q4_K_S [N x 5120] (20 sb)", 5120, 144, 4},
        {"Q4_K_S [N x 17408] (68 sb)", 17408, 144, 4},
        {"IQ3_S  [N x 5120] (20 sb)", 5120, 112, 4},
        {"IQ3_S  [N x 17408] (68 sb)", 17408, 112, 4},
        {"IQ2_XXS[N x 5120] (72B sb)", 5120, 72, 4},
        {"IQ2_XXS[N x 5120] (80B 16a)", 5120, 80, 4},
        {"IQ2_XS [N x 5120] (80B sb)", 5120, 80, 4},
        {"Q8_0   [N x 5120] (272B sb)", 5120, 272, 4},
        {"F16    [N x 4096] (8192B Po2)", 4096, 512, 4},
        {"F16    [N x 5120] (10240B)", 5120, 512, 4},
    };
    const int pads[] = {0, 64, 128, 256, 512};
    for (const auto& c : cases) {
      const int nsb = c.K / 256;
      const int win_bytes = 2 * c.sb_bytes;
      const int win_calls = (win_bytes + 63) / 64;
      const int native_row = nsb * c.sb_bytes;
      const int nsteps = nsb / 2;
      const int nrows = (int)(buf_bytes / (native_row + 512)) / (c.grows * 16) * (c.grows * 16);
      const double useful_bytes = (double)nrows * native_row;

      std::string opts = "-cl-std=CL3.0 -DGROWS=" + std::to_string(c.grows) + " -DWIN_CALLS=" + std::to_string(win_calls);
      cl_program p = cl.build(kProbeCl, opts);
      cl_kernel k = cl.kernel(p, "row_stream_block");
      const uint32_t sgs = (uint32_t)((nrows + c.grows - 1) / c.grows);
      const uint32_t gws = (sgs * 16 + 255) / 256 * 256;

      printf("  %-26s | %5d B | %7d B |", c.label, c.sb_bytes, native_row);
      for (int pad : pads) {
        int stride = native_row + pad;
        int nr = nrows, ns = nsteps, wb = win_bytes;
        set_args(k, 0, db, nr, ns, wb, stride, bo);
        const double t = time_kernel(k, gws, 256);
        printf(" %6.1f GB/s|", useful_bytes / t / 1e9);
      }
      printf("\n");
      clReleaseKernel(k);
      clReleaseProgram(p);
    }
    printf("\n");
  }

  // --- Section 3: Per-Lane Vector-Load Row Stream (`gemv_v` pattern) Across Layouts & Strides ---
  {
    printf("--- 3. Per-Lane Vector-Load Row Stream (`gemv_v` pattern) — Alignment & Padding Trio ---\n");
    printf("  %-38s | %7s | %9s | %9s | %9s | %9s\n",
           "Pattern / Layout", "SB (B)", "NativeRow", "Pad +0 B", "Pad +64B", "Pad+512B");
    printf("  ---------------------------------------+---------+-----------+-----------+-----------+----------\n");

    struct VCase {
      const char* label;
      int vmode;
      int K;
      int sb_bytes;
      int grows;
      int logical_sb_bytes;
    };
    const VCase vcases[] = {
        {"IQ2_S   GGUF unregrouped (82B, 2B-align)", 5, 5120, 82, 4, 82},
        {"IQ2_S   inferzero native (88B, 8B-align)", 6, 5120, 88, 4, 82},
        {"IQ2_S   16B-repacked SB  (96B, 16B-align)", 6, 5120, 96, 4, 82},
        {"IQ2_XXS GGUF unregrouped (66B, 2B-align)", 2, 5120, 66, 4, 66},
        {"IQ2_XXS inferzero native (72B, 8B-align)", 1, 5120, 72, 4, 66},
        {"IQ2_XXS 16B-repacked SB  (80B, 16B-align)", 1, 5120, 80, 4, 66},
        {"IQ2_XS  GGUF unregrouped (74B, 2B-align)", 2, 5120, 74, 4, 74},
        {"IQ2_XS  inferzero native (80B, 16B-align)", 1, 5120, 80, 4, 74},
        {"IQ3_XXS GGUF unregrouped (98B, 2B-align)", 2, 5120, 98, 4, 98},
        {"IQ3_XXS inferzero native (104B, 8B-align)", 1, 5120, 104, 4, 98},
        {"IQ3_XXS 16B-repacked SB  (112B, 16B-align)", 1, 5120, 112, 4, 98},
        {"Q6_K    GGUF unpadded    (210B, 2B-align)", 3, 5120, 210, 1, 210},
        {"Q6_K    inferzero native (216B, 8B-align)", 3, 5120, 216, 1, 210},
        {"Q6_K    16B-repacked SB  (224B, 16B-align)", 3, 5120, 224, 1, 210},
        {"IQ4_XS  inferzero native (136B, 8B-align)", 0, 5120, 136, 2, 136},
        {"IQ4_XS  16B-repacked SB  (144B, 16B-align)", 0, 5120, 144, 2, 136},
        {"F16     power-of-two row [N x 4096] (8KB)", 4, 4096, 512, 2, 512},
        {"F16     model width row  [N x 5120] (10KB)", 4, 5120, 512, 2, 512},
    };
    const int pads[] = {0, 64, 512};
    for (const auto& vc : vcases) {
      const int nsb = vc.K / 256;
      const int native_row = nsb * vc.sb_bytes;
      const int nrows = (int)(buf_bytes / (native_row + 512)) / (vc.grows * 16) * (vc.grows * 16);
      const double logical_bytes = (double)nrows * nsb * vc.logical_sb_bytes;

      std::string opts = "-cl-std=CL3.0 -DVMODE=" + std::to_string(vc.vmode) + " -DGROWS=" + std::to_string(vc.grows);
      cl_program p = cl.build(kProbeCl, opts);
      cl_kernel k = cl.kernel(p, "row_stream_vec");
      const uint32_t sgs = (uint32_t)((nrows + vc.grows - 1) / vc.grows);
      const uint32_t gws = (sgs * 16 + 255) / 256 * 256;

      printf("  %-38s | %5d B | %7d B |", vc.label, vc.sb_bytes, native_row);
      for (int pad : pads) {
        int stride = native_row + pad;
        int nr = nrows, ns = nsb, sb = vc.sb_bytes;
        set_args(k, 0, db, nr, ns, sb, stride, bo);
        const double t = time_kernel(k, gws, 256);
        printf(" %6.1f GB/s|", logical_bytes / t / 1e9);
      }
      printf("\n");
      clReleaseKernel(k);
      clReleaseProgram(p);
    }
    printf("\n");
  }

  // --- Section 4: Real `decode.cl` GEMV Kernels (`gemv`, `gemv_v`, `gemv_mt`, `gemv_mt_v`) ---
  // Note: decode.cl includes decode_w8_* and h8r_decode before gemv*; strip unused kernels so OpenCL JIT
  // compiles only the target GEMV kernel in milliseconds instead of minutes.
  {
    printf("--- 4. Production `decode.cl` Kernels (Full Decode + DP4a Dot, 600+ MiB Footprint) ---\n");
    printf("  %-34s | %9s | %9s | %9s | %9s | %9s\n",
           "Kernel & Weight Type [N x K]", "Kernel", "(a)Native", "(b)+64B", "(b)+512B", "(c)16B-SB");
    printf("  -----------------------------------+-----------+-----------+-----------+-----------+----------\n");

    const std::string full_src = read_file("zero/kernels/decode.cl");
    // Cut out decode_w8_sb .. decode_w8_sl (between `__kernel void decode_w8_sb` and `// Single-token matrix-vector product`)
    std::string base_src = full_src;
    {
      const size_t a = base_src.find("__kernel void decode_w8_sb");
      const size_t b = base_src.find("// Single-token matrix-vector product");
      if (a != std::string::npos && b != std::string::npos && b > a) {
        base_src.erase(a, b - a);
      }
      const size_t c = base_src.find("__kernel void decode_w8_vb");
      const size_t d = base_src.find("__attribute__((intel_reqd_sub_group_size(16)))\n__kernel void gemv_v");
      if (c != std::string::npos && d != std::string::npos && d > c) {
        base_src.erase(c, d - c);
      }
    }

    auto tabs = build_iq_tables();
    cl_mem d_tabs = cl.buffer(CL_MEM_READ_ONLY, tabs.size() * 4, tabs.data());
    const int maxK = 17408, M = 16;
    cl_mem d_aq = cl.buffer(CL_MEM_READ_WRITE, (size_t)M * maxK);
    cl_mem d_asc = cl.buffer(CL_MEM_READ_WRITE, (size_t)(maxK / 32) * M * 4);
    cl_mem d_abs = cl.buffer(CL_MEM_READ_WRITE, (size_t)(maxK / 16) * M * 2);
    cl_mem d_c = cl.buffer(CL_MEM_READ_WRITE, (size_t)262144 * 8 * 4);
    fill_random(cl, d_aq, (size_t)M * maxK);
    {
      std::vector<float> ones((size_t)(maxK / 32) * M, 0.001f);
      CL_CHECK(clEnqueueWriteBuffer(cl.queue, d_asc, CL_TRUE, 0, ones.size() * 4, ones.data(), 0, nullptr, nullptr));
      std::vector<uint16_t> z((size_t)(maxK / 16) * M, 1);
      CL_CHECK(clEnqueueWriteBuffer(cl.queue, d_abs, CL_TRUE, 0, z.size() * 2, z.data(), 0, nullptr, nullptr));
    }

    struct RealCase {
      const char* label;
      uint32_t type;
      const char* kname;
      int N;
      int K;
      int grows;
      int T;
    };
    const RealCase rcases[] = {
        {"Q4_K_S ffn_gate [17408 x 5120] T=1", GGML_Q4_K, "gemv", 17408, 5120, 4, 1},
        {"Q4_K_S ffn_down [5120 x 17408] T=1", GGML_Q4_K, "gemv", 5120, 17408, 2, 1},
        {"Q4_K_S attn_qkv [10240 x 5120] T=1", GGML_Q4_K, "gemv", 10240, 5120, 4, 1},
        {"IQ2_S  ffn_gate [17408 x 5120] T=1", GGML_IQ2_S, "gemv_v", 17408, 5120, 4, 1},
        {"IQ2_S  ffn_down [5120 x 17408] T=1", GGML_IQ2_S, "gemv_v", 5120, 17408, 4, 1},
        {"IQ3_S  ffn_down [5120 x 17408] T=1", GGML_IQ3_S, "gemv", 5120, 17408, 2, 1},
        {"IQ3_XXS ffn_gate[17408 x 5120] T=1", GGML_IQ3_XXS, "gemv_v", 17408, 5120, 4, 1},
        {"IQ2_XXS ffn_gate[17408 x 5120] T=1", GGML_IQ2_XXS, "gemv_v", 17408, 5120, 4, 1},
        {"IQ2_XS  ffn_down[5120 x 17408] T=1", GGML_IQ2_XS, "gemv_v", 5120, 17408, 4, 1},
        {"Q6_K   lm_head  [64000 x 5120] T=1", GGML_Q6_K, "gemv_v", 64000, 5120, 1, 1},
        {"IQ2_S  ffn_gate [17408 x 5120] T=4", GGML_IQ2_S, "gemv_mt_v", 17408, 5120, 4, 4},
        {"IQ2_S  ffn_down [5120 x 17408] T=4", GGML_IQ2_S, "gemv_mt_v", 5120, 17408, 4, 4},
        {"IQ2_XXS ffn_gate[17408 x 5120] T=4", GGML_IQ2_XXS, "gemv_mt_v", 17408, 5120, 4, 4},
        {"IQ3_S  ffn_down [5120 x 17408] T=4", GGML_IQ3_S, "gemv_mt_v", 5120, 17408, 4, 4},
        {"Q6_K   lm_head  [64000 x 5120] T=4", GGML_Q6_K, "gemv_mt_v", 64000, 5120, 1, 4},
    };

    for (const auto& rc : rcases) {
      const int nsb = rc.K / 256;
      const size_t bb = gguf_sb_bytes(rc.type);
      const size_t bp = raw_padded_sb_bytes(rc.type);
      const size_t bp16 = (bb + 15) / 16 * 16;
      const size_t mat_bytes = (size_t)rc.N * nsb * bb;
      const size_t max_row_stride = (size_t)nsb * bp16 + 512;
      const size_t slice_stride = ((size_t)rc.N * max_row_stride + 4095) / 4096 * 4096;
      const int nslices = std::max(1, (int)(buf_bytes / slice_stride));
      std::vector<cl_mem> slices;
      for (int s = 0; s < nslices; ++s) {
        cl_int err = 0;
        cl_buffer_region reg{(size_t)s * slice_stride, slice_stride};
        slices.push_back(clCreateSubBuffer(db, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err));
      }

      auto run_variant = [&](int row_pad, bool align16_sb) -> double {
        std::string patched = base_src;
        if (align16_sb) {
          const std::string old_bbp = "#define BBP ((BB + 7) / 8 * 8)";
          const std::string new_bbp = "#define BBP ((BB + 15) / 16 * 16)";
          if (auto pos = patched.find(old_bbp); pos != std::string::npos) {
            patched.replace(pos, old_bbp.size(), new_bbp);
          }
        }
        const std::string defs = "-cl-std=CL3.0 -cl-mad-enable -DQT=" + std::to_string(rc.type) +
                                 " -DGROWS=" + std::to_string(rc.grows) +
                                 (row_pad > 0 ? " -DROW_PAD=" + std::to_string(row_pad) : "") +
                                 (rc.T > 1 ? " -cl-intel-256-GRF-per-thread -DTB=" + std::to_string(rc.T) : "");
        cl_program p = cl.build(patched, defs);
        cl_kernel k = cl.kernel(p, rc.kname);
        const uint32_t lws = 256;
        const uint32_t gws = (uint32_t)(((size_t)(rc.N + rc.grows - 1) / rc.grows * 16 + lws - 1) / lws * lws);
        ZE_CHECK(zeKernelSetGroupSize(k->k, lws, 1, 1));
        ze_group_count_t gc{gws / lws, 1, 1};
        const bool mt = rc.T > 1;
        int N = rc.N, K = rc.K, MM = M, T = rc.T, ldc = rc.N, c_off = 0, acc = 0;

        auto pass = [&] {
          const auto t0 = clk::now();
          for (cl_mem sl : slices) {
            if (mt) set_args(k, 0, sl, N, K, d_tabs, d_aq, d_asc, d_abs, MM, T, d_c, ldc, c_off, acc);
            else set_args(k, 0, sl, N, K, d_tabs, d_aq, d_asc, d_abs, MM, d_c, c_off, acc);
            ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
          }
          zrt::sync();
          return std::chrono::duration<double>(clk::now() - t0).count();
        };
        pass();  // warmup
        double best = 1e30, spent = 0;
        int reps = 0;
        while (spent < 0.4 || reps < 3) {
          const double t = pass();
          best = std::min(best, t);
          spent += t;
          ++reps;
        }
        clReleaseKernel(k);
        clReleaseProgram(p);
        return (double)nslices * mat_bytes / best / 1e9;
      };

      const double gbs_nat = run_variant(0, false);
      const double gbs_p64 = run_variant(64, false);
      const double gbs_p512 = run_variant(512, false);
      const double gbs_16b = (bp == bp16) ? gbs_nat : run_variant(0, true);

      printf("  %-34s | %-9s | %6.1f GB/s| %6.1f GB/s| %6.1f GB/s| %6.1f GB/s\n",
             rc.label, rc.kname, gbs_nat, gbs_p64, gbs_p512, gbs_16b);

      for (cl_mem sl : slices) clReleaseMemObject(sl);
    }

    clReleaseMemObject(d_tabs);
    clReleaseMemObject(d_aq);
    clReleaseMemObject(d_asc);
    clReleaseMemObject(d_abs);
    clReleaseMemObject(d_c);
  }

  clReleaseMemObject(db);
  clReleaseMemObject(bo);
  return 0;
}
