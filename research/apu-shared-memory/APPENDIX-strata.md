# Strata (Niko1221/Strata) — code-level analysis for inferzero / MTL Level Zero

Repo: `/home/bschooley/.hermes/cache/scratch/Strata` (MIT). Analysis is from current source + current docs at this
checkout, not from blog posts. Every claim below carries `file:line`.

> **Deliverable location.** The task asked for this content at
> `/home/bschooley/.hermes/cache/scratch/research-notes/strata-findings.md`. Under this task's read-only rule the only
> file I may write is this findings file, so the full deliverable lives here. Copy it verbatim to that path if the
> scratch copy is wanted.

---

## 0. Executive summary — the three answers that change our plan

1. **Strata has no Intel *iGPU* path.** The SYCL backend is an Arc **discrete** card port (Alchemist DG2 / Battlemage
   BMG-G21/G31). Integrated Intel Arc is explicitly listed as untested **and** as having no memory planning:
   `docs/INTEL_ARC.md:177-179` — *"Not tested by anyone: … integrated Arc GPUs (the B390 / Panther Lake in #515;
   shared-memory planning does not exist)"*. Device classification knows only two Intel generations
   (`engine/sycl/include/strata/sycl_queue.hpp:120-158`, enum `IntelGpuGen { Unknown, Alchemist, Battlemage }`). A
   Meteor Lake iGPU falls into `Unknown`, so every Battlemage/Alchemist tuning default is off.
   → **Nothing about VRAM-less iGPU allocation is solved in Strata. That part of our engine is genuinely ahead.**
2. **Strata's decode kernels are dp4a, not DPAS.** `dpct::dp4a` via `STRATA_DP4A`
   (`engine/sycl/include/strata/kernels/dp4a.hpp:31-40`) is what every decode matvec uses. DPAS appears only in the
   ESIMD **prompt**-expert kernels and the XMX GEMM, and the s8 DPAS combination they need is an **Xe2** feature
   (`docs/INTEL_INT8_EXPERTS.md:43-46`: *"s8 8x16x32 matrix combination (Xe2: Battlemage, Lunar Lake) … DP4a fallback:
   none"*). Xe-LPG (our MTL iGPU) is Xe1-class, so Strata's int8 DPAS path would not even be available to us.
   → **Our "DP4a beats DPAS on MTL" measurement is architecturally expected for *decode*, not a tuning miss.** Strata
   independently found grouped XMX 11-15% slower on Battlemage (`docs/INTEL.md:504`) and -24% to -27% for XMX prompt
   attention (`docs/INTEL.md:404`). **But matrix cores do win prefill on APUs** — ×1.59-1.72 on Strix Halo
   (`docs/STRIX_HALO_WINDOWS.md:69-75`), ×1.6-2.0 on a 780M (`docs/RADEON_780M.md:88-89`) — so our DPAS result was
   probably measured in the decode-shaped regime and is worth retesting at T ≥ 256, where we are at ~33% of DP4a TOPS.
3. **The single most transferable thing for us is the MTP draft chain structure**, because our binding constraint is
   the sequential draft chain, not bandwidth. Strata submits the whole draft round as **pre-captured graphs on one
   queue, with device-side token handoff and host polling of a doorbell**, and can gate steps by draft confidence
   (`engine/sycl/src/core/mtp.cpp:1645-1752` vs `:1753-1833`). Metal goes further and fuses the per-step host work
   into kernels (`engine/metal/src/engine/msl/mtp_chain.metal:1-8`).
4. **Our model is dense, and that changes which of their tricks pay.** 66.5 GB/s ÷ 3.05 tok/s = **21.8 GB per token** =
   one full read of a 40B model at ~4.4 bits/weight. Strata's whole design assumes a 512-expert MoE where a window
   re-reads only 1-3 GB of experts. So: their MoE tiering and expert-cache work does **not** transfer, while their
   weight-stationary "decode once, apply to every token of the window" trick (`engine/metal/kernels/gemv.metal:1-7`)
   pays us *more* than it pays them — every extra accepted token per window is a straight multiplier on effective
   bandwidth. See §8.8 and §9 q5.

A second documentation sweep (added to the brief mid-task) is in **§7b** and it **corrected two of my rankings**: the
760-line `docs/A770_SPEED_SPEC.md` is an autopsy of a failed Intel tuning campaign with measured negative results for
drafter/verify overlap (`:254-258`) and large GRF on BF16 (`:528-530`), and it shows the A770's decode was ~20% GPU-busy
with 87% of the window spent waiting on a CPU handshake (`:687-752`). That log is the most valuable single document in
the repo for us, mostly for what *not* to try.

No NPU support exists anywhere in the repo (grep for `npu|VPU|openvino|zeGraph|intel_npu` returns only false positives
from the substring `npu` inside `input`). Strata does not touch the Intel NPU.

---

## 1. The `engine/sycl/` directory — what the Intel path actually is

### 1.1 Provenance: it is a dpct-migrated CUDA engine, not a hand-written SYCL engine

- `docs/INTEL.md:6-9`: *"Strata's engine is CUDA. On an Intel Arc it runs as Strata's own engine ported to SYCL
  (`sycl/`) … Everything Intel-specific lives in `sycl/`; no shared upstream file is changed."*
- The port is mechanical-then-repaired: `engine/sycl/tools/migrate.sh` runs SYCLomatic/dpct over the CUDA translation
  units, `engine/sycl/tools/fixups.py` repairs what dpct got wrong, `engine/sycl/tools/build.sh` builds with icpx
  (`docs/INTEL.md:337-348`).
- Consequence visible in source: thousands of `DPCTxxxx` diagnostic comments, e.g.
  `engine/sycl/src/core/mtp.cpp:1700-1717`, and `dpct::get_in_order_queue()` used as CUDA's null stream
  (`engine/sycl/include/strata/sycl_queue.hpp:12-14`).
- Layout: `engine/sycl/src` and `engine/sycl/include` hold **only the files the port changes**;
  `engine/sycl/CMakeLists.txt` takes every other source from the original location (`docs/INTEL.md:334-335`).
  The decode kernels therefore live at `engine/sycl/src/kernels/cuda/*.dp.cpp` — "cuda" is the *source tree* name.

### 1.2 Xe kernel inventory — what exists, and what silicon it needs

| Kernel family | File | Xe feature | Notes |
|---|---|---|---|
| Decode quantized matvec (all formats) | `engine/sycl/src/kernels/cuda/native_mmvq.dp.cpp` (6,023 lines) | `dp4a`, 32-lane sub-group butterfly | the hot decode path; per-format block structs at `:75-128` with `static_assert` layout checks `:129-148` |
| Q6_K "wide" decode (16-byte loads) | same file `:1503-1708` | 16-byte vector loads, `grf_size<256>` | the alignment story, §2 |
| Q4_K / Q5_K / IQ4_XS wide decode | same file `:1709-1760+` | 16-byte loads | `:1711-1713` gives the before/after bandwidth |
| Prompt experts, fused int8 | `engine/sycl/src/prefill/esimd_moe.dp.cpp` | **ESIMD + s8 DPAS 8x16x32 (Xe2 only)**, 256 GRF, sub-group 16 | described in `docs/INTEL_INT8_EXPERTS.md` |
| Prompt experts, grouped XMX | `engine/sycl/src/prefill/xmx_moe.dp.cpp` | XMX / joint_matrix | default on Alchemist only (`docs/INTEL.md:262-264`) |
| Prompt GEMM (bf16) | `engine/sycl/src/prefill/gemm.dp.cpp`, `gemm_bf16_parity.dp.cpp` | oneMKL | oneMKL 4 GiB operand bug worked around (§2.5) |
| DPAS peak probe | `engine/sycl/src/kernels/esimd_dpas_probe.cpp` | ESIMD `xmx::dpas` | standalone microbenchmark, `:1-5`, `:17-46` |
| Attention / QSA / DeltaNet / router / sampler / KV | `engine/sycl/src/kernels/*_parity.cpp` + `engine/sycl/src/core/*` | portable | three PTX kernels were **not** ported (`docs/INTEL.md:374-379`) |

Subgroup tricks in use:
- 32-lane sub-groups are forced: `-fsycl-default-sub-group-size=32` because *"the kernels are written for warps"*
  (`docs/INTEL.md:353`), plus explicit `[[sycl::reqd_sub_group_size(32)]]` at each launch, e.g.
  `native_mmvq.dp.cpp:1675`, `:1683`, `:1695`.
- Butterfly reduction: `sycl::permute_group_by_xor(sg, v, o)` for `o = SG/2 … 1`
  (`native_mmvq.dp.cpp:1619-1628`); a 16-lane variant `sg_sum<SG>` at `:165-170`.
- Sub-group-size experiment that lost: *"`STRATA_MMVQ_SG`, SIMD16 sub-group variants of the decode mmvq, were removed
  in 0.2.13: no gain in the engine"* (`native_mmvq.dp.cpp:1630`).
- Prefetch experiment that lost: *"(STRATA_MMVQ_PF, a prefetching wide Q6_K kernel … were removed in 0.2.13: no gain in
  the engine.)"* — same line, `:1630`. This matters: explicit software prefetch did **not** pay on Xe.

ESIMD bring-up notes (compiler bugs worth knowing if we ever use ESIMD on MTL):
`docs/INTEL_INT8_EXPERTS.md:74-80` — dynamic index into a kernel-argument array returns garbage; `t % n` / `t / n` with
runtime `n` folds to `t`, so kernels use 2-D grids; `gather<T,N,VS>` with `VS > 1` unavailable on Battlemage; a lambda
capturing a `simd` by reference lands in private memory.

### 1.3 Memory allocation on Intel — no iGPU story, and the pool bug is documented

There is **no** shared/USM-iGPU allocation strategy, because the target is a discrete card:

- **No virtual memory management at all.** `engine/sycl/src/core/vmm.cpp:1-6`: *"SYCL port: no virtual memory
  management. Upstream's vmm.cpp reaches the CUDA driver's `cuMem*` … Level Zero has no equivalent through dpct. Every
  entry point therefore reports 'not available'"*. `vmm_available()` returns `false` (`:11`); `VmmRange::reserve`
  returns `false` (`:16`). Consequently `--vram-elastic` and elastic KV (`--kv-grow`) **refuse** on the SYCL backend.
  Reinforced at `engine/sycl/src/core/expert_cache.cpp:12`: `#define STRATA_EC_NO_VMM 1` — *"--vram-elastic is
  CUDA-only"*.
- **Device memory is plain `sycl::malloc_device`**, wrapped only for two Intel driver bugs
  (`engine/sycl/include/strata/sycl_queue.hpp:264-322`):
  1. `max_mem_alloc_size` is reported and a too-large request produces a *named* error instead of a silent null
     (`:281-293`, citing #5557: *"`sycl::malloc_device` answers null, not an error, for one allocation above the GPU's
     limit"*). Measured limits in the comment: *"3.77 GiB on an 8 GB A750, ~4 GiB on a 16 GB A770 under NEO, 1 GiB
     under WSL2 GPU-PV"* (`:267-269`).
  2. **Aliased pages**: allocations ≥ 32 MiB are written with a per-64 KiB tag and read back; on failure the allocation
     is retried behind a growing 2·n MiB spacer (`:171-213`, `:294-319`). `xe` returned a 22 GiB arena with two 2 MiB
     pages mapped onto the same memory (write at +2 MiB appeared at +1022 MiB → NaN logits). Escape hatch
     `STRATA_ARENA_ALIAS_CHECK=0`.
- **Host memory that a running kernel polls** is special-cased: ordinary host USM does not work as a doorbell on
  Alchemist/i915 even with system-scope atomics, so buffers come from
  `zeMemAllocHost(..., ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED)` (`sycl_queue.hpp:42-78`,
  `host_malloc_polled`). `STRATA_HOST_UNCACHED=0` disables.
- **Driver-dependent host-memory access**: *"an Arc A750 (i915) reads ordinary host memory a kernel is handed; an Arc
  Pro B70 (xe) page-faults on it (ccs0, FaultType 0, address in the CPU's mmap range) and the card times the job out"*
  (`sycl_queue.hpp:94-96`). **`xe` is the driver Meteor Lake uses by default**, so if inferzero hands a pageable host
  pointer to a kernel it should expect exactly this fault. (Runtime question — verify which driver our box binds.)
- **Large fills cannot use the blitter**: one `queue.memset` of many GiB times out on `xe` ("Engine memory CAT error",
  GT reset), so `big_fill_zero` does the fill as compute kernels of ≤ 256 MiB each, waited one at a time
  (`sycl_queue.hpp:18-39`).
- **Over-allocation past the pool is a machine-level hazard, and it is exactly our iGPU situation**:
  `docs/INTEL.md:142-144` — *"On the `xe` driver an allocation past VRAM does not fail: the driver evicts buffers into
  host RAM, the kernel runs out of memory, and the machine can livelock until its hardware watchdog resets it."*
  Repeated at `docs/INTEL_ARC.md:129-130`. On a shared pool there is no "past VRAM" wall at all, so this is the one
  Intel-specific warning that transfers directly to us: budget, never rely on allocation failure.
- **Bounded spins are mandatory**: a device spin that never sees its flag makes `xe` time out the queue and reset the
  GT (`docs/INTEL.md:149-151`; bound is `STRATA_SYCL_SPIN_MAX`, 2,000,000 reads, 20,000 for AOT `bmg` builds).
- **Pinned mirror limits**: a single `sycl::malloc_host` above a few GB fails on the A750
  (`docs/INTEL.md:250-252`), so the mirror is chunked on `xe` (`engine/sycl/setup_intel.py:54-57`).
- Copies staged through `pinned` buffers because *"SYCL has no `cudaHostRegister` and a direct copy out of pageable
  memory could hang a prompt"* (`docs/INTEL.md:265-267`).

### 1.4 Intel-specific docs on Meteor Lake / Lunar Lake / APUs

- **Meteor Lake**: exactly one mention in the whole tree, and it is about **CPU** topology, not the GPU —
  `docs/release-notes-0.2.1.md:47`: *"Linux hybrid CPUs with low-power E cores (Core Ultra 'Meteor Lake' /
  'Arrow Lake-H', #3262). The engine and setup now count the cores listed in `cpu_lowpower` as efficiency cores. A
  Core Ultra 7 255H used to look like 8 P and 8 E cores, so the hybrid plan did not apply and 15 workers ran; it now
  plans 10. The reporter measured the 15-worker default about 35% slower than 9."* Implementation:
  `engine/include/strata/kernels/cpu/pool_plan.hpp:52-65` (`hybrid_cpu_is_efficiency`, reads
  `/sys/devices/cpu_lowpower/cpus`) and `engine/src/kernels/cpu/pool.cpp:483-487`.
- **Lunar Lake**: one mention, as Xe2 silicon for the s8 DPAS combination (`docs/INTEL_INT8_EXPERTS.md:43`). No
  measured numbers, no config, no tested-card row.
- **Panther Lake / integrated Arc**: `docs/INTEL_ARC.md:177-179` (untested + "shared-memory planning does not exist").
- **APU behaviour that *is* implemented is AMD-only** — see §3.4. There is no Intel analogue.
- **Intel NPU: absent.** No NPU/VPU/OpenVINO/`zeGraphCreate` symbol anywhere; the image reader is CUDA/HIP/Vulkan/CPU
  only (`docs/IMAGES.md:61-65`). Strata's Intel path never leaves the GPU.

---

## 2. Cross-arch decode-GEMV patterns: SYCL vs Metal vs CUDA

The CUDA and SYCL paths are **the same kernel family** — `engine/sycl/src/kernels/cuda/native_mmvq.dp.cpp:4-6` says it
is adapted from llama.cpp's `mmvq.cu` (pinned commit `3cf03257`), and the CUDA original is
`engine/src/kernels/cuda/native_mmvq.cu` with identical constants (`QK=256, Q8K=32, WARPS=4, WARP=32` at
`native_mmvq.cu:59-65` vs `native_mmvq.dp.cpp:67-73`) and identical block layouts (`:67-80` vs `:75-128`). So the real
three-way comparison is **(CUDA ≈ SYCL) vs Metal**, plus the SYCL-only load-width work.

### 2.1 The measured bandwidth evidence (this is the payload)

All on an Arc Pro B70, from `engine/sycl/src/kernels/cuda/native_mmvq.dp.cpp`:

| Line | Measurement |
|---|---|
| `:1505-1506` | *"bw probe on the B70: 4-byte loads per lane stream at **395 GB/s**, 16-byte at **596**; the shared kernel above does 4-byte loads and measures **130-140 GB/s** on the 8192-wide Q6_K projections."* |
| `:1509-1512` | *"a 16-byte load from a 2-byte aligned address … the B70 splits a misaligned vector load into pieces: the wide kernel streamed **145 GB/s** that way against **640 GB/s** for the same blocks repacked at a 16-byte stride. Both aligned loads stay inside the 210-byte block … (last byte read <= block + 207)."* |
| `:1711-1713` | *"The multi kernel these replace does 4-byte loads (Q4_K/Q5_K, **230-280 GB/s**) or global-memory codebook lookups (IQ4_XS, **~100 GB/s**)."* |
| `:1635-1639` | Large GRF (`grf_size<256>`): *"on the A770 they spill at 128, and `mmvq_bench` Q6_K 4 columns goes **70-80 → 150-237 GB/s** with the same checksum."* |
| `:1630` | prefetching wide Q6_K kernel + SIMD16 sub-group variants **removed**: "no gain in the engine". |

Two conclusions that bear directly on our 89.9 vs 57.3 GB/s result:
- The **word size is not the problem; alignment is.** Strata gets 1.5x from 16-byte loads *when aligned* and 4.4x
  *loss* when the same 16-byte load is issued from a 2-byte-aligned address. Our 16-byte-load regression is consistent
  with misalignment (GGUF block sizes 18/34/110/144/210 bytes are not 16-aligned) or with a driver splitting the
  vector message. **Fix the layout, not the width.**
- 4-byte-per-lane *scalar streams* already reach 395 GB/s, i.e. 62% of what 16-byte reaches — the memory-level
  parallelism gap is recoverable by issuing more independent loads per lane (which is what row-reuse does), not only by
  widening.

### 2.2 The SYCL/CUDA decode kernel shape (row reuse + sub-group-per-row)

`native_mmvq_q6k_wide_kernel` (`native_mmvq.dp.cpp:1550-1629`):

- **8 lanes share one 256-element block**; lane `g` owns `iqs = 4g..4g+3`, so `ql` (16 B), `qh` (16 B) and each
  column's q8 ints are **one vector load each**; a 32-lane sub-group therefore has **four blocks in flight**
  (`:1506-1508`, index math `:1559-1564`).
- **One row per sub-group, `WARPS` (4) rows per work-group** (`:1554-1557`, `NWG = WARPS*WARP/SG`).
- **`RPW` rows per sub-group share the activation loads**: the activation `int4`s are loaded **once per column** and
  applied to all `RPW` rows (`:1583-1597`), with the stores/reduction unchanged. Dispatched as `RPW ∈ {1,2,4}`
  (`:1702`, `STRATA_MMVQ_WIDE_RPW`, default 1 at `:1631-1634`).
- **dp4a chain, 8 per column-block**: `strata::dp4a(...)` over 4 `int4`s for each of the two halves
  (`:1609-1616`), i.e. `dp4a` is the compute primitive, not DPAS.
- Work-group launcher: `nd_range<3>(range(1,1,blocks) * range(1,WARPS,WARP), range(1,WARPS,WARP))` — a **flat 2-D
  thread block of 128 with one dimension unused**, a direct artifact of the CUDA warp layout (`:1672-1676`).
- Optional `grf_size<256>` property at the launch (`:1674`), with an extra shape-guarded dispatch for Battlemage
  (`STRATA_Q6_SHAPE_GRF`, `:1656-1671`).

The **realignment shim** is the reusable artifact: `load16_a2(const void*)` (`:1526-1544`) loads the containing
aligned 16-byte chunk plus (only when unaligned) the next one, selects 5 dwords by ternary (no indexed register
access — a compiler requirement), and shifts by 2 bytes when needed. Its safety argument is stated at `:1531-1532`:
*"the second chunk only when the address is unaligned: then it holds needed bytes, so (16-byte chunks never straddle a
page) it cannot fault even at the very end of an allocation."* Companion helpers: `ld_i4_a4` (aligned 4-int load,
`:1515-1519`) and a SWAR `q6_sub32` replacing the saturating char4 subtract in 4 integer ops (`:1520-1525`).

### 2.3 The Metal shape (the differentiated one)

`engine/metal/kernels/gemv.metal:1-7` — header: *"y[t][r] = W[r] · x[t] for a window of 1..8 tokens … Shape of a
threadgroup: SG simdgroups; KS of them split the row's units (KS = 1: no split), the other SG/KS cover different rows.
In a simdgroup, LPR lanes share a row and 32/LPR rows run side by side; each lane holds NR rows. **The weights of a
unit are decoded once and applied to every token.** Partial sums of a K split are added in a fixed order
(deterministic)."*

- Template space `gemv<F, LPR, NR, SG, KS>` with a **19-point instantiation grid** per format
  (`GV_SET`, `:93-96`): `LPR ∈ {32,16,8,4,2,1}`, `NR ∈ {1,2,4}`, `SG ∈ {4,8}`, `KS ∈ {1,2,4,8}`. 13 formats × 19
  shapes (`:97-101`). `NTOK` is a **function constant** so the token loop is specialized (`:18`).
- **Weight-stationary, token-broadcast**: per unit, decode `q[NR][UNIT]` and scales for all rows, then loop tokens
  inside (`:60-70`). A probe variant `gemv_probe == 3` inverts the loops (rows outer, tokens inner, x re-read from L1)
  purely to *measure* whether the broadcast wins (`:46-58`) — i.e. this was A/B'd, not assumed.
- **Codebook staged in threadgroup memory** (`threadgroup uint tab[fmt<F>::TAB]`, `:30-33`) and a **K-split
  cross-simdgroup reduction** in a fixed order for determinism (`:71-90`).
- A paired-gate/up fused dispatch `gemv_pair` writes `silu(Wg·x) * (Wu·x)` in one pass, two formats, two codebooks
  (`:103-150+`).

### 2.4 Which pattern gets closest to peak, and why (their words)

- 16-byte loads with **8 lanes per block, 4 blocks in flight per sub-group** and the realignment shim is the closest:
  596 GB/s probe / 640 GB/s repacked, vs 130-140 GB/s for the ported 4-byte kernel (`:1505-1511`).
- The reasons stated are *load width* and *alignment*, plus **row reuse of activations** (`:1688-1689`).
- **Prefetch did not help** (`:1630`), and **smaller sub-groups did not help** (`:1630`) — both negative results we
  can skip re-testing.
- Metal's contribution is orthogonal: **amortize one weight decode across the whole verify window** rather than
  reduce bytes. Quantified in `docs/NVFP4.md:276-277`: *"Each weight is decoded once for all the tokens of a verify
  window, bit for bit the same results. Before the kernel work, the RTX 3060 took 2.26 ms at 4 tokens"* → 0.766 ms
  (2.95x), with 89% of DRAM bandwidth at 1 token and 79% at 4 (`:270-274`).

### 2.5 Other SYCL↔CUDA deltas worth knowing

- oneMKL reads the last partial tile of an operand from the wrong place when it lies > 4 GiB into an allocation, so the
  GEMM **stages** any operand that `device_offset_end` reports > 4 GiB (`sycl_queue.hpp:248-263`, `:405` docs).
  Related build flag `-ze-opt-greater-than-4GB-buffer-required` (`docs/INTEL.md:152-154`).
- `-fp-model=precise` and `-cl-fp32-correctly-rounded-divide-sqrt` are load-bearing: icpx defaults to a fast FP model
  and the Arc's fp32 divide is not correctly rounded, which breaks byte-exactness against ggml (`docs/INTEL.md:345-347`).
- Copies on the wrong engine: §3.5.

---

## 3. Memory budgeting, layer placement, "one-click on any consumer hardware"

The whole model is described as a **two-tier + SSD** system, not a partial-offload `n_gpu_layers` heuristic
(`docs/HOW_IT_WORKS.md:8-33`):

- GPU/VRAM: attention + DeltaNet mixers, gated-residual weights, routers, shared experts, output head, MTP draft layer,
  KV (from 64K only its most-read part), and an **expert cache** that fills the rest of VRAM.
- RAM: **all 24,576 experts, pinned**; the CPU computes the experts the GPU does not hold, **in place, concurrently**
  with the GPU.
- SSD: the 28.8 GB n-gram (PLE) table, read a few rows per token.
- Sizing rule of thumb: *"every extra GB holds ~700 more experts"* (`HOW_IT_WORKS.md:25`).

### 3.1 The placement decision

- `--stream-experts` (Arc default): no resident host copy of the experts; every expert goes GGUF → VRAM through a
  staging ring. **Result: the engine runs in 23 GiB of RAM where upstream needs 32 GB** (`docs/INTEL.md:110`).
- Under `STRATA_VERIFY_NO_HOST=1` there is **no SSD expert tier**: every expert must be in VRAM or in the pinned mirror
  (`STRATA_MIRROR_MIB`), and the engine **refuses to start** with `"REFUSED: N experts are neither in VRAM nor
  mirrored"` (`docs/INTEL.md:133-136`). This is a hard admission-control gate, not best-effort paging.
- **Expert profile** = a *ranked* list, so the cache holds what a given workload actually routes to. On `xe`, setup
  writes `terminal/data/expert-profile-sycl.bin` built from the routing of 45 varied prompts
  (`docs/INTEL.md:204-206`); shipped variants exist for base/coder and an `--expert-profile-sycl.bin` in
  `terminal/data/`. Ranking reader: `engine/sycl/src/core/expert_cache.cpp:152+` (`read_expert_profile`).
- **Prompt can borrow cache slots** and refill after (`--prefill-borrow`, `STRATA_LEND_MIRROR`,
  `STRATA_PREFILL_STREAM_ALL`) — decode after the prompt is 10-23% slower without borrowing but the prompt is 2-5%
  faster (`docs/INTEL.md:187-189`).
- **PCIe share heuristic**: the fraction of missed experts *copied* to the GPU instead of computed by the CPU is a
  continuous function of the measured H2D bandwidth, not a fixed constant —
  `engine/sycl/src/program/generate.cpp:1695-1704` (`pcie_frac_for_gbps`), calibrated at `:2782-2805`; below 20 GB/s it
  scales linearly, so *"a laptop's narrow link wants less"* (`docs/DETAILS.md:144`). For our box the analogue is the
  fabric, and the *shape* of the rule (measure, then scale continuously) is what to copy.
- NVFP4 gets its own share (0.25 not 0.55) because its expert blob is 2.76 MB vs 1.51 MB; *"At 0.55 the copy kernel was
  40% of all GPU kernel time"* (`docs/NVFP4.md:64-69`).

### 3.2 Refusal and reserve behaviour on `xe`

- VRAM reserve: 1,024 MiB up to 32K context, 2,048 above, 300 MiB under 12 GB; a user value is kept
  (`docs/INTEL.md:41`).
- Fit ladder: *"A model that does not fit the card at the default context gets a context down the ladder 32K, 16K, 8K,
  a smaller prompt chunk and a lower reserve until it does. Setup prints one plain line per change."*
  (`docs/INTEL.md:42`).
- Companion rule from the reports: forced reserves cost real throughput — a forced 1536 MiB reserve cut one reporter's
  decode 80 → 50 tok/s (`docs/release-notes-0.2.1.md:44`, `:53`).

### 3.3 Where free memory is actually read (the part that matters for a shared pool)

`device_free_bytes()` — `engine/sycl/src/core/expert_cache.cpp:69-145`:

1. Driver first: `dpct::get_current_device().get_memory_info(free_b, total_b)` (`:76`).
2. If the driver says **0 free with a known total** (xe without `ext_intel_free_memory`, A770 #1545), fall back to
   `total − own_drm_local_bytes()` (`:77-89`).
3. `own_drm_local_bytes()` parses `/proc/self/fdinfo` for `drm-total-local0` (i915) / `drm-total-vram0` (xe)
   (`:30-64`) — *"Level Zero's free-memory query reads the constant total on an Arc A750 (i915): the engine then sized
   its caches from the whole card and over-allocated (UR_RESULT_ERROR_OUT_OF_HOST_MEMORY at the first launch)"*
   (`:31-33`).
4. If the device is **integrated**, take `MemAvailable − headroom` when that is larger
   (`dpct::get_device(dev).get_integrated()`, `:101-142`), headroom default **6 GiB**
   (`STRATA_UMA_HEADROOM_GIB`, `:128-138`), justified by NVIDIA's DGX Spark guide §5.5 *"Memory reporting on UMA
   systems"*, and explicitly **excluding swap** (`:66-68`).

That is the correct model for a VRAM-less iGPU, and it is four lines of policy: *driver number, own-client correction,
then `MemAvailable − 6 GiB` on integrated parts.* Our engine should adopt step 4 before its first allocation.

### 3.4 The APU (AMD) placement model — the closest thing to an iGPU plan Strata has

`terminal/setup.py:4450-4464` (`amd_apply_uma`), docstring verbatim:

> *"Strix Halo is an APU with unified memory. Its 'dedicated VRAM' is a BIOS carve-out (512 MB to 96 GB; 2 GB on the
> maintainers' box) and the GPU also reaches shared system memory … The shared part IS the RAM the model's experts live
> in, so it is not extra room beside the RAM. g gets: uma, dedicated_gb (the carve-out: the only part that adds to the
> RAM, which the low-RAM mode counts) and vram_gb = the memory the GPU can use in all (the carve-out plus the shared
> pool less what the OS keeps)."*

- `UMA_OS_LEFT_GB = 6` (`setup.py:4400`); `shared = min(GTT, ram − 6)` (`:4461-4462`).
- **Double counting is the bug they fixed**: `terminal/tools/strata_mcp.py:1362-1364` — *"RAM + VRAM together. An
  APU's shared memory is the RAM itself, so only its BIOS carve-out counts as VRAM."*
- Per-desktop reserve: *"the larger of 8 GiB and 8% of the RAM"* (`docs/STRIX_HALO.md:167-169`,
  `STRATA_UMA_RESERVE_GIB`).
- The expert cache is given an **explicit slot count** on an APU rather than `auto`, because `auto` would take the room
  the OS needs (`docs/STRIX_HALO.md:38-40`).
- APU prompt buffers share the pool with the expert cache, so a `--prefill auto:32768` tip is suppressed on APUs
  (`setup.py:4444-4447`).
- Engine runs at `nice +5` on a Linux APU so the desktop keeps its CPU (`docs/STRIX_HALO.md:175-176`).
- Intel equivalent: **none.** `docs/INTEL_ARC.md:177-179`.

### 3.5 Copies on the wrong engine — a Level Zero trap we share

`engine/sycl/include/strata/copy_queue.hpp:1-15` (header comment):

> *"With the Level Zero 'v2' adapter (the default on Xe2 / Battlemage and newer), every SYCL queue, including a queue
> that only holds host-to-device copies, runs on the compute engine: a 2 MB expert copy (~80 us at 25 GB/s) blocks the
> kernels behind it for that long. A prompt chunk streams ~10,000 experts: **~0.8 s of the 3.8 s of a 4K prompt** on an
> Arc Pro B70 is the compute engine waiting for its own copies (unitrace: every H2D copy sits on 'L0 Compute
> Engine<0,0>' and none overlaps a kernel). The device does expose a copy-only queue group
> (`zeDeviceGetCommandQueueGroupProperties`: group 1, flags COPY, 1 queue); the v2 adapter just does not use it for
> in-order queues."*

Fix: `make_copy_engine_queue()` creates an immediate command list on that group and wraps it with
`sycl::make_queue<ext_oneapi_level_zero>` (`:10-12`, `:22`). Gate: `STRATA_COPY_ENGINE` (default on for Battlemage).
**This is a 21% prompt-time win from a queue-group selection, and it applies to any Level Zero device that reports a
COPY group** — our MTL iGPU very likely does.

### 3.6 CPU-side placement (E-core aware)

`engine/include/strata/kernels/cpu/pool_plan.hpp:40-43`: *"With more E-cores than P-cores the default is the P-cores
but the host's plus half of the E-cores (#642: i9-14900KF 8P+16E, 15 workers 165 / 116 tok/s against 106 / 84 with all
23; Core Ultra 7 265, 8P+12E, RTX 3060 decode at 4K: 12-16 workers +12.6% over the 19 of one-per-core, 24 workers
-44%)"*. Meteor Lake LP-E detection at `pool_plan.hpp:52-65` / `pool.cpp:483-487`; the reported MTL result is 15 workers
→ 10 and *"about 35% slower than 9"* (`docs/release-notes-0.2.1.md:47`). For a Core Ultra 7 155H (6P + 8E + 2 LP-E)
the naive "one worker per physical core" plan is 16 and is wrong.

---

## 4. Speculative decoding / MTP

### 4.1 Strategy

- **The model's own MTP layer drafts; the full model verifies.** `docs/HOW_IT_WORKS.md:39-46`: *"the model's own MTP
  layer drafts up to 3 tokens; one pass over all 48 layers checks them, 2.4-3.2 tokens per pass on average"*, giving
  *"1.6-1.8x sooner"*.
- **A second drafter: prompt lookup.** *"prompt lookup drafts up to 5 tokens from the earlier copy, but only where its
  measured acceptance and cost say it pays: code edits 6-11% faster, other text unchanged"* (`HOW_IT_WORKS.md:44-46`);
  `--lookup-chain-min M`, default 3 (`docs/DETAILS.md:383`).
- **Depth and confidence**: `--spec N` and `--spec-min-p P`; *"Setup's default is `--spec 4 --spec-min-p 0.5`"*
  (`docs/DETAILS.md:381`). `--spec-min-p` is *"How sure the draft layer must be to add another guess to a check"*
  (`docs/DETAILS.md:145`).
- **The MTP layer is optional and costs VRAM**: *"The server also runs without it … every token is still verified, and
  the draft layer's ~0.7-1 GiB of VRAM goes to the expert cache"* (`docs/DETAILS.md:380`). On Intel it is not even
  built: *"The draft layer (MTP) is not built for Intel cards"* (`docs/INTEL_ARC.md:28-29`).
- **Draft vocabulary subset**: `--draft-vocab en|cyrillic|fr|latin`, *"`en` keeps the English and code subset (~110 MiB
  less VRAM, English 1-2% faster, CJK gets almost no drafts)"* (`docs/DETAILS.md:382`); tables in
  `terminal/data/draft_vocab{,_en,_fr,_cyrillic}.bin`.
- **Draft head down-quantization**: `--mtp-q4` requantizes the draft head's subset rows to Q4_0, printing the bytes
  read per step: *"draft head over N tokens as Q4_0 (X MiB read per step, was Y)"*
  (`engine/sycl/src/core/mtp.cpp:270-297`). Guarded against a silently-empty drafter — *"measured: 0 drafts, decode
  73.8 → 35.3 tok/s"* (`:276-279`).
- **Draft attention window**: `--mtp-window 8192` — *"the draft layer attends to the last 8,192 cells"*
  (`docs/DETAILS.md` via `docs/STRIX_HALO.md:160-161`); on Strix Halo `--mtp-window 8192` gave 64K output +3.5% and
  32K -0.4% (`docs/STRIX_HALO.md:242`).
- **Acceptance observability**: `GET /metrics` exposes `drafts_offered`, `drafts_accepted`
  (`docs/DETAILS.md:355-360`). Reported acceptance: 66% prose / 90% code on Arc IQ3_S (`docs/INTEL.md:219-220`);
  81-87% with the MTP layer (`docs/INTEL.md:430`); 88.7% on a Q2_0 draft layer over 225 test requests
  (`docs/NVFP4.md:204`).

### 4.2 Code: how the chain is actually run (`engine/sycl/src/core/mtp.cpp`)

Round entry `MtpDrafter::draft(...)` at `:1637`. Two mutually exclusive execution strategies:

**(a) Whole chain submitted up front, host polls** (`:1701-1752`):
```
for j in 1..max_steps-1:  cs_->ext_oneapi_graph(step_exec_[j])     // :1702-1704 — no wait between steps
cs_->ext_oneapi_empty(); prefetch_ple(tokens[a]);                  // :1719-1720
for j in 0..max_steps-2:  while (h_out_[j] < 0) _mm_pause();       // :1721-1733 — doorbell poll only
                          drafts[j] = h_out_[j]; prefetch_ple(drafts[j]);
cs_->wait();                                                       // :1734
```
The token handoff between steps is **device-to-device**: step `j+1`'s graph reads step `j`'s pick from device memory,
and the host only consumes results. `h_out_`/`h_prob_` come from `mapped()` → `host_malloc_polled` → `zeMemAllocHost`
with `BIAS_UNCACHED` (`:116-127`), which is why the spin can see the store at all.

**(b) Confidence-gated chaining** (`:1753-1833`): submit `round_exec_[T]` (`:1754`), wait for draft 0, then
`for (j = 1; j < max_steps && pj >= min_p; ++j) { submit step_exec_[j]; wait_step(j); ... }` (`:1793-1817`) — stops
spending GPU time as soon as the drafter is unsure, at the cost of a host round-trip per step.

Both are built from **CUDA-graph captures taken at first use**: `capture_round(T, cp, err)` (`:1645`) then
`capture_step(j, cp, err)` for each `j` (`:1646-1648`), replayed as `ext_oneapi_graph` (dpct's graph extension).
`max_steps = min(max_t_ - 1, max_drafts_)` (`:1646`).

Other structure that matters:
- **Rejected rows' K/V are not caught up**: `T = a + 1` unless `STRATA_MTP_CATCHUP_ALL=1` (`:93-100`, `:1641-1643`) —
  the round only runs for cells up to the accepted row, *"the K/V of the rejected rows is not caught up - no later read
  reaches a cell past the one being drafted and the next round writes it first"*.
- **KV streaming for the drafter** is a ring of `window + 4*max_t + 64` cells over a host copy (`:444-453`), and
  `--kv k8v4` never applies to the drafter (`:449-451`).
- **Overlap with verify**: `chain_live_` forbids a second in-flight chain (`:1640`), and on Intel a commit graph is left
  running while the drafter runs on its own queue (`docs/INTEL.md:432`, 46.5 → 49.8 tok/s). Two-card pipelining goes
  further: card 1 starts the next window while card 2 checks this one (`--pipeline-windows 2`,
  `docs/INTEL_ARC.md:137-143`, +17-23%).
- **Metal fuses the per-step host work away**: `engine/metal/src/engine/msl/mtp_chain.metal:1-8` — *"the two steps that
  keep a whole MTP draft round on the GPU (one command buffer for every draft step … `mtp_pick` draft j: the pick's
  token id … and its probability, the softmax of the draft head's logits at the pick (the host computed it after every
  step before); `mtp_embed_F` the next step's input: the picked token's embedding row dequantized … (the host
  dequantized it before)"*. Kernels at `:22-41` (`mtp_pick`) and `:63-86` (`mtp_embed`, per-format `Dec<F>` dequant).

### 4.3 Can drafting run on CPU while the GPU verifies?

**No — not in this engine, and the docs are unambiguous about the split.** The drafter is a GPU object bound to the
device (`mtp.cpp:309-311`: *"a layer split's last stage on another GPU: the drafter lives there"*); the CPU's role is
the *expert* pool, not drafting (`HOW_IT_WORKS.md:27-28`). The only CPU-side draft work is the host's pick/softmax/
embedding step, and Metal's `mtp_chain.metal:1-8` exists specifically to move *that* onto the GPU. So Strata's answer
to our question is the inverse of what we hoped: **they moved the last host work in the draft chain onto the device**,
because the host step was costing them measurable window time. The one measured figure for the draft's share of a
window is `docs/QUANTS.md:106-113`: of a 34.3 ms decode window (2.0 tokens/window), *"plus the MTP draft (5.8 ms
measured)"* — 17% of the window on a 644 GB/s card, with the whole window running *"at ~35-40% of that bound"*.

---

## 5. Published APU / small-pool benchmark numbers

### 5.1 What exists at all

| Pool | Where | Data quality |
|---|---|---|
| Strix Halo (gfx1151, Radeon 8060S, 128 GB unified) | `docs/STRIX_HALO.md:182-244`, `docs/COMMUNITY_BENCHMARKS.md:49-52` | **good** — 3-6 interleaved pairs, one fresh process per run, identical token ids |
| Strix Halo on Windows | `docs/STRIX_HALO_WINDOWS.md` | referenced, +59-72% prompt over 0.2.7 (`STRIX_HALO.md:146-152`) |
| Radeon 780M (gfx1103, 12 CU, DDR5) | `docs/RADEON_780M.md:66-120`, `COMMUNITY_BENCHMARKS.md:74-75` | **medium** — one machine, one round per cell |
| Radeon 890M/880M (gfx1150) | mentioned, `igpu` defaults in `settings-register.json:14326` | no published table |
| Intel Arc A750 / A770 / B-series | `docs/INTEL.md:178-298`, `INTEL_ARC.md:162-172` | discrete only |
| **Meteor Lake** | — | **none** |
| **Lunar Lake** | `docs/INTEL_INT8_EXPERTS.md:43` | **no measurements** (cited only as Xe2 sDPS-capable silicon) |
| **Panther Lake / integrated Arc** | `docs/INTEL_ARC.md:177-179` | **untested, no memory plan** |

### 5.2 Strix Halo — measured (gfx1151, Linux, 128 GB, engine 0.1.40 + their fast config, `--spec 4 --mtp
--lookup-chain 3 --mtp-q4 all --prefill 16384`, int8 KV, all experts in unified memory; medians of 3,
`docs/STRIX_HALO.md:197-213`)

| Model / quant | Ctx | Prompt tok/s | Decode tok/s | Expert blob | Weights touched |
|---|---|---:|---:|---:|---|
| UD-IQ4_XS | 8K | 1,293 | **53.8** | 2.42 MB | 55.4 GiB experts + 4.8 GB dense |
| UD-IQ4_XS | 128K | 1,320 | **51.4** | 2.42 MB | same |
| UD-Q4_K_XL | 8K | 1,169 | **52.6** | 3.13 MB | 71.7 GiB experts + 4.8 GB dense |
| UD-Q4_K_XL | 128K | 1,248 | **43.3** | 3.13 MB | same |
| IQ3_S | 8K | 1,242 | **59.7** | 1.97 MB | 46 GB experts + ~4.8 GB dense |
| IQ3_S | 128K | 1,312 | **42.8** | 1.97 MB | same |
| IQ3_XXS | 8K | 1,238 | **51.5** | 1.75 MB | 43 GB experts |
| IQ3_XXS | 128K | 1,252 | **50.9** | 1.75 MB | same |

Expert blob sizes: IQ3_S 1.97 MB (`docs/INTEL.md:193`), IQ4_XS/Q4_K_XL derived from `docs/QUANTS.md:18-19`
(55.4 GiB and 71.7 GiB over 24,576 experts), IQ3_XXS from `docs/MODELS.md:69` (43 GB / 24,576).

### 5.2b Strix Halo on Windows (Qwen IQ3_S, 0.2.7 build, `docs/STRIX_HALO_WINDOWS.md:21-28`)

| Test | Result |
|---|---|
| Short chat, decode | **44.0 tok/s** (range 40.6-59.5; code-like answers faster than prose) |
| 4K / 16K / 32K prompt | **755 / 821 / 792 tok/s** (TTFT 5.4 / 19.6 / 40.6 s) |
| Coding-agent session, 6 turns | answers 47 tok/s, 69% of the prompt from the conversation cache |

Effective bandwidth at 44.0 tok/s on IQ3_S with the §5.5 formula: 2.95/2.45 GB per token → **130 / 108 GB/s**, i.e.
51 / 42% of a 256 GB/s pool — noticeably below the Linux fast-config rows (69%), which is what `STRATA_SH_STREAM` off
and the missing opt-in set cost on that box (`STRIX_HALO_WINDOWS.md:89-104`: decode 43.6 → 46.6 tok/s just from turning
the shared-expert stream off).

**The most important number in that doc for us:** the matrix-core prompt set on the APU gave
**4K 688.8 → 1,192.8 tok/s, 8K 787.0 → 1,283.5, 32K 775.4 → 1,245.1, 64K 748.5 → 1,188.9 (×1.59 to ×1.72)**
(`STRIX_HALO_WINDOWS.md:69-75`), and on the 780M 1K 98.5 → 157.3 (×1.60) and 4K 115.1 → 233.3 (×2.03)
(`docs/RADEON_780M.md:88-89`). **Matrix cores pay on iGPUs for prefill** — see §8.6 for why that does not contradict
our DPAS result.

### 5.3 70 W laptop Strix Halo (HP ZBook Ultra G1a, 128 GB, ROCm 7.14.1, UD-IQ4_XS, `--spec 5 --mtp-window 8192`,
medians of 4-6 fresh processes, `docs/COMMUNITY_BENCHMARKS.md:49-52`)

| Ctx | Prompt tok/s | Decode tok/s |
|---|---:|---:|
| 8,192 | 969 | **59.2** |
| 32,768 | 1,067 | **57.8** |
| 131,072 | 1,031 | **42.0** |

*"Prompt is power-limited (70 W package limit)"* — the single most relevant analogue to our 28 W-class MTL part:
**decode survives a power cap; prompt does not.**

### 5.4 Radeon 780M (gfx1103, Ryzen 7 8745HS, 60 GB DDR5, Q2_0 + MTP, 32K, int8 KV)

| Config | Decode tok/s | Prompt tok/s | Source |
|---|---:|---:|---|
| `--spec 2` (best window on this chip) | **15.65** | — | `docs/RADEON_780M.md:111-113` |
| `--spec 4` | 14.18 | — | same |
| `--spec 6` | 11.16 | — | same |
| with the 17 exact switches | 17.17 | 211-212 @ 4K | `docs/RADEON_780M.md:71-72` |
| code / story, iGPU prompt set | 21.99 / 15.05 | 233 @ 4K | `docs/RADEON_780M.md:88-89` |
| community port, 4K | 14.3 | 277 (WMMA opt-ins) | `docs/COMMUNITY_BENCHMARKS.md:74-75` |

Two findings here that price our own options:
- *"`--spec 2` is the best MTP window on this chip: 15.65 tok/s decode … against 14.18 at `--spec 4` and 11.16 at
  `--spec 6`"* — **on a bandwidth-starved iGPU, deeper speculation costs more than it returns.** Depth is a
  bandwidth/latency trade, not a free win.
- *"gfx1103 was outside the `dp4a` guard …, so every quantized dot product ran a four-step byte loop instead of
  `v_dot4_i32_iu8`. gfx1103 now compiles the native instruction (decode +55% measured: 8.70 → 13.44 tok/s)"*
  (`docs/RADEON_780M.md:12-14`). A one-line dispatch guard was worth 55% — the same class of win as our DP4a-vs-DPAS
  finding.

### 5.5 Derived effective bandwidth

Model: one decode window touches the **dense side once** (`docs/QUANTS.md:28`: *"The dense side the GPU reads once per
verify window is about 4.8 GB in every file"*) plus **10 experts × 48 layers × expert blob** per token in the window,
and the window carries `A` accepted tokens (`A = 2.4-3.2`, `docs/HOW_IT_WORKS.md:43`). So

```
bytes/token = D/A + 480·E          effective GB/s = tok/s × bytes/token
D = 4.8 GB, E = expert blob bytes
```

The formula and `A` are **my derivation, not Strata's claim** — Strata publishes tok/s and the window budget, not a
per-APU GB/s figure. Two values of `A` bracket the uncertainty.

| System | Pool (theoretical) | Model / quant | Decode tok/s | bytes/token @A=2.4 | @A=3.2 | **Eff. GB/s @2.4 / @3.2** | % of pool |
|---|---|---|---:|---:|---:|---|---:|
| Strix Halo 395, 8K | 256 GB/s (LPDDR5X-8000, 256-bit) | IQ3_S | 59.7 | 2.95 GB | 2.45 GB | **176 / 146** | 69 / 57% |
| Strix Halo 395, 8K | 256 | UD-Q4_K_XL | 52.6 | 3.50 | 3.00 | **184 / 158** | 72 / 62% |
| Strix Halo 395, 8K | 256 | UD-IQ4_XS | 53.8 | 3.12 | 2.62 | **168 / 141** | 66 / 55% |
| Strix Halo 395, 128K | 256 | IQ3_S | 42.8 | 2.95 | 2.45 | **126 / 105** | 49 / 41% |
| Strix Halo 70 W laptop, 8K | 256 | UD-IQ4_XS | 59.2 | 3.12 | 2.62 | **185 / 155** | 72 / 61% |
| Strix Halo 70 W laptop, 128K | 256 | UD-IQ4_XS | 42.0 | 3.12 | 2.62 | **131 / 110** | 51 / 43% |
| Radeon 780M, 32K | ~89.6 GB/s (DDR5-5600, 2 ch)* | Q2_0 | 17.17 | 2.58 | 2.16 | **44 / 37** | 49 / 41% |
| Radeon 780M, 32K | ~89.6 | Q2_0 (code) | 21.99 | 2.58 | 2.16 | **57 / 47** | 63 / 53% |
| Radeon 780M, 4K | ~89.6 | Q2_0 | 14.3 | 2.58 | 2.16 | **37 / 31** | 41 / 35% |

\* the 780M's memory configuration is not stated; dual-channel DDR5-5600 is the assumption, so treat its percentages as
±20%.

Calibration check: the one place Strata states a decode-window budget directly is R9700 (discrete, 644 GB/s):
34.3 ms window, 2.0 tokens, per-component bytes and times, *"plus the MTP draft (5.8 ms measured)"*, and *"Decode runs
at ~35-40% of that bound for every file"* (`docs/QUANTS.md:106-113`). Feeding the same formula: 2.7 + 2.84 + 0.67 +
0.5 + 1.05 GB ≈ 7.76 GB per 2-token window → 3.88 GB/token × (2.0/0.0343) = 226 GB/s = **35% of 644 GB/s**. The formula
reproduces their stated efficiency exactly, which is why I trust it for the APU rows above.

**What this tells us for MTL.** Our iGPU block-read peak is 89.9 GB/s and we run Q4 at 3.05 tok/s (66.5 GB/s measured).
If our engine were at Strata's efficiency (35-70% of pool) it would be at 31-63 GB/s of *effective* traffic — i.e. we
are already operating in the right band, which is the point: **the 780M row (49% of pool, 17.17 tok/s on a 2-bit quant
with MTP) is the honest peer for our 3.05 tok/s raw / 7.95 tok/s MTP numbers, and Strata's edge comes from the spec
loop and the pack, not from a bandwidth trick we are missing.**

---

## 6. `tools/research/` — internal experiments worth borrowing

Small, curated, dated; three areas:

1. **`tools/research/tools/ab-harnesses-2026-10/`** (`README.md:1-10`):
   - `h2d_bench.cu` (10 Oct) — *"What a stream of expert-sized host-to-device copies costs end to end (issue, copy,
     host function, sync)."* Directly reusable shape for our fabric measurements; it times the **four components
     separately**, which is what distinguishes a driver cost from a bandwidth cost.
   - `quant_bound.py` (12 Oct) — *"Decode bounds of a quant from its real per-layer formats (the output of
     `tools/quant_formats.py --json`)."* A **predictive** bound calculator per quant from the GGUF headers alone. This
     is the tool behind the QUANTS.md window-budget table and the most immediately useful thing to borrow: it turns
     "which quant fits my bytes/token budget" into arithmetic before downloading 90 GB.
   - `idle_gpu_ab.py` (20 Oct) — *"how idle are the GPUs, and does an existing switch close the gap? Interleaved A/B of
     whole engine runs with a per-card utilisation sampler."*
   - `arc/ab.py`, `arc/b.py`, `arc/kl.py` (30 Oct) — interleaved A/B of engine arms on the Arc box, plus **KL between
     first-token logit dumps**.
2. **`tools/research/stage-reserve-proposal/`** (`README.md:1-26`): an unused proposal to grow a later layer-split
   stage's reserve with the number of window sizes — `proposed_window_mib = kWindowMib + (sizes-1)·6 MiB + 48 MiB`,
   *"For `--spec 4` (windows of 1..6 tokens) the term is 174 MiB instead of 96"*. The transferable idea for us is that
   **graph capture per window size has its own VRAM cost** (6 MiB per captured graph, once 42 MiB), which must be in
   the budget before the first allocation. It also documents *why* it is not shipped: no model gate, no mixed-pair A/B.
3. **`tools/research/tools/live-probes-2026-10/`** and **`dgx-spark-2026-10/`**: HTTP-level probes (stream ownership,
   cancellation/parking, session pressure, batch-group cache) and a whole DGX Spark qualification suite
   (`dgx_quality24.py`, `dgx_needle.py`, `dgx_concurrent.py`, `dgx_kvresident.py`). Relevant only as a **test-surface
   template**, not as kernels.

Adjacent, in-tree experiment worth reading: `engine/src/kernels/mmvq_il_parity.cpp:1-21` — an interleaved q8_1 copy
layout plus P1/P2 prefetch variants, each bit-checked against `native_mmvq`, with a `--gap MB` mode that *"reads MB
megabytes of OTHER weights … before every timed call"* to model cache pressure. The tuning result is a per-cell table
of *"the rows count (1/2/4) that takes at least 3% off native_mmvq's time, else 0"* (`:21`) — i.e. **row-reuse is
applied as a per-shape lookup table, not a single global choice** (see also `native_mmvq.dp.cpp:1656-1660` for the
shape-guarded GRF dispatch).

---

## 7. Quantization ladder and quality checks

### 7.1 The ladder

- **Base packs** (ISTA-DASLab GSQ-RCO, `README.md:35-44`): Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Coder IQ1_M. Footprints
  (RAM+VRAM) 37.6 / 39.2 / 47.0 / 54.8 GB (`docs/MODELS.md:38-43`); experts 34 / 35.5 / 43 / 50 GB
  (`docs/MODELS.md:65-71`).
- **Unsloth K-quants** (`docs/QUANTS.md:16-22`): UD-IQ4_XS 93.7 GB, UD-Q4_K_XL 111.3, UD-Q5_K_XL 158.3,
  UD-Q6_K_XL 169.2, Q8_0 188.2. Routed experts per token (10 × 48 layers): 1.16 / 1.50 / 1.92 / 2.13 / 2.51 GB.
  Everything else (attention, SSM, hyper-connections, shared expert, PLE k/v, embedding, head) is Q8_0; the PLE table
  is IQ4_NL (26.8 GiB) or Q8_0 (50.7 GiB) (`:24-28`).
- **NVFP4** — three flavours (`docs/NVFP4.md:18-22`): Strata's GPTQ pack, NVIDIA's ModelOpt pack, and NVIDIA's
  FP4-QAD retrained model. Format: *"64-value blocks of E2M1 codes with four UE4M3 sub-block scales, plus one FP32
  `weight_scale_2` per expert and projection"* (`:6-7`). Expert blob **2.76 MB** vs 1.51 MB for IQ2_XS (`:66`) —
  twice the bytes, half the cache slots, hence PCIe share 0.25 instead of 0.55.
  **Setup does not offer NVFP4 on AMD or Intel** (`:181-182`).
  A concrete numerical trap worth remembering: folding `s_down` into `up` looks algebraically free but pushes the
  hidden to ~1e-5, where `amax/127` lands in FP16 subnormals and one expert's output went from 1.1% to **2-12%**
  error (`:76-83`).
- **Gyro-S / Gyro-M** trellis-coded experts: 58.5 / 92.0 GB download, 27.6 / 35 GB experts (`docs/MODELS.md:162-163`).
- **Prompt-path precision ladder for NVFP4** (`docs/NVFP4.md:87-114`): `w4a8` (default, KL 0.0018, 2,503 tok/s) vs
  `w4a4` (KL 0.0080, 2,904 tok/s) vs `fp16` (reference, 1,905 tok/s) — *"`w4a8` is the default. It is 4.5x closer to
  the reference than `w4a4`, at 86% of its speed."* Note the direction: on Ampere/Ada, **w4a4 is 210% faster but 4x
  less accurate**, and the FP16 path is the *slowest* of the three.

### 7.2 Quality checks — the actual yardstick

`terminal/tools/quant_quality.py` + `terminal/data/quant_quality/v1` (code, docs, English, German/French/Japanese/
Chinese, an agent transcript; 5 × 1,000 tokens, `set.ids.json`). Method: *"Every token is read through the decode
windows (teacher-forced) with the experts on the CPU (one VRAM slot), no expert adaptation, no PCIe share, no route
tail skip, FP16 K/V; per position the engine writes the log-probability of the text's next token and its 64 most likely
tokens (`STRATA_LOGPOS`, `STRATA_LOGPOS_TOPK=64`). Top-1 agreement … KL: KL(Q8_0 ‖ quant)"*
(`docs/QUANTS.md:56-61`). Sorted, 4,995 positions (`:71-79`):

| Quant | Top-1 = Q8_0 | Mean KL | KL p99 | PPL (Q8_0 4.052) |
|---|---:|---:|---:|---:|
| UD-Q6_K_XL | 94.61% | 0.033 | 0.494 | 4.020 |
| UD-Q5_K_XL | 94.55% | 0.038 | 0.636 | 4.024 |
| UD-Q4_K_XL | 93.09% | 0.056 | 0.829 | 4.068 |
| UD-IQ4_XS | 88.91% | 0.101 | 1.596 | 4.120 |
| IQ3_S | 87.69% | 0.144 | 1.975 | 4.097 |
| IQ3_XXS | 84.46% | 0.202 | 2.433 | 4.119 |
| Q2_0 | 80.60% | 0.315 | 3.643 | 4.475 |

**The comparison we should steal:** any new kernel is gated against this yardstick and the *noise floor* is measured
too — e.g. int8 expert path *"KL mean 0.0002 nats …, top-1 99.7%, perplexity ratio 1.0004"* against *"a chunk-size
change (`--prefill 8192` against 4096) moves the first-token distribution by 0.01-0.03 nats"*
(`docs/INTEL_INT8_EXPERTS.md:84-88`). That is the right framing: **express every kernel change's error relative to a
benign configuration change**, not relative to "exact".

Second gate: **bit-identity + interleaved A/B**. e.g. `docs/INTEL.md:499-505` reports each switch's effect against the
same run with identical output tokens, and `docs/INTEL_ARC.md:137-143` states *"The answers are those of a serial
`--spec 2` run (100 of 100 agent requests); a serial `--spec 4` run differs in a rare near-tie (3 of 100)"*.

### 7.3 Benchmark harnesses

- `tools/bench.sh:1-13` and `tools/BENCH.bat:1-19`: 5-line wrappers over
  `terminal/tools/strata_bench.py`, which *"starts the model with the settings you already have … sends a fixed suite
  through the HTTP API, three runs each, and reports medians with ranges"* (`docs/COMMUNITY_BENCHMARKS.md:103-106`).
  Tests: chat, 4K and 16K fresh prompts, an agent session (4 tools + 5 turns), `--clients N`, `--long` 32K
  (`:108-114`). Speeds are the **engine's own counters** (`prompt_per_second`, `predicted_per_second`), TTFT is
  client-side, every prompt carries a unique tag so the cache cannot help (`:116-118`).
- The **standard** protocol is `strata bench run` (fixed prompts, thinking on, twelve sanity checks, 2-4 min), writing
  `results.json` + `outputs.jsonl` + settings + engine log, with `complete|partial|nonconforming` status
  (`docs/COMMUNITY_BENCHMARKS.md:6-10`, `:77-94`).
- A/B of two builds: `terminal/tools/ab_engine.py`; recall: `terminal/tools/needle_bench.py`
  (`docs/COMMUNITY_BENCHMARKS.md:141-142`, `:214-226`).
- Their own warning about the A/B harness's sensitivity: *"`terminal/tools/ab_engine.py`'s three short chats are below
  this machine's noise floor"* (`docs/COMMUNITY_BENCHMARKS.md:43-44`).

### 7.4 Test surface (for reference)

`engine/tests/` is organized by concern and is the model I'd copy for our kernels: `tests/quant/`
(`quant_parity.cpp`, `quant_selftest.cpp`, `type_geometry_oracle.cpp`, `nvfp4_decode_model_test.cpp`),
`tests/program/` (the placement/planning rules: `split_rules_test.cpp`, `stage_reserve_test.cpp`,
`prefill_plan_test.cpp`, `adaptive_range_test.cpp`, `batch_read_test.cpp`, `q36_context_fit_test.cpp`, …),
`tests/prefill/` (30+ rule tests, one per heuristic: `stream_floor_rule_test.cpp`, `fit_out_rule_test.cpp`,
`loan_rules_test.cpp`, `own_only_switches_test.cpp`, …), `tests/platform/` (`numa_balancing_test.cpp`,
`io_priority_test.cpp`, `host_limits_test.cpp`, `direct_file_async.cpp`), `tests/cuda/`
(`prefill_decode_golden.cu`, `native_moe_gated_test.cu`, `hc_q8_stream_test.cu`), plus the Intel parity suite
`engine/sycl/src/kernels/*_parity.cpp` (25 tests; *"19 of the 22 kernel parity tests passed byte for byte or within
tolerance"* on the B70, `docs/INTEL.md:510-512`) driven by `-DSTRATA_SYCL_PARITY=ON` / `ctest --test-dir build-sycl`
(`docs/INTEL_ARC.md:111`). Notable: `tools/test_check_no_mmq.py`, `tools/test_check_isa_guard.py`,
`tools/test_check_cuda_spills.py`, `tools/test_prefetch_gate.py` — lint-style guards that keep a kernel from silently
regressing (spill counting, ISA guards, "no MMQ" invariant). Cheap and high-leverage for us.

---

## 7b. Second-pass docs sweep — `A770_SPEED_SPEC`, `BATCHING`, `KVA`, and the DMA/KV/rotation notes

These docs were added to the brief after the first pass. Two of them **change my ranking**, so read this section before
§8.

### 7b.1 `docs/A770_SPEED_SPEC.md` — a 760-line engineering log that overturns the naive reading

This is not a spec, it is a failed-campaign autopsy on an A770 (16 GB, `xe`, PCIe 3.0 **x4 = 1.7-1.8 GB/s measured**
`:42-44`), and it is the closest thing in the repo to our situation (small Intel card, most experts off-VRAM, MTP on).

- **The bottleneck was not kernels, and not bandwidth.** `gputop` sampled the A770's compute engine during a 700-token
  decode at *"0% 0% 4.1% 14.8% 41.4% 2.8% 0% 8.5% 65.0% 84.3% … mostly 0-20%, spiking occasionally, mean over the run
  ~20%. The GPU is **not saturated during decode**"* (`:687-699`). The stage table then showed why: `waitA`/`waitB` are
  *"the GPU **spinning for the CPU expert-pool flags**"* and together with the result copy they are *"~87% of the
  window"* (`:749-752`).
- **How they got that table is itself the transferable part.** Both in-engine profilers were broken
  (`STRATA_VERIFY_PROFILE` / `STRATA_VERIFY_EAGER` → `verify: layer 0 rang but its payload never arrived whole`,
  HTTP 400, `:715-731`), so they added an opt-in `STRATA_DOORBELL_NOCHECK` to make `doorbell_wait_payload` trust the
  ring and finally got a per-stage split (`:735-752`). Lesson: **a doorbell/payload protocol needs a bypass switch, or
  you cannot measure it.** They explicitly label the numbers inflated and say the next step is an *honest* profiler.
- **Every lever they exhausted, with numbers** — this is the single most valuable thing in the repo for us, because it
  is a list of things *not* to spend time on:
  - spec window width: wider is worse (`:260-264`);
  - `--spec-min-p` sweep: **0.0 → 10.4/10.5/10.4 tok/s (52% acceptance); 0.5 → 12.2/12.0/11.7 (68%); 0.7 →
    11.8/11.4/11.6 (93%)** (`:243-248`). Higher thresholds raise acceptance but offer fewer drafts; 0.5 best. **This is
    direct evidence that acceptance rate is a bad objective on its own** — 93% acceptance is *slower* than 68%.
  - `--spec-split` (overlap drafter with verify): negative (`:254-258`, quoted in §8.3);
  - `--adapt-swaps 0`: worse, 11.2/10.9/10.5 vs 12.0 (`:250-252`);
  - `--kv q4_0`: neutral, 12.0/12.3/12.6 vs int8 — *"KV VRAM was not the binding constraint"* (`:266-269`);
  - shrinking the VRAM expert cache: 4,717 slots → 10.9, 3,200 → 10.8, 2,400 → 9.6 tok/s, and GPU-reach wait *grew*
    (`:579-588`);
  - `--pool-workers` 6 vs 7: wash (`:682-685`);
  - `STRATA_EXPERT_LANES` 8 vs 16: null, bit-identical (`:596-608`);
  - `STRATA_MMVF_ROWS=1` (4-row BF16 GEMV): **1.4-6x slower** (`:509-512`);
  - **`GRF=256` on BF16 projections: 10-58% slower** (`:528-530`) — the opposite sign from the Q6_K matvec result in
    `native_mmvq.dp.cpp:1635-1639`. Large GRF is **family-specific**, not a general lever.
- **Standalone kernel bandwidth on this card** (useful as an iGPU-class datapoint): hyper-connection read
  (`fused_gr_read_multi`) *"180.4 / 179.0 / 203.7 / 218.0 / 241.7 / 259.8 us for 1..6 tokens (**73 GB/s at 1 token down
  to 51 GB/s at 6**)"* (`:661-664`); grouped expert kernel 0.33-0.38 ms/layer, ~24 ms/window, **not** the bottleneck
  (`:635-651`). Note the shape: a small tensor read at 1 token runs at 73 GB/s and **degrades** as rows are added —
  the small-kernel regime we live in.
- **Dual-GPU layer split is blocked by single-context graph capture** (`:277-306`): `UR_RESULT_ERROR_DEVICE_LOST` then
  *"Cannot submit to a queue with a dependency from a graph that is associated with a different context"*. Relevant to
  us only as a warning that graph capture is per-context.
- **`--pcie-frac` is inert in resident-experts mode** (`:232-242`): the counter never moved; the only PCIe expert path
  is the pinned mirror. Don't wire a hybrid CPU+PCIe split expecting the flag to do it.
- **Their measurement protocol is worth copying wholesale** (`:214-229`, `:403-438`): rank changes by **ms per round on
  a controlled token stream with identical round/acceptance counts**, arms interleaved in one window, std dev < 0.02 ms,
  *"one-side probes and single-kernel rankings mislead (our own router microbench 3.3x gave no end-to-end gain)"*.

### 7b.2 `docs/BATCHING.md` — batching economics on a memory-bound engine, and per-row launch cost

- **Batching does not add throughput when the engine is memory-bound; it buys waiting time.** One RTX 5070 12 GB, Q2_0:
  4 concurrent requests one-at-a-time = 70.7 tok/s total, with `parallel 4` = 63.1 tok/s total (*"**-11%**"*), while
  the fourth request's wait fell from 11.2 s to 1.8 s (`:653-671`). Explanation given: *"a batch window over 4
  conversations reads 24 CPU experts per layer against ~8 for one, so it costs about what the 4 tokens cost one after
  the other (54 ms per 4-row window, ~20 ms per 1-row window)"* (`:667-671`).
- **Per-row launch overhead is ~2.4 ms, cut to 1.7-1.9 ms by merging launches** — see §8.9. The mechanism is stated:
  block scores/top-k/attention and the GDN recurrence ran *one launch per row* (`:691-693`).
- **Draft depth in a batch window**: `STRATA_PIPED_MTP=1` beat `=2` beat `=3`, with *"about 55% of the checked proposals
  were kept at =2 and about 67% at =1. The acceptance falls as the window grows, which is why =1 or =2 is the choice
  and =3 is not"* (`:509-524`). Same lesson as the 780M's `--spec 2`: **deeper windows lose on acceptance *and* on
  bandwidth**.
- **A bit-exactness gotcha worth knowing**: `STRATA_IQ_MT_MIN=1` is required for exact comparisons because *"by default
  an expert's rows round differently alone than in a group, so the output depends on how many rows of a window share an
  expert"* (`:613-618`). If we ever compare our engine's output across different draft depths, this is the class of bug
  that will look like nondeterminism.

### 7b.3 `docs/KVA.md` — approximate the prompt read for late layers (prompt-only, 1.3-1.7x)

KVA reads a long prompt exactly through early layers, then for late layers builds **only their recurrent state** — GDN
recurrence and convolution history, QSA K/V pages, indexer keys — and skips the attention output, output projection,
the whole MoE half and its expert transfers, *"because nothing reads their results"* (`:1-8`). The last **2,048 tokens
are read exactly** and decode then runs the ordinary full path (`:12-14`, `:62`).

- **Key insight (generalizable beyond this model):** a layer's K/V depends on the hidden state *entering* it, not on
  its own output — so for a cold long prompt you can populate every late layer's K/V pages from the boundary residual
  and skip the expensive half of each late layer.
- Measured (P100, split 24): 1.31x at 5.6K, 1.54x at 14.6K, 1.69x at 30K prompt tokens, decode unchanged; first-token
  KL 0.018-0.020 nats against a 0.0013-0.0032 control floor (`:132-153`). Shipped opt-in is **split 36**: +11% at 14K,
  +18% at 30K prompt on an RTX 5070 with agents 6/10 (exact 4 and 7) and extraction 585/606 facts (exact 585)
  (`:28-30`).
- **The cost lands on decode, not prompt**: on an R9700, prompt 2411 → 3086 tok/s (+28%) but *"decode 94.7 to 84.3 tok/s
  (−11%, tokens per round 4.08 to 3.33)"* — *"the draft layer's guesses are accepted less often once most of the context
  was approximated"* (`:33-38`). On a slow GPU the difference is within noise.
- Split 24 (more layers approximated) *silently dropped records* on extraction tasks (4/6 vs 6/6, 249/270 facts) and
  invented a row on a 30K answer (`:21-27`). **Split choice is a quality decision, and the failure mode is silent
  content loss, not visible garbage.**
- **Not available on SYCL** (`:47`, `:102-104`): *"the SYCL engine has its own prompt path without KVA"*. We would be
  porting from scratch, with the quality gate as the real cost.

### 7b.4 `docs/BATCHED_DMA.md` — a clean negative result: do not chase copy submission

`STRATA_DMA_BATCH=1` batches a verifier group's expert uploads through `cudaMemcpyBatchAsync` (CUDA 13+). Result:
**submission latency −82.7% (0.0936 → 0.0162 ms), total transfer latency −1.3% (2.5589 → 2.5248 ms)**, submitting-thread
cycles −84.9%, and **no model speedup**: one coding request 30.3 tok/s individual vs 30.0 batched, 29.7/31.4/30.3/30.2
across an A-B-B-A (`:45-93`). The 63-case correctness matrix is thorough and the doc is explicit that transfer
correctness is not text determinism. **Conclusion for us: H2D submission overhead is not worth engineering time**;
bandwidth and engine selection are.

### 7b.5 `docs/KV_PREFETCH.md` — overlap streamed KV upload with prefill compute

Opt-in `STRATA_KV_PREFETCH=1` starts the next attention layer's KV upload on a separate stream while the current layer
computes, reusing the existing staging pool (no extra buffer) (`:9-11`). Measured ABBA on 2x RTX 3080 20 GB:
prefill 16.989 → 16.082 s at 32K (−5.3%) and 32.604 → 31.259 s at 64K (−4.1%); **no 8K improvement and no decode
change**; ~939 ms / 1,553 ms of serial KV-stage wait moved off the critical path (`:34-47`). Small, safe, prompt-only —
relevant if we ever stream KV from RAM on long contexts.

### 7b.6 `docs/EXCHANGE_ROTATION.md` — pointer rotation instead of a host memcpy

When the resident RAM complement swaps an expert with the GPU cache, the old code copied the evicted expert's bytes into
the promoted expert's old slot. `STRATA_EXCHANGE_ROTATE=1` instead **makes the temporary buffer resident and the old slot
the next temporary buffer** — zero host copy, ownership flip only (`:1-8`). Requires the resident complement to be fully
pinned and GPU-mapped, and keeps spare buffers **per block size** because a pack may have seven to nine distinct expert
sizes (`:51-67`). Measured: +8.1% decode on one native token-identical pair (104.90 → 113.44 tok/s, 1,024 identical
output IDs) and +13-15% on 23 agent turns (`:93-111`, `:172-178`), because *"the copy path moved about 50 MiB of RAM per
window on the decode thread"* (`:107-111`). **The generalizable idea is budget-neutral and exact: swap ownership, never
copy.** Their verification (identical output token IDs + identical `--window-hashes` for all 434 windows, `:99-103`) is
the standard I would hold our own swap paths to.

### 7b.7 `docs/DISJOINT_EXPERT_CACHE.md` — multi-GPU only

Keeps helper-GPU-owned experts out of the primary cache's promotion candidates, live-checked via
`RemoteExperts::holds` rather than a stale mask (`:9-19`). 15.5% generation gain on two 3080s (`:37-44`). **Not
applicable to a single iGPU**; noted only so it is not re-investigated.



## 8. Top-10 transferable techniques for inferzero's MTL Level Zero engine

Ranked by expected effect on *our* two measured constraints (BW-bound raw decode at 66.5 GB/s; MTP chain-bound at
7.95 tok/s). Revised after the §7b sweep: item 3 was demoted on A770 evidence, item 9 replaced.

### 1. Align the weight stream to 16 bytes; do not "just widen" [HIGH]
`engine/sycl/src/kernels/cuda/native_mmvq.dp.cpp:1509-1512`: a 16-byte load from a 2-byte-aligned address streamed
**145 GB/s against 640 GB/s** for the same blocks repacked at a 16-byte stride. And `:1505-1506`: 4-byte scalar per-lane
395 GB/s vs 16-byte 596 GB/s. Our 16-byte path measured *slower* (57.3 vs 89.9), which is the misalignment signature,
not a width result. Action: repack each quant type's blocks to a 16-byte-multiple stride (or per-type stride override,
as `BS` at `:1550`), then re-measure. Copy `load16_a2` (`:1526-1544`) only as the fallback for tails.

### 2. Submit the whole MTP draft chain in one go; hand tokens off on-device; poll a doorbell [HIGH]
`engine/sycl/src/core/mtp.cpp:1701-1733` — all `step_exec_[j]` graphs submitted with no waits, then host spins on
`h_out_[j]`. Our sequential chain is the binding constraint; this converts N host round-trips into N back-to-back graph
launches. Requires the draft token to be consumed from device memory by the next step (no host touch), and the doorbell
buffer to be host-visible (`host_malloc_polled`, `BIAS_UNCACHED`, `sycl_queue.hpp:63-78`). Cap every spin
(`docs/INTEL.md:149-151`).

### 3. Overlap the next window's draft with this window's verify — test it, do not assume it [MEDIUM, was HIGH]
B70 evidence is positive: `chain_live_` guard + separate drafter queue (`mtp.cpp:1640`; `docs/INTEL.md:432`: 46.5 →
49.8 tok/s from leaving the commit graph running beside the drafter on its own queue); cross-device `--pipeline-windows 2`
+17-23% (`docs/INTEL_ARC.md:137-143`).
**But the A770 log is negative and it is the closer analogue to us**: *"`--spec-split` HURTS — 9.9 / 10.2 / 10.3 tok/s
vs `--no-spec-split` 11.7 / 12.1 / 12.6. So overlapping the drafter with the verify window is negative on this A770
(opposite the fork's B70 result)"* (`docs/A770_SPEED_SPEC.md:254-258`). The difference tracks VRAM/link: when the
drafter's reads and the verify's reads contend for the same pool, overlapping adds pressure instead of hiding latency.
For a single shared-memory iGPU our prior should be **negative or wash**. Run it as an A/B, cheap, expect it to lose.

### 4. Confidence-gated draft depth, and shrink depth on a bandwidth-starved part [HIGH]
`for (j = 1; j < max_steps && pj >= min_p; ++j)` (`mtp.cpp:1793`), `--spec-min-p` (`docs/DETAILS.md:145`, `:381`).
Measured on an APU: `--spec 2` 15.65 > `--spec 4` 14.18 > `--spec 6` 11.16 tok/s (`docs/RADEON_780M.md:111-113`) —
**deeper is worse on a small pool**. We run `--mtp-drafts 3`; test 2.

### 5. Put copies on the copy-only queue group, not the compute engine [HIGH]
`engine/sycl/include/strata/copy_queue.hpp:1-15`: with the L0 v2 adapter all copies land on
`L0 Compute Engine<0,0>`; **0.8 s of a 3.8 s prompt** was the compute engine waiting for its own H2D. Fix is
`zeDeviceGetCommandQueueGroupProperties` → group with `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY` → immediate command
list → `sycl::make_queue<ext_oneapi_level_zero>`. On MTL, if any staging/prefill copy is on the compute queue it is
serialized behind our own kernels.

### 6. dp4a for decode; matrix cores for prefill — the crossover is rows, not the device [HIGH]
`dp4a.hpp:31-40` (`dpct::dp4a` is the decode primitive); s8 DPAS 8x16x32 needs Xe2 (`docs/INTEL_INT8_EXPERTS.md:43-46`);
grouped XMX 11-15% slower on Battlemage (`docs/INTEL.md:504`); XMX prompt attention -24 to -27% (`docs/INTEL.md:404`).
Strata states the rule explicitly: *"On the A770 its ordinary matvec stayed faster at two/three columns; therefore
determine a measured crossover rather than sending every batch to matrix hardware"* (`docs/A770_SPEED_SPEC.md:364-368`).
**But on iGPUs matrix cores are a large prefill win**: Strix Halo (Windows) 4K 688.8 → 1,192.8 tok/s, 64K 748.5 →
1,188.9 (×1.59-1.72, `docs/STRIX_HALO_WINDOWS.md:69-75`); 780M 1K ×1.60, 4K ×2.03 (`docs/RADEON_780M.md:88-89`).
→ Our "DPAS slower than DP4a on MTL" result is almost certainly **decode-shaped** (T ≤ 4, bandwidth-bound). Given our
prefill sits at ~33% of DP4a TOPS, re-test DPAS on **prefill-shaped GEMMs (T ≥ 256)** before writing it off. Do not
spend more time on DPAS at T = 1-4.

### 7. Row-reuse: one activation load per column, shared across N output rows [HIGH]
`native_mmvq.dp.cpp:1583-1597` + `:1688-1702` (`RPW ∈ {1,2,4}`), applied as a **per-shape table** rather than one
global choice (`engine/src/kernels/mmvq_il_parity.cpp:21`, `native_mmvq.dp.cpp:1656-1660`). Complements the MTL
"memory-level-parallelism gap" directly: it multiplies independent loads in flight per lane without changing the load
width — exactly the axis where we measured 16-byte loads failing.

### 8. Weight-stationary verify: decode each weight unit once, apply to all window tokens [HIGH]
Metal's core design (`engine/metal/kernels/gemv.metal:1-7`, `:60-70`), quantified at `docs/NVFP4.md:276-277`
(2.26 → 0.766 ms at 4 tokens, 2.95x; 89% → 79% of DRAM bandwidth). Our `--mtp-drafts 3` verify window is the ideal
case for this and it is *orthogonal* to the draft-chain fix.
**This matters more for us than for Strata, because our model is dense** (§9 q5): a dense 40B re-reads the *entire*
21.8 GB (Q4) or ~11 GB (IQ2_M) every window, so N accepted tokens per window is a straight N× multiplier on effective
bandwidth. Strata's MoE only re-reads ~1-3 GB of experts per window and amortizes a ~4.8 GB dense side, so their gain
from this trick is capped; ours is not. **Acceptance rate is therefore our bandwidth-efficiency knob** — which is the
same conclusion as their `--spec-min-p` sweep (`docs/A770_SPEED_SPEC.md:243-248`), approached from the other side.

### 9. Merge the verify window's per-row launches into one launch per stage [HIGH]
`docs/BATCHING.md:689-724`: `STRATA_BATCH_ATTN_ROWS=1` + `STRATA_BATCH_GDN_GROUPS=1` run block-score/top-k/attention and
the GDN recurrence **for all rows of a window in one launch each**, instead of one launch per row. Measured: the window
cost **14.3 ms for 4 rows → 12.2 ms**, and **24.6 ms for 8 rows → 19.9 ms**, i.e. *"about 1.7-1.9 ms per added row
instead of 2.4 ms"* — roughly **0.5-0.7 ms of pure per-row launch overhead**. Rows still read their own slot state,
passed by value so the captured graph stays valid (`:689-700`). Our MTP verify window is 4 rows (`--mtp-drafts 3`): at
2.4 ms/row that is ~9.6 ms of launch overhead per window, against a window budget in the tens of ms. This is the
cheapest structural win in the whole repo for a small-row-count decode.

### 10. Treat Level Zero's memory numbers as wrong until proven, and read `MemAvailable` on an iGPU [MEDIUM]
`engine/sycl/src/core/expert_cache.cpp:69-145`: driver `get_memory_info` → if `free == 0 && total > 0` use
`total − own fdinfo` (`:77-89`, from `drm-total-vram0`/`drm-total-local0`, `:30-64`) → if integrated, use
`MemAvailable − 6 GiB` (`:101-142`). On a shared pool the driver's "VRAM" number is the carve-out, not the budget, and
over-allocation does not fail — it livelocks (`docs/INTEL.md:142-144`).

**Runners-up (worth one experiment each):**
- `[HIGH — method, do this first]` **Prove the GPU is actually busy before optimising kernels.** `gputop`/`ze`-level
  compute-engine sampling during our decode would settle whether our 3.05 tok/s is bandwidth (as we believe) or
  handoff/launch stalls. Strata's A770 campaign spent three work packages on kernels before sampling showed the GPU at
  ~20% busy and `waitA`/`waitB` at 87% of the window (`docs/A770_SPEED_SPEC.md:687-752`). Add a doorbell/payload
  bypass switch so the profiler is usable (`STRATA_DOORBELL_NOCHECK`, `:735`).
- `[MEDIUM-HIGH, prompt only, approximation]` **KVA** — approximate the late-layer prompt read from the boundary
  residual, keeping a 2,048-token exact tail: 1.3-1.7x prompt, but −11% decode on a fast card and silent content loss
  at aggressive splits (`docs/KVA.md:1-40`, `:132-153`). Not in SYCL (`:102-104`). Real work is the quality gate, not
  the code.
- `[MEDIUM, family-specific]` Large GRF (`grf_size<256>`): +1.9-3.0x on Q6_K narrow-column matvec on an A770
  (`native_mmvq.dp.cpp:1635-1639`) but **10-58% slower** on BF16 projections on the same card
  (`docs/A770_SPEED_SPEC.md:528-530`). Test per kernel family with a spill check (`tools/check_cuda_spills.py`), never
  as a global flag.
- `[MEDIUM]` **Exchange rotation** — swap buffer ownership instead of memcpy'ing the evicted expert: exact (+8.1% on one
  token-identical pair), needs pinned+mapped pools and per-block-size spares
  (`docs/EXCHANGE_ROTATION.md:1-8`, `:51-67`, `:172-178`).
- `[MEDIUM]` Draft vocabulary subset: `--draft-vocab en` = ~110 MiB less VRAM and +1-2% English, at the cost of
  near-zero CJK drafts (`docs/DETAILS.md:382`); with `--mtp-q4` the per-step head read is printed and shrinks
  proportionally (`mtp.cpp:296`).
- `[MEDIUM]` E-core-aware CPU worker count: 15 workers → 10, and *"15-worker default about 35% slower than 9"* on MTL
  (`docs/release-notes-0.2.1.md:47`, `pool_plan.hpp:40-65`). For a 155H, do not plan 16.
- `[MEDIUM]` Continuous, measured PCIe/fabric share instead of a constant: `pcie_frac_for_gbps`
  (`generate.cpp:1695-1704`, `:2782-2805`).
- `[LOW]` Uncached host memory for any device-polled flag (`sycl_queue.hpp:42-92`); on `xe` ordinary host memory
  page-faults in the kernel (`:94-96`).
- `[LOW]` `SYCL_CACHE_PERSISTENT=0` (persistent JIT cache segfaults on Xe2, `docs/INTEL.md:140-141`);
  `-ze-opt-greater-than-4GB-buffer-required` for >4 GiB buffers (`:152-154`); aliased-page guard
  (`sycl_queue.hpp:171-213`); chunked fills instead of one big `memset` (`:18-39`).
- `[LOW]` Skip two experiments Strata already lost: software prefetch (`native_mmvq.dp.cpp:1630`) and 16-lane
  sub-groups for decode (same line).

---

## 9. Observed facts vs inference vs open runtime questions

**Observed (cited):** everything in §1-§7. The load-width and alignment numbers, the copy-engine queue finding, the
`dp4a`-not-DPAS primitive, the MTP submission structure, the UMA memory policy, the APU tok/s tables and quant
quality table, the absence of NPU code, and the absence of any Intel iGPU plan.

**Inference (mine, labelled):**
- The effective-GB/s table in §5.5 depends on my `D/A + 480·E` formula and on `A` and on assumed memory widths. It
  reproduces Strata's own stated 35-40%-of-bound for the R9700, which is why I consider it sound, but it is not a
  Strata-published number.
- The 780M's 2-channel DDR5 assumption.
- That our 16-byte-load regression is an alignment effect — strongly suggested by `:1509-1512`, not proven on our box.
- That two of Strata's wins will *not* transfer to a single shared pool: drafter/verify overlap lost on the A770
  (`docs/A770_SPEED_SPEC.md:254-258`) and large GRF lost on A770 BF16 (`:528-530`). Both losses are contention effects,
  and an iGPU has strictly less contention headroom than a 16 GB card with a 1.8 GB/s link. Prior: negative or wash.
- My overall read: **Strata's advantage over our 3.05/7.95 tok/s is the spec loop, the pack ladder and the measurement
  discipline — not a bandwidth trick we are missing.** Our measured 66.5 GB/s at 3.05 tok/s sits inside the band their
  own window model predicts for a pool of our size (§5.5), so kernel-level bandwidth work on our side is close to done;
  the remaining wins are structural (draft chain, per-row launches, alignment).

**Open runtime questions for the parent:**
1. Which kernel driver does our MTL iGPU bind — `xe` or `i915`? It decides whether
   `sycl_queue.hpp:94-96` (kernel page-faults on ordinary host memory, job timeout) applies to us. Check
   `/sys/class/drm/card*/device/driver`.
2. Does our device expose a `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY` group (technique 5)? Enumerate with
   `zeDeviceGetCommandQueueGroupProperties`.
3. What does `zeDeviceGetMemoryProperties` report for `max_mem_alloc_size` on our iGPU (Strata sees 3.77 GiB on an 8 GB
   A750)? `sycl_queue.hpp:267-269` suggests it can be far below the pool and is a hard wall for one allocation.
4. Does Level Zero's free-memory query return 0 on our MTL part (the A770 #1545 failure mode,
   `expert_cache.cpp:77-89`)? If so our budget code needs the `MemAvailable` fallback, mandatorily.
5. Arithmetic worth stating explicitly, because it reframes the target: 66.5 GB/s ÷ 3.05 tok/s = **21.8 GB of weights per
   token**, which is about a 40B model at ~4.4 bits/weight — i.e. our Qwen3.8-40B Q4 run reads the *whole model once per
   token*, as a **dense** model does. The §5.5 MoE formula (10 of 512 experts) therefore does **not** apply to us, and our
   run is textbook bandwidth-bound. With IQ2_M + MTP at 7.95 tok/s the per-window traffic drops to ~11 GB and the implied
   bandwidth is ~36 GB/s at ~2.4 tokens/window — well under 66.5, which is exactly why the draft chain, not bandwidth, is
   binding there. Confirm the IQ2_M pack's byte count before quoting these.
6. Does our verify window issue **one launch per row** for attention/GDN? If yes, technique 9 is worth ~0.5-0.7 ms × rows
   per window and is probably the cheapest remaining structural win.
7. Is Qwen3.8-40B hybrid (any gated-recurrence/SSM layers)? KVA's state-only path needs recurrent state to build; a pure
   attention model would have to skip only the attention out-proj + MLP half, which is less of the layer.
