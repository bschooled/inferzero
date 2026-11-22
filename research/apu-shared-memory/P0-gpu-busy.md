# M2 — P0: GPU-Busy Fraction & Roofline Measurement During the IQ2_M + MTP d3 Decode Window

**Date:** 2026-11-21
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16 — Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz, 32 GB shared LPDDR5x-7467, driver 26.40.40196, kernel 7.3.5-2-cachyos, `i915` DRM driver)
**Model:** `Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf` (13.53 GiB / 14.53 GB on-disk weights; 13.74 GB main-model raw weights streamed per pass)
**Flag:** `INFERZERO_GPU_BUSY`
**Probe & harness:** `zero/clrt.h`, `zero/engine.h`, `zero/engine_drafters.h`, `zero/engine_sessions.h`, `tools/gpu_busy_sampler.cpp`, `bench/gpu_busy.sh`
**Raw logs & CSVs:**
- `research/apu-shared-memory/results/p0-counter-probe.log`
- `research/apu-shared-memory/results/p0-gpu-busy-prose.{log,csv}`
- `research/apu-shared-memory/results/p0-gpu-busy-structured.{log,csv}`
- `research/apu-shared-memory/results/p0-gpu-busy-ar-prose.{log,csv}`
- `research/apu-shared-memory/results/p0-external-sampler-{prose,structured}.csv`

---

## METHOD

To determine whether the Meteor Lake iGPU sits idle during the IQ2_M + MTP d3 decode window (the Strata A770 nightmare scenario of ~20% GPU-busy / 87% CPU handshake) or is genuinely busy on-device, we implemented all three measurement paths from the brief:

### Path A: In-Engine Phase Timing & L0 Event Timestamps (`INFERZERO_GPU_BUSY=1`)
1. **Zero hot-path perturbation when off:** Gated behind `INFERZERO_GPU_BUSY=1` (default OFF). When off (`INFERZERO_GPU_BUSY=0` or unset), `want_busy` is false, no extra events are requested on `clEnqueueNDRangeKernel`, and no timing hooks execute.
2. **Bit-identical verification:** Confirmed via `runz --dump-logits` over a 16-token MTP d3 generation on `prose.txt`:
   - `sha256(INFERZERO_GPU_BUSY=0) = ac0488197f73f353ea226195b47bd9724f3f710b2c63217d351ada4112334afe`
   - `sha256(INFERZERO_GPU_BUSY=1) = ac0488197f73f353ea226195b47bd9724f3f710b2c63217d351ada4112334afe`
3. **Phase decomposition:** Each speculative decode window is split into three phases:
   - `draft` (`Engine::mtp_draft`): the serial `for (int k = 0;; ++k)` MTP draft chain (`mtp_rows` over pending accepted tokens + `tok`, blocking `clEnqueueReadBuffer` of `248,320` logits per step, host `max_element` + `softmax_den`, and chained `mtp_rows` for drafts 2..3).
   - `verify` (`Engine::verify`): main-model 96-layer forward pass over `T = 1 + drafts.size()` rows (typically 4 rows for d3), `head_rows` over `T` rows, and blocking `clEnqueueReadBuffer` of `T * 248,320 * 4` bytes (~3.97 MB).
   - `verify_done` (`Engine::verify_done`): state commit (`hver_ -> hlast_`, `hpend_`) and, whenever `accepted < drafted`, Gated DeltaNet state rollback/replay (`gdn_recur` + `conv_state_update` across the 72 recurrent layers).
4. **Metrics recorded per phase and exported to `INFERZERO_GPU_BUSY_CSV`:**
   - Phase wall time (`steady_clock`), sum of on-device kernel execution durations from Level Zero hardware context timestamps (`zeEventQueryKernelTimestamp` via `ClRt::event_ms`), sum of GEMM/GEMV kernel execution time (`gemm_*`, `lm_head`, `mtp_eh_proj`), host submission CPU time (`zeKernelSetArgumentValue` + `zeCommandListAppendLaunchKernel` + `zeCommandListAppendMemoryCopy`), host synchronization wait time (`zeCommandListHostSynchronize`), and `gap_ms = max(0, wall_ms - gpu_ms)`.

### Path B: External Hardware Engine-Busy Sampling (`tools/gpu_busy_sampler.cpp`)
We built a standalone C++20 external sampler (`bin/gpu_busy_sampler`) that samples every 100 ms alongside `runz`:
1. **i915 PMU perf events (`perf_event_open` on `/sys/devices/i915`, type 14):** `ccs0-busy` (`config=0x4000`, ns), `rcs0-busy` (`config=0x0`, ns), `bcs0-busy` (`config=0x1000`, ns), and `software-gt-awake-time-gt0` (`config=0x100004`, ns).
2. **Per-process DRM client `fdinfo` (`/proc/<pid>/fdinfo/*`):** `drm-engine-compute`, `drm-engine-render`, and `drm-engine-copy` cumulative nanoseconds for the `runz` process's `i915` DRM client.
3. **Sysfs GT RC6 residency & clocks (`/sys/class/drm/card1/gt/gt0/`):** `rc6_residency_ms` (duty cycle of GT0 awake time = `1 - d_rc6_ms / dt_ms`) and `rps_act_freq_mhz` (actual GPU slice clock).

#### What the node could NOT measure (and why):
- **`xe` kernel driver counters:** The node runs the `i915` kernel driver (`/sys/class/drm/card1/device/driver -> i915`), not `xe`.
- **Level Zero Sysman engine activity (`zesEngineGetActivity`):** Although `zesDeviceEnumEngineGroups` succeeds and enumerates 7 engine groups (types 5..9), `zesEngineGetActivity` returns `activeTime = 0` (`ts` small constant) on all 7 groups because the Linux `i915` Sysman backend does not wire up engine utilization counters.
- **Unprivileged `perf_event_open` / `intel_gpu_top`:** `intel_gpu_top`, `gputop`, and `perf` are not pre-installed on the node, and `/proc/sys/kernel/perf_event_paranoid = 2` (with `/proc/sys/dev/i915/perf_stream_paranoid = 1`) blocks unprivileged PMU opens (`EACCES`). Running `sudo ./bin/gpu_busy_sampler` (or reading `/proc/<pid>/fdinfo/*`, which is world-readable) bypasses both limitations cleanly without installing packages.

### Path C: Gufo Roofline Self-Check Ratio
Measured peak iGPU memory read bandwidth with `bin/gpu_bench` (90.0 GB/s block reads, 57.6 GB/s 16-byte loads) and compared pure weight-stream time (`rd_bytes / 89.9 GB/s`) against actual decode GEMV/GEMM kernel execution time, total GPU kernel time, and full window wall time for both `T = 1` (autoregressive) and `T = 4` (MTP d3 verify).

---

## NUMBERS

### 1. In-Engine Phase Split (`INFERZERO_GPU_BUSY=1`, `tg200`, 60 s cooldown)

| Metric | `prose` (MTP d3, 69 windows) | `structured` (MTP d3, 54 windows) | `prose` Raw AR (`T=1`, 50 steps) |
|---|---:|---:|---:|
| **Generated tokens / wall time** | 200 tok / 25,018.1 ms (**7.99 tok/s**) | 200 tok / 18,985.2 ms (**10.53 tok/s**) | 50 tok / 12,662.9 ms (**3.95 tok/s**) |
| **MTP drafts accepted** | 130 / 182 (**71.4%**, 2.90 tok/pass) | 145 / 160 (**90.6%**, 3.70 tok/pass) | — (1.00 tok/pass) |
| **Mean window wall time** | **360.40 ms/win** (24,867.3 ms total) | **349.64 ms/win** (18,880.5 ms total) | **252.74 ms/step** (12,636.8 ms total) |
| **Mean window GPU kernel time** | **313.47 ms/win** (21,629.7 ms total) | **302.73 ms/win** (16,347.4 ms total) | **228.95 ms/step** (11,447.6 ms total) |
| **Mean window GEMM/GEMV GPU time** | **266.21 ms/win** (18,368.3 ms total) | **256.77 ms/win** (13,865.7 ms total) | **202.82 ms/step** (10,140.9 ms total) |
| **Mean window host gap (`wall - gpu`)** | **46.92 ms/win** (3,237.6 ms total) | **46.91 ms/win** (2,533.1 ms total) | **23.78 ms/step** (1,189.2 ms total) |
| **Overall Window GPU-Busy % (L0 events)** | **87.0% busy** (13.0% gap) | **86.6% busy** (13.4% gap) | **90.6% busy** (9.4% gap) |

#### Per-Phase Breakdown (`prose` vs `structured`, MTP d3)

| Phase | Prompt | Share of Window Wall | Wall Time (ms/win) | L0 Kernel GPU Time (ms/win) | Phase GPU-Busy % | GEMM GPU Time (ms/win) | Host Gap (ms/win & %) | Launches / Copies / Syncs per win |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`draft` (`mtp_draft`)** | `prose` | **13.7%** | 49.48 ms | 42.18 ms | **85.2% busy** | 41.01 ms | 7.30 ms (14.8%) | 53.9 / 7.5 / 10.3 |
| **`draft` (`mtp_draft`)** | `structured` | **14.7%** | 51.48 ms | 44.58 ms | **86.6% busy** | 43.15 ms | 6.90 ms (13.4%) | 61.6 / 7.9 / 10.9 |
| **`verify` (`verify`)** | `prose` | **83.6%** | 301.39 ms | 264.46 ms | **87.7% busy** | 225.19 ms | 36.93 ms (12.3%) | 1544.6 / 290.0 / 5.0 |
| **`verify` (`verify`)** | `structured` | **84.1%** | 294.14 ms | 255.33 ms | **86.8% busy** | 213.62 ms | 38.81 ms (13.2%) | 1560.2 / 290.0 / 5.0 |
| **`verify_done`** | `prose` | **2.6%** | 9.52 ms | 6.84 ms | **71.8% busy** | 0.00 ms | 2.69 ms (28.2%) | 73.0 / 38.2 / 1.0 |
| **`verify_done`** | `structured` | **1.1%** | 4.02 ms | 2.81 ms | **70.0% busy** | 0.00 ms | 1.21 ms (30.0%) | 29.3 / 16.5 / 1.0 |

*Note on `host_submit` in `verify` (~129 ms/win):* `verify()` enqueues 288 `clEnqueueCopyBuffer` calls across the 72 Gated DeltaNet layers (`L.conv_state -> L.conv_pre`, `bq -> L.qkvin`, `cq -> L.xin`, `gq -> L.abin` at `zero/engine_qwen.h:59-62`) without host synchronization. On `i915` immediate command lists, when the in-flight queue depth saturates, `zeCommandListAppendLaunchKernel` / `zeCommandListAppendMemoryCopy` blocks inside the driver waiting for ring space while the GPU is actively executing earlier kernels in the same pass. That driver backpressure is why `host_submit` (~129 ms) overlaps with GPU execution rather than adding to `gap_ms` (~37 ms).

*Note on profiling overhead:* Running `prose` (`tg200`) with `INFERZERO_GPU_BUSY=0` vs `INFERZERO_GPU_BUSY=1` (without `--prof-gen`) yielded `23,386.9 ms` (`8.55 tok/s`, `277.1 ms/verify`) vs `24,546.2 ms` (`8.15 tok/s`, `290.5 ms/verify`) — a 4.7% timestamp-event overhead across ~1,667 kernel launches per window. Thus the unperturbed engine has an even smaller host gap (~9% instead of ~13%).

---

### 2. External Hardware Counter Sampling (Path B — Steady-State Decode Window)

Sampled at 100 ms intervals via `bin/gpu_busy_sampler` concurrently with the `tg200` runs above:

| External Hardware Counter Source | `prose` Steady-State Decode (24.96 s, 246 samples) | `structured` Steady-State Decode (17.97 s, 177 samples) | Agreement with L0 In-Engine Timestamps |
|---|---:|---:|---|
| **i915 PMU `ccs0-busy` (`config=0x4000`)** | **97.42% busy** | **97.48% busy** | Confirms compute engine is ~97.5% active (includes the 290 async device-to-device buffer copies per window not timed by `cl_event` kernel timestamps) |
| **Per-process DRM `fdinfo` (`drm-engine-compute`)** | **97.69% busy** | **96.89% busy** | Matches system-wide `ccs0-busy` within 0.6%, confirming `runz` owns 100% of GPU compute load |
| **i915 PMU `software-gt-awake-time-gt0`** | **100.00% awake** | **100.00% awake** | GT0 never sleeps during decode |
| **Sysfs GT0 RC6 (`1 - d_rc6 / dt`)** | **99.84% awake** | **99.63% awake** | GT0 is out of RC6 power-gate 99.6–99.8% of the window |
| **Sysfs GT0 Actual Clock (`rps_act_freq_mhz`)** | **2196 MHz mean** (2050–2250 MHz) | **2165 MHz mean** (1950–2250 MHz) | Holds 96–98% of 2250 MHz max clock throughout decode |

**Comparison to Strata's A770 Autopsy:**
Strata's A770 (`A770_SPEED_SPEC.md`) showed `gputop` compute engine busy at **~20%** (mostly 0–20% with 87% of the window spent spinning on CPU expert-pool handshakes). On Meteor Lake running `inferzero` IQ2_M + MTP d3, the iGPU is **86.6–87.0% busy in kernel execution alone** and **97.4–97.5% busy on `ccs0` including in-list memory copies**. The GPU is **not** sitting idle waiting on the host.

---

### 3. Cross-Check: Pass Rate × Weight Bytes vs BASELINE.md

- **On-disk model weight bytes:** `13.53 GiB = 14.53 GB` (`14,529,495,552` bytes).
- **Main-model raw weight bytes streamed per verify pass (`eng.rd_bytes`):** `13.74 GB` (`13,744,291,840` bytes; excludes the MTP head's ~3.1 GB of weights read during `mtp_draft`, which brings total DRAM read traffic per window to `16.68–17.04 GB`).
- **Pass-rate × 14.53 GB cross-check:**
  - `prose`: `2.775 passes/s × 14.53 GB = 40.32 GB/s` (or `2.775 × 13.74 GB = 38.14 GB/s`).
  - `structured`: `2.860 passes/s × 14.53 GB = 41.56 GB/s` (or `2.860 × 13.74 GB = 39.31 GB/s`).
  - **Mean across `prose` + `structured`:** `2.818 passes/s × 13.74 GB = 38.7 GB/s` (`× 14.53 GB = 40.9 GB/s`), landing right on `BASELINE.md`'s `~38.1 GB/s` (`7.96 tok/s ÷ 3.03 tok/pass = 2.627 passes/s × 14.53 GB = 38.17 GB/s`).

---

### 4. Gufo Roofline Ratio & Where the Window Time Actually Goes (Path C)

Using `gpu_bench` on this node:
- **Peak iGPU block-read bandwidth (`K_READ_B`, 2 KB in flight/sub-group):** **90.0 GB/s** (89.9 GB/s in `REPORT.md`).
- **Peak iGPU 16-byte per-lane load bandwidth (`K_READ`):** **57.6 GB/s** (57.3 GB/s in `REPORT.md`).
- **Pure weight-stream floor for one main-model pass (`13.744 GB ÷ 89.9 GB/s`):** **152.88 ms** (or `161.62 ms` using `14.53 GB` on-disk size).

| Decode Regime | Main-Model Weight Stream Floor (@ 89.9 GB/s) | Actual Verify GEMM/GEMV Time | Verify Total GPU Kernel Time | Verify Phase Wall Time | Full Window Wall Time | **Gufo Roofline Ratio (`GEMM_ms / stream_ms`)** | **End-to-End Window Ratio (`win_wall / stream_ms`)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **IQ2_M Raw AR (`T = 1`)** | 152.88 ms | **202.82 ms** (67.8 GB/s) | 228.95 ms (60.0 GB/s) | 252.74 ms (54.4 GB/s) | 252.74 ms | **1.33×** (75.4% of peak BW) | **1.65×** (60.5% of peak BW) |
| **IQ2_M + MTP d3 (`prose`, `T ≈ 3.64`)** | 152.88 ms | **225.19 ms** (61.0 GB/s) | 264.46 ms (52.0 GB/s) | 301.39 ms (45.6 GB/s) | 360.40 ms | **1.47×** (67.9% of peak BW) | **2.36×** (42.4% of peak BW) |
| **IQ2_M + MTP d3 (`structured`, `T ≈ 3.96`)** | 152.88 ms | **213.62 ms** (64.3 GB/s) | 255.33 ms (53.8 GB/s) | 294.14 ms (46.7 GB/s) | 349.64 ms | **1.40×** (71.6% of peak BW) | **2.29×** (43.7% of peak BW) |

#### Decomposition of the 2.36× (`prose`, 360.4 ms vs 152.9 ms ideal) Gap:
Why does `BASELINE.md` see only `~38.1 GB/s` effective main-model weight bandwidth when the GPU is 87%–97.5% busy? Because a 360.4 ms MTP d3 window spends its time in **five distinct buckets on an almost continuously busy GPU**:
1. **152.9 ms (42.4% of window):** Ideal main-model weight-stream time at 89.9 GB/s (`13.74 GB`).
2. **+72.3 ms (20.1% of window) — Multi-token verify GEMV + `lm_head` overhead (1.47× roofline ratio):**
   - For `T = 1` AR, main-model GEMVs take `202.8 ms` (**67.8 GB/s**, 1.33× roofline — close to Gufo's 1.26×).
   - For `T = 4` verify, `gemv_mt_v` / `gemv_mt` + `lm_head` take `225.2 ms` (**61.0 GB/s**, 1.47× roofline). Critically, `lm_head` (Q6_K, 248,320 rows, forced `GROWS=1` at `engine.h:1423`) jumps from **10.63 ms/pass** at `T=1` (`531.6 ms / 50`) to **31.8 ms/pass** at `T=4` (`~3002 ms` total across verify + draft, or ~11.9 ms per `lm_head` launch), because `gemv_mt_v` with `GROWS=1` on Q6_K does less work-sharing and reads `lm_head` 4.6× per window (once in `verify` over 4 rows + 2.6–3.0 times in `mtp_draft` over 1 row!).
3. **+39.3 ms (10.9% of window) — Non-GEMM verify kernels (`264.5 - 225.2 ms`):**
   - `gdn_recur` (~14–18 ms/win), `rmsnorm_quant` (~5.4 ms/win), `attention` + `attn_prep` + `attn_combine` (~10.0 ms/win), `gdn_conv` + `conv_state_update` + `gated_norm_quant` (~8.2 ms/win), `swiglu_quant` (~3.4 ms/win).
4. **+36.9 ms (10.2% of window) — Verify non-kernel time (`301.4 - 264.5 ms`):**
   - Includes the 288 per-layer `clEnqueueCopyBuffer` state saves (`L.conv_pre`, `L.qkvin`, `L.xin`, `L.abin` — which keep `ccs0` 97.5% busy!) plus the final blocking `clEnqueueReadBuffer` of `4 × 248,320 × 4 = 3.97 MB` of verify logits.
5. **+49.5 ms (13.7% of window) — Serial MTP draft chain (`mtp_draft`) + +9.5 ms (2.6%) `verify_done`:**
   - `mtp_draft` is **85.2–86.6% GPU-busy** (42.2–44.6 ms GPU kernel time out of 49.5–51.5 ms wall, with only 6.9–7.3 ms of host gap for `clEnqueueReadBuffer` + `softmax_den` across 3 serial steps).
   - Why does `mtp_draft` take 42–45 ms of GPU time? Because each draft step runs `mtp_eh_proj` + 1 layer + **the full 248,320-row Q6_K `lm_head` (~1.05 GB)**! Across ~2.7–3.0 draft steps per window, the drafter streams ~3.0–3.3 GB of weights on the GPU (mostly `lm_head` three times in a row).
   - `verify_done` adds 6.8 ms GPU / 9.5 ms wall on `prose` (when drafts are rejected, replaying `gdn_recur` across all 72 recurrent layers).

---

## VERDICT

**POSITIVE**
Flag: `INFERZERO_GPU_BUSY`

**GPU busy ~86.6–87.0% (L0 kernel timestamps) and ~97.4–97.5% (external `i915` PMU `ccs0-busy` & `fdinfo`) during all decode phases (draft phase 85.2–86.6% GPU-busy, verify phase 86.8–87.7% GPU-busy) → overhead is ON-device (kernel structure & redundant device work), lean toward P1/P4-launch work.**

### Priority Gate Decision for P1–P4:
1. **Not the Strata A770 scenario:** Host launch/handshake gaps (`wall_ms - gpu_ms`) are only **13.0–13.4% of the window** (~46.9 ms out of ~355 ms, and ~9% without event-profiling overhead), and external PMU `ccs0-busy` is **97.4–97.5%** when counting the in-list device-to-device copies. Moving drafting to the CPU or chasing L0 doorbell graphs to hide host launch latency (Strata's A770 fix) will **not** yield large speedups because the iGPU is already busy ~87–97.5% of the time and the host gap inside `mtp_draft` is only **6.9–7.3 ms/window (2.0% of window wall)**.
2. **Where the real on-device wins are:**
   - **P1 (Kernel bandwidth / roofline 1.33× → 1.47× & `lm_head` Q6_K):** `gemv_mt_v` / `gemv` runs at 61.0–64.3 GB/s (68–72% of the 89.9 GB/s block-read ceiling). Closing the 1.40–1.47× roofline gap on `gemv_down`/`gemv_gate`/`gemv_up`/`gemv_qkv` and `lm_head` directly cuts the 214–225 ms GEMM block that dominates 62% of the entire window.
   - **On-device draft & verify waste (P2/P3/P4 on-device):**
     - `lm_head` alone consumes **2.46–3.00 seconds (14–15% of all GPU time)** because every serial MTP draft step reads all 248,320 rows of `lm_head` (~11.5 ms GPU time × 3 drafts = ~34.5 ms/window just in draft `lm_head`). Enabling reduced-vocabulary drafting (`ARC_MTP_VOCAB`) or skipping low-value draft steps (P2 adaptive draft depth / P3 zero-cost prompt-lookup copy drafting, which bypasses `mtp_draft`'s 44 ms of GPU work entirely when matched) saves **on-device GPU time**, not host idle time.
     - In `verify`, the 288 per-layer `clEnqueueCopyBuffer` calls (`L.conv_pre`, `L.qkvin`, `L.xin`, `L.abin`) + rejected-draft `verify_done` replay (`gdn_recur` × 72 layers) account for ~30–35 ms/window of non-GEMM device traffic.
