# M2 — P0: measure GPU-busy fraction during the decode/draft window (gates P2–P4 priority)

Goal: determine whether the MTL iGPU sits IDLE during parts of the IQ2_M+MTP d3 decode
window (handshake/launch stalls) or is genuinely busy. REPORT.md §2 says the champion
passes pull only ~37–38 GB/s while BASELINE.md now pins the run at 7.96 tok/s /
~38.1 GB/s — but that number cannot tell idle-GPU from lean-GPU. Strata's A770 autopsy
(A770_SPEED_SPEC.md: decode ~20% GPU-busy, 87% of window in CPU handshake) is the
nightmare scenario. Your measurement decides whether P2/P3/P4 (draft-chain/verify
overhead) or P1 (bandwidth) is the right top priority.

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Create branch
  `p0-gpu-busy` from current main (c4246bd or newer). **Do not push, do not touch gh
  auth, do not merge.** Leave clone on `p0-gpu-busy` when done.
- Read first: `research/apu-shared-memory/REPORT.md` §1/§2/§8-q1,
  `research/apu-shared-memory/BASELINE.md`, `APPENDIX-strata.md` (doorbell/graph +
  A770 sections), `zero/engine_drafters.h` (~line 502 the serial `for t<nd` draft
  chain), `zero/clrt.h` (L0 wrapper: launches, events, kernel cache),
  `zero/engine.h` dispatch paths for decode.
- Node: `ssh bschooley@192.168.50.48` (passwordless), ALWAYS wrap `bash -lc '...'`.
  Passwordless sudo works. Node tree `~/inferzero` is OLD (120c311) — do NOT modify it.
  For instrumented builds: `rsync` (or scp -r) the branch source tree to
  `~/inferzero-p0` on the node and build there with `./build.sh runz serverz` (g++13+,
  level-zero-loader + intel-compute-runtime already installed). Never touch `~/models`.
- Champion model: `/home/bschooley/models/Qwen3.8-40B-...-IQ2_M.gguf` (13.53 GiB,
  14.53 GB on-disk weights).

## What to measure (pick the strongest 1–2 paths; report what was impossible)

A. **In-engine phase timing (preferred — gives the phase split):** instrument the
   decode loop behind env `INFERZERO_GPU_BUSY=1` (default OFF; when off, zero behavior
   change — do not perturb the hot path). Per decode window (one verify pass + its d3
   draft chain) record: wall time, per-kernel submit→complete intervals via L0 events
   or `zeEventQueryPhysicalTimer`-style timestamps from clrt, and the sum of
   in-flight GPU time vs window wall time. Export per-window CSV: draft_ms, verify_ms,
   gap_ms (submit latency/host work between kernels), accept info. Run: `runz <IQ2_M>
   --mtp --mtp-drafts 3 --n 200 --show` (or serverz single request) with the env set.
B. **External engine-busy sampling:** check what MTL/xe exposes: `intel_gpu_top -l`
   / `gputop` availability (install `intel-gpu-tools` via sudo only if quick), xe
   perf/OA busy %, or `/sys/class/drm/card*/` busy counters. Sample during the same
   run. If OA unavailable, say so — do not rabbit-hole.
C. **Roofline ratio (cheap, do it regardless):** ratio of decode GEMM kernel time to
   pure weight-stream time (gufo method). gpu_bench's block-read GB/s (89.9 measured)
   against engine-pulled GB/s (~38) bounds the inefficiency even without counters.

## Discipline

- Two-phase rule: ONE screen pass (tg~200, 1 repeat, 60 s cool before) is enough for
  busy%; 2nd repeat only if noisy. No other GPU process (`pgrep -x serverz; pgrep -x
  runz` empty; kill only yours). No CPU-heavy work concurrently (L0 watchdog
  0x70000001). Do NOT run a full-power all-CPU lane next to GPU load.
- Cross-check: measured window pass-rate × 14.53 GB should land near BASELINE's
  ~38 GB/s — if wildly off, your timing is wrong, fix before concluding.
- Instrumented engine changes go on branch `p0-gpu-busy` behind the env flag; keep
  default path untouched (iGPU path must stay bit-identical with flag off).

## Files you own

- Probe/instrumentation code committed on `p0-gpu-busy`: engine env-flag timing AND/OR
  a standalone probe — at least one committed NON-.md path must contain `busy`
  (e.g. `bench/gpu_busy.sh`, `tools/gpu_busy.cpp`) — the goal checker looks for it.
- `research/apu-shared-memory/P0-gpu-busy.md` with `METHOD` / `NUMBERS` / `VERDICT`
  sections: per-phase busy% (draft phase vs verify phase vs host gaps), the roofline
  ratio, pass-rate×bytes sanity line, and an explicit priority VERDICT: either
  "draft phases show GPU idle ≥X% → P2/P3/P4 confirmed top priority" or
  "GPU busy ~X% during all phases → overhead is ON-device (kernel structure), lean
  toward P1/P4-launch work".
- Raw CSV/logs force-added under `research/apu-shared-memory/results/`.

## Completion condition (stop after this)

Branch `p0-gpu-busy` with 1–3 commits (engine flag, probe file(s), P0-gpu-busy.md +
logs). Report: busy% per phase, gap/host fraction, roofline ratio, the VERDICT line,
and anything the node could NOT measure (e.g. no OA counters — say so explicitly).
Do not push. Do not start P1.
