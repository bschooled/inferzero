# M3 — P1: alignment/stride audit of decode GEMV streams + load-time padding experiment

Goal: find and fix memory-layout waste that keeps inferzero decode GEMVs at 61–68 GB/s
instead of the 89.9 GB/s iGPU block-read ceiling. P0 (read it first:
`research/apu-shared-memory/P0-gpu-busy.md`) measured roofline ratios of **1.33x for
T=1 AR GEMV (67.8 GB/s)** and **1.40–1.47x for d3 verify GEMVs (61.0–64.3 GB/s)**, and
the GEMM block is ~62% of the decode window. Sibling precedents: Strata repacked Q6_K
210-B blocks from 145 → 640 GB/s by making the row stride 16-byte aligned
(`native_mmvq.dp.cpp:1505-1512`); gufo padded power-of-two F16 rows by 512 B at LOAD
time and doubled stream rate (103 → 203 GB/s) with bit-identical output. Our own
16-byte-load vs block-read gap (57.3 vs 89.9 GB/s in gpu_bench) says alignment + MLP
matter here.

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Branch
  `p1-alignment` from current main (397ac0d or newer). **Do not push, do not touch gh
  auth, do not merge.** Leave the clone on `p1-alignment`.
- Node: `ssh bschooley@192.168.50.48`, always `bash -lc '...'`. Rsync the branch tree
  to `~/inferzero-p1` and build there: `./build.sh runz serverz gpu_bench`. Do NOT
  touch `~/inferzero` or `~/inferzero-p0` or `~/models`. GPU hygiene: `pgrep -x
  serverz; pgrep -x runz` empty before/after; kill only yours; 60 s cool between full
  runs.
- Read first: P0-gpu-busy.md (phase/GEMV timings), REPORT.md §6.2 + §8 q2/q5,
  APPENDIX-strata.md (alignment probe), APPENDIX-gufo.md (padding experiment),
  `zero/clrt.h` (buffer creation/upload), `zero/engine_load*.h` / engine_setup /
  engine_alloc (GGUF → device buffer layout), decode GEMM kernels `zero/kernels/
  gemm.cl`, `decode.cl` (how they address rows/strides).

## Tasks

1. **Audit (write it down even if nothing is misaligned):** behind env
   `ARC_ALIGN_AUDIT=1` (or a small tool), log for every decode-relevant weight buffer:
   base address mod 16 (and mod 64), row stride bytes, stride mod 512 — for Q4_K_S
   blocks (144 B), their scales/min blocks, IQ2_M superblocks (256 B) if cheap, and
   all F16 tensors read per token. Print a summary table into the run log. Answer
   §8-q2 concretely: are IQ2_M sub-block reads misaligned like Strata's Q6_K case?
2. **Microbench:** extend gpu_bench or add `bench/align_probe.cpp` that streams the
   SAME quant block layout at (a) native stride, (b) +512 B padded stride, (c)
   16-B-aligned repacked layout. Report GB/s for 16-byte-load and block-read patterns.
   This bounds the win before touching the engine.
3. **Engine experiment (only if the microbench shows >=5% signal):** implement
   load-time padding of suspect streams behind env `ARC_PAD_ROWS=N` (bytes of pad per
   row; 0 = off = default). Padding must not change numerics — logits stay
   bit-identical (diff `runz --dump-logits` with pad on/off) and the W3 startup GEMM
   self-check must stay green.

## Measurement (two-phase rule)

- Baselines to compare against (BASELINE.md): Q4_K_S raw full-pass median **3.17**
  (prose 3.18 / structured 3.17); IQ2_M+MTP d3 7.96.
- Screen: Q4 raw tg150, 1 repeat, prose+structured, pad OFF vs best pad candidate,
  30 s cool. Then a screen with IQ2_M+MTP d3 for insight (IQ suspect path).
- Confirm (only if screen >=5%): full pass = 3 repeats tg300 prose+structured,
  60 s cool, medians. Success bar: Q4 raw **>3.4 tok/s** full-pass median.
- Also record bytes/pass unchanged and effective GB/s (pass-rate × 22.72 GB for Q4)
  so the verdict says which regime moved.

## Files you own

- Audit + padding code on branch `p1-alignment` (env-gated, default off).
- `bench/align_probe.cpp` (or gpu_bench extension) + its build hookup.
- `research/apu-shared-memory/P1-alignment.md` with METHOD / NUMBERS / VERDICT.
  VERDICT line: `POSITIVE` + `Flag: <name>` if a padding toggle is merged and works;
  otherwise honest `NEGATIVE`/inconclusive with the measured numbers (an audit that
  finds everything already 16-B-aligned and a microbench showing stride padding does
  nothing on MTL is a VALID, valuable outcome — record it and stop).
- Microbench logs + run logs force-added under `research/apu-shared-memory/results/`.

## Completion condition (stop after this)

Branch `p1-alignment`, 1–4 commits. Report: audit table (aligned/misaligned streams),
microbench GB/s trio, screen+confirm Q4 tok/s, bit-exact check result, VERDICT line.
Do not push. Do not start P2.
