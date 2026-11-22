# P1 — Alignment / stride audit of decode weight streams + load-time padding experiment

**Date:** 2026-11-21 · **Node:** Spectre (192.168.50.48), `~/inferzero-p1` (branch `p1-alignment`)
**Hypothesis tested (REPORT §6.2 / §8 q2,q5):** inferzero decode GEMV streams suffer
Strata-style stride aliasing (Q6_K 210 B rows → 145 vs 640 GB/s) or gufo-style
power-of-two row aliasing (103 → 203 GB/s with +512 B pad), explaining part of our
57.3 → 89.9 GB/s load-width gap and lifting Q4 raw 3.17 → >3.4 tok/s.
**Raw logs:** `results/p1-align-probe.log`, `p1-audit-q4ks.log`, `p1-audit-iq2m.log`,
`p1-bitexact-*.log`, `speedbench-Q4_K_S-raw-p1-pad{0,64,512}.log`.

## METHOD

1. **Audit** — engine env `ARC_ALIGN_AUDIT=1` (default off) dumps, for every decode
   weight/view tensor at load: base address mod 16/64, superblock size (BBP), row
   stride mod 16/64/512, power-of-two check, and sub-block field offsets/alignments.
   Run on both Q4_K_S and IQ2_M.
2. **Microbench** — new `bench/align_probe.cpp` (built via `build.sh align_probe`),
   four sections on the MTL iGPU: raw read ceiling; block-read row streams
   (`gemv` pattern) across padding 0/64/128/256/512 B; per-lane vector-load streams
   (`gemv_v`) comparing GGUF unregrouped (2-B aligned) vs inferzero native (8-B) vs
   16-B-repacked superblocks; and the **production decode.cl kernels** (DP4a dot,
   600+ MiB footprint, T=1 and T=4).
3. **Engine padding experiment** — `ARC_PAD_ROWS=<bytes>` adds load-time row padding
   to weight row strides (default 0/off). Bit-exactness: `runz --dump-logits` sha256
   compared pad 0 vs 512 for Q4 and IQ2.
4. **Screen** (two-phase rule) — Q4_K_S raw, tg150, 1 repeat, prose+structured
   (+code/fiction screen), 30 s cooldown, pad ∈ {0, 64, 512}.

## NUMBERS

**Audit (Q4_K_S, 742 tensors / 20.29 GiB):** every base 4096-B aligned; **0/742**
mod-16 violations; one anomaly — `lm_head` Q6_K repacked 210→216 B BBP: odd
superblocks start at 8 mod 16, stride 4320 mod 64 = 32. Q4_K/Q5_K streams: qs fields
16-B aligned, strides 2880/9792 all mod-16 clean.
**Audit (IQ2_M, 681 tensors / 13.71 GiB):** 468/681 streams (9.85 GiB) have
BBP % 16 == 8 (odd superblocks at 8 mod 16 — the §8-q2 suspect), 396/681 row strides
mod-64 != 0, 0 power-of-two strides.

**Microbench on MTL (peak block-read 89.5 GB/s during the same session):**
- Block-read row streams vs padding (Q4_K [N×5120]): 90.4 / 90.6 / 91.3 / 91.0 GB/s
  at pad 0/64/256/512 — **flat**; every quant and shape in the table is flat ±2%
  across all paddings, including F16 8192-B power-of-two rows (87.9 → 92.3 at +256,
  noise). **No stride-aliasing signature exists on Xe-LPG/MTL.**
- Vector-load layout trio: GGUF unregrouped (2-B aligned) IQ2_S = **52.9 GB/s**;
  inferzero native 8-B regrouped = **85.3**; hypothetical 16-B repack = 78.7 (worse).
  Q6_K: ungrouped 91.3 / native 89.3 / 16-B repack 86.1. **inferzero's load-time
  regrouping already harvested the Strata 145→640-class win; further repacking hurts.**
- Production kernels: Q4_K_S gemv T=1 = 79.1–83.7 GB/s (at ~90% of ceiling);
  **IQ2_S gemv_v T=1 = 69.2–70.7 GB/s, T=4 = 66.0–67.2** — IQ dequant ALU caps the
  champion's streams ~15–20% below Q4's; Q6_K lm_head gemv_v = 86.8 (T=1) / 78.9 (T=4).

**Screen (Q4 raw, tg150, prose / structured, srv tok/s):** pad 0 = 3.03 / 3.10;
pad 64 = 3.13 / 3.15 (+2.4 / +1.6%, noise); pad 512 = 2.85 / 2.85 (**−7 / −8%**).
**Bit-exact:** pad 0 vs 512 logits sha256 identical (Q4: `4c0447e2…184cea` both runs).

## VERDICT

**NEGATIVE.** Weight-stream alignment/stride padding is a dead end on Meteor Lake:
the aliasing pathology measured on gfx1151 (gufo) and discrete Xe (Strata) does not
exist on Xe-LPG shared memory — block-read streams sit at 87–93 GB/s regardless of
stride, and 512 B padding actively costs ~7% end-to-end. §8-q2 answered: IQ2 sub-blocks
ARE 8-mod-16 aligned but it costs nothing measurable on this GPU; §8-q5: no tREF/stride
penalty visible at our strides. The 57.3 → 89.9 width gap is per-lane load width +
memory-level parallelism, not alignment — and inferzero's load-time regrouping
(52.9 → 85.3 GB/s vs unregrouped) already captured the layout win the siblings found.

Two forward leads this file hands to P4/follow-ups (not re-litigated here):
1. **IQ dequant ALU, not bandwidth, caps champion decode** — IQ2_S `gemv_v` streams
   at 66–71 GB/s vs Q4 `gemv` at 79–84; cheaper IQ2 block decode (or per-lane
   load-width change) is where IQ2's window time hides.
2. The P0 window roofline gap (1.33×) is therefore NOT layout — it is inside-kernel
   ALU + non-GEMM device work.

Code kept behind envs default-off (`ARC_ALIGN_AUDIT`, `ARC_PAD_ROWS`) + `bench/align_probe.cpp`
for future GPUs (B-series users may see the opposite signature — re-run the probe).
