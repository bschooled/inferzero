# LADDER-FINAL — the champion ladder with regime arithmetic, end of campaign

Campaign: REPORT.md §6 queue P0-P5 on the MTL node (fork main `7420c82`).
Protocol: `bench/speedbench.sh` full confirm passes (tg300, 3 repeats, medians,
60 s cool, prompts code/fiction/prose/structured). Session drift on this laptop
is ±3-5 %; the only valid comparisons are same-session paired runs — marked
"paired" below. Baseline numbers from `BASELINE.md`; post-P4 ladder from
`LADDER-P4.md`.

## The ladder

| config | BASELINE (Nov21 a.m.) | post-merge ladder (Nov21 p.m.) | window bytes (est) | window wall | effective GB/s | regime |
|---|---:|---:|---:|---:|---:|---|
| IQ2_M + MTP d3 (all flags off) | **7.96** | 7.52 | 17.19 GB | 0.386 s | ~44.5 | overhead-bound |
| **IQ2_M + MTP d3 + `--mtp-draft-vocab 32768`** | — | **8.19** (paired **+8.9 %** vs same-session 7.52) | 14.88 GB | 0.341 s | ~43.7 | overhead-bound |
| IQ3_M + MTP d3 | 7.33 | 7.06 (prose+structured pair 7.30 vs 7.33: no regression) | 20.97 GB | 0.421 s | ~49.8 | overhead-bound |
| Q4_K_S raw (no MTP) | 3.17 | 3.14 | 22.72 GB | 0.318 s | ~71.3 | BW-bound, 79 % of iGPU peak |

Byte convention: one model-file read/window + one output-head read per draft
step (full Q6_K head 0.886 GB; gathered 32,768-row head 0.117 GB).

## What each merged change moved

| change | merged | effect on the ladder | regime statement |
|---|---|---|---|
| P0 GPU-busy instrumentation | yes (`INFERZERO_GPU_BUSY`) | none (profiler; +4.7 % only while on) | proved GPU 87-97 % busy: no host idle to reclaim |
| P1 alignment/stride padding | NEGATIVE, no flag needed | none | streams 87-93 GB/s flat vs pad 0-512 B; layout win already in native regroup |
| P2 priced adaptive depth | merged, default off | none selected (−1.4 % screen) | d3 is the cost optimum at 56-76 % acceptance — depth is not the lever |
| P3 copy-draft union | merged, default off | none (−0.5 %) | proposals too rare (3.7-26.9 % of windows) |
| **P4 reduced-vocab draft head** | **merged, opt-in** | **+2.7 % (child session) to +8.9 % (paired ladder)** on the champion; IQ3/Q4 untouched | 2.31 GB/window deleted from the fabric at flat ~44 GB/s — pure byte deletion in the overhead-bound regime |
| P5 DPAS prefill T>=256 | NEGATIVE (flag routes nothing) | none | Xe-LPG has no matrix path (`intel_sub_group_i8_i8_matrix_mad_k32` does not compile); prefill stays DP4a/gemv |

pp numbers (same ladder session): pp medians 24.0 (IQ2) / 23.1 (IQ3) / 12.2
(Q4) t/s in-server; `runz --bench-pp`: IQ2 69.8/73.5/71.8 and Q4 47.6/62.1/71.9
at pp 512/1024/2048 — DPAS flag moved them by +0.1 %.

## Final state

Champion (recommended): **IQ2_M + `--mtp --mtp-drafts 3 --mtp-draft-vocab
32768`** — 8.19 tok/s in the paired ladder session; expect ~8.0-8.7 across
sessions. Default champion (no flags) unchanged at ~7.5-8.0. The docs rule is
satisfied: the single merged speed change (P4) has its `docs/before-after.md`
row and CHANGELOG entry; everything else is a documented NEGATIVE with its own
METHOD/NUMBERS/VERDICT file.

What the campaign left as open leads (in the P-files, unranked): draft-vocab
ranking-list tail quality (acceptance recovery, esp. prose/creative); IQ2/IQ3
dequant ALU cap (66-71 GB/s vs 87-93 GB/s raw block-read); prefill on
B-series/XMX hardware where a matrix engine exists (P5 probe code is reusable
there).
