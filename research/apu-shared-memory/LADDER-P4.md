# LADDER-P4 — post-merge ladder after the draft-vocab head (M6/P4)

Same-session ladder on the node (`~/inferzero-p4`, merged main code verified
`git diff p4-draft-lmhead main -- zero tools bench` empty), one pass, four
configs, full-confirm protocol: tg300, 3 repeats, 60 s cooldown, all four suite
prompts through `bench/speedbench.sh`, `--slots 2 --ctx 8192`. Logs:
`results/speedbench-*-ladder-p4{a,b,c,d}.log`.

## The ladder

| Config | Median tok/s (12 runs) | code | fiction | prose | structured | MTP accepted | pp median |
|---|---:|---:|---:|---:|---:|---:|---:|
| a: IQ2_M + MTP d3 (flag **off**) | 7.52 | 8.01 | 6.41 | 7.49 | 8.93 | 2229/3523 (63.3%) | 24.0 |
| b: IQ2_M + MTP d3 + `--mtp-draft-vocab 32768` | **8.19** | 8.66 | 7.05 | 7.77 | 9.73 | 2160/3627 (59.6%) | 26.6 |
| c: IQ3_M + MTP d3 (regression check) | 7.06 | 7.76 | 6.39 | 6.46 | 8.17 | 2265/3441 (65.8%) | 23.1 |
| d: Q4_K_S raw (regression check) | 3.14 | 3.14 | 3.14 | 3.14 | 3.14 | — (AR) | 12.2 |

**Paired a→b (the valid comparison — same session, interleaved protocol):**
**+8.9 % median** (code +8.1, fiction +10.0, prose +3.7, structured +9.0), at
−3.7 pp acceptance. No regressions elsewhere: IQ3 prose+structured median
(7.30) matches BASELINE full-pass 7.33; Q4 3.14 vs 3.17 (−0.9 %, flat across
prompts as a bandwidth-bound config must be). Session drift is real though:
this session's flag-off champion sits at 7.52 vs BASELINE's 7.96 (−5.5 %), and
IQ3's creative-prompt acceptance ran low this session (fiction 55.1 %) —
cross-session tok/s moves of 3–5 % are machine state, not code.

## Regime arithmetic (bytes the fabric no longer carries)

Convention as in BASELINE: window bytes ≈ model file read once (14.53 GB) plus
one `lm_head` read per MTP draft step beyond the target pass. `lm_head` rows are
Q6_K, 3,570 B: full head 248,320 rows = 0.886 GB; gathered 32,768-row head =
0.117 GB. Three draft steps: **a carries +2.66 GB a window, b carries +0.35 GB —
2.31 GB/window deleted.**

| | a (off) | b (32K) |
|---|---:|---:|
| tok/verify window (from acceptance) | 2.90 | 2.79 |
| window wall (tok/win ÷ tok/s) | 0.385 s | **0.340 s (−11.6 %)** |
| window bytes (GB) | 17.19 | 14.88 |
| effective weight BW (bytes ÷ window) | 44.7 GB/s | 43.8 GB/s |

The regime story is exactly what P0 predicted for an overhead/overdraft-bound
champion: **effective GB/s does not move** (still ~44, ~half the 89.9 iGPU
peak) — the window got faster because **bytes left the window**, not because
any stream got more efficient. Draft GPU ms per window 35.4→12.6 and draft
`lm_head` 11.5→1.5–2.2 ms/step per `P4-draft-lmhead.md` (GPU_BUSY CSV).
Q4 stays the bandwidth-bound reference: 22.72 GB/window at 3.14 tok/s →
72.0 GB/s (80 % of peak), untouched by this change as expected.

## Notes

- The paired +8.9 % here is larger than the child session's +2.7 % confirm; both
  are ≥0 across 24 paired runs, and the direction is identical per prompt.
  Machine state at 300-token depths moves tok/s; acceptance ratios did not.
- fiction anomaly, kept honest: acceptance DROPS 52.0→47.6 % with the flag yet
  fiction gains the most (+10 %). Shorter draft heads make failed windows
  cheaper faster than they cost acceptances on creative text. The ranking-list
  tail (not row count N) is the follow-on lever, per P4 file.

## Standing recommendation

`--mtp-draft-vocab 32768` (needs `scripts/draft-vocab/vocab-32768.txt` gathered
at load; 115 MB) is the champion candidate pending LADDER-FINAL; default stays
off until then.
