# NEXT ITERATION — the shared-memory (Spectre/MTL) campaign handoff

Iteration 1 (goal queue P0-P5, REPORT.md §6) closed 2026-11-22, reviewed pass,
fork `bschooled/inferzero` main `4c2a0df`. This file is the entry point for
iteration 2: what is settled, what to run next, and how to re-run anything.

## Settled (one line each; full METHOD/NUMBERS/VERDICT in the linked file)

| item | verdict | moved the ladder? |
|---|---|---|
| P0 GPU-busy instrumentation ([P0](P0-gpu-busy.md)) | GPU 87-97 % busy in all decode phases; host gap 13 % | no — but it killed the "hide host latency" branch and pointed at on-device bytes |
| P1 alignment / stride padding ([P1](P1-alignment.md)) | NEGATIVE — flat 87-93 GB/s vs pad 0-512 B | no |
| P2 priced adaptive draft depth ([P2](P2-adaptive-mtp.md)) | NEGATIVE — d3 is the cost optimum at 56-76 % acceptance | merged `--mtp-adaptive`, off |
| P3 copy-draft union ([P3](P3-copy-draft.md)) | NEGATIVE — proposals too rare (3.7-26.9 % of windows) | merged `--mtp-copy-union`, off |
| **P4 reduced-vocab draft head** ([P4](P4-draft-lmhead.md), [LADDER-P4](LADDER-P4.md)) | **POSITIVE** — draft GPU −64 %, window −8 % | **yes: 7.52 -> 8.19 paired (+8.9 %; session range +2.7..+8.9)** |
| P5 DPAS prefill T>=256 ([P5](P5-dpas-prefill.md)) | NEGATIVE — Xe-LPG has no matrix path (`-DXMX` does not even compile) | no |

Champion line: `serverz --mtp --mtp-drafts 3 --mtp-draft-vocab 32768`
(`champion-vocab.sh`). Quality gates: [QUALITY-GATES](QUALITY-GATES.md) —
everything default-off is bit-identical to the pre-campaign engine
(four-way sha `c1f52324…` on the final tree). Final ladder + regime
arithmetic: [LADDER-FINAL](LADDER-FINAL.md).

## Iteration-2 queue (ranked, with the evidence that ranks it)

1. **M-A — draft-vocab ranking-list tail quality.** P4 left acceptance at
   −3.7 pp (63.3 -> 59.6 %) and prose/creative is where it leaks (fiction
   acceptance 47.6 % with the flag); upstream D1 saw Chinese −15 % from the
   same tail. Regenerate/re-rank `scripts/draft-vocab/` (corpus already in
   repo) toward the node's own generations, and/or a context-aware candidate
   set (top-N global ∪ recent-context tokens). Cheapest remaining tok/s on
   the champion: acceptance recovery at zero extra bandwidth.
2. **M-B — IQ dequant ALU cap.** decode GEMVs on IQ2/IQ3 stream at 66-71 GB/s
   while raw block reads do 87-93 GB/s on the same fabric (P1 §forward-leads,
   P0 roofline 1.40-1.47x): ~20 % of the IQ-class window is dequant ALU, not
   bytes. Kernel work in the IQ GEMV paths; Q4-class stays the BW-bound
   reference. Biggest structural ceiling left on this box.
3. **M-C — XMX prefill on cards that have it (hand-off, not this node).** The
   P5 probe (`ARC_PROMPT_DPAS`, wide-tile gate) is merged and inert on MTL;
   strata/gufo measure prefill x1.6-2.0 from matrix cores on APUs that have
   XMX. Ask B-series owners (issue #4 / #45 style) to run the gate +
   `xw_peak`/`--bench-pp` sweep there.
4. **M-D — compose the merged flags.** `--mtp-adaptive` x `--mtp-draft-vocab`
   interaction never measured (adaptive skips cost observation on copy-used
   windows; its linear verify-cost model lacks the verify_done term the P4
   CSVs now give). Bands 1-3 of the cost table also untested (long contexts).

Skip list for MTL (measured here — do not re-litigate): stride/alignment
padding (P1), draft-depth pricing (P2), copy-draft union (P3), DPAS/matrix
prefill (P5). Plus the siblings' negatives inherited in REPORT.md §6 (NT
loads, naive overlap, GRF=256 on BF16, NPU).

## How to re-run anything (all committed)

- Ladder (the protocol this campaign used): `bash bench/speedbench.sh
  <model.gguf> --mtp --mtp-drafts 3 [--mtp-draft-vocab 32768] --full
  --repeats 3 --cooldown 60 --max-tokens 300`; screens: `--screen-*` options;
  the exact four-config post-merge runner: `bench/ladder.sh`.
- GPU-busy phases: `INFERZERO_GPU_BUSY=1` + `bench/gpu_busy.sh`
  (+ `tools/gpu_busy_sampler.cpp` external PMU cross-check).
- Prefill: `runz --bench-pp N` (bounded `--n`; never `--n 0`).
- Acceptance/launch audit: `--mtp-adaptive` CSV, per-request copy stats,
  `ARC_LAUNCH_STATS`.
- Criteria re-check: `bash research/apu-shared-memory/checks/check.sh all`.
- Raw evidence: 78 logs in `results/`, force-added; every report names its
  log files.

## Discipline (learned the hard way, keep it)

- Session drift on this laptop is ±3-5 %: **only same-session paired runs
  count**; alternating A/B beats A-block-then-B-block; medians over >=3 for
  confirms, 1x tg150 screens for screening.
- One GPU user at a time; `pgrep -ax 'runz|serverz'` empty before and after
  every run; no unbounded bench flags; a crashed child left a bench looping
  for 11 h once — bound everything.
- 60 s cooldown between confirm runs (clocks sag), 30 s is enough for screens.
- Node: `ssh bschooley@192.168.50.48`, one `~/inferzero-<tag>` tree per
  experiment tree, never touch `~/models` or other iterations' trees.
