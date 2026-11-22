# M4 — P2: priced adaptive draft depth (replaces static --mtp-drafts 3)

Goal: make draft depth a runtime decision. Today `--mtp-drafts 3` is static; the data
says it is wrong half the time: M1 measured acceptance 74% on structured/code but
**56% on fiction** at 7.96 tok/s average, and P0 showed the draft phase costs
13.7–14.7% of window GPU time with `lm_head` alone ~34.5 ms/window (every draft step
re-reads all 248,320 vocab rows). Gufo ships exactly this control (`mtp_costs.hpp`
per-context×width ms table + `mtp_policy.hpp` cost-aware depth controller with
failed-depth mask). Implement a native equivalent behind a flag.

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Branch
  `p2-adaptive-depth` from current main (88fc5c8 or newer). **Do not push, do not
  touch gh auth, do not merge.** Leave clone on the branch.
- Node: `ssh bschooley@192.168.50.48` with `bash -lc '...'`. Rsync branch to
  `~/inferzero-p2`, `./build.sh runz serverz`. Never touch `~/inferzero`, `~/models`,
  or other experiment trees. GPU hygiene: `pgrep -x serverz; pgrep -x runz` empty
  before/after; 60 s cool between full runs, 30 s between screens.
- Read first: `research/apu-shared-memory/REPORT.md` §6.3, `APPENDIX-gufo.md`
  (mtp_costs / mtp_policy sections), `BASELINE.md` (per-prompt acceptance),
  `P0-gpu-busy.md` (draft phase timing, `INFERZERO_GPU_BUSY` CSV format — **reuse
  that instrumentation to price costs**), `zero/engine_drafters.h` (`mtp_draft`,
  verify/verify_done, existing `opt.copy_draft`).

## Implementation

1. **Costing**: with `INFERZERO_GPU_BUSY=1`-style timestamps (already in main),
   measure per-window: draft ms per step, verify ms at T=1..kMaxDrafts, across ctx
   bands (≤512, ≤2048, ≤8192) for the running quant. Either calibrate online (first
   windows per band) or price a compact table at startup — cheap and transparent wins.
2. **Controller** per window: choose d maximizing expected tok/s =
   (1 + ê(d)) / (draft_ms(d) + verify_ms(d+1) − verify_ms(0)), where ê(d) is an EMA of
   observed acceptance at candidate depth d. Add gufo's failed-depth mask (depth
   rejected 3 consecutive times → cap below it until context/acceptance changes),
   and AR backoff (if even d=1 windows go negative EV, run pure AR briefly).
   Small hysteresis to avoid per-window thrashing.
3. **Flag**: `--mtp-adaptive` option (runz + serverz) / env `ARC_MTP_ADAPTIVE=1`;
   **default off** = current static path untouched (bit-identical). Log chosen depth
   per window (cheap stderr or CSV) so acceptance/cost data is auditable.

## Measurement (two-phase rule — and fast)

- Screen first (tg150, 1 repeat, 30 s cool): prose, structured, fiction — static d3
  vs adaptive, same server session if possible (separate serverz runs are fine).
- Confirm if screen shows adaptive ≥ +2% average or big win on low-acceptance
  prompts: full pass (3× tg300, 60 s cool) on prose + structured + fiction for BOTH
  static d3 and adaptive; medians; report tok/s per prompt + depth distribution +
  overall acceptance. Success: adaptive ≥ static d3 on the full-pass average, no
  regression >2% on structured; expect fiction/prose gains.
- Honest NEGATIVE (adaptive ≈ static) is fine — the cost table + acceptance data
  becomes input for P4; say so and stop.
- Ladder note: if adaptive wins, ALSO run the IQ2_M full-pass ladder row with
  adaptive (that is the future champion config) but do NOT flip defaults —
  propose in VERDICT, parent decides.

## Files you own

- `zero/engine_drafters.h` (+ engine/options plumbing, runz.cpp / server chat option),
  all additive, default-off.
- `research/apu-shared-memory/P2-adaptive-mtp.md` METHOD/NUMBERS/VERDICT (POSITIVE
  + `Flag: <name>` line if merged working; honest NEGATIVE otherwise).
- CSVs/logs force-added under `research/apu-shared-memory/results/p2-*`.

## Completion condition (stop after this)

Branch `p2-adaptive-depth`, 1–4 commits. Report: per-prompt static vs adaptive
screens + confirm if run, depth distribution, VERDICT line. Do not push. Do not
start P3/P4.
