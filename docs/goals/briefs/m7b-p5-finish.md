# M7b — finish P5 (DPAS prefill at T>=256) — continuation after a provider crash

The M7 child crashed mid-run (provider outage) with the ENGINE WORK DONE and
partial numbers only in its dead transcript. Do not redo the engine. Finish the
measurement protocol, write the verdict, commit.

## Already true (verified by the parent)

- Branch `p5-dpas-prefill` (checked out in `/home/bschooley/local-dev/inferzero`):
  `9551dbb` = `ARC_PROMPT_DPAS` engine change (default off, wide prompt tiles),
  `358421b` = unrelated LADDER-P4.md (leave it). Work continues ON THIS BRANCH.
- `research/apu-shared-memory/P5-dpas-prefill.md` exists UNTRACKED with the
  dead child's WIP notes — read it first; it already contains (from its runs):
  A-default IQ2_M pp1024 = 74.5/73.5/68.5 (median 73.5 t/s),
  A-default Q4_K_S pp2048 = 72.1/71.4/71.9 (median 71.9 t/s).
- The crashed child left a stuck `runz --bench-pp 2048 --n 0` on the node for
  11 h; the parent killed it. Node is CLEAN now (no serverz/runz).

## Hygiene (non-negotiable, this is why you're a short brief)

- Every `runz --bench-pp` runs in the FOREGROUND with a bounded timeout sized to
  the run (~40 s per pp2048 run; give 120 s per call, never `--n 0`, never a
  sleep-poll loop over a background bench). If you must background, KILL your
  processes before you finish and prove it with `pgrep -ax runz; pgrep -ax
  serverz` returning empty.
- Node: ssh bschooley@192.168.50.48 `bash -lc '...'`; tree `~/inferzero-p5`
  (re-rsync branch code, `./build.sh runz serverz` if code changed). Never touch
  other ~/inferzero-* or ~/models. 60 s cooldown between bench runs.

## Protocol (complete it)

1. Build/verify flag: `ARC_PROMPT_DPAS=1` routes prompt GEMMs to DPAS only for
   tile T>=256 (threshold env-tunable if you exposed it). Confirm default-off
   `runz --dump-logits` sha256 on IQ2_M equals the same run from branch base
   (report both shas).
2. **B-set**: pp 512, 1024, 2048 on IQ2_M and Q4_K_S with the flag ON, 3 runs
   each, medians (that is 18 bench runs; each ~15-40 s; total well under an
   hour). Cooldown 60 s between runs.
3. Compare A vs B per length/config (A-default for IQ2 pp512/Q4 pp512/pp1024
   and IQ2 pp2048 you still need too if the WIP file lacks them — fill gaps
   same-session where possible).
4. Gate: POSITIVE needs pp >= +30% at some length (target 70 -> 120+). If B
   loses or ties: one T-threshold probe (ARC_PROMPT_DPAS_T=128 and =512 on the
   losing length, 1 run each) then honest NEGATIVE.
5. If POSITIVE: one speedbench full pass (prose,structured, 3x tg300) with the
   flag for the e2e number; `Flag:` line; `docs/before-after.md` row +
   CHANGELOG entry; W3 self-check green note.

## Files you own

`research/apu-shared-memory/P5-dpas-prefill.md` (finish it: METHOD/NUMBERS/
VERDICT), results force-added `results/p5-*`, `docs/before-after.md` +
`CHANGELOG.md` (POSITIVE only), engine files ONLY if a threshold bug blocks
measurement.

## Completion condition

1-3 commits on `p5-dpas-prefill`; report A/B table + VERDICT line + final node
`pgrep` proof. Do not push. Do not touch drafting code or the ladder.
