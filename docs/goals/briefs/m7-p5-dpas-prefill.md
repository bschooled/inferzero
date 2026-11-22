# M7 — P5: prefill DPAS retest at T>=256 tiles only

Goal: prefill. The generation side of the queue is done; pp is the last ranked
item (§6 P5): today pp is ~70–77 t/s, and the sibling engines get prefill
x1.6–2.0 from matrix-core GEMMs on APUs (strata docs). An earlier inferzero DPAS
attempt LOST — but it was decode-shaped (small-T, GEMV-ish tiles). Retest only
where tiles are actually wide: **T>=256 token tiles in the prompt GEMMs**.

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Branch
  `p5-dpas-prefill` from current main (4656d99 or newer). **Do not push, no gh
  auth, no merge.** Leave clone on the branch.
- Node: ssh bschooley@192.168.50.48, `bash -lc '...'`. Work tree `~/inferzero-p5`
  (rsync your branch, `./build.sh runz serverz`). Never touch other ~/inferzero-*
  or ~/models. **The parent is running a GPU ladder in `~/inferzero-p4` right
  now (~45 min): before EVERY GPU measurement loop until
  `pgrep -x serverz; pgrep -x runz` is empty (check every 120 s; builds and code
  reading meanwhile). One GPU user at a time, always.**
- Read first: `REPORT.md` §6 item 5 + §1; `docs/lessons-intel-arc-igpu.md` and
  `docs/before-after.md` for the EARLIER DPAS attempt and why it lost
  (find its numbers); `zero/kernels/gemm.cl` (DP4a 16-lane), prompt-path GEMM
  dispatch in `zero/engine.h` / `engine_setup.h` (where T decides the kernel),
  `zero/xe2/engine_xe2.h` only as pattern reference (discrete cards; MTL must
  stay bit-identical when their code is added — AGENTS.md rule).

## Tasks

1. Find the prior DPAS-attempt numbers and the current prompt-GEMM kernel choice
   rule. Write a 5-line summary in the research file of what lost before and why
   tiles now differ.
2. Behind `ARC_PROMPT_DPAS=1` (default off): route prompt GEMMs to the DP4a/DPAS
   path ONLY for tiles with T>=256 (make the threshold env-tunable, default 256).
   If the old attempt already exists in-tree behind a dead switch, revive and
   constrain it rather than writing new kernels.
3. Measure `runz --bench-pp` at pp 512 / 1024 / 2048 (>=3 repeats, medians,
   60 s cool) on IQ2_M and Q4_K_S, default vs flag. Then one speedbench full
   pass (prose,structured, 3x tg300) with the flag to confirm end-to-end
   (pp is ~10% of these runs — pp-only gain is fine, state the e2e delta too).

## Gates

- Default off: `runz --dump-logits` sha256 identical to main for IQ2_M.
- Flag ON changes GEMM math order, so ON-output MAY differ: run `runz --kl-ref`
  if the repo's reference recordings cover this model (check `runz --help`);
  if not available at 40B on this node, record a flag-on vs flag-off logits
  diff at prompt end + greedy sample and state the gap plainly. W3 startup
  self-check must stay green.
- iGPU path bit-identical with flag off (AGENTS.md).

## Success / VERDICT

- POSITIVE: pp >= +30% at any tested length (target 70 -> 120+) with the flag
  merged; `Flag:` line + `docs/before-after.md` row + CHANGELOG entry.
- Otherwise honest NEGATIVE with the T-threshold sweep numbers (try T=128 and
  T=512 if 256 is a bust — one sweep, not five).

## Files you own

Whatever kernel/dispatch files the switch needs under `zero/`,
`research/apu-shared-memory/P5-dpas-prefill.md` (METHOD/NUMBERS/VERDICT),
results force-added `results/p5-*`, `docs/before-after.md` + `CHANGELOG.md`
(POSITIVE only).

## Completion condition (stop after this)

Branch `p5-dpas-prefill`, 1–4 commits. Report: pp table per length/config, e2e
delta, VERDICT line. Do not push. Do not touch drafting code.
