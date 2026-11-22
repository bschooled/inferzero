# M4 — P2: Priced Adaptive MTP Draft Depth (Screen, Honest Negative)

**Date:** 2026-11-21
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16 — Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz, 32 GB shared LPDDR5x-7467, driver 26.40.40196, kernel 7.3.5-2-cachyos)
**Model:** `Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf` (IQ2_M, 96 layers, 13.53 GiB)
**Branch:** `p2-adaptive-depth` (from main `2e76bee`), measured tree `~/inferzero-p2`
**Flag:** `--mtp-adaptive` (runz and serverz) / `ARC_MTP_ADAPTIVE=1`; **default off** = the static `--mtp-drafts` path, verified bit-identical against main
**Harness:** `bench/speedbench.sh` driving `serverz` (greedy, `temperature: 0.0`, `cache_prompt: false`, `--slots 1 --ctx 8192 --no-prefix-cache --session-cache-gb 0`)
**Raw logs & CSV:** `results/p2-speedbench-adaptive-screen.log`, `results/p2-speedbench-static-screen.log`, `results/p2-adaptive-screen.csv`

---

## METHOD

### Implementation (what was measured)

Draft depth became a per-window decision inside `Engine::mtp_draft` (`zero/engine_drafters.h`), modelled on gufo's
`MtpLengthController` (`APPENDIX-gufo.md` §2.4) but priced from this engine's own timings:

- **Acceptance, direct:** greedy verification makes the accepted prefix independent of how many drafts were
  requested, so for every depth `d` the controller keeps an EWMA (decay 0.9) of `min(accepted, d)` over the
  windows that drafted at least `d` — `e[d]`. No conditional-independence assumption and no estimator for a
  censored chain. Seeds `e[d] = 0.7·d` (P0's ~2.0 accepted drafts at d3).
- **Cost, measured:** `verify_ms(T) = a + b·T` fitted online (decayed least squares, P0's `T=1 252.74 ms`,
  `T=4 301.39 ms` as anchoring priors) per context band (`≤512`, `≤2048`, `≤8192`, `>8192`); `draft_step_ms` =
  EMA of the draft phase's wall time per draft step per band. Timings are steady clocks around `mtp_draft` /
  `verify`, not GPU events (no profiling overhead).
- **Decision:** `EV(d) = (1 + e[d]) / (verify_ms(1) + d·(draft_step_ms + verify_row_ms))`, `EV(0) = 1/verify_ms(1)`.
  The depth with the largest EV wins; 1.5 % hysteresis. `--mtp-drafts N` with `--mtp-adaptive` caps the depth
  (default ceiling 7).
- **gufo's failed-depth mask, priced:** three consecutive rejections at index `d` *and* a marginal
  (`e[d] − e[d−1]`) below 0.20 (under the 0.24–0.28 break-even at 7–8 tok/s) masks the depth; it is retried every
  16 windows. After a fully accepted (censored) window and ≥16 windows since the last probe, one deeper depth is
  tried. `EV(0)` winning starts 12 pure-AR windows, then one probe draft (implemented in runz's loop and serverz's
  single-slot generation; `--slots > 1` keeps the static cap).
- **Audit:** `ARC_MTP_ADAPTIVE_CSV=path` writes one row a window (`window,band,pick,drafted,accepted,vrows,
  draft_ms,verify_ms,dstep,v1,vrow,e1..e7`); a summary prints when the engine is released.

### Measurement protocol

The **two-phase rule**: screen first; confirm only if the screen showed adaptive **≥ +2 % average** or a large
win on the low-acceptance prompts. The screen is 1× `tg150` per prompt, 30 s cool, all four suite prompts
(`prose`, `structured`, `fiction`, `code`), static d3 then adaptive (separate `serverz` runs, one model load
each). No stray `serverz`/`runz` before or after either run (`pgrep -x`). **No confirm pass was run** — the
screen did not meet the rule (see below), which the brief allows as an honest negative.

Commands (node, `~/inferzero-p2`):

```
SPEEDBENCH_DATE=p2-adaptive bash bench/speedbench.sh <IQ2_M> --mtp --mtp-adaptive \
  --two-phase --full-prompts prose,structured,fiction --repeats 1 --max-tokens 150 --cooldown 30 \
  --screen-repeats 1 --screen-tokens 150 --screen-cooldown 30 --out-dir ~/p2-results
SPEEDBENCH_DATE=p2-static   bash bench/speedbench.sh <IQ2_M> --mtp --mtp-drafts 3 ... (same parameters)
```

---

## NUMBERS

### 1. Screen — per prompt (server-reported t/s, 1× tg150, 30 s cool)

| Prompt | Static d3 (t/s) | Static MTP accepted | Adaptive (t/s) | Adaptive MTP accepted | Δ |
|---|---:|---:|---:|---:|---:|
| `fiction` | **6.97** | 86/153 (56.2 %) | **6.73** | 86/154 (55.8 %) | **−3.4 %** |
| `prose` | **8.09** | 95/137 (69.3 %) | **7.98** | 99/157 (63.1 %) | **−1.4 %** |
| `structured` | **8.79** | 101/141 (71.6 %) | **8.78** | 113/192 (58.9 %) | **−0.1 %** |
| `code` | **8.66** | 99/131 (75.6 %) | **8.84** | 111/175 (63.4 %) | **+2.1 %** |
| **3-prompt full-pass median** (fiction, prose, structured) | **8.09** | 282/431 (65.4 %) | **7.98** | 298/503 (59.2 %) | **−1.4 %** |
| **All 4 runs median** | **8.38** | 381/562 (67.8 %) | **8.38** | 409/678 (60.3 %) | **0.0 %** |

`fiction` ran first in both configs, so both carry the first-request warm-up dip (6.97 of a 3-repeat median that
is 6.76 in `BASELINE.md`); the delta between the two runs is still the comparison.

### 2. Depth the controller chose (per request, from the CSV)

| Request | Chosen depths (windows) | Mean drafts / window | Windows |
|---|---|---:|---:|
| `fiction` | d3:60, d4:3, d1:1 (tail cap) | 2.45 | 64 |
| `prose` | d4:35, d3:7, d5:8, d2:1 | 3.0 | 51 |
| `structured` | d5:10, d6:19, d7:8 | 4.7 | 37 |
| `code` | d7:32, d6:2, d5:1, d2:1, d1:1 | 4.5 | 37 |

The controller learns across requests (the server keeps one engine): after prose it pushed `structured`/`code`
(acceptance > 60 %) to d5–d7 and kept `fiction` (56 %) at d3. It never chose d2 for fiction: its learned
`e[2] ≈ 1.3` vs `e[3] ≈ 1.6` made d3 worth ~3 % more than d2 at the measured 36–38 ms per extra draft.

### 3. Window economics per chosen depth (CSV, all prompts; `draft_ms + verify_ms`, no `verify_done`)

| Chosen depth | Windows | Mean accepted | tokens/win | draft+verify ms/win | implied tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.50 | 1.50 | 296.6 | 5.06 |
| 2 | 2 | 0.50 | 1.50 | 309.0 | 4.85 |
| 3 | 67 | 1.37 | 2.37 | 343.7 | 6.91 |
| 4 | 38 | 2.00 | 3.00 | 367.9 | 8.16 |
| 5 | 19 | 2.84 | 3.84 | 408.1 | 9.41 |
| 6 | 21 | 2.43 | 3.43 | 466.6 | 7.35 |
| 7 | 40 | 3.35 | 4.35 | 447.6 | 9.72 |

Deeper windows do produce more tokens per window, and at high acceptance the marginal token roughly pays for
its ~36–38 ms — which is exactly why `structured`/`code` moved to d5–d7 with **no net gain** (structured 8.79 →
8.78): `tokens/pass` +32 % and `window wall` +31 % cancel. The learned band-0 costs at the end of the run:
`dstep 22.7 ms`, `v1 242.6 ms`, `vrow 22.4 ms` (vrow drifts up with context; at `pos≈500` it was 22 ms).
All four screen prompts stayed in context band 0 (`pos ≤ 490`); bands 1–3 kept the P0-seeded costs and are
untested here.

### 4. Controls

- **Default-off bit-identity:** `runz --mtp --mtp-drafts 3 --n 48 --dump-logits "Explain quicksort in one
  paragraph."` on main's `2e76bee` build vs this branch's: **sha256 `935e2e4f40352cb6d550196cb8cc0494…d80ac1`
  on both** (`results/p2-bx-main.log`, `results/p2-bx-branch.log`).
- **Same tokens as static:** with `--mtp-adaptive`, the 48-token greedy ids were identical to static d3 on the
  quicksort prompt (`results/p2-runz-smoke-adaptive.log` vs `p2-runz-smoke-static.log`); speculative verification
  keeps the text plain's.
- **Auditability:** the CSV confirmed the picked depth, drafted/accepted counts, per-window draft/verify ms and
  the fitted costs for every window (189 rows for the screen); the 96-token probe run shows the censoring probe
  trying d4 and then choosing it (d4 in 4 of 37 windows, two fully accepted; `results/p2-runz-probe.csv`).

---

## VERDICT

**NEGATIVE — adaptive ≈ static.** On the screen the 3-prompt full-pass median is **7.98 vs 8.09 tok/s (−1.4 %)**:
`fiction −3.4 %`, `prose −1.4 %`, `structured −0.1 %`, `code +2.1 %`; the all-run median is equal (8.38). The
screen did not reach the +2 % average (or a low-acceptance-prompt win) that the two-phase rule requires, so the
3× tg300 confirm pass was **not** run.

The mechanism works and is auditable — no crashes, the same text, a default-off path bit-identical to main — but
at this model's acceptance profile (56–76 % a draft) the static d3 is already near the cost optimum. The
controller's value is in the *data*, which is the P4 input:

- `e[d]` at d3 ends ~1.6–2.2 accepted drafts (fiction ~1.6, structured/code ~2.2–2.3);
- the marginal draft costs ~36–38 ms (22 ms draft step + ~14–22 ms verification row) and pays for itself only
  above ~0.24–0.28 acceptance — i.e. **structured/code can profitably run deep (d5–d7), prose/fiction cannot**;
- at d5–d7 the extra tokens and the extra window time cancel on `structured`, so “deeper is free” is false there;
- fiction never wanted d2 with this estimator: `e[3]−e[2]` stayed above the break-even.

**Flag:** `--mtp-adaptive` / `ARC_MTP_ADAPTIVE=1` (serverz + runz), default off; `ARC_MTP_ADAPTIVE_CSV=<path>`
for the per-window audit. Reusable per-depth acceptance and cost tables are in
`results/p2-adaptive-screen.csv`.

### For P4

1. Price **verification rows non-linearly** (or include `verify_done` replay) before trusting an EV that sends
   prompts to d6–7; the linear `a+bT` fit under-prices deep windows by ~10–15 %.
2. The per-depth `e[d]` and the observed `dstep/v1/vrow` should seed P4's cost table instead of re-learning them.
3. The most promising target is **not** shallower drafting on fiction (d3 is already optimal) but the
   **draft-phase `lm_head` cost** (34.5 ms/window in P0) and reduced-vocabulary drafting, which this data says
   would move `structured`/`code` (where d5–d7 is chosen) as much as the depth itself.
