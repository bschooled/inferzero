# M5 — P3: copy drafting measured, and a union window (screen, honest NEGATIVE)

**Date:** 2026-11-21
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16 — Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz,
32 GB shared LPDDR5x-7467, driver 26.40.40196, kernel 7.3.5-2-cachyos)
**Model:** `Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf` (IQ2_M, 96 layers, 13.53 GiB)
**Branch:** `p3-copy-draft` (from main `43802b9`), measured tree `~/inferzero-p3`
**Flags:** `--copy-draft` (R1, unchanged) and **`--mtp-copy-union`** (new, default off);
`ARC_COPY_UNION_MIN` (shortest copy key, default 3 tokens), `ARC_COPY_UNION_MAX` (most copy proposals per window,
default 4)
**Harness:** `bench/speedbench.sh` driving `serverz` (greedy, `temperature: 0.0`, `cache_prompt: false`,
`--slots 1 --ctx 8192 --no-prefix-cache --session-cache-gb 0`)
**Raw logs:** `results/p3-screen-{static,copy,union}.log`, `results/p3-bitexact.log`, `results/p3-smoke-union.log`

---

## METHOD

### What was already there (R1) and what was measured

`Engine::mtp_draft` (zero/engine_drafters.h) already had copy drafting: with `--copy-draft`, a decode window whose
last 8…4 tokens (longest first, `ARC_COPY_NGRAM` / `ARC_COPY_MIN`) appeared earlier in the context drafts the tokens
that followed there and returns them **instead of** asking the MTP block. The verifier treats them as ordinary drafts,
so the text cannot change. Nobody had measured whether it fires on our four suite prompts.

### Union mode (the missing piece)

`--mtp-copy-union` (default off; MTP-block path only — a DFlash2 or Gemma drafter keeps the copy-only behaviour):

1. the copy lookup runs with keys from `ARC_COPY_UNION_MIN` (default **3** tokens, longest first up to
   `ARC_COPY_NGRAM`, default 8) and returns at most `min(--mtp-drafts, ARC_COPY_UNION_MAX)` proposals
   (default cap **4**);
2. those proposals open the verify window; the MTP chain then drafts `--mtp-drafts − <copy proposals>` more tokens, so
   the window is `[copy… , MTP…]` and never longer than the static window;
3. if the copy proposals fill the window, no MTP draft is made, but the block's KV rows of the pending accepted
   drafts and `tok` are still written (as the copy-only path does), so the block stays in step;
4. `verify()` / `verify_done()` are **untouched** — drafts are proposals either way.

`copy_used_` is set when a window starts with copy proposals, so the P2 adaptive controller (not required to compose
here) skips that window's MTP cost observation rather than mixing two draft costs.

### Stats added

- `serverz`, per request, only when copy drafting is on: `| copy X of Y windows, A/B proposals accepted`
  (`Y` = drafted windows, `B` = copy proposals, `A` = the proposals the model accepted). Parsed from the existing
  `[chatcmpl-arcN]` line in the speedbench logs.
- The engine's `copy_report()` at release: proposals per drafted window and the union count (runz prints the same for
  its run).
- The existing `INFERZERO_GPU_BUSY_CSV` already records `copy_draft` per window; unchanged.

### Measurement protocol

Screen per the brief: **1× tg150, 30 s cool, all four prompts** (`code`, `fiction`, `prose`, `structured`, the
script's alphabetical order — the same order in every mode), one `serverz` run per mode (one model load each).
Three modes: **static d3** (`--mtp --mtp-drafts 3`), **copy-only** (`+ --copy-draft`), **union**
(`+ --copy-draft --mtp-copy-union`). `pgrep -x serverz` / `runz` empty before and after every run; the W3 start-up
self-check printed no warning (no old/bad driver message).

The two-phase rule: a confirm pass (3× tg300, 60 s cool, prose + structured + fiction) only if a mode wins **≥ +5 %
average or ≥ +10 % on structured/code with ≤ 2 % prose regression**. **No confirm pass was run** — the screen did not
meet the rule (see below), which the brief allows as an honest negative. Single 150-token runs carry run-to-run noise
of roughly ±3 % on this iGPU, which is the size of every delta seen.

---

## NUMBERS

### 1. Copy-proposal rate per prompt domain (copy-only and union)

From the per-request server lines (`results/p3-screen-copy.log`, `results/p3-screen-union.log`):

| Prompt | Copy-only: windows with a proposal | Proposals accepted | Union: windows with a proposal | Proposals accepted |
|---|---:|---:|---:|---:|
| `code` | 5 / 54 (9.3 %) | 8 / 15 (53 %) | 7 / 55 (12.7 %) | 11 / 21 (52 %) |
| `fiction` | 4 / 64 (6.3 %) | 7 / 12 (58 %) | 4 / 64 (6.3 %) | 7 / 12 (58 %) |
| `prose` | 2 / 54 (3.7 %) | 3 / 6 (50 %) | 3 / 54 (5.6 %) | 3 / 9 (33 %) |
| `structured` | 14 / 52 (26.9 %) | 26 / 40 (65 %) | 15 / 52 (28.8 %) | 27 / 43 (63 %) |

Copy drafting **does fire on all four prompts**, but rarely on prose/fiction/code (2–7 windows out of 52–64), and it
proposes ~2.9 tokens per hit window. The union's shorter key (3 vs 4 tokens) finds one or two more windows and a few
more (usually spurious) proposals; the acceptance rate is close.

### 2. Screen — per prompt (server-reported t/s, 1× tg150, 30 s cool)

| Prompt | Static d3 | Copy-only | Δ copy | Union | Δ union | MTP accepted (static / copy / union) |
|---|---:|---:|---:|---:|---:|---|
| `code` | **8.25** | 8.05 | **−2.4 %** | 7.94 | **−3.8 %** | 98/133 · 95/138 · 94/142 |
| `fiction` | **6.90** | 7.07 | **+2.5 %** | 7.12 | **+3.2 %** | 86/153 · 86/152 · 86/152 |
| `prose` | **8.01** | 8.16 | **+1.9 %** | 8.23 | **+2.7 %** | 95/137 · 95/137 · 95/137 |
| `structured` | **8.68** | 8.45 | **−2.7 %** | 8.53 | **−1.7 %** | 101/141 · 97/149 · 97/149 |
| **Median of 4** | **8.13** | 8.11 | **−0.2 %** | 8.09 | **−0.5 %** | |
| **Mean of 4** | **7.96** | 7.93 | **−0.3 %** | 7.96 | **0.0 %** | |

All deltas are inside the single-run noise band except the pattern that matters: `structured` — the only prompt where
copy fires often (27 % of windows) — is **slower** in both copy modes (−2.7 %, −1.7 %), and `code` is slower still
under union (−3.8 %) because a wrong copy proposal at slot 0 rejects the whole window while its MTP drafts are wasted.

### 3. Why union rarely is a *mix*

The suite runs `--mtp-drafts 3`, so a window has three slots. Copy hit windows average ~2.9 proposals, i.e. most of
them fill the window completely; the MTP chain then drafts at most `3 − ncopy` tokens (often zero). Copy-only and
union therefore behave alike on hit windows — union's numbers track copy-only, not static. (At a deeper window, say
`--mtp-drafts 7`, mixing would happen more; this screen is the d3 case the brief pins.)

### 4. Mechanism smoke (repetitive text, not a speed comparison)

A 47-token prompt asking for the same line three times, union mode, 40 tokens: **9 of 12 drafted windows proposed**
(75 %), 27 proposals, 19 accepted (70 %), 3.33 tokens per pass, 10.72 t/s in a single run
(`results/p3-smoke-union.log`; no static baseline in that run, so not a speed claim). The text is the same repeated
lines as plain greedy — verification keeps the output plain's.

### 5. Controls

- **Default-off bit-identity:** `runz --mtp --mtp-drafts 3 --n 48 --dump-logits "Explain quicksort in one paragraph."`
  on a build of main `43802b9` and on this branch: **sha256 `935e2e4f40352cb6d550196cb8cc0494336dec46276d5ef22519792317d80ac1` on both**
  (`results/p3-bitexact.log`). Greedy token md5 equal.
- **Same tokens in every mode:** on that prose prompt the 48 greedy ids are identical under static d3, copy-only and
  union (and the MTP draft/accept counts are identical: 51 drafts, 29 accepted), as speculative verification requires.
- **W3 startup self-check:** green in every load (no driver warning in any screen or smoke log).
- **Hygiene:** `pgrep -x serverz` / `runz` empty before and after every run; each mode is one `serverz` load.

---

## VERDICT

**NEGATIVE — copy drafting is real but too rare on our prompts, and the union window does not pay.**

Copy proposals fire on **3.7–9.3 %** of drafted windows on prose/fiction/code and **26.9 %** on structured (copy-only;
5.6–28.8 % with the union's 3-token key), and the proposals are accepted 50–65 % of the time. But on the screen the
copy-only median is **8.11 vs static 8.13 tok/s (−0.2 %)** and the union median **8.09 (−0.5 %)**; by mean, 7.93
(−0.3 %) and 7.96 (0.0 %). The one prompt with frequent hits, `structured`, is the one that loses (−2.7 %, −1.7 %),
and `code` loses under union (−3.8 %). No mode reached +5 % average or +10 % on structured/code, so the 3× tg300
confirm pass was **not** run.

The default-off path is bit-identical to main (dump-logits sha256 `935e2e4f…`), and all three modes produce the same
greedy tokens; the acceptance machinery was not touched. The union implementation and its two env tunables are kept
behind `--mtp-copy-union` as the brief asked, with the per-request copy-proposal stat.

**Flag:** `--mtp-copy-union` (serverz + runz), default off; `ARC_COPY_UNION_MIN=3`, `ARC_COPY_UNION_MAX=4`.
Copy-only stays `--copy-draft` / `ARC_COPY_DRAFT=1` and per-window `copy_draft` stays in `INFERZERO_GPU_BUSY_CSV`.

### What the numbers say for a next attempt

1. **Where copy can pay is repetition, not these one-shot suite prompts.** The smoke (75 % of windows, 70 % of
   proposals accepted) is the regime gufo's 35.79 → 116.93 tok/s came from: copying a line that is being edited or a
   tool result echoed back, not free-form prose. A benchmark built from a real edit/tool round is the honest test.
2. **The union's ordering is the weak point.** The MTP drafts appended after copy proposals are chained from `tok`
   (positions `pos+1…`), not from the copy prefix, so when copy proposal 0 is wrong the window is pure loss. A variant
   that advances the MTP block over the copy prefix before drafting the tail would give the appended drafts a chance;
   with `--mtp-drafts 3` and `ARC_COPY_UNION_MAX 4` there is usually no room for a tail anyway, so a deeper window
   (`--mtp-drafts 7`) is where a union could show anything.
3. **3-token keys buy proposals, not accuracy.** The union's 3-token minimum added 1–2 hit windows per prompt with
   acceptance around 33–52 %; nothing suggests going below 4 is worth it on mixed content.
