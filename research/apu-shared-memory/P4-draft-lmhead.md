# M6 — P4: the reduced-vocabulary draft head on the IQ2_M champion (measured, POSITIVE at 32768)

**Date:** 2026-11-21
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16, Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz,
32 GB shared LPDDR5x-7467, driver 26.40.40196, kernel 7.3.5-2-cachyos)
**Model:** `Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf` (96 layers, n_embd 5120,
vocab 248,320, `lm_head` Q6_K: 3,570 B a row, 886 MB; GPU buffers 16.86 GiB)
**Branch:** `p4-draft-lmhead` (from main `ea5de85`), measured tree `~/inferzero-p4`
Flag: mtp_draft_vocab (the `EngineOptions` field; set by `--mtp-draft-vocab N`, new, or by `ARC_MTP_VOCAB=N`)
**Raw logs:** `results/p4-bitexact.txt`, `results/p4-bitexact-text.txt`, `results/p4-gpu-busy-{static,16k,32k}.{log,csv}`,
`results/p4-gpu-busy-means.txt`, `results/p4-smoke.txt`, `results/p4-screen-{static,v16k,v32k}.log`,
`results/p4-confirm-{static,v32k}.log`

---

## METHOD

### What already existed, and what P4 added

The brief's first task was the reduced-vocabulary draft head "or the existing `ARC_MTP_VOCAB` if present: measure and
tune it instead of reinventing". **It is present** (QUEUE D1, 2026-11-21): `Engine::load_draft_head`
(`zero/engine_drafters.h`) gathers N rows of `lm_head` plus every control token into `draft_head_` at load, the draft
step's argmax runs over those N rows and maps back through `draft_ids_`, and `verify()` is untouched, so a draft
outside the N is simply never proposed. The ranking comes from `scripts/draft-vocab/vocab-<vocab>.txt` (this
repository's prose, Markdown, code and 6,912 tokens of the model's own Indonesian, Chinese and Spanish answers).

So P4 added no new drafting mechanism. It made the head reachable from the command line and measured it on the
champion:

- `EngineOptions::mtp_draft_vocab` (0 = off), seeded from `ARC_MTP_VOCAB` when the flag is absent;
- `--mtp-draft-vocab N` in `runz` and `serverz` (`--help`, `--help-all`), validation of a negative N;
- `load_draft_head()` reads the option instead of a process-wide `getenv` and says so when it cannot apply
  (no `--mtp`, a split model, a Gemma drafter) or the rank list for the vocabulary is missing.

With the flag absent nothing is allocated and `load_draft_head` returns at its first line.

### Measurement protocol

1. **Default off, bit for bit:** `runz --mtp --mtp-drafts 3 --n 16 --dump-logits` on the prose prompt, main
   `ea5de85` against this branch, both with `ARC_MTP_VOCAB` unset, sha256 of the two files.
2. **Lossless with the head on:** prose, n=96, static against `--mtp-draft-vocab 16384`; the generated text
   (`--show`) of both runs hashed. Decoding is a function of the token ids, so equal text = equal ids. Speculation is
   lossless by construction (verification is the full head), and this checks it end to end.
3. **Draft-phase ms:** `INFERZERO_GPU_BUSY=1` (`INFERZERO_GPU_BUSY_CSV`), prose, tg120, 30 s cool between runs, one
   run a config: static, 16384, 32768. The profiler's ~4.7 % overhead is the same in all three, so the ratios hold.
4. **Screen:** `bench/speedbench.sh`, `serverz` (greedy, `temperature: 0.0`, `cache_prompt: false`), tg150 x 1,
   30 s cool, all four prompts, one model load a config: static, 16384, 32768. Acceptance x/y per prompt from the
   server's own `MTP x/y` line.
5. **Confirm** (run because the screen met the gate): 3 x tg300, 60 s cool, 60 s between configs; `bench/speedbench.sh
   --full` seats all four prompts in the confirm pass, so it is 12 runs a config.
6. **Launch audit:** per-window launch, copy and sync counts and the phase gaps from the `INFERZERO_GPU_BUSY` CSV of
   step 3 (`draft_launches`, `draft_copies`, `draft_syncs`, `draft_gap_ms`), windows 1 and later (window 0 carries
   the prefill's tail).
7. `pgrep -x serverz` / `runz` empty before and after every run; the W3 start-up GEMM self-check printed no warning
   in any run (only the node's Mesa/libclc OpenCL notice).

---

## NUMBERS

### 1. Gates

- Default off, `--dump-logits` sha256, main `ea5de85` vs `p4-draft-lmhead` (n=16, MTP d3, prose):
  `0d848bc56af122598ff556aba4ccf54e7e1679b8bf785858c659068b8cf2f640` **both** (15,892,480 bytes both). IDENTICAL.
- Head on (16384) against static, prose, n=96: generated text sha256
  `167cbeb5392a396025baa54ac0579702766d3e18223b9da7831069e10bfc280f` **both** (499 bytes). SAME TEXT.
- Verify path unchanged in code and in time: verify GPU 251.5 / 250.5 / 252.7 ms a window (static / 16K / 32K).
- Memory: 32768 rows x 3,570 B = **115 MB** (13 % of the 886 MB head); 16384 = 58 MB. Trivial next to the 13.5 GiB
  model in 32 GB shared. Nothing is allocated with the flag off.
- W3 startup self-check: green.

### 2. Draft and window GPU time (`INFERZERO_GPU_BUSY=1`, prose, tg120)

| config | windows | window wall ms | window GPU ms | draft wall ms | **draft GPU ms** | draft GEMM ms | tok/win | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static d3 | 49 | 342.3 | 295.2 | 41.6 | **35.40** | 34.34 | 2.43 | 70/119 (58.8 %) |
| `--mtp-draft-vocab 16384` | 57 | 317.9 (-7.1 %) | 273.4 (**-7.4 %**) | 14.7 | **11.00 (-69 %)** | 9.94 | 2.09 | 62/139 (44.6 %) |
| `--mtp-draft-vocab 32768` | 51 | 314.5 (-8.1 %) | 271.5 (**-8.0 %**) | 15.7 | **12.60 (-64 %)** | 11.55 | 2.33 | 68/124 (54.8 %) |

The draft head cuts exactly the bytes it was meant to: 11.5 ms of `lm_head` a draft step at 248,320 rows becomes
0.76 ms at 16,384 and 1.5-2.2 ms at 32,768, and the rest of a draft step (the MTP block's own ~3.1 ms) is untouched.
The window saving is the whole draft saving: 295.2 -> 271.5 ms.

### 3. Screen (tg150 x 1, 30 s cool, serverz, four prompts)

| prompt | static d3 tok/s (accepted) | 16384 tok/s (accepted) | 32768 tok/s (accepted) |
|---|---:|---:|---:|
| code | 8.60 (98/133, 73.7 %) | 8.63 (94/143, 65.7 %) | 8.98 (96/136, 70.6 %) |
| fiction | 7.14 (86/153, 56.2 %) | 6.23 (70/174, 40.2 %) | 7.46 (84/153, 54.9 %) |
| prose | 8.26 (95/137, 69.3 %) | 7.17 (82/173, 47.4 %) | 8.88 (95/140, 67.9 %) |
| structured | 8.94 (101/141, 71.6 %) | 9.12 (97/147, 66.0 %) | 9.56 (100/143, 69.9 %) |
| **median** | **8.43** (380/564, 67.4 %) | **7.90** (-6.3 %; 343/637, 53.8 %) | **8.93 (+5.9 %;** 375/572, 65.6 %) |

16,384 is not usable on this model: adding the list's ranks 16,384-23,906 buys 20 percentage points of acceptance on
prose. That is the same effect the R1/D1 measurements saw on Indonesian (59 % -> 45 %) and Chinese (-15 %): the
ranking fits English and code, and the places where it does not fit are where the drafts fall out of the top N.
32,768 is the smallest N that holds: acceptance within 3.1 pp on every prompt, +5.9 % median.

### 4. Confirm (3 x tg300, 60 s cool, 12 runs a config)

| prompt | static d3 tok/s (accepted) | 32768 tok/s (accepted) | tok/s | acceptance |
|---|---:|---:|---:|---:|
| code | 8.14 (567/846, 67.0 %) | 8.54 (558/855, 65.3 %) | +4.9 % | -1.7 pp |
| fiction | 6.85 (504/969, 52.0 %) | 6.94 (480/1008, 47.6 %) | +1.3 % | -4.4 pp |
| prose | 7.67 (546/876, 62.3 %) | 7.65 (519/918, 56.5 %) | -0.3 % | -5.8 pp |
| structured | 9.11 (612/834, 73.4 %) | 9.63 (603/846, 71.3 %) | +5.7 % | -2.1 pp |
| **median** | **7.83** (2,229/3,525, 63.2 %) | **8.04 (+2.7 %)** (2,160/3,627, 59.6 %) | **+2.7 %** | **-3.6 pp** |

The three repeats of a cell agree to within 0.2 % (8.41 / 8.55 / 8.54), so these are not noise; the effect really is
smaller at 300 tokens than at 150 (+2.7 % against +5.9 %). Over 300 tokens prose drifts further from the ranked
list (acceptance -5.8 pp) and its gain disappears; code and structured keep +5-6 %.

### 5. Launch audit (static d3 window, windows 1 and later, from the CSV)

| phase | launches / window | copies / window | syncs / window | gap / window | largest gap |
|---|---:|---:|---:|---:|---:|
| draft (`mtp_draft`) | 49.0 | 6.5 | 8.3 | 5.27 ms (1.5 % of the window) | 8.9 ms typical, one 47 ms host stall in 48 windows |
| verify | 1,534.6 | 290.0 | 4.0 | 37.7 ms | - |
| verify_done | 90.0 | 46.6 | 1.0 | 3.2 ms | - |

The draft window's gap is not launch-count shaped: 49 launches at ~20 us are ~1 ms of it, and the rest is the 2.4
blocking `clEnqueueReadBuffer` calls of the logits (0.99 MB each) and the blocking `mtp_e_` writes, which the step
needs before its argmax. The copies are two 51 KB `h`-row copies (pending accepted drafts, `hlast_`) plus a 3-row
`mtp_e_` write a step: no redundant copy to coalesce, no >3 ms lump to merge. **Recorded, no code.** (L0 graph
capture is explicitly not this milestone.)

---

## VERDICT

**POSITIVE (qualified): the reduced-vocabulary draft head works and pays, but only at 32,768 rows and only on code
and structured text.**

`Flag: mtp_draft_vocab`

- The head does what P0 predicted: draft GPU time 35.4 -> 12.6 ms a window (-64 %), window GPU time -8.0 %
  (295.2 -> 271.5 ms), verification and the text untouched (bit-identical default off; the same generated text with
  the head on).
- Screen gate met (+5.9 % median tok/s, acceptance within 3.1 pp on every prompt, window GPU -8.0 %).
- Confirm: +2.7 % median (code +4.9 %, structured +5.7 %, fiction +1.3 %, prose -0.3 %), acceptance -3.6 pp
  overall, -5.8 pp on prose at 300 tokens. Real but small; it is a code/structured-text win, not a general one.
- 16,384 is a loss (-6.3 %) on this model even though it halves the bytes again: the ranking's tail carries 20 pp of
  the acceptance. Do not go under 32,768 with this list.
- Left opt-in (default off, `--mtp-draft-vocab 32768`), like D1 before it, because creative prose at 300 tokens is
  flat to slightly negative and Chinese/Indonesian were already negative at this list.
- For the ladder: 32,768 as the variant (`-fastdraft`/`mtp_vocab`), not the default; the parent re-measures it
  against the champion rows.
