# M6 — P4: cut on-device draft waste — reduced-vocabulary draft lm_head (+ launch audit)

Goal: the last standing decode lever. P0 proved the GPU is 87–97% busy (no host
idle to reclaim), P1 killed layout fixes, P2 killed depth tuning, P3 killed free
copy drafting. What remains is that the work itself is wasteful: **each of the 3
MTP draft steps spends ~11.5 of its ~14.7 GPU ms running `lm_head` over ALL
248,320 vocab rows (~34.5 ms of every ~355 ms window = ~10%)**, and P1 found IQ2
dequant GEMVs are ALU-capped at 66–71 GB/s. This milestone attacks the draft
`lm_head` bytes; the IQ-dequant kernel project is explicitly OUT of scope
(separate follow-on).

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Branch
  `p4-draft-lmhead` from current main (ea5de85 or newer). **Do not push, no gh
  auth, no merge.** Leave clone on the branch.
- Node: ssh bschooley@192.168.50.48, `bash -lc '...'`. Rsync branch to
  `~/inferzero-p4`, `./build.sh runz serverz gpu_bench`. Never touch other
  ~/inferzero-* or ~/models. `pgrep -x serverz; pgrep -x runz` empty before/after.
- Read first: `P0-gpu-busy.md` (phase ms + `INFERZERO_GPU_BUSY` CSV format — reuse
  it), `P1-alignment.md` forward leads (lm_head gemv_v is BW-EFFICIENT ~86.8 GB/s
  at T=1 — the saving is BYTES, not efficiency), `P2-adaptive-mtp.md` (draft step
  cost table), `zero/engine_drafters.h` (`mtp_draft` → where draft logits/argmax
  come from; `mtp_hw`, `pend_toks_`), `zero/kernels/` (lm_head GEMV kernels: gemv_v
  / gemv_mt_v), and grep the tree for `ARC_MTP_VOCAB` (P0 mentioned it — if it
  already exists, measure/tune it instead of reinventing).

## Tasks

1. **Reduced-vocab draft head** behind `--mtp-draft-vocab` (or the existing
   `ARC_MTP_VOCAB` if present; default OFF = current full-vocab behavior):
   draft steps compute lm_head logits ONLY over a bounded candidate token set
   (e.g. unique tokens in current context + last ~256 generated tokens, capped
   ~16k). Host knows the token set cheaply; device side can gather candidate rows
   into a persistent buffer (rows rarely change — incremental per window is fine).
   Verify path and target-side logits UNTOUCHED: speculation stays lossless —
   drafts outside the candidate set simply don't get proposed.
2. **Measure**: with `INFERZERO_GPU_BUSY=1` (accept its ~4.7% profiling overhead
   for the ratio), before/after draft-phase ms per window. Then tok/s screens
   WITHOUT the profiler: static d3 vs draft-vocab, 4 prompts, tg150×1, 30 s cool.
   Track acceptance x/y per prompt — acceptance must stay within ~5 pp of static.
   Success gate for confirm: window GPU-time −≥8% AND screen tok/s ≥ +5% average.
   Confirm only then: 3× tg300 prose+structured+fiction medians, both configs, +
   the IQ2 champion full-pass row (this becomes the new champion candidate —
   parent handles ladder re-measure and docs rows).
3. **Launch audit (cheap, no code unless obvious)**: from the GPU_BUSY CSV /
   ARC_LAUNCH_STATS, count per-window launches and find any >3 ms/window of
   launch-shaped gaps or redundant `clEnqueueCopyBuffer` calls in the d3 window
   (`mtp_rows`/`mtp_sync_rows`); fix ONLY trivial merges (e.g. coalescing copies);
   otherwise record and skip. L0 graph capture is NOT this milestone.

## Gates

- Default-off bit-identical: `runz --dump-logits` sha256 vs main `ea5de85`.
- W3 startup self-check green; acceptance per prompt reported (drafts differ
  legitimately — output stays verify-correct).
- Memory: candidate lm_head rows must fit alongside the model on 32 GB shared
  (16k×216 B ≈ 3.4 MB — trivial; state the budget anyway).

## Files you own

`zero/engine_drafters.h`, `zero/engine.h`, kernel files under `zero/kernels/` only
if a gathered-rows GEMV variant is needed, `zero/runz.cpp`, `tools/server.cpp`/
`server_chat.h` flag plumbing, `research/apu-shared-memory/P4-draft-lmhead.md`
(METHOD/NUMBERS/VERDICT; POSITIVE must carry `Flag:` line), results p4-*.

## Completion condition (stop after this)

Branch `p4-draft-lmhead`, 1–4 commits. Report: draft-phase ms before/after,
acceptance per prompt, screen table, confirm if gate met, VERDICT line.
Do not push. Do not start P5.
