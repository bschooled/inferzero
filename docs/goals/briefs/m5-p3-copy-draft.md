# M5 — P3: copy/prompt-lookup drafting — evaluate, tune, and union into the MTP window

Goal: free tokens. The engine ALREADY has R1 copy drafting: in
`zero/engine_drafters.h::mtp_draft()`, when `hist && copy_on()` it looks for the
recent token sequence earlier in the context and proposes the tokens that followed
there (early return, `copy_used_`), and `--copy-draft` exists in the server option
table. Nobody has measured what it does on our diverse prompts, and gufo's
prompt-lookup drafter turned 35.79 → 116.93 tok/s on repetitive decode. Two jobs:
(1) MEASURE the existing feature honestly; (2) implement the missing piece — a
UNION mode where copy proposals and MTP drafts share one verify window
(copy-first, MTP fills the remaining slots up to the cap), behind a flag.

## Where things are

- Working clone (COMMIT HERE): `/home/bschooley/local-dev/inferzero`. Branch
  `p3-copy-draft` from current main (43802b9 or newer). **Do not push, no gh auth,
  no merge.** Leave clone on the branch.
- Node: ssh bschooley@192.168.50.48, `bash -lc '...'`. Rsync to `~/inferzero-p3`,
  `./build.sh runz serverz`. Never touch other ~/inferzero-* trees or ~/models.
  `pgrep -x serverz; pgrep -x runz` empty before/after each server run.
- Read first: `zero/engine_drafters.h` mtp_draft() copy path + copy_on() + how
  verify()/verify_done() treat drafts (semantics: drafts are just proposals —
  acceptance machinery is drafting-agnostic); `APPENDIX-gufo.md` prompt_lookup
  section; `BASELINE.md` + `P2-adaptive-mtp.md` (static d3 references and cost
  numbers); `tools/server_chat.h` for how --copy-draft is currently plumbed.

## Tasks

1. **Measure existing copy drafting** (screen, tg150 ×1, 30 s cool, all 4 prompts):
   static d3 vs `--copy-draft` alone. Does it fire at all on prose/fiction/
   structured/code? Add a cheap stat if none exists: copy-proposal rate per window
   (stderr/CSV, gated with the existing profiling flags).
2. **Union mode** behind `--mtp-copy-union` (default off): build the draft list as
   [copy proposals first, capped; then MTP drafts to fill the window up to
   max_drafts]. Tunables (env): min key length (default 3 tokens), max copy
   proposals (default 4). Verify path UNCHANGED — drafts are just tokens.
3. Screen union mode (same 4 prompts). If any mode wins ≥5% average OR ≥+10% on
   structured/code with ≤2% prose regression: confirm (3× tg300, 60 s cool) top two
   modes on prose + structured + fiction; medians.

## Gates

- Default-off bit-identical: `runz --dump-logits` sha256 vs main for the champion.
- Greedy speculative semantics untouched: acceptance machinery not modified
  (diff review + the existing verify tests/flow); W3 startup self-check green.
- Do NOT touch adaptive-depth controller internals (P2 lane); composing with
  `--mtp-adaptive` may be noted but is not required to work this milestone.

## Files you own

`zero/engine_drafters.h`, `zero/engine.h` (opt), `zero/runz.cpp`, `tools/server.cpp`
/ `tools/server_chat.h` (flags), `research/apu-shared-memory/P3-copy-draft.md`
(METHOD/NUMBERS/VERDICT — POSITIVE + `Flag:` line if a mode wins; honest NEGATIVE
with copy-proposal rates otherwise), results force-added under
`research/apu-shared-memory/results/p3-*`.

## Completion condition (stop after this)

Branch `p3-copy-draft`, 1–4 commits. Report: copy-proposal rates per prompt domain,
per-mode screen table, confirm if run, VERDICT line. Do not push. Do not start P4/P5.
