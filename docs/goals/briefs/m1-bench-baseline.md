# M1 — Diverse-prompt bench harness + baseline ladder re-measure

Goal: build a SPEED-Bench-style diverse-prompt benchmark client for inferzero and use it
to re-measure the Qwen3.8-40B champion ladder on the remote Spectre node, committing the
harness and the baseline numbers to the repo. Repo conventions: AGENTS.md at repo root.

## Where things are

- Working clone (YOU COMMIT HERE): `/home/bschooley/local-dev/inferzero`.
  Create branch `bench-speedbench` from current main and commit there. **Do not push,
  do not touch gh auth, do not merge.** Leave the clone checked out on
  `bench-speedbench` when done.
- Benchmark node: `ssh bschooley@192.168.50.48` (passwordless key). The node's login
  shell may be fish — always wrap remote commands as
  `ssh bschooley@192.168.50.48 "bash -lc '...'"` or scp a .sh and run it.
  Node repo `~/inferzero` with built binaries `~/inferzero/bin/{runz,serverz,gpu_bench}`.
  Do NOT git-pull or rebuild on the node — use the existing binaries as-is (first run
  `~/inferzero/bin/runz --help` and confirm `--mtp`, `--mtp-drafts`, `--show`, `--slots`,
  `--ctx` exist; if they don't, STOP and report).
- Models (never move/delete): exact paths on the node
  - IQ2_M: `/home/bschooley/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf`
  - IQ3_M: `...-IQ3_M.gguf`, Q4: `...-Q4_K_S.gguf` (same stem).

## What to build (files you own, nothing else)

1. `bench/speedbench.sh` — bash, must work in bash 4+/5 (node runs Arch/CachyOS). Usage:
   `speedbench.sh <model.gguf> [--mtp --mtp-drafts N]` driving `serverz` on the node:
   start one serverz per model (loads once), `--slots 1 --ctx 8192`, send each prompt
   from `bench/prompts/*.txt` as a chat request (curl to the OpenAI-style
   `chatcmpl` endpoint, streaming), cap generation ~300 tokens, compute tok/s per
   request (client-side timing of streamed content is fine; also capture the server
   log's own `t/s` and `MTP x/y accepted` lines). **Two-phase (user-authorized):**
   the FULL pass (tg~300, ≥3 repeats → median, 60 s cool) runs per config on only TWO
   representative prompts (`prose.txt`, `structured.txt`); every other prompt gets a
   single screen pass (tg~150, 1 repeat, 30 s cool) for prompt-shape only. Write raw lines to
   `results/speedbench-<config>-<date>.log` and print a per-prompt + per-config median
   table. Keep it simple bash+curl+awk — no Python anywhere (repo rule).
2. `bench/prompts/` — ≥4 diverse prompt files, one user message each:
   `prose.txt` (open-ended essay ask), `fiction.txt` (continue a story),
   `code.txt` (write a nontrivial function with tests), `structured.txt`
   (repetitive boilerplate-rich output, e.g. convert a list to JSON lines / a table),
   plus optionally `summary.txt`. Each 40–120 words. Plain ASCII.
3. `research/apu-shared-memory/BASELINE.md` — sections `METHOD`, `NUMBERS`, `VERDICT`
   (VERDICT = "baseline established"). Table: per config (IQ2_M+MTP d3, IQ3_M+MTP d3,
   Q4_K_S raw no-MTP), per-prompt tok/s, the 3-repeat median, MTP accepted x/y where
   shown, and a sanity note comparing medians to the prior records: IQ2_M+MTP d3
   7.95 standalone / 7.76 server, IQ3_M 7.12, Q4 raw 3.05 tok/s. Also add the word
   "median" explicitly in the table header.

## Measurement discipline (non-negotiable)

- Before every server start: `ssh ... "bash -lc 'pgrep -x serverz; pgrep -x runz'"` must
  be empty (a stray GPU process poisons numbers; `pgrep -f` self-matches — use `-x`).
  Kill only processes you started. When finished, kill your serverz and verify it is gone.
- One config at a time, GPU otherwise idle; 60 s cool between runs.
- Long loops: write a driver .sh on the node (`~/bench-m1/`), run it in the background
  with a log file, poll the log — do not block a single ssh call past ~10 min.
- Never run CPU-heavy work concurrently with the GPU runs (L0 watchdog trips at
  device-lost 0x70000001).

## Expected numbers / sanity

Medians should land within ~10% of 7.9 / 7.1 / 3.0 tok/s. If far off, check stray
processes and GPU clock hold (`~/inferzero/bin/gpu_bench` quotes Sysman clock holding)
before believing the number. Q4 raw tg300 takes ~100 s per run — budget the time
(~1.5–2 h total wall is normal).

## Completion condition (stop after this)

Three commits on branch `bench-speedbench` in the working clone: harness+prompts,
results log(s) (git add -f if results/ is gitignored — they are research evidence,
force-add like existing `research/apu-shared-memory/results/` logs), BASELINE.md.
Report: branch name, commit hashes, the per-config medians, MTP acceptance counts, and
any anomalies. Do not push. Do not start other experiments.
