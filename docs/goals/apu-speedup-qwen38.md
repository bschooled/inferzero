# Qwen3.8-40B generation speed on the Spectre (MTL) node

**Created:** 2026-11-21 · **Mode:** focused · **Session-scoped**

## Objective

Make Qwen3.8-40B generation measurably faster on the HP Spectre x360 node
(bschooley@192.168.50.48 — Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EU, 128-bit shared
LPDDR5x-7467, no VRAM) by executing the ranked action queue §6 (P0–P5) of
`research/apu-shared-memory/REPORT.md`. Every experiment leaves a
METHOD/NUMBERS/VERDICT file in `research/apu-shared-memory/`; helpful code is committed
behind a flag; the champion ladder (IQ2_M+MTP d3 7.95 / IQ3_M 7.12 / Q4 raw 3.05 tok/s)
is re-measured after every merge. The repo is the record: commits in
`/home/bschooley/local-dev/inferzero`, pushed to the fork `github.com/bschooled/inferzero`
(`gh auth switch --user bschooled` before pushing, switch back after). Chat is not.

## Design brief

- **Source of truth:** `research/apu-shared-memory/REPORT.md` (§1 HW ceilings:
  theoretical 119.5 GB/s, iGPU block-read 89.9, 16-byte-load 57.3, CPU 67.8, concurrent
  ~100; §2 regime finding: IQ2_M+MTP champion is OVERHEAD-bound at ~37 GB/s pulled —
  the serial MTP draft chain `zero/engine_drafters.h:502` sets pass time; Q4 raw is the
  BW-bound reference at 66.5 GB/s) + `APPENDIX-gufo.md` / `APPENDIX-strata.md`
  (cited sibling-engine analyses; clones at `~/.hermes/cache/scratch/{gufo,Strata}`).
- **Measurement discipline (non-negotiable):** ≥3 repeats, medians, 60 s cooldown
  between runs (MTL clocks sag); diverse prompts (SPEED-Bench-style client first —
  single tg300 sweeps cannot resolve 10% effects); report tok/s AND bytes/pass AND
  effective GB/s for every result so each number says which regime moved.
- **Regimes:** move the overhead-bound champion by cutting draft-chain/verify overhead
  (P2, P3, P4); move the BW-bound Q4 path by alignment/padding (P1).
- **SKIP LIST (measured negatives — do not re-litigate):** non-temporal/streaming loads
  (>50% BW loss); naive drafter/verify overlap before handshake latency is fixed
  (`--spec-split` lost on A770); GRF=256 on BF16; CPU-run drafting unless P0 shows
  GPU-idle draft phases (and then E-cores only, `taskset -c 12-15`); NPU (Linux driver
  lacks sysmem extensions, `zeGraphCreate2` → 0x78000003); MoE expert tiering (dense
  40B); Strata-style auto layer-split as default.
- **Safety:** never run a full-power all-core CPU lane concurrently with BW-saturated
  GPU load — trips the L0 device-lost watchdog (0x70000001). Never touch `~/models`;
  never delete server logs; kill stray `serverz`/`runz` before any GPU test
  (`pgrep -x serverz; pgrep -x runz` — `pgrep -f` self-matches ssh).
- **Gates:** greedy speculative output must equal autoregressive output bit-exactly;
  the W3 startup GEMM self-check must stay green; every merged speed change gets a row
  in `docs/before-after.md` + a CHANGELOG entry. One branch per experiment in the
  working clone; merges to main happen after the ladder re-measure.
- **Node quirks:** ssh wraps `bash -lc '...'` (node may default to fish); remote repo
  `~/inferzero` (binaries there; node tree is behind the local clone — use its
  existing `bin/` unless deliberately syncing); models `~/models/*.gguf`; runz tg
  timing with `--mtp --show` prints `MTP: x/y accepted`.

## Criteria

| # | Criterion | Verification |
|---|---|---|
| 1 | Diverse-prompt benchmark harness (`bench/speedbench.sh` + ≥4 prompt files under `bench/prompts/`) committed, and baseline ladder re-measured with it: ≥3-repeat medians for IQ2_M+MTP d3, IQ3_M+MTP d3, Q4_K_S raw on the node, written to `research/apu-shared-memory/BASELINE.md` (expect ≈7.95 / 7.12 / 3.05 tok/s). | `bash research/apu-shared-memory/checks/check.sh 1` |
| 2 | P0: GPU-busy fraction measured during a decode/draft window via L0 or i915/xe perf counters (probe code committed, any path containing `busy`), with `research/apu-shared-memory/P0-*.md` giving per-phase busy% and a VERDICT that gates P3/P4 priority. | `bash research/apu-shared-memory/checks/check.sh 2` |
| 3 | P1: alignment/stride audit of decode GEMV weight streams + load-time padding experiment (gufo 512 B precedent), microbench numbers AND Q4 raw tg result in `research/apu-shared-memory/P1-*.md`; success bar: Q4 raw 3.05 → >3.4 with the padding toggle merged behind a flag; honest NEGATIVE allowed. | `bash research/apu-shared-memory/checks/check.sh 3` |
| 4 | P2: gufo-style priced adaptive draft depth (startup cost table per context×width, per-window depth choice, failed-depth mask) behind a flag, vs static d3 on diverse prompts, in `research/apu-shared-memory/P2-*.md`; ladder re-measured. | `bash research/apu-shared-memory/checks/check.sh 4` |
| 5 | P3: prompt-lookup/n-gram union drafter (zero bandwidth cost) behind a flag, measured on structured AND prose prompts in `research/apu-shared-memory/P3-*.md`; no prose regression. | `bash research/apu-shared-memory/checks/check.sh 5` |
| 6 | P4: draft-chain surgery — merged per-row verify launches and/or whole-draft-chain L0 command-graph capture with device-side handoff — with the drafting tax (12–17%) measured before/after in `research/apu-shared-memory/P4-*.md`; ladder re-measured. | `bash research/apu-shared-memory/checks/check.sh 6` |
| 7 | P5: prefill DPAS retest at T≥256 tiles only, pp measured before/after in `research/apu-shared-memory/P5-*.md` (target pp 70→120+; honest NEGATIVE allowed). | `bash research/apu-shared-memory/checks/check.sh 7` |
| 8 | Quality gates recorded green in `research/apu-shared-memory/QUALITY-GATES.md`: W3 startup GEMM self-check + greedy spec-vs-AR bit-exactness on at least the champion config, after the merges. | `bash research/apu-shared-memory/checks/check.sh 8` |
| 9 | `research/apu-shared-memory/LADDER-FINAL.md`: final ladder for IQ2_M/IQ3_M/Q4 with regime arithmetic (tok/s, bytes/pass, effective GB/s) for the baseline and for every merged change; every merged speed change has a `docs/before-after.md` row and CHANGELOG entry. | `bash research/apu-shared-memory/checks/check.sh 9` |
