# QUALITY GATES — after all queue merges (fork main 7420c82)

Every merge in this campaign was gated on the same three checks. Recorded green
throughout; per-merge primary evidence is linked.

## 1. W3 startup GEMM self-check

Green in every run logged since P0: `results/*.log` carry the line pattern
"--- W3 / driver lines (none = the start-up self-check is green) ---" with no
driver lines after it (see `P0-gpu-busy.md`, and the model-load headers in every
`speedbench-*` and `p5-bench` log). No merge ever tripped it.

## 2. Bit-exactness of default-off paths (`runz --dump-logits`)

| merge | compared | sha256 | verdict |
|---|---|---|---|
| P0 `INFERZERO_GPU_BUSY` | flag 0 vs 1 | `ac0488197…ada4` both | identical |
| P1 `ARC_PAD_ROWS` | pad 0 vs on (Q4) | `4c0447e2…184cea` both | identical |
| P2 `--mtp-adaptive` | branch vs main | `935e2e4f…d80ac1` both | identical |
| P3 `--mtp-copy-union` | branch vs main | `935e2e4f…` chain | identical |
| P4 `--mtp-draft-vocab` | branch vs `ea5de85` | `0d848bc5…f640` both | identical |
| P5 `ARC_PROMPT_DPAS` | 4-way: base `dd2e6a9`, flag off, flag on, `ARC_XMX_TRY=2` stand-in | `c1f52324…123a6` all four (16,885,760 B each) | identical |

The P5 four-way dump is the strongest end-to-end statement: the fully merged
tree (everything after P0-P4 included) is bit-identical to the pre-campaign
engine with all five flags off/default — the iGPU path is untouched, per
AGENTS.md.

## 3. Greedy speculative == autoregressive output

Speculation semantics were never modified: `verify()` / `verify_done()` were
left untouched in P2, P3 and P4 (diffs in those files' code commits).
Behavioral proofs on record: P3 — "all three modes (static d3, copy-only,
union) produce the same greedy tokens"; P4 — same generated text with the
32K draft head on (`167cbeb5…`), i.e. narrower drafts propose less, never
wrong. The draft-vocab head can only omit proposals; a token outside the
candidate set is verified by the full head exactly as before.

## Residual honesty

- P4's flag ON legitimately changes which tokens are DRAFTED (text stays
  full-head text; acceptance moves −3.7 pp, measured).
- P5's `ARC_XMX_TRY=2` DP4a stand-in shows why no draft-side numeric risk
  existed: on Xe-LPG it routes nothing — the "-DXMX" kernel does not compile.
- No merge has any known logits effect when its flag is off. All flags default
  off except where a user opts in.
