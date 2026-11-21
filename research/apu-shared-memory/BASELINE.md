# M1 Baseline Ladder Re-Measure — Diverse-Prompt Speedbench on Spectre (MTL)

**Date:** 2026-11-21
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16 — Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz, 32 GB shared LPDDR5x-7467, driver 26.40.40196, kernel 7.3.5-2-cachyos)
**Harness:** `bench/speedbench.sh` + `bench/prompts/{prose,structured,code,fiction}.txt`
**Raw logs:** `results/speedbench-{IQ2_M-mtp-d3,IQ3_M-mtp-d3,Q4_K_S-raw}-20261121.log` (also copied to `research/apu-shared-memory/results/`)

---

## METHOD

To replace single-prompt `tg300` sweeps with a prompt-diverse benchmark while keeping experimental wall time bounded, `bench/speedbench.sh` implements a **two-phase measurement protocol** driving `serverz` (`--slots 1 --ctx 8192 --no-prefix-cache --session-cache-gb 0`, greedy `temperature: 0.0`, `"cache_prompt": false`, OpenAI `/v1/chat/completions` SSE streaming):

1. **Server lifecycle & hygiene:** Before starting `serverz`, the harness checks `pgrep -x serverz` and `pgrep -x runz` and aborts if any GPU inference process is present. One `serverz` instance is started per model configuration (loaded once) and verified dead after the config finishes. A 60 s cooldown separates configurations.
2. **Phase B — Full confirmation pass (representative prompts: `prose.txt`, `structured.txt`):**
   - `max_tokens = 300`, **3 repeats per prompt**, **60 s cooldown** between runs so Meteor Lake iGPU/fabric clocks recover from thermal/power sag.
   - Captures both server-reported generation throughput (`serverz` `[chatcmpl-arcN]` log line: `gen N tok T s (X.XX t/s)` and `MTP x/y drafts accepted`) and client-side streamed SSE decode throughput (`(gen_tok - 1) / (t_last_chunk - t_first_chunk)`).
   - Computes the **3-repeat median** per prompt and the **full-pass config median** across the two representative extremes (open-ended analytical prose vs repetitive structured JSONL/Markdown table conversion).
3. **Phase A — Screen pass (remaining prompts: `code.txt`, `fiction.txt`):**
   - `max_tokens = 150`, **1 repeat per prompt**, **30 s cooldown** between runs.
   - Captures prompt-to-prompt shape across the broader suite (C++20 LRU cache + unit tests in `code.txt`; narrative continuation in `fiction.txt`) without spending full 3x300 wall time on every prompt.

Prompt suite (`bench/prompts/*.txt`, plain ASCII, 57–74 words):
- `prose.txt` (57 words, 126 tokens with chat template): disaggregated vs monolithic database storage engine essay.
- `structured.txt` (63 words, 340 tokens with chat template): 6-row service inventory converted to JSON Lines + sorted Markdown table.
- `code.txt` (74 words, 155 tokens with chat template): C++20 arena-backed intrusive LRU cache with linear-probing hash table and `assert()` test suite.
- `fiction.txt` (65 words, 131 tokens with chat template): maritime night-watch narrative continuation.

---

## NUMBERS

### 1. Per-Config, Per-Prompt Results and 3-Repeat Medians

| Config | Phase | Prompt | Gen Cap | Individual Runs (srv tok/s) | 3-repeat median srv tok/s | median client tok/s | median pp tok/s | MTP accepted x/y (median run & total) |
|---|---|---|---:|---|---:|---:|---:|---|
| **IQ2_M + MTP d3** | full (3x) | `prose` | 300 | 7.15, 7.29, 7.29 | **7.29** | 7.31 | 22.7 | 182/292 (62.3%; sum 546/876) |
| **IQ2_M + MTP d3** | full (3x) | `structured` | 300 | 8.64, 8.68, 8.67 | **8.67** | 8.71 | 41.1 | 204/278 (73.4%; sum 612/834) |
| **IQ2_M + MTP d3** | screen (1x) | `code` | 150 | 8.26 | **8.26** | 8.34 | 27.0 | 98/133 (73.7%) |
| **IQ2_M + MTP d3** | screen (1x) | `fiction` | 150 | 6.76 | **6.76** | 6.71 | 24.5 | 86/153 (56.2%) |
| **IQ2_M + MTP d3** | **FULL-PASS MEDIAN** | **`prose` + `structured` (6 runs)** | **300** | — | **7.96** | **8.00** | **31.9** | **sum 1158/1710 (67.7%)** |
| **IQ2_M + MTP d3** | **ALL-PROMPTS MEDIAN** | **all 4 prompts (8 runs)** | — | — | **7.78** | **7.83** | **25.8** | **sum 1342/1996 (67.2%)** |
| **IQ3_M + MTP d3** | full (3x) | `prose` | 300 | 6.39, 6.47, 6.48 | **6.47** | 6.49 | 20.7 | 174/280 (62.1%; sum 522/840) |
| **IQ3_M + MTP d3** | full (3x) | `structured` | 300 | 8.18, 8.19, 8.18 | **8.18** | 8.15 | 40.6 | 205/271 (75.6%; sum 615/813) |
| **IQ3_M + MTP d3** | screen (1x) | `code` | 150 | 7.96 | **7.96** | 8.03 | 25.1 | 101/136 (74.3%) |
| **IQ3_M + MTP d3** | screen (1x) | `fiction` | 150 | 6.38 | **6.38** | 6.42 | 22.6 | 88/157 (56.1%) |
| **IQ3_M + MTP d3** | **FULL-PASS MEDIAN** | **`prose` + `structured` (6 runs)** | **300** | — | **7.33** | **7.33** | **30.6** | **sum 1137/1653 (68.8%)** |
| **IQ3_M + MTP d3** | **ALL-PROMPTS MEDIAN** | **all 4 prompts (8 runs)** | — | — | **7.22** | **7.26** | **23.9** | **sum 1326/1946 (68.1%)** |
| **Q4_K_S raw (no-MTP)** | full (3x) | `prose` | 300 | 3.16, 3.18, 3.18 | **3.18** | 3.18 | 10.8 | — (no MTP) |
| **Q4_K_S raw (no-MTP)** | full (3x) | `structured` | 300 | 3.17, 3.17, 3.17 | **3.17** | 3.17 | 24.3 | — (no MTP) |
| **Q4_K_S raw (no-MTP)** | screen (1x) | `code` | 150 | 3.17 | **3.17** | 3.17 | 13.2 | — (no MTP) |
| **Q4_K_S raw (no-MTP)** | screen (1x) | `fiction` | 150 | 3.17 | **3.17** | 3.17 | 11.5 | — (no MTP) |
| **Q4_K_S raw (no-MTP)** | **FULL-PASS MEDIAN** | **`prose` + `structured` (6 runs)** | **300** | — | **3.17** | **3.17** | **17.6** | **—** |
| **Q4_K_S raw (no-MTP)** | **ALL-PROMPTS MEDIAN** | **all 4 prompts (8 runs)** | — | — | **3.17** | **3.17** | **12.4** | **—** |

### 2. Champion Ladder Summary & Regime Arithmetic

| Config | Weight Bytes / Pass | Full-Pass 3-Repeat Median (tok/s) | All-Prompts Median (tok/s) | Prior Record (tok/s) | Delta vs Prior | Mean Tokens / Verify Pass | Implied Weight BW (GB/s) | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **IQ2_M + MTP d3** | 13.53 GiB (14.53 GB) | **7.96** | **7.78** | 7.95 standalone / 7.76 server | +0.1% vs standalone / +2.6% vs server | 3.03 tok/pass (67.7% acc) | ~38.1 GB/s | Overhead-bound (draft chain + verify launches) |
| **IQ3_M + MTP d3** | 17.05 GiB (18.31 GB) | **7.33** | **7.22** | 7.12 standalone | +2.9% (+1.4% all-prompts) | 3.06 tok/pass (68.8% acc) | ~43.9 GB/s | Overhead-bound |
| **Q4_K_S raw (no-MTP)** | 21.16 GiB (22.72 GB) | **3.17** | **3.17** | 3.05 standalone | +3.9% | 1.00 tok/pass (AR) | **72.0 GB/s** (67.1 GiB/s) | **Bandwidth-bound** (~80% of 89.9 GB/s iGPU peak) |

### 3. Sanity Note vs Prior Records & Prompt-Shape Observations

- **Agreement with prior records:** All three configurations land well within the expected ±10% window of the prior records (`7.95 / 7.76`, `7.12`, and `3.05` tok/s):
  - **IQ2_M + MTP d3:** Full-pass median is **7.96 tok/s** (client median **8.00 tok/s**) and all-prompts median is **7.78 tok/s**, matching the prior `7.95` standalone / `7.76` server numbers to within 0.1–2.6%.
  - **IQ3_M + MTP d3:** Full-pass median is **7.33 tok/s** and all-prompts median is **7.22 tok/s**, within +1.4% to +2.9% of the prior `7.12` tok/s record.
  - **Q4_K_S raw (no-MTP):** Both full-pass and all-prompts medians are **3.17 tok/s** (`3.16–3.18` across every prompt and repeat), +3.9% above the prior `3.05` tok/s single-run figure thanks to clean 60 s cooldowns and zero background contention.
- **Repeat-to-repeat stability:** With `"cache_prompt": false`, `temperature: 0.0`, and 60 s cooldowns, repeats 2 and 3 are nearly identical (e.g., IQ2_M `prose`: `7.15, 7.29, 7.29`; IQ2_M `structured`: `8.64, 8.68, 8.67`; IQ3_M `structured`: `8.18, 8.19, 8.18`; Q4_K_S `structured`: `3.17, 3.17, 3.17`). Repeat 1 of the first prompt after server start (`prose` rep 1) shows a slight ~1–2% warmup dip before settling on repeats 2–3.
- **Prompt-to-prompt shape under MTP vs Raw AR:**
  - Under **Q4_K_S raw (no-MTP)**, decode speed is completely invariant to prompt domain (`3.17–3.18 tok/s` across `prose`, `structured`, `code`, and `fiction`), confirming a pure memory-bandwidth-bound regime.
  - Under **MTP d3**, throughput varies strongly with draft acceptance rate across domains:
    - `structured` (73.4–75.6% draft acceptance) and `code` (73.7–74.3% acceptance) run at **8.26–8.67 tok/s** on IQ2_M and **7.96–8.18 tok/s** on IQ3_M.
    - `prose` (62.1–62.3% acceptance) runs at **7.29 tok/s** (IQ2_M) and **6.47 tok/s** (IQ3_M).
    - `fiction` (56.1–56.2% acceptance) runs at **6.76 tok/s** (IQ2_M) and **6.38 tok/s** (IQ3_M).
  - This 28% spread between creative prose/fiction and structured/code output demonstrates why single-prompt benchmarks can mislead and why pairing `prose.txt` + `structured.txt` in the full confirmation pass brackets both ends of the MTP acceptance distribution.

---

## VERDICT

**baseline established**
