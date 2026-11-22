# P5 — prefill DPAS retest at T ≥ 256 tiles (M7, 2026-11-21 / finished 2026-11-22)

**Question (REPORT §6 item 5):** inferzero's prefill runs at ~70–77 tok/s ≈ 6.2 T int8/s, 33 % of this iGPU's
DP4a peak (18.4 TOPS). Sibling engines get ×1.6–2.0 on APUs from matrix-core GEMMs (Strata: ×1.59–1.72 on Strix
Halo, ×1.6–2.0 on a 780M). Is there prefill left in DPAS here?

**Answer:** not on this node, and not by this route: the MTL iGPU (Xe-LPG) has no matrix engine at all — its
driver exposes no DPAS, so `gemm.cl -DXMX` does not even build here. The switch is in the tree
(`9551dbb`, default off) for the GPUs that do have one; the measurements below bound what the *tile shape* alone
buys on Xe-LPG, and it is a loss, not a gain.

**Date:** 2026-11-21 22:09 – 22:56 and 2026-11-22 10:24 – 10:42 (PST)
**Node:** `bschooley@192.168.50.48` (HP Spectre x360 16 — Intel Core Ultra 7 155H, Arc Xe-LPG iGPU 128 EUs @ 2250 MHz,
32 GB shared LPDDR5x-7467 (30.7 GiB visible), driver 26.40.40196 (NEO, OpenCL 3.0; Mesa 26.3.1 / Rusticl),
kernel 7.3.5-2-cachyos)
**Models:** `Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf` (IQ2_M, 96 layers, 13.53 GiB)
and `…-Q4_K_S.gguf` (21.16 GiB), the two the brief names
**Branch:** `p5-dpas-prefill` @ `9551dbb` (the switch), measured tree `~/inferzero-p5`; the branch base `dd2e6a9` in a
worktree at `/tmp/p5base`
**Flags:** `ARC_PROMPT_DPAS=1` (DPAS for prompt tiles of `ARC_PROMPT_DPAS_T` tokens and more, default 256, DP4a below;
unset = the engine's own verdict, the code path before the switch), `ARC_XMX_TRY=2` (the DP4a stand-in for DPAS)
**Harness:** `runz <model> --bench-pp N --n 0` — one model load per invocation, the tool times 3 prompt passes itself
**Raw logs:** `results/p5-bench.log` (24 bench invocations, 68 timed passes), `results/p5-dump.log` (the four
logits runs), `results/p5-logits.sha256`, `results/p5-prompt.txt` (the dump prompt, 443 tokens)

---

## 1. What lost before, and why the tiles now differ (5 lines)

1. The earlier DPAS loss was on the **Arc 140T (Arrow Lake H, 8-lane DPAS)**, 27B IQ3_S, `--bench-pp 2048`:
   **DP4a 178–182 tok/s against XMX 113–120** — GEMMs 1.6× slower (`docs/before-after.md`, 2026-11-07 row;
   `queued` in `zero/engine.h:442-444`).
2. Its shape was **decode-shaped**: a 2K prompt in 512-token chunks whose GEMM tiles were 16–32 tokens wide, i.e.
   the same short, load-bound tiles decode uses (REPORT §1: "DPAS slower than DP4a in decode-shaped GEMMs").
3. The verdict was taken **once per process** from a 64×64×1024 shape (later a 512×17408×5120 one, ≥3 % rule:
   `zero/engine_setup.h:802-863`), and that one answer bound **every** prompt tile, wide or narrow.
4. On this laptop the question cannot even be asked: **Xe-LPG has no XMX** (`docs/lessons-intel-arc-igpu.md:30`;
   `zero/engine_setup.h:534-536`). The peer engines' ×1.6–2.0 is real matrix hardware (AMD WMMA), not Xe-LPG.
5. What differs now is only the **dispatch**: `ARC_PROMPT_DPAS=1` asks per launch — DPAS for chunks of
   `ARC_PROMPT_DPAS_T` tokens (default 256) and more, DP4a below — instead of one verdict for the whole prompt
   path. That is the part this node can test; the instruction itself is missing.

## 2. METHOD

- **Code under test.** `prompt_dpas_for(tokens)` / `prompt_dpas_t()` in `zero/engine_setup.h`, called by the plain
  prompt GEMM (`zero/engine.h`, `xd = prompt_dpas_for(Mp)`) and H8's (`zero/engine_h8.h`). It returns −1 (the
  engine's verdict, the pre-switch call), 1 (DPAS) or 0 (DP4a). It returns 0 outright when the GPU's driver offers
  no matrix engine, so on Xe-LPG the flag cannot route anything: the start-up line says so
  (`inferzero: ARC_PROMPT_DPAS: this GPU's driver offers no DPAS (Xe-LPG has none): prompt GEMMs on DP4a`).
- **Is there DPAS at all?** No, twice over: the driver's device extensions carry no
  `cl_intel_subgroup_matrix_multiply_accumulate`, and the engine's own `-DXMX` build fails at
  `intel_sub_group_i8_i8_matrix_mad_k32` ("use of undeclared identifier"; `results/p5-xmx-try1.log`,
  `ARC_XMX_TRY=1`). Xe-LPG has no XMX units — this is a hardware fact, not a driver gap.
- **Protocol per cell.** `runz <model> --bench-pp {512,1024,2048} --n 0`: the model is loaded fresh, a 64-token
  warm-up runs, then three timed prompt passes (`eng.reset()` between them). The median of the three is the cell.
  60 s of idle between invocations, model reloaded each time — the published `--bench-pp` protocol.
  - **A (default):** the flag unset.
  - **B (flag):** `ARC_PROMPT_DPAS=1`.
  - **C (stand-in):** `ARC_PROMPT_DPAS=1 ARC_XMX_TRY=2` — `gemm.cl` built with `-DTP=8` and DP4a in place of the
    DPAS calls (the DPAS kernel's 8-token step, on the vector units). This is the only configuration on this node
    where the routing has any effect, and it is **bit-identical to A** (§3.4): its speed difference is the kernels'
    shape, not their arithmetic — though §3.3 shows the try reaches past the prompt GEMMs too.
- **Bit-identical check.** `runz <model> "$(cat results/p5-prompt.txt)" --n 8 --dump-logits F`: branch base
  (`dd2e6a9`, no switch in the source), flag off, flag on, stand-in — 17 rows × 248 320 f32 = 16 885 760 bytes each.
- **Hygiene.** `pgrep -ax runz; pgrep -ax serverz` empty before and after every block. The crashed child had left
  its driver script `~/p5-bench/ppbatch.sh` running for 12 h 15 min (it was what restarted the "stuck" runz after
  the parent killed one); it and its runz were killed at 10:24, before any measurement here, and nothing respawned
  (no cron, no systemd unit). Every bench ran in the foreground under `timeout -s TERM`, never `--n 0` in the
  background, no polling loop.
- **Cache control.** The engine streams the weights (`W8 just-in-time`): a pass reads the whole model, so the
  page cache decides the number when the other model was touched last. Part 1 ran IQ2_M first and Q4_K_S after it
  (Q4 loads 20–95 s, its first cells are the noisiest); the repeat pair and the threshold probes warm the model
  file (`cat > /dev/null`) first.

## 3. NUMBERS

### 3.1 A (default) against B (`ARC_PROMPT_DPAS=1`) — 3 runs each, medians, tok/s

| model | pp | A default | B flag | B − A |
|---|---|---|---|---|
| IQ2_M | 512 | 70.3 / 69.8 / 69.7 → **69.8** | 70.1 / 69.9 / 69.8 → **69.9** | +0.1 % |
| IQ2_M | 1024 | 74.5 / 73.5 / 68.5 → **73.5** | 74.1 / 73.6 / 68.1 → **73.6** | +0.1 % |
| IQ2_M | 2048 | 72.6 / 71.6 / 71.8 → **71.8** | 77.0 / 71.7 / 71.9 → **71.9** | +0.1 % |
| Q4_K_S | 512 | 50.8 / 47.1 / 47.6 → **47.6** | 52.6 / 52.4 / 48.7 → **52.4** | +10.1 % (noise, see 3.5) |
| Q4_K_S | 1024 | 63.0 / 62.1 / 61.7 → **62.1** | 64.2 / 62.1 / 62.5 → **62.5** | +0.6 % |
| Q4_K_S | 2048 | 72.1 / 71.4 / 71.9 → **71.9** | 74.4 / 72.0 / 72.0 → **72.0** | +0.1 % |

Eighteen invocations, 54 timed passes. Five of the six cells are the same number to 0.1 %; the sixth is noise.

### 3.2 A against C (the DPAS kernel's shape, 8 tokens a step, on DP4a)

| model | pp | A default | C stand-in | C − A |
|---|---|---|---|---|
| IQ2_M | 512 | 69.8 | 43.6 / 44.3 / 41.7 → **43.6** | −37.5 % |
| IQ2_M | 1024 | 73.5 | 47.3 / 45.5 / 44.9 → **45.5** | −38.1 % |
| IQ2_M | 2048 | 71.8 | **no number** — see 3.4 | — |
| Q4_K_S | 512 | 47.6 | 39.2 / 39.2 / 39.2 → **39.2** | −17.6 % |
| Q4_K_S | 1024 | 62.1 | 50.9 / 48.3 / 48.1 → **48.3** | −22.2 % |
| Q4_K_S | 2048 | 71.9 | 61.7 / 58.8 / 54.5 → **58.8** | −18.2 % |

### 3.3 The threshold probes (IQ2_M, pp512, warm page cache, 3 runs each)

| routing | tok/s | against A (69.8) |
|---|---|---|
| `ARC_PROMPT_DPAS_T=4096` — nothing is wide enough, every prompt GEMM on DP4a | 19.7 / 18.3 / 19.8 → **19.7** | −71.8 % |
| `ARC_PROMPT_DPAS_T=512` — the whole 512-token chunk on the stand-in | 44.5 / 41.7 / 42.0 → **42.0** | −39.8 % |
| `ARC_PROMPT_DPAS_T=128` | not run: at pp512 the prompt is one 512-token GEMM, so 128 and 256 route identically | — |

(The same `T=4096` probe read **13.4 tok/s** on its first attempt, 10:34, with the IQ2_M file pushed out of the page
cache by the Q4_K_S run before it: 13 744 MB of weights in 38.1 s is 0.36 GB/s, i.e. the disk. That is why the
probes above are run twice, the second time with the model file warmed.)

`ARC_PROMPT_DPAS_T=4096` was meant as the control that isolates the routing ("nothing routed, so it must equal A").
It does not: under `ARC_XMX_TRY=2` the engine also takes its other XMX paths (`xmx_` gates H8,
`zero/engine.h:489`, the XMX attention kernels, `xe2_`'s 2D loads and the MoE tile kernel,
`zero/engine_setup.h:553-557`), and those stay on the emulated kernels whatever the threshold says. So **C is not a
clean isolation of the wide prompt tiles** — it is "the whole prefill with the DPAS-shaped tiles where they apply,
and the try's other paths on their stand-ins elsewhere". The direction is unambiguous (every routed configuration
is slower), the exact share of the loss that belongs to the wide prompt GEMM alone is not measurable here.

### 3.4 Bit-identical, and the one cell that has no number

- Branch base `dd2e6a9`, flag off, flag on, stand-in: **all four dumps sha256
  `c1f52324fe6bc148d1530fd412cfb0f4f42accbd46f0e646469117a4249123a6`** (16 885 760 bytes each; same top-10, same
  greedy tokens). The child's own three dumps (its 444-token prompt) agree the same way: `ae831fcf…c040a` for base,
  off and on. So: the switch off is the code path before it, the switch on is a no-op on this GPU, and the stand-in
  computes the same numbers as DP4a.
- **C at IQ2_M pp2048 has no number.** The stand-in run at 22:15 exited 1 after loading (no `pp2048 run` line), and
  the `T=4096` variant at 22:56 never returned — it hung until it was killed at 10:23 the next morning (the 11 h
  process in the brief). Both are the emulated wide-tile kernel at 2048 tokens tripping the driver on this node;
  the same kernel is fine at 512 and 1024 (Q4_K_S 2048 too, at 58.8 tok/s). Not chased further: it is a proxy, and
  the verdict does not rest on it.

### 3.5 Noise, and why the Q4 pp512 cell is not a win

The within-invocation spread is 5–10 % on this node (A pp512 IQ2: 70.3 → 69.7; B pp512 Q4: 52.6 → 48.7), and the
engine streams the whole model per pass, so the page cache decides a cell when the other model was touched last.
Q4_K_S pp512 is the worst case: part 1 measured A 47.6 against B 52.4 (+10.1 %) right after a 90 s load, and the
repeat pair, warmed, measured A 49.2 against B 53.0 (+7.7 %). With **the same kernel and the same arguments** (the
flag only turns `gemm_kernel`'s `xmx` argument from −1 into 0, and on a GPU with `xmx_` false both select the same
DP4a kernel — proved by the identical dumps) an 8–10 % difference cannot be the switch. It is the cell to re-measure
on a quieter node if anyone wants a clean number for Q4 at 512.

## 4. VERDICT

**NEGATIVE.** The gate asked for ≥ +30 % prefill at some length (70 → 120+ tok/s). Nothing here is within 30 points
of that: B ties A in five of six cells (+0.1 %, +0.1 %, +0.1 %, +0.6 %, +0.1 %) and the sixth is noise; every
configuration in which the routing is observable is 18–40 % **slower**.

1. **There is no DPAS on this node to route to.** Xe-LPG has no XMX units and its driver exposes no DPAS
   (`gemm.cl -DXMX` does not build). The switch is a dispatch change; with nothing to dispatch to, B is A bit for
   bit and tok/s for tok/s. The ×1.6–2.0 the sibling engines get is WMMA hardware, and this iGPU does not have it.
2. **The tile shape alone does not pay here.** The DPAS kernel's 8-token step on DP4a, bit-identical output, is
   18–38 % slower than the DP4a default at every length that could be measured. So "wide tiles were the missing
   ingredient" is not supported: wide tiles on the vector units are worse, which is what the 2026-11-07 140T loss
   said in the first place.
3. **What remains unmeasured is the switch's actual question** — DPAS on wide prompt tiles on a GPU that has DPAS.
   That needs a B-series card, an Arc 140T or a Lunar/Panther Lake iGPU. The switch stays in the tree, default off,
   with `ARC_PROMPT_DPAS_T` to tune the threshold there; the same six-cell table is the measurement to repeat.
4. **What the prefill time actually is, this result does not change.** ~6.2 T int8/s at 70 tok/s against an
   18.4 TOPS DP4a peak is a question about the dot loops and the traffic they wait on (the 2026-11-08 row's gate
   GEMM at ~3.9 T MAC/s, and the load-bound dots), not about tile width. If prefill is to gain ×1.6 on this class
   of iGPU, the evidence so far points at reading fewer bytes (quantisation, sparsity, H8's rotation), not at a
   different GEMM tile.

No `docs/before-after.md` row and no CHANGELOG entry: nothing ships and nothing changes by default (`9551dbb` is
already committed with the switch off). A one-bullet note under "What did not pay off" is the natural home if the
ladder wants one; the numbers above are what it would cite.

## 5. Files

- `zero/engine_setup.h` — `prompt_dpas_on()` / `prompt_dpas_t()` / `prompt_dpas_for()` (commit `9551dbb`)
- `zero/engine.h`, `zero/engine_h8.h` — the two call sites (commit `9551dbb`)
- `results/p5-bench.log` — the 24 bench invocations, raw
- `results/p5-dump.log`, `results/p5-logits.sha256`, `results/p5-prompt.txt` — the bit-identical check, raw
- `results/p5-xmx-try1.log` — the `-DXMX` build failing on this GPU (no DPAS extension to compile against)
