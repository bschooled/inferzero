# Gufo (Strix Halo / gfx1151) — code-level findings for an Intel Meteor Lake inferzero engine

Repo analysed: `/home/bschooley/.hermes/cache/scratch/gufo` (in-tree `version.txt` = **0.35.0**).
Everything below is read from that checkout; every claim carries `file:line`.
Inferences are marked **[inference]**; anything I could not establish is under *Unresolved*.

> Note on the requested deliverable path: the task asked for
> `/home/bschooley/.hermes/cache/scratch/research-notes/gufo-findings.md`, but this session is
> read-only outside this findings file, so the full report lives here. Copy it to
> `research-notes/gufo-findings.md` unchanged.

Scale sense: Gufo is a single-machine, single-GPU engine for one chip (`README.md:14-19`,
principle "**One machine.** Every kernel is tuned for Strix Halo with 128 GiB", `README.md:115`).
It is HIP/ROCm, Wave32 RDNA3.5, and deliberately not portable. What transfers is *mechanism and
method*, not code.

---

## 0. Where the time actually goes (needed to read the rest)

Gufo publishes a per-draft-width **cycle cost table** measured on gfx1151 at C1/d0
(`src/models/qwen38_flash_next/mtp_costs.hpp:18-26`), milliseconds per full speculative cycle:

| width (drafts verified) | 0 (headless AR + catch-up) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| C1, context 0 | 38.5 | 47.3 | 56.1 | 65.7 | 74.1 | 82.2 | 89.7 | 96.8 |

Two facts fall out, and they frame everything else:

* Predicted AR decode = 1/0.0385 s = **26.0 tok/s** vs **28.29 tok/s** measured
  (`docs/internal/models/qwen3.8-flash-next/BENCHMARKS.md:17`) — the table is the real thing.
* Each additional MTP draft token costs **~8.7 ms** at d0/C1, ~23 % of an AR step (deltas of that
  row, `mtp_costs.hpp:21`). That is a *sequential chain* cost, not a bandwidth cost — exactly the
  shape of inferzero's binding constraint (`engine_drafters.h:502`).

Gufo did **not** fix the sequential draft chain by parallelising it. It fixed the problem around it:
(a) verify the whole block in one pass, (b) decide per depth whether a draft is worth its 8.7 ms,
(c) add a zero-weight second drafter (copy lookup) that skips the chain entirely. §2.3, §2.4, §2.5
are the payload for inferzero.

---

## 1. Unified memory pool management

### 1.1 Host/BIOS configuration is part of the design (directly analogous to MTL)

`docs/install.md:26-52`:

* BIOS iGPU carve-out set to **minimum 512 MiB**; the GPU must use **GTT** (system memory it can
  map), not a reserved carve-out.
* Kernel parameters `amdgpu.gttsize=126976 ttm.pages_limit=32505856` — i.e. *RAM in MiB − 4096* and
  *that × 256*. "For 128 GiB of RAM they give the GPU 124 GiB and leave 4 GiB to the system."
* `ulimit -l unlimited` / `LimitMEMLOCK=infinity`.
* Why (`docs/install.md:54-57`): "with a GTT limit well below your RAM the RAM cache cannot borrow
  memory that sessions have reserved and not used yet, and with locked memory limited, GPU work can
  stall for seconds while the kernel compacts memory."

**So the `--ulimit memlock=-1` in the README container snippet (`README.md:73`) is not for
weights.** It exists because Gufo pins *host buffers* long-term (§1.4). Weights are never mlocked.
That distinction matters for MTL, where "pin the weights" is usually the wrong reflex.

### 1.2 Weights: mmap → prefault → staged upload or registered host pages

`src/models/qwen/hip/model_loader.cpp`:

* GGUF shards are `mmap`ed read-only (`src/core/platform/mapped_file.cpp:262-275`, `MADV_SEQUENTIAL`
  at `:270`).
* **Prefault** (`src/core/platform/prefault.hpp:67-143`): 16 threads × 16 MiB chunks issuing
  `MADV_POPULATE_READ` (`:97-98`, `:120-122`). Rationale `:60-66`: "Parallel faults expose disk
  queue depth without making another weight copy."
* **Upload** (`model_loader.cpp:85-154`): eight workers each `memcpy` 64 MiB chunks into their own
  `hipHostMalloc` staging buffer, then `hipMemcpy`. The comment is the finding: *"a shard in the page
  cache uploads at memory speed. A `hipMemcpy` straight from the pageable mapping pins it page by
  page and runs at a small fraction of that"* (`:88-90`). Fallback is one 16 MiB staging buffer
  (`:134-152`).
* **Placement decision**, per shard (`:204-249`):
  1. free − shard ≥ **4 GiB dispatch reserve** → `hipMalloc` device region (`:216-226`; constant
     `src/core/hip/heap_placement.hpp:8-16`);
  2. else → `MapRegisteredRegion` (`:36-67`): clone the mapping into anonymous pages,
     `hipHostRegister(Mapped|ReadOnly)`, use the device pointer — **one copy, GPU reads host
     pages** (`:62-66`);
  3. registration refused → device region filled by `WeightUpload` with bounded host residency
     (`:156-189`; comment `:156-160`: "Copying through a mapping faults the whole shard into host
     memory first").
* Why device memory is preferred when it fits (`:209-215`): *"Device memory is not on the kernel's
  page LRU, so compaction never migrates it. A registered anonymous copy is."*
* Managed memory is last resort (`src/core/hip/managed_alloc.hpp:15-33`): `hipMallocManaged` +
  immediate `hipMemAdviseSetCoarseGrain`, because fine-grain would keep GPU coherent with CPU on
  every kernel touch. `:46-49` gives the APU sizing rule: the device heap is a cap
  (`ttm.pages_limit`, half of RAM by default) while managed allocations draw on the whole system
  pool — "Both are the same DRAM".

There is **no weight eviction or paging policy**: the model must fit, residency decided once at
load. Deliberate, from "one model per use case" (`README.md:21`).

### 1.3 KV / session state: reserve address space, map pages on demand

`docs/internal/state-memory.md` is the most useful document in the tree for our problem.

* A **region** is one virtual range for the full context, split into segments that grow from their
  first row; `Ensure(rows)` maps 2 MiB pages for `[0, rows)`, `MapAhead(rows)` maps in background.
  "The address never changes, so captured graphs and kernel arguments stay valid and growth copies
  nothing." (`:10-18`)
* Models call `Ensure` at one choke point per model, map 8,192 rows ahead so the scheduler normally
  makes no HIP call, back a 2,048-row tile past the frontier because attention reads are
  tile-rounded (`:20-29`).
* **Page pool** (`:38-56`): creates 2 MiB pages ahead (up to **8 GiB** at load, before the heap
  fragments), keeps 256-512 free, and only creates pages from free 2 MiB blocks in `/proc/buddyinfo`
  above a **free floor**, because "Creating a 2 MiB page with a few GiB free forces compaction, and
  compaction that touches GPU-visible pages evicts all of the process's GPU queues" (`:46-49`).
  Measured cost of getting this wrong: 1-6.5 s of queue eviction per run vs 3 ms with the floor
  (`:105-111`).
* **Ledger** (`:58-73`): bytes *promised* to regions at full size, *mapped*, *pooled*; every
  admission/cache budget subtracts outstanding bytes so budgets "see the memory they saw when
  sessions allocated their whole context at creation".
* **Session sharing** (`:276-334`) when N full contexts do not fit: sessions share
  `full_sessions × session_bytes − sessions × fixed_bytes` of demand-mapped state, per-session
  accounts charge pages; scheduler does admission → reclaim → queue → decode claim → preempt →
  refuse with distinct HTTP codes.

### 1.4 Host buffer pinning (the real meaning of memlock)

`state-memory.md:429-442`: GPU-visible host buffers are **anonymous pages pinned long-term via
io_uring buffer registration**, not `hipHostMalloc`, because "Compaction keeps trying to migrate
pinned `hipHostMalloc` pages; each failed attempt invalidates their GPU mapping and evicts every GPU
queue of the process (a probe froze the GPU for 45-109 s)". Pinning counts against
`RLIMIT_MEMLOCK`; where it fails (systemd default 8 MiB, Windows) it falls back to `hipHostMalloc`.
Never register/unregister/free pinned host memory while serving: 60-547 ms of queue eviction per
2 GiB.

### 1.5 The RAM cache borrows device pages (dma-buf, not pinned host memory)

`state-memory.md:75-121`: cache payloads live in pool pages mapped for GPU, exported as dma-buf, and
mapped for CPU at the same address, so one pointer serves memcpy, kernels and copies. Measured
gfx1151 Nov 2026 (`:80-85`):

| path | GB/s |
|---|---|
| CPU read of these pages | 28 |
| CPU write | 26 |
| GPU copy into them | 107 |
| registered host pages (comparison) | 51 |

A **4× bandwidth difference for the cache tier**, which is why they bother.

---

## 2. Speculative / MTP implementation

### 2.1 Separate head GGUF, but it *shares* the target's embedding and output matrix

`README.md:56-64` downloads `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,786,568,256 B =
2.79 GB; `src/models/qwen38_flash_next/models.lock.json:48-50`) next to the 4-shard Q4 target.

`src/models/qwen/hip/mtp/model.cpp:247-248` is the meaning of "shared":

```cpp
weights->token_embedding = target_weights.token_embd;
weights->output          = target_weights.output;
```

The head keeps only its fusion/attention/FFN matrices and norms; embedding and the Q8_0 output
projection are **borrowed by pointer**. Its own matrices are dequantized once and repacked to
**bf16** in 32-row chunks (`model.cpp:105-138`, `kPackChunkRows` `:19`), ~2.6 GiB resident.

**inferzero check**: if our `mtp-*.gguf` carries embed/output copies, GiBs and a load pass are being
wasted; sharing by pointer is trivial.

### 2.2 Drafting is a **sequential autoregressive loop** — same shape as ours

`src/models/qwen/hip/mtp/backend.cpp:123-161`:

```cpp
auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_, checkpoint, true);
proposed_tokens_.push_back(token);
for (std::uint32_t index = 1; index < count; ++index) {
  token = executor_->ForwardFeedback(token, checkpoint + index, true);
  proposed_tokens_.push_back(token);
}
```

No tree, no parallel heads, no verification-free lookahead. One MTP-layer forward per draft token,
each `Run` ending in **GPU argmax + one D2H copy + full stream synchronize**
(`src/models/qwen/hip/mtp/executor.cpp:264-271`). The MTP executor keeps **its own KV cache** sized
to the full context (`executor.cpp:100-102`) on its own stream (`:79-83`), and drafts run **on the
same GPU**; there is no CPU drafter in the MTP path.

After verification the backend **rewinds and replays** only the accepted rows with their *target*
hidden states, because tentative draft hidden states must not feed the head
(`backend.cpp:163-193`, comment `:175-178`).

Budget: `count = min(max_tokens, config_.max_draft_tokens)` (`backend.cpp:138`) with
`kMaxMtpDraftTokens = 7` (`src/models/qwen38_flash_next/mtp_policy.hpp:15`).

### 2.3 Verification: **one batched pass over the whole block**

Production sets batched verification on (`src/models/qwen/runner.cpp:1489-1495`; also
`src/models/qwen/drivers.cpp:287,439,666`). `SpeculativeVerifier::VerifyStep`
(`src/core/speculative/speculative_verifier.cpp:249-272`) builds
`inputs = {current_token} ∪ proposal.tokens` (`:426-435`), calls
`ForwardVerificationChunk(inputs, position, ...)` **once**, then walks per-row predictions accepting
until the first mismatch (`:536-595`). On a partial accept it restores and re-commits only accepted
rows (`:596-661`; `SaveState` `:434`; `RestoreState`/`CommitVerificationChunk` `:655-660`).

Widths above 8 rows are supported: `kMaxWideVerificationRows = 64`
(`src/models/qwen/hip/executor.hpp:518-522`) — "A chain longer than eight rows (a prompt-lookup
copy) shares every projection across its rows and runs attention in **eight-row segments**, so each
row has the arithmetic of consecutive ordinary verifications; the replay ring grows to hold it."

There is a deliberate **non-batched** path for greedy/unpenalised decoding
(`speculative_verifier.cpp:254-259` → `VerifySequentialStep`, `:887-964`): token by token, break at
first mismatch so rejected rows cost nothing, then replay the accepted prefix. Library default is
`use_batched_verification{false}` (`speculative_verifier.hpp:26`) — the *library* default optimises
wasted work; the *model runners* override to batched. **On a bandwidth-bound iGPU batched is the
correct default**, which is also inferzero's conclusion.

Lossless sampling machinery exists (`VerifySampledToken`, `speculative_verifier.hpp:144-167`); a
copy proposal is point mass `q=1` (`src/models/qwen/hip/dflash/backend.cpp:589-595`), keeping
rejection sampling exact.

### 2.4 Draft length is adaptive and *cost-aware*, not acceptance-only

Core path (`speculative_verifier.cpp:143-166`): 16-entry rolling window of acceptance, `+1` above
0.70, `−1` below 0.35, clamped to `[1,7]` (`speculative_verifier.hpp:19-28`).

The real one is `MtpLengthController` (`src/models/qwen38_flash_next/mtp_policy.hpp:50-197`):

* per-depth `successes[7]`/`failures[7]`, EWMA decay 0.75 (`:98-103`), `p(depth)=s/(s+f)`
  (`:164-170`);
* a `failed_depths` bitmask: **once a depth has ever rejected, its probability is 0** (`:117-118`,
  `:165-167`);
* choose `argmax_d E[tokens](d) / cost(d)`, cost from `MtpCycleCosts(context, concurrency)[d]`
  normalised to width 0 (`:171-197`; table `mtp_costs.hpp:18-43`);
* **censoring is explicit**: only the first rejection is an observed failure; a fully accepted,
  budget-limited chain is censored, so the controller *probes* deeper —
  `probe_depth = min(2×drafted, limit)` every 16 cycles (`:104-114`);
* best width 0 ⇒ 16-token AR backoff (`:119-120`), decremented per AR token (`:129-132`), then one
  probe draft to detect recovery (`:73-77`);
* a batch controller picks **one width for the physical batch** (≤8 rows) from measured ms per width
  (EWMA 0.25, two warm samples), context bins of 4096 doubling, and **periodic headless cycles**
  (`cells % 32 < 2 → width 0`) that keep measuring the true AR alternative including "predictor
  catch-up" (`:208-265`, `:230-231`). It refuses cost updates after a copy cycle because the
  following catch-up pollutes them (`:267-272`, `:288-293`).

### 2.5 A second, zero-weight drafter: prompt-lookup copies

`src/core/speculative/prompt_lookup.hpp` — per-session index over committed tokens, no GPU work:

* key = **3 tokens**, packed as 3×21 bits into a u64 (`:43`, `:319-327`); open-addressed hash of
  key → most recent occurrence plus a `prev[]` chain of earlier ones (`:232-309`);
* minimum match **12** greedy / **24** sampled (`:44-47`); up to **64** proposal tokens (`:48`); 16
  candidate occurrences examined (`:49`);
* **resume**: after a mostly-accepted copy only **5** matching tokens are needed inside a 512/8-token
  window around the last copy's end (`:50-54`, `:114-116`, `:201-210`) — for output that edits a line
  of the text it is copying;
* message-framing guard rejects candidates spanning a boundary token so a repeated closing
  tag/header is not copied (`dflash/backend.cpp:552-564`);
* sampled requests index only their own starting context, because "earlier parts of a freshly
  generated answer are especially poor sources for new prose/code" (`prompt_lookup.hpp:22-23`).

A hit **short-circuits neural drafting** (`from_lookup`, `dflash/backend.cpp:525-601`); the verifier
reports `lookup_n` / `lookup_n_accepted` (`docs/models/qwen3.8-flash-next.md:43-45`). Published
effect: repetitive d0 decode **35.79 → 116.93 tok/s** vs mixed (`flash-next/BENCHMARKS.md:37`); a
later build improved repetitive 32K from 24.02 → 53.93 tok/s purely from "prompt lookup skips neural
drafting for verified copies" (`27b/BENCHMARKS.md:38-40`).

---

## 3. gfx1151 kernel tuning for a shared-LPDDR iGPU

### 3.1 Measured bandwidth ceilings (the numbers to calibrate against)

| quantity | value | source |
|---|---|---|
| sustained decode bandwidth, dense Q4 27B | **~218 GB/s** (implied, §5) | `27b/BENCHMARKS.md:114` |
| read-only stream of one kernel's weights | **~224 GB/s** | `deepseek-v4-flash/EXPERIMENTS.md:28` |
| 8 KiB F16 rows, power-of-two stride | 103 GB/s | `deepseek-v4-flash/EXPERIMENTS.md:16` |
| same rows, +512 B padding per row | **203 GB/s** | same |
| device→device copy | 104-110 GB/s | `docs/internal/design/cache-redesign/RFC.md:1769-1772` |
| pinned host copy | ~85 GB/s; pageable 61-83 | same |
| kernel that "sustains 124 GB/s in the model" | 400-860 GB/s when measured in a repeat loop | `docs/internal/performance.md:346-349` |

Strix Halo is 256-bit LPDDR5X-8000 (~256 GB/s theoretical), so decode runs at **85-88 % of
theoretical**, ~97-100 % of a measured weight-stream ceiling. **The calibration for inferzero**: our
66.5 GB/s is 74 % of our 89.9 GB/s measured peak; the Gufo-grade equivalent is ~78-80 GB/s.

### 3.2 Measurement discipline (why most iGPU microbenchmarks lie)

`docs/internal/performance.md:346-354`, verbatim:

> "Strix Halo has a 32 MB MALL, and a single decoder projection is 4-25 MB. Timing one shape in a
> repeat loop leaves it cache resident and reports 400-860 GB/s for a kernel that sustains 124 GB/s
> in the model. A decode-bandwidth harness must therefore allocate the whole per-token weight
> footprint and time a full token pass… `__builtin_nontemporal_load` bypasses the MALL, and on every
> streaming kernel measured so far it has cost more than half the achieved bandwidth rather than
> helping."

Enforced in the sweep tool: "Each launch uses another copy of the weights, enough copies for at least
512 MiB, so no run reads from the 32 MiB MALL cache. Without that, matrices under ~35 MB look faster
than the memory allows." (`tools/qwen27b/verify_sweep/README.md:53-55`).

**MTL has no MALL, but both of our headline numbers (89.9 GB/s block read, 57.3 GB/s at 16-byte
loads) are candidates for the same artefact**, and the 57.3 GB/s figure especially should be
re-measured with a rotated, full-footprint stream before we conclude MLP is the problem.

### 3.3 What the decode kernels actually do

* **Wave32** for decode; Wave64 only for wide verification attention ("wave64 verification
  attention", `BENCHMARKS.md:8`; `HardwareWaveSize` template `small_batch_gemm.hpp:84-93`;
  `attention_decode_verify_wave64.hip:16-36`).
* **16-byte weight loads**: `uint4`/`PackedWeights` (4×u32) per lane per row-group
  (`small_batch_gemm.hpp:137-155`); activations staged as `float4` (`:122-129`).
* **LDS staging of activations, strided to break bank conflicts**: `kStride = kValuesPerVector + 4`
  = 12 floats (`:99-103`); Q8_0 `kStride = kQ8_0BlockSize + 4` (`:260`) plus a transposed "XOR
  stage" layout fitting 16 activation rows in 64 KiB LDS (`:258-262`, `:287-290`). Comment `:71-73`:
  "These GEMMs exchange data through LDS only. Global inputs are read-only, and output stores follow
  the final tile" — hence `s_waitcnt lgkmcnt(0)` + barrier, no global fence.
* **Multiple output rows per wave** (`RowsPerWave`) so one activation read feeds several dots
  (`sums[RowsPerWave][Batch]`, `:110`, `:156-175`).
* **8-value dot per lane per row**, hand-unrolled (`:162-175`) — no dp4/dot8 intrinsic on the F32
  path; K-quant paths use integer block-dot helpers (`kernels/gemv_quant.hip:67-88`).
* **Cheap reductions**: `__builtin_amdgcn_ds_swizzle` instead of `__shfl_xor` ("Immediate XOR masks
  avoid the lane-address calculation of `__shfl_xor`", `small_batch_gemm.hpp:27-39`); DPP row
  exchanges for 16-lane heads (`attention_decode_verify_wave64.hip:19-20`); `__shfl_xor` lowers to
  `ds_bpermute_b32` i.e. the LDS path (`kernels/ssm_row_split.hip:118`).
* **Token groups** share weight reads across independent sequences in one launch
  (`small_batch_gemm.hpp:250-253`).
* **Load-hint chosen at dispatch, compile-time**: "Select the cache hint at dispatch: a runtime
  branch here makes the large feature projection slower, even when it is uniform." (`:142-153`).
* **Verification attention >8 rows**: split at 8 rows, 16-lane heads, 2 KV heads/block, split-K over
  the context, `head_dim` fixed 512 with a `false` return otherwise
  (`attention_decode_verify_wave64.hip:16-62`).
* **Launch-count reduction matters**: "Each launch costs about 1.6-2 µs of GPU gap on gfx1151"
  (`deepseek-v4-flash/EXPERIMENTS.md:40`); folding quantize+RoPE+KV-store into producers cut
  1,720 → 1,205 launches and gave 18.10 → 18.48 tok/s (+2.1 %) at d0.

### 3.4 The per-shape launch-geometry table (biggest kernel-level win found)

`src/families/qwen/kernels/verify_gemm_27b.hip:12-16` documents the file as launch geometries
"measured faster on gfx1151 … outputs are bit-identical to `LaunchBatchedQuantGEMMFp32`". The Linux
table (`:578-605`) lists, per (type, m, k, batch), a
`Launch<T, Batch, WavesPerBlock, RowsPerWave, TokenGroups, …>` with its measured gain. Q4_K
`attn_k/attn_v` (m=1024, k=5120), 2-8 rows:

| batch | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| gain vs generic | +15 % | +19 % | +14 % | +55 % | +50 % | +41 % | +65 % |

Method (`tools/qwen27b/verify_sweep/README.md:44-57`): real GGUF tensors at their file offsets; every
candidate compared **bit for bit** with production, inexact candidates never timed; alternating
rounds, median, 20 launches/sample after 3 warm-ups; ≥512 MiB of weight copies per launch to defeat
the MALL; a shape is listed only if it beats generic by **≥5 % in two independent sweeps**
(`:72-87`). Toolchains differ enough that a new compiler needs a new sweep (up to 29 % regression
when a Windows shape is used on Linux, `:96-100`).

### 3.5 Cheap exact wins to check on MTL first

1. **Row padding.** `deepseek-v4-flash/EXPERIMENTS.md:16`: 8 KiB F16 rows read at **103 GB/s** with a
   power-of-two stride and **203 GB/s** with 512 B (256 halves) padding, +122 MiB, outputs
   bit-identical; ratio-4 compressor pair 1.97×; tg64 17.59 → 18.66 at 32K. Highest gain per line of
   code in the repo.
2. **Skip LDS staging for small decode GEMV inputs.** `deepseek-v4-flash/EXPERIMENTS.md:15`: read `x`
   from global at 16 rows/block instead of staging; small shapes 1.11-1.45× faster. (Opposite of the
   big-kernel rule — shape-dependent, must be measured.)
3. **Diagnose with a stream-ratio.** `deepseek-v4-flash/EXPERIMENTS.md:36`: "The kernel runs at
   **1.26× the time of a read-only stream of its weights**, so little decode latency is left to
   hide" — that ratio tells you whether prefetch/overlap work exists at all.
4. **Reject** bandwidth-neutral micro-optimisations: GPU argmax to avoid a D2H (`:29`, +0.03-0.12 %,
   inside spread); FP4 mirror of keys when the kernel is not BW-bound (`:22`).

---

## 4. Prefix / context cache ("built for coding agents that resend long conversations")

Design in `docs/cache.md` + `docs/internal/cache.md`.

* The retained unit is a **checkpoint of all model state**, not just K/V: append-only rows (shared
  across checkpoints) + private state (recurrent/sliding-window/frontier) that is copied
  (`internal/cache.md:23-33`). Private state exists only at its exact boundary, so **reuse requires
  an exact token prefix match** (`:126-129`) — a miss can agree on 7,094 of 7,103 tokens and still
  re-prefill everything.
* **Live frontier first**: if a session's state already ends at a prefix of the new prompt, reuse in
  place with no restore and no device copy, including generated tokens (`:106-113`).
* Checkpoints: up to **4 intermediate** per request on a **2,048-token grid**, plus branching
  fallback and complete prompt; capture runs asynchronously (`:216-231`).
* **Learned divergence points**: when a new prompt's common prefix with a retained checkpoint exceeds
  what it could already restore by **≥512 tokens**, prefill stops there and the position is retained
  at "stable boundary" priority — this is what makes N conversations sharing one 4,500-token system
  prompt each restore the whole thing from the third one on (`:233-260`).
* **128 immutable checkpoint records**, independent of `--sessions` (`:262-266`).
* Eviction order (`:273-306`): full-prompt retry copy → intermediate copy with a stable fallback →
  older stable boundary covered by a newer one → oldest-used. A short checkpoint whose fixed state
  costs > 8× its rows leaves before idle deep conversations, but only after a sixteenth of the budget
  was admitted since its last use.
* **Concurrent shared-prefix admission** (`:158-212`): a cold request sharing ≥512 tokens with a
  request already prefilling waits; the leader publishes a checkpoint mid-prefill; the follower
  restores. Measured at `--sessions 4` with a 3.8k shared system prompt: **13.4 s → 4.6 s**.
* **Hit-rate / value**:
  * second turn, same conversation: 101,545 tokens **280.4 s → 2.4 s** (27B); 203,047 tokens
    **184.4 s → 0.7 s** (Flash-Next) (`internal/cache.md:63-66`; `docs/cache.md:14-17`);
  * a recorded 28-turn `pi` coding session **reused 95.9 % of its prompt tokens**; with reuse broken
    it "reprocessed 9.3× the tokens and took 2.5× the wall time, at only 15k tokens" (`:68-70`);
  * disk restore of 24,866 tokens after restart **1.8 s vs ~50 s from scratch** (`docs/cache.md:101`).
* Budgets: RAM cache default = half of available RAM at load, cap 32 GiB, always leaving 4 GiB
  (`docs/cache.md:65-78`); disk tier off by default, 8 GiB cap, single-owner directory (`:80-99`);
  `cache_prompt` / `cache_persist` per request (`:104-112`); reported via `timings.cache_n` /
  `timings.prompt_n` (`:114-118`).
* **For inferzero**: worth more than any kernel change for agent workloads, and nothing here is
  exotic — exact-prefix checkpoints + divergence learning + leader/follower admission. Our
  prefill at 33 % of DP4a TOPS makes it worth more here than there.

---

## 5. Benchmarks and derived effective bandwidth

Method (`docs/benchmarks.md:9-28`): release build, Strix Halo 128 GB, dated; pp = 2,048-token prompt,
tg = 128 generated tokens, greedy, thinking off, measured **through HTTP**, repeated against each
model's reference (llama.cpp / antirez ds4); "Depth" rows start from a long cached prefix because
decode slows with context.

Single user, greedy, depth 0 (`docs/internal/models/*/BENCHMARKS.md:17-18,114,131,152,220`;
`flash-next:17,37`; `deepseek:18,39`):

| model | quant | weights | AR tok/s | spec mode | spec tok/s (mixed / repetitive) | implied GB/s (dense: W×tok/s) | implied per-token bytes @224 GB/s |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | UD-Q4_K_XL 16.45 GiB | 17.66 GB | **12.37** | DFlash2 | 26.91 / 64.74 | **218.5** | 17.7 GB (all weights) |
| Qwen3.8-27B | UD-Q8_K_XL 29.30 GiB | 31.46 GB | **7.19** | DFlash2 | 16.44 / 47.69 | **226.2** | 31.5 GB (all weights) |
| Qwen3.8-Flash-Next | UD-Q4_K_XL 103.69 GiB | 111.33 GB | **28.29** | MTP (+copies) | 35.79 / **116.93** | 3,150 → impossible (MoE) | **7.9 GB** ⇒ 7.1 % of weights/token |
| DeepSeek V4 Flash | IQ2XXS 80.76 GiB | 86.71 GB | **17.65** | DSpark (+copies) | 23.79 / 29.57 | 1,530 → impossible (MoE) | **12.7 GB** ⇒ 14.6 % of weights/token |

Derived facts worth carrying over:

* Dense Q4/Q8 land at **218 / 226 GB/s** = 85 % / 88 % of a 256 GB/s bus, ~97-100 % of the measured
  ~224 GB/s weight-stream ceiling. **Dense single-token decode is at the bus limit; nothing is left
  there.**
* DFlash2 mixed 26.91 vs AR 12.37 = **2.18 tokens per target pass**; 26.91 × 17.66 GB = 475 GB/s
  would be impossible, so verification rows share the weight read, as designed.
* For MoE the number that matters is **per-token active bytes** (7.9 GB, 12.7 GB above), not file
  size. Any tok/s ↔ GB/s claim for Qwen3.8-40B must use active bytes.
* Concurrency (rates summed, `BENCHMARKS.md:16`): 27B Q4 AR 12.42 → **74.95** at C8; Flash-Next AR
  28.24 → **125.48**, MTP mixed 35.74 → 127.02, MTP repetitive 116.50 → **245.81** at C8
  (`flash-next:58-62,74-80`).
* Prefill: 1,801 / 649 / 475 tok/s vs llama.cpp 490 / 358 / 237 — Gufo's prefill win is +80 to
  +270 % (`:17,114`). Their prefill is a WMMA I-gemm path, not the decode GEMV path.
* Memory: Flash-Next peaks at 88.78 GiB for a 103.7 GiB model at context 133,121 AR — **less than
  the file size**, because weights are mmapped/registered, not doubly resident (`:102-105`).
* Load: Flash-Next ready 16.42 s vs 117.83 s; 27B Q4 6.27 s vs 23.33 s (`:87-93,247-251`).

---

## 6. Quality-verification discipline and the startup self-check

### 6.1 What "verified" means

`docs/benchmarks.md:30-36`: "A speedup counts only if the output stays correct. Each model has a
quality report that compares Gufo with an independent implementation of the original model, token by
token, and checks that speculative decoding with greedy sampling produces exactly the autoregressive
output."

The mechanism that makes this cheap: **drafts only propose**. `QUALITY.md:3-9`: "greedy verification
takes the target's argmax and sampled verification keeps exact rejection sampling with the proposal's
own distribution, so the output distribution is unchanged." Used aggressively: the draft head's
prefix rows become a **Q4_K copy of the Q8_0 head** (141.6 MiB,
`flash-next/README.md:17-24`), accepted because the copy reconstructs at 7.465 % relative RMS (ggml
reference 7.467 %) with byte-identical greedy/sampled controls (`QUALITY.md:11-33`, +2.6 %
throughput). **Anything in the drafter can be quantised arbitrarily at zero quality cost.**

Reports keep bit-exact operator checks vs a host reference, full-logit sweeps, greedy-token parity,
seeded-sampling replay, snapshot/restore exactness, and an explicit statement of what was *not* run
(e.g. `QUALITY.md:56-66`: "these observations remain unqualified; this is not a global timing
pass"). Promotion gates: `docs/internal/performance.md:466-480`.

### 6.2 The self-check concept

Three layers, none loading a model:

1. **`gufo diagnose`** (`src/cli/diagnose/diagnose.cpp:95-228`): sections `platform`, `gpu`, `npu`,
   `inventory`, `extensions`; checks driver, `/dev/kfd` permissions, GTT/heap size, NPU node and
   whether XRT opens it, toolchain; non-zero exit on failure. `docs/troubleshooting.md:14-17`:
   "checks the driver, device permissions, memory and the extensions compiled in, without loading a
   model… it can't see the state of a server that is already running; use `/health` and `/ready`."
2. **`gufo diagnose --benchmark bandwidth`** (`src/core/diagnostics/bandwidth.h:14-38`): CPU and HIP
   paths, 256 MiB working set, 3 warm-up + ≥10 measured iterations, 2 s minimum per path, reporting
   median/p95/min/max **and every raw repetition**, with a checksum sentinel verified outside the
   timed region.
3. **`gfx1151_peak`** (`tools/bench/gfx1151_peak.hip:1-9`): "Achievable-peak calibration… Every
   accumulator is consumed, every thread has a separate checksum, and host references validate the
   work outside the timed region. These are measured ceilings, not model throughput predictions."
   Reports sustained WMMA INT8/BF16/FP16, VALU FP32 FMA, LDS reads, DRAM read/write/copy.
   `performance.md:295-322`: "Score every kernel against measured ceilings, not spec-sheet numbers."

Plus a **kernel-resource gate in CI**: `tools/ci/check-kernel-resources.py` reads code objects out of
the binary (no GPU) and fails if a kernel exceeds its scratch/VGPR baseline
(`performance.md:234-253`; `tests/tools/kernel_resources_test.py`).

---

## 7. RDMA dual-Strix-Halo project

**Not in this tree.** Only references:

* `README.md:136-138`: "Community projects built on Gufo include
  [Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo), which runs models across two
  machines and **isn't merged yet**."
* `docs/patches/batched-hc-q8-upstream.md:10-13`: the upstreamed patch "contains no RDMA
  implementation and reuses no private fork ABI."

No `rdma`/`ibv`/`verbs`/transport code under `src/`. The engine is single-device by construction and
by stated principle (`README.md:115`), with every state owning one GPU address and streams, queues
and mappings created per device (`docs/internal/state-memory.md:10-18`).

**[inference]** If someone added it, the seams are already cut: `ISpeculativeTargetExecutor`
(`src/core/speculative/speculative_verifier.hpp:81-236`) is a pure virtual interface whose contract is
"given token ids and a position, return next predictions plus optional hidden/logits, with
`SaveState`/`RestoreState`/`CommitVerificationChunk`"; weights are already partitioned per GGUF shard
with their own device-pointer table (`model_loader.cpp:191-272`), so a shard→machine assignment is a
data-structure change. What would *not* survive is the bandwidth-locality assumption: every draft and
verification row would cross the link.

**Conclusion: nothing to borrow on RDMA. Do not spend time here.**

---

## 8. Top-10 transferable techniques (ranked, tagged)

| # | Technique | Evidence | Transferability to inferzero (MTL, Level Zero) |
|---|---|---|---|
| 1 | **Verify the whole draft block in ONE forward pass**, projections shared across rows, attention segmented at 8 rows; production default is batched | `speculative_verifier.cpp:249-272,426-435`; `executor.hpp:518-522`; `runner.cpp:1493`; marginal 8.7 ms is the *draft*, not verify (`mtp_costs.hpp:21`) | **HIGH.** Our 7.95 tok/s at ~37 GB/s says our passes are not amortising the weight read across draft rows. Highest-value structural change. |
| 2 | **Prompt-lookup (copy) drafter** — 3-token key, 12/24-token min match, ≤64 proposals, resume-at-5, boundary guard; skips the neural chain on a hit | `prompt_lookup.hpp:43-54,114-116,157-218`; `dflash/backend.cpp:525-601`; `executor.hpp:522`; 35.79→116.93 tok/s (`flash-next/BENCHMARKS.md:37`); 24.02→53.93 (`27b:38-40`) | **HIGH.** Pure CPU, no weights, no quality risk (verified anyway). Bypasses our sequential MTP chain on repeat-heavy turns — the exact bottleneck. |
| 3 | **Cost-aware per-depth draft controller**: measured ms-per-width table, decayed success/failure per depth, failed-depth bitmask, censored-tail probing (2× depth / 16 cycles), AR backoff | `mtp_policy.hpp:50-197`; `mtp_costs.hpp:18-43`; simple core version `speculative_verifier.cpp:143-166` | **HIGH.** Replaces fixed `--mtp-drafts 3` with `E[tokens]/cost` vs AR. The table is 24 numbers we can measure in an afternoon. |
| 4 | **Per-shape launch-geometry sweep** for 2-8-row projections, ≥5 % gate over two sweeps, bit-exact | `verify_gemm_27b.hip:578-605` (+14…+65 %); method `verify_sweep/README.md:44-87` | **HIGH.** Our verification/prefill shapes are fixed; the sweep is mechanical and the 2-8-row wins are the largest kernel numbers in the repo. |
| 5 | **Pad weight rows to break power-of-two stride aliasing** (512 B per 8 KiB F16 row) | `deepseek-v4-flash/EXPERIMENTS.md:16` (103→203 GB/s, exact, +122 MiB) | **HIGH.** One-line loader change, huge; LPDDR channel hashing on MTL has the same failure mode. Test on our worst width first. |
| 6 | **Weight upload through pinned staging** (64 MiB × 8 workers) + parallel `MADV_POPULATE_READ` prefault, never memcpy from a pageable mapping | `model_loader.cpp:85-154,88-90`; `prefault.hpp:60-143`; 16.42 s vs 117.83 s load (`flash-next:92`) | **HIGH.** Level Zero analogue: `zeMemAllocHost` bounce buffers + parallel copy commands; our load is a serial DRAM copy. |
| 7 | **Single-copy weight residency**: registered/mapped anonymous host pages when the device heap is short, so the GPU reads the *only* copy | `model_loader.cpp:36-67,204-249`; rationale `:209-215`; `heap_placement.hpp:8-16` (4 GiB reserve) | **HIGH (concept), MEDIUM (mechanism).** On MTL "device" memory *is* DRAM, so a host-allocated USM buffer read by the iGPU can halve residency. Needs a measured comparison of `zeMemAllocDevice` vs `zeMemAllocHost`/shared. |
| 8 | **Roofline self-check**: multi-path bandwidth benchmark (medians + raw reps, sentinel-verified) and peak calibration, plus the *kernel time ÷ pure weight stream* ratio | `bandwidth.h:14-38`; `gfx1151_peak.hip:1-9`; `performance.md:295-322`; ratio `deepseek:36`; MALL artefact `performance.md:346-354` | **HIGH (method).** Our 89.9 and 57.3 GB/s numbers were not measured with a rotated full-footprint harness; re-derive before optimising against them. |
| 9 | **Draft head shares the target's embedding + output**, own matrices repacked bf16 at load; the drafter may itself be quantised (Q4_K copy of a Q8_0 head) because drafts only propose | `mtp/model.cpp:105-138,247-248`; `QUALITY.md:3-33`; 141.6 MiB / +2.6 % (`flash-next/README.md:17-24`) | **HIGH.** Check whether our `mtp-*.gguf` duplicates embed/output (GiBs + a load pass), then quantise the drafter freely. |
| 10 | **Pool discipline**: ledger of promised vs mapped vs pooled bytes; demand-mapped state regions; page sharing between checkpoints and sessions; give memory back at idle | `state-memory.md:10-36,58-73,131-201,276-334,336-368` | **MEDIUM.** Level Zero has `zeVirtualReserve`/`zePhysicalMemCreate`/`zePhysicalMemMap` on Xe-LPG, so the mechanism exists, but with 128 GiB shared LPDDR and one session our pressure is lower. The ledger idea is the portable part. |
| — | *(Method)* **Context cache**: exact-prefix checkpoints, 2,048 grid, learned divergence at ≥512-token gain, leader/follower shared-prefix admission | `internal/cache.md:60-70,233-260,158-212`; 95.9 % reuse, 9.3× token saving | **HIGH for agent workloads**, orthogonal to kernels; worth more to us than to them because prefill is 33 % of DP4a TOPS. |
| — | *(Skip-list — negative results)* non-temporal loads on streaming kernels cost >50 % bandwidth (`performance.md:352-354`); GPU greedy argmax +0.03-0.12 %, inside noise (`deepseek:29`); FP4 key mirror when not BW-bound (`:22`); 2-step-ahead branch-free prefetch when already at 1.26× stream (`:36`). | | **HIGH value as a skip-list.** |

---

## 9. Test surface (what they hold themselves to)

* Speculative core: `tests/core/test_speculative_verification.cpp` (batched vs sequential, adaptive
  length, sampled acceptance, batch plans; ~1,500 lines, drives both modes).
* MTP: `tests/models/qwen38_flash_next/{mtp_ops_test,mtp_sampling_test,mtp_audit,wide_verification_test,verification_sampling_test}.cpp`.
* Copy drafter: `tests/models/qwen38_flash_next/prompt_lookup_test.cpp`.
* Cache/state identity: `tests/models/qwen38_flash_next/{snapshot_test,snapshot_copy_test,session_test,continuation_identity_test,continuation_adapter_test}.cpp`;
  helper `tests/support/continuation_identity.hpp`; golden fixtures under
  `tests/models/qwen27b/fixtures/continuation_identity/` encoding every draft-policy option.
* Memory: `tests/core/hip_memory_visibility_test.hip` (asserts a restricted `RLIMIT_MEMLOCK` yields
  `PinnedBytes()==0`, i.e. the fallback is real); `tests/functional/{cache_disk_faults,device_loss_faults}.cpp`;
  `tests/models/qwen38_flash_next/host_pressure_probe.cpp`.
* CI gates: `tests/tools/kernel_resources_test.py` + `tools/ci/kernel-resources.json`.
* Benchmarks as tests: `tests/models/qwen27b/verify_sweep/`, `tools/qwen27b/verify_sweep/`.

---

## 10. Unresolved / not established here

1. Whether inferzero's `mtp-*.gguf` duplicates embed/output (our repo was not read in this pass).
2. Per-token *active* bytes for Qwen3.8-40B — required before any tok/s ↔ GB/s comparison, as Gufo's
   MoE rows make explicit.
3. Whether Level Zero on MTL exposes a host-mapped allocation the iGPU reads at device-allocation
   bandwidth; that decides technique #7.
4. Whether our 89.9 GB/s peak survives a rotated, full-footprint harness (technique #8).
5. Gufo's NVMe/cache-tier numbers (`RFC.md:1769-1790`) are Strix-Halo-specific and unverified for
   transfer; a different subsystem from decode.
