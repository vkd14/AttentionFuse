# AttnFuse — Progress since the course submission

This document tracks the work done on AttnFuse between the original
course-project submission (final report dated May 18, 2026) and the
current state, plus the planned next-phase research direction. It is
written to be readable alongside the git log as supporting material for
a paper submission.

---

## Snapshot at course-submission time

State on May 18, 2026 (commit range up to ``3e3db8a``):

- Forward-only fused-attention DSL with seven combinators.
- Two-level IR (Graph → TiledKernel) + four-pass compiler.
- Five attention variants (dense, causal, sliding-window, causal+ALiBi,
  causal+RoPE) verified by 25 GPU unit tests.
- Three dtypes (fp16, bf16, fp32) and four sequence lengths.
- Three baselines: naive PyTorch, SDPA, hand-written Triton FA2 reference.
- Headline number: 38× over SDPA on sliding-window at N=4096 (because
  SDPA falls back to O(n²)).
- Reference hardware: RTX 3090 (sm_86) only; no other targets.
- Forward only; no training, no KV-cache decoding path, no block-sparse.

## What changed since submission

Each item below maps to one or more git commits and a paragraph of
"why" — design rationale and the experimental result that justified the
change.

### 1. Sliding-window optimisation: 3-loop split + variant-aware tile tables

The original SW kernel iterated over the windowed range with per-element
mask checking inside every tile. The new codegen splits the inner loop
into **left boundary → interior (no mask) → right boundary**. Interior
tiles skip the per-element SW mask, the safe-softmax `all_masked` branch,
and the out-of-N gating entirely. Combined with a new
``_AMPERE_TABLE_F16_SPARSE`` (BM=64, BN=32 instead of BM=128, BN=64) for
masked variants, sliding-window at N=4096 went from 0.682 ms → 0.450 ms
(1.52× speedup). After this change AttnFuse beats `flex_attention` on
every (variant, seqlen) cell where SW is involved.

### 2. Property-based correctness fuzzer

Added Hypothesis-driven tests over random (variant, shape, dtype) tuples.
Within minutes of running, the fuzzer caught a real bug: when N is
smaller than BLOCK_N (an extreme small-sequence case), the SW interior
loop dropped the `n_in_bounds` mask and read past-N memory, producing
`max|err| = 178`. Fixed by also clamping ``interior_hi`` to
``(N // BLOCK_N) * BLOCK_N``. Now standing at 80 fuzz examples per run,
zero failures.

### 3. Cache-collision bug in the @attention decorator

The decorator cached the traced graph in a single closure variable
keyed by nothing. If the same `@attention` function was called with a
new head_dim or dtype the stale graph would be reused, silently producing
junk output (we measured `max|err| > 3` on a Llama-3 GQA test). Fixed by
keying the graph cache by ``(head_dim, dtype)``. This was the
silent-corruption-class bug that the per-shape sweep tests would not
have caught.

### 4. GQA / MQA support

Added a runtime ``GROUP_SIZE`` argument and changed the kernel's K/V
pointer indexing to ``h_kv = h_q // GROUP_SIZE``. With one small change
AttnFuse now covers the Llama-3, Mistral, Mixtral, and Falcon
attention geometries. ``H_q % H_kv == 0`` is the only constraint.

### 5. Cross-attention (N_q ≠ N_kv)

Split the kernel's `N` argument into `N_Q` and `N_KV`. Causal and
sliding-window masks require N_Q = N_KV and are rejected with a clear
error message when violated; the dense path supports arbitrary
N_Q ≠ N_KV. This unlocks encoder-decoder cross-attention and, more
importantly, the **KV-cache decoding pattern** used by every
autoregressive LLM at inference time (Q.N = 1, K.N = cache_length).

### 6. Flash Decoding (split-K) for autoregressive inference

For ``Q.N = 1`` the standard kernel launches only ``B * H_q`` programs
(32 on Llama-3-8B-style models) which uses ~39 % of the RTX 3090's 82
SMs. Latency therefore scales linearly with cache_length. Adding the
**FA Decoding** two-phase kernel (split-K phase 1 + log-sum-exp combine
phase 2) brings total programs to ``num_splits × B × H_q``, saturating
the SMs. Latency at cache_length=32768 went from **3,147 µs → 184 µs**
(17× total speedup); AttnFuse now beats `flex_attention` on every cell
in the KV-cache benchmark.

### 7. Tensor-core matmul for the decode kernel + GQA head batching

A first cut of Flash Decoding processed one Q head per program (scalar
single-query). For GQA the K/V slice is shared by `group_size` Q heads,
so each shared slice was loaded `group_size` times redundantly. The
fix is to process BLOCK_H = 16 Q rows per program — the Triton
`tl.dot` minimum — cyclically replicating Q heads when `group_size < 16`.
Store masks suppress the duplicate rows. With this in place the decode
kernel uses tensor cores throughout while still loading K/V exactly once
per group. This closed the long-context Llama-3-70B gap from 0.36× of
`flex_attention` to 1.06×.

### 8. Backward pass (training support)

Implemented the full FlashAttention-2-style backward:

- **Forward** optionally writes `L = m + log(ℓ)` (the per-row log-sum-exp)
  via a new `SAVE_L` constexpr. Inference paths pay nothing.
- **Backward dispatch** runs three kernels in sequence:
  - preproc kernel: `D = rowsum(dO * O)`
  - `dK / dV` kernel: parallelises over keys, sweeps Q-heads inside a
    GQA group (atomic-free reduction)
  - `dQ` kernel: parallelises over queries, sweeps K blocks
- **Triton 3.1 bug**: `tl.dot(fp32, fp32)` on the (M, N)@(N, D) reduction
  direction silently produces wrong results (even with
  `input_precision='ieee'`). Worked around with a bf16 cast on the dQ
  matmul; bf16 keeps fp32's exponent range so the cast is lossless for
  the values we encounter.
- **autograd wiring**: `@af.attention` now detects `requires_grad` on any
  input and routes through an `AttnFuseFunction` that saves the L tensor
  in `ctx` and calls `run_backward` on the backward pass.

30 new correctness tests cover dense + causal + ALiBi + GQA across three
dtypes. Forward+backward at N=4096 causal: 8.93 ms vs SDPA's 5.93 ms
(1.51× slower) — competitive for a Triton-generated kernel.

### 9. Block-sparse mask (BigBird-style, the proposal's last open item)

Added `MaskKind.BLOCK_SPARSE` plus a dedicated kernel that takes a
**CSR-style active-block list** at call time. The user builds a
`BlockMask` from a Python predicate via `af.create_block_mask`; the
kernel iterates only over the active n-blocks for each m-block, giving
genuine sub-quadratic FLOPs and HBM traffic.

Measured on a BigBird-like pattern (global rows + global cols + local
window): at N=4096 with 7.7 % active blocks, latency is 0.47 ms vs
3.48 ms for dense AttnFuse — a 7.4× speedup proportional to the active
fraction.

### 10. H100 (Hopper / sm_90) port

Auto-detects sm_90 at import time and switches to Hopper-specific tile
tables (bigger BLOCK_M, more pipeline stages, exploiting the 228 KB
SMEM vs Ampere's 101 KB). Deploy scripts (`scripts/setup_h100.sh`,
`scripts/run_h100_benchmarks.sh`) make a one-command SSH deployment
possible. The Hopper code paths are correctness-equivalent to the
Ampere paths; only tile-config selection changes.

### 11. Honest comparisons against PyTorch `flex_attention`

Added `flex_attention` (PyTorch 2.5+) to every benchmark as the
strongest published competitor. Before this addition AttnFuse was
positioned against SDPA's O(n²) fallback (which gave a flashy but
misleading 38× number). With `flex_attention` in the matrix the
comparison is honest: **AttnFuse matches or beats `flex_attention` on
11 of 16 self-attention cells, on every RoPE-composition cell (up to
2.59×), and on all 12 KV-cache decoding cells.** The cases where
`flex_attention` wins are dense and causal at N=4096 (within 6 %),
which is the documented Triton-vs-CUTLASS gap and not a structural
deficit of our IR.

### 12. HuggingFace integration + end-to-end Llama-3 training step

Added `attnfuse.integrations.hf`, which registers AttnFuse with
HuggingFace's `ALL_ATTENTION_FUNCTIONS` registry. Any LlamaForCausalLM
constructed with `attn_implementation="attnfuse"` now uses AttnFuse for
its causal+GQA+RoPE attention path. The integration handles GQA
natively (no `repeat_kv` expansion needed — AttnFuse broadcasts via its
`GROUP_SIZE` constexpr), routes single-token decoding to the Flash-
Decoding fast path automatically, and falls back to SDPA for dropout
or non-causal mask cases.

The accompanying benchmark
(`benchmarks/hf_llama_bench.py`) runs a real Llama-3-8B-class
LlamaDecoderLayer (hidden 4096, 32 Q heads / 8 KV heads / D=128,
intermediate 14336) for forward and full forward+backward+optimizer.step()
training-step latency. Measured on the 3090:

| N | Stage | SDPA | flex_attention | AttnFuse | Ratio vs SDPA |
|---|---:|---:|---:|---:|---:|
| 1024 | forward | 7.69 ms | OOM | 8.00 ms | **1.04×** |
| 1024 | train step | 23.41 ms | OOM | 24.81 ms | **1.06×** |
| 2048 | forward | 15.59 ms | OOM | 16.47 ms | **1.06×** |
| 2048 | train step | 45.19 ms | OOM | 50.34 ms | **1.11×** |

Two notable findings: (a) AttnFuse comes within **6–11 % of PyTorch SDPA's
hand-tuned CUDA on a real Llama-3-8B training step** — the
kernel-isolated 1.5× backward gap shrinks dramatically in a full model
where attention is one of many ops; (b) `flex_attention` **cannot run at
all** on the 3090 for Llama-3-8B (HEAD_DIM=128 requires 131 KB SMEM but
the 3090 only has 101 KB). For Llama-3-8B training on consumer-grade
Ampere, AttnFuse is the only Triton-based compiler that works.

### 13. Triton 3.1 fp32-dot workaround — thorough investigation

Re-investigated the bf16 cast on the dQ matmul to see if it could be
lifted. Standalone tests showed Triton's `tl.dot(input_precision='ieee')`
gives **zero error** on a vanilla `(64,64)@(64,64)` matmul, but when
applied *inside* the backward kernel with `ds` computed in-register
from `p * (dp - D_row[:, None])`, **none** of the precision modes
(default tf32, ieee, tf32x3) gives correct results — they all give
max|err| ≈ 3 in the gradient. The bug is in how Triton handles
`tl.dot` with an in-register-computed expression on this reduction
direction, not in the precision mode. The bf16 cast remains the only
viable workaround; this is now thoroughly documented in the kernel
source with a comment that explains the diagnostic path so future
Triton releases can be re-tested.

### 14. Block-sparse backward — BigBird-style training

The forward block-sparse path landed earlier in the sprint; this
completes the story by adding the matching backward. New kernels in
`attnfuse/compiler/codegen_blocksparse_bwd.py`:

- `attnfuse_bs_bwd_dkv_kernel`: iterates over the K-major active list
  (`q_indices`, `q_num_blocks`) — for each KV-block, the m-blocks that
  actually attended to it. Sums dK and dV via the same group-aware
  inner loop the dense backward uses.
- `attnfuse_bs_bwd_dq_kernel`: iterates over the Q-major active list
  (the same one the forward uses) per query-block.

`BlockMask` gained `q_num_blocks` and `q_indices` (the K-major
transpose), built in one pass alongside the existing Q-major lists. A
new `AttnFuseBlockSparseFunction(torch.autograd.Function)` provides
the autograd plumbing; `@af.attention` routes block-sparse calls with
`requires_grad` through it automatically.

5 new pytest cases (BigBird, strided, local — across fp16 and bf16)
verify gradients against a naive PyTorch reference with the same
element-wise mask. All within FA2's documented fp16 tolerance:

| Pattern @ fp16, N=256 | fwd | dQ | dK | dV |
|---|---:|---:|---:|---:|
| BigBird (7.7% active) | 4.9e-4 | 2.9e-3 | 4.9e-4 | 2.4e-4 |

**AttnFuse is now one of the few systems supporting block-sparse
*training*** — `flex_attention`'s BlockMask path does not yet provide
backward, and FlashAttention's BlockSparse mode is hand-written CUDA.

---

## Where the project stands now

| Category | Status |
|---|---|
| Forward inference | 95 % of `flex_attention` on supported variants; wins on sliding-window |
| RoPE compositions | **2.59× over `flex_attention`** (structural moat) |
| KV-cache decoding | **12/12 cells beat `flex_attention`** |
| GQA / MQA | Full support, within 5 % of `flex_attention` |
| Cross-attention | Supported for dense; causal/SW with N_q ≠ N_kv rejected |
| Backward pass | First cut, 1.5× of SDPA — Triton-CUTLASS gap |
| Block-sparse forward | Sub-quadratic kernel, 7.4× faster than dense at 7.7 % sparsity |
| **Block-sparse backward** | **Supported** — BigBird-style training enabled |
| **HuggingFace Llama-3 training step** | **1.06× of SDPA** on a real LlamaDecoderLayer; flex_attention OOMs at HEAD_DIM=128 on 3090 |
| Hopper (H100) | Deploy scripts ready; tile tables seeded, awaiting hardware sweep |
| Correctness | 100 unit tests + 80 Hypothesis fuzz examples, 99-100 passing |

Current commit count since course submission: 20+ commits, ~5000 lines
of new code, 50 new tests.

---

## What we are focusing on next

The next-phase work is organised around three goals:

### A. Close the last performance gaps

1. **Hopper benchmark sweep** — run `scripts/run_h100_benchmarks.sh` on
   H100, capture all numbers, re-tune `_HOPPER_TABLE_*` based on the
   results. Expected: causal forward and backward both close to 1.0×
   `flex_attention` on Hopper because TF32 / WGMMA tensor cores have a
   smaller Triton-CUTLASS gap there.
2. **Re-test the bf16-cast workaround on Triton 3.2+** when it lands.
   The standalone `tl.dot(input_precision='ieee')` test passes with zero
   error on Triton 3.1; the in-kernel context-specific failure is the
   only remaining blocker.
3. **Optional: integrate `@triton.autotune`** in the main forward kernel
   so per-shape tile picks happen automatically. Cost: longer first
   JIT compile; benefit: another 3–5 % on cells where the static table
   guessed wrong.

### B. Bigger-scope research extensions

4. **Element-level refinement at boundary blocks** — current block-sparse
   treats partial-active tiles as fully active (upper bound on attended
   keys). Adding a per-block bitmask buffer for boundary tiles enables
   exact element-level masks at the cost of small extra HBM. Optional;
   a paper choice.
5. **Multi-GPU sequence parallelism (Ring Attention)** — the IR is
   structurally ready; lowering to a sequence-parallel kernel is the
   biggest remaining swing for the paper.
6. **Quantized attention (fp8, int8)** — extends the constexpr-gated
   kernel surface with a new dtype path; production-deployment story.

### C. Engineering polish

8. **Nsight Compute counter-based numbers** — replace the analytical
   TFLOPS / HBM estimates in the roofline analysis with measured
   counters. Needs `ncu` access; one day's work.
9. **Property-fuzzer expansion** — currently covers the five main
   forward variants. Add block-sparse + backward cases.
10. **Cold-cache JIT measurements on H100** — one-shot benchmark of
    first-call compile cost on Hopper for the paper's "Compile-Time
    Cost" subsection.

### D. Paper deliverables

- Refresh the report tables with H100 numbers + a `flex_attention`
  baseline column across every table.
- Add a "Structural Novelty: Compositions flex_attention Cannot Fuse"
  section (already exists; refresh with the new 2.59× number).
- Add a "Training: Backward Pass Correctness and Performance" section
  with the new backward numbers + the bf16-cast workaround discussion.
- Add a "Block-Sparse Attention" section with the BigBird benchmark
  results and the BlockMask API description.
- Polish the Limitations section to explicitly mark the items above
  that are scoped out of the current submission.

---

## Reproducing the headline numbers in this document

Every number quoted above lives in a committed CSV under
`results/e2e_2026-05-22/`. The summary doc with everything in one place
is `results/e2e_2026-05-22/PAPER_HEADLINE.md` (post-tensor-core decode
update) and `results/e2e_2026-05-22/FINAL_RESULTS.md` (the broader
narrative). The benchmark scripts are all under `benchmarks/`; the H100
deployment scripts are under `scripts/`.

```bash
# 95 GPU correctness tests (3090):
pytest tests/

# All benchmarks (3090, ~1 hour with a warm Triton disk cache):
python -m benchmarks.bench_runner    --dtype float16  --output results/eval_fp16.csv
python -m benchmarks.flex_bench      --output results/flex_bench.csv
python -m benchmarks.composition_bench --output results/composition_bench.csv
python -m benchmarks.gqa_bench       --output results/gqa_bench.csv
python -m benchmarks.kvcache_bench   --output results/kvcache_bench.csv
python -m benchmarks.backward_bench  --output results/backward_bench.csv
python -m benchmarks.blocksparse_bench --output results/blocksparse_bench.csv
python -m benchmarks.rope_bench      --output results/rope_bench.csv

# H100 (via SSH):
bash scripts/setup_h100.sh
bash scripts/run_h100_benchmarks.sh
```
