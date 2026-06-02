# AttnFuse — Architecture and Design

This document is the 10-minute tour of how AttnFuse is put together,
written for readers who want to understand the engineering before
diving into the paper or the code.

## What problem AttnFuse solves

Modern attention research keeps inventing new combinations of score,
mask, bias, and positional-encoding operations: sliding-window, ALiBi,
RoPE, block-sparse, GQA / MQA, and various compositions thereof. Each
new variant today requires hand-written CUDA or Triton to run with
respectable throughput.

PyTorch's `flex_attention` (introduced in 2.5) addresses part of this
gap via a `score_mod` Python callback compiled by TorchInductor. The
remaining gap is structural: `score_mod` fires *after* `Q @ K^T`, which
means it cannot fuse positional encodings that must be applied *before*
the score matmul. Rotary Position Embedding (RoPE) — the encoding used
by every major recent LLM (LLaMA, Mistral, Falcon, Qwen, DeepSeek) — is
the most consequential example.

AttnFuse is an embedded Python DSL that closes that gap with composable
combinators and compiles down to Triton.

## The five layers

```
┌──────────────────────────────────────────────────────────┐
│  USER CODE                                               │
│  @af.attention                                           │
│  def llama_attn(Q, K, V):                                │
│      s = af.rope(Q, K)                                   │
│      s = af.causal(s)                                    │
│      return af.softmax(s) @ V                            │
└──────────────────────┬───────────────────────────────────┘
                       │ symbolic tracing (`attnfuse/dsl/tracer.py`)
                       ▼
┌──────────────────────────────────────────────────────────┐
│  HIGH-LEVEL IR  (attnfuse/ir/high_level.py)              │
│  Graph DAG: TensorSym → ScoreOp → MaskOp →               │
│             BiasOp → NormOp → MatMulPV                   │
│  Pure dataclasses; no tensor data; SHA-1 signature       │
└──────────────────────┬───────────────────────────────────┘
                       │ four passes (`attnfuse/compiler/`)
                       │   1. fuse        (recognise pattern)
                       │   2. tile        (BLOCK_M, BLOCK_N, warps, stages)
                       │   3. lower       (Graph → TiledKernel)
                       │   4. codegen     (TiledKernel → Triton source)
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TILED IR  (attnfuse/ir/tiled.py)                        │
│  TiledKernel record: HEAD_DIM, MASK_KIND, BIAS_KIND,     │
│                     NORM_KIND, ROPE_KIND, BLOCK_M, ...   │
└──────────────────────┬───────────────────────────────────┘
                       │ Triton source string
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TRITON KERNEL  (attnfuse/compiler/codegen.py)           │
│  Single parameterised kernel, every variant gated by     │
│  `tl.constexpr` flags → Triton JIT specialises a         │
│  distinct PTX binary per unique combination →            │
│  zero runtime branching in the hot loop.                 │
└──────────────────────┬───────────────────────────────────┘
                       │ launch
                       ▼
┌──────────────────────────────────────────────────────────┐
│  RUNTIME  (attnfuse/runtime/)                            │
│  • dispatch.py        forward launch                     │
│  • flash_decode.py    Q.N=1 fast path (two-phase)        │
│  • backward.py        FA2-style dK/dV + dQ + preproc     │
│  • block_mask.py      block-sparse forward + backward    │
│  • autograd.py        torch.autograd.Function bindings   │
│  • kernel_cache.py    LaunchBundle keyed by graph sig    │
└──────────────────────────────────────────────────────────┘
```

## What lives in each layer

### Layer 1 — User-facing DSL

Eight combinators are exposed as Python functions that return IR
nodes. The `@af.attention` decorator traces the function once with
symbolic `TensorSym` placeholders and caches the resulting `Graph` keyed
by `(head_dim, dtype)` so re-tracing only happens when the shape class
changes.

The DSL surface (`attnfuse/dsl/api.py`) is intentionally narrow:

| Combinator | Effect |
|---|---|
| `scaled_dot_product(Q, K)` | $S = QK^\top / \sqrt{d}$ |
| `rope(Q, K)` | Fused RoPE inside the inner loop (the structural-novelty case) |
| `causal(s)` | Lower-triangular mask |
| `sliding_window(s, W)` | Local attention window |
| `full(s)` | No-op mask |
| `block_sparse(s)` | User-supplied BlockMask at call time |
| `alibi(s, num_heads)` | ALiBi linear positional bias |
| `additive_bias(s)` | External $(B, H, N, N)$ bias tensor |
| `softmax(s)` | Online numerically stable softmax |
| `relu_attention(s)` | ReLU normalisation |

These compose freely. The decorator detects `requires_grad` on inputs
and routes through `torch.autograd.Function` when present.

### Layer 2 — High-level IR

Six node types, all pure dataclasses:

- `TensorSym` — leaf node, carries `(batch, num_heads, seqlen, head_dim, dtype)`
- `ScoreOp` — the matmul, with `rope: bool` and `scale: Optional[float]`
- `MaskOp` — `MaskKind` enum (FULL, CAUSAL, SLIDING_WINDOW, BLOCK_SPARSE)
- `BiasOp` — `BiasKind` enum (ALIBI, ADDITIVE)
- `NormOp` — `NormKind` enum (SOFTMAX, RELU)
- `MatMulPV` — the value-projection root

`Graph.signature()` is a SHA-1 hash of the structural content (op kinds,
dtype, head_dim, RoPE flag, mask window, etc.). Two graphs with the
same signature share a compiled kernel; this is what makes per-shape
re-tracing free.

### Layer 3 — Compiler

Four passes in `attnfuse/compiler/`:

1. **Fuse** (`fuse.py`) — structural recogniser that validates the
   graph fuses into a single online-softmax loop. Rejects unsupported
   compositions at compile time, not at kernel launch.
2. **Tile** (`tiling.py`) — picks BLOCK_M, BLOCK_N, num_warps,
   num_stages from a hardware-tuned lookup table. Five tables exist:
   `_AMPERE_TABLE_F16`, `_AMPERE_TABLE_F16_SPARSE`,
   `_AMPERE_TABLE_F16_ROPE`, `_AMPERE_TABLE_F32`,
   `_AMPERE_TABLE_F32_ROPE`, plus matching Hopper tables for sm_90.
   The "SPARSE" table is used for causal / sliding-window / ALiBi
   variants which want smaller per-program tiles (more SMs busy).
3. **Lower** (`lowering.py`) — collapses the graph into a flat
   `TiledKernel` record.
4. **Codegen** (`codegen.py`, `codegen_decode.py`,
   `codegen_backward.py`, `codegen_blocksparse.py`,
   `codegen_blocksparse_bwd.py`) — emits Triton source strings.

### Layer 4 — Triton kernel

One parameterised Triton kernel template per major code path
(forward, decode-split, decode-combine, backward-preproc, backward-dKdV,
backward-dQ, blocksparse-forward, blocksparse-dKdV, blocksparse-dQ).
Each `tl.constexpr` flag (`MASK_KIND`, `BIAS_KIND`, `NORM_KIND`,
`ROPE_KIND`, `BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `GROUP_SIZE`, `SAVE_L`)
gets a distinct PTX binary from Triton's JIT, so the hot loop has
zero runtime branching.

Source strings live under `attnfuse/_generated/` and are imported via
`importlib.util.spec_from_file_location` (Triton requires the source
to be on disk for its caching to hash correctly).

### Layer 5 — Runtime

The runtime translates a real Q/K/V batch into a kernel launch.
`dispatch.py` handles the main forward path; it auto-routes to the
Flash Decoding two-phase kernel when `Q.N == 1` and the KV cache is
large enough. `flash_decode.py` implements the two-phase split-K with
cyclic Q-head replication for GQA. `backward.py` orchestrates the
three backward kernels. `autograd.py` wraps everything for
`torch.autograd`.

`kernel_cache.py` maintains a process-local `LaunchBundle` per graph
signature: precomputed dispatch metadata + the JIT-compiled function
+ cached placeholder tensors. Per-call dispatch overhead is under
**1 μs** thanks to this cache.

## Three things that are subtle

### 1. The fused-RoPE kernel

`flex_attention`'s `score_mod` runs on the post-matmul score tile and
therefore cannot fuse RoPE. AttnFuse rotates Q once *before* the inner
loop and rotates each K tile *inside* the inner loop, using a gather
over `rot_offs_d` indices that maps `d → d + D/2` for `d < D/2` and
`d - D/2` otherwise. The whole rotation happens in registers — no extra
HBM round-trip for the rotated Q and K. This gives up to **2.10×**
speedup over `flex_attention` on `causal + RoPE` compositions at
N=4096.

### 2. Flash Decoding with cyclic Q-head replication

For `Q.N == 1` autoregressive decoding the standard kernel launches
only `B × H_q` programs (32 for Llama-3-8B), wasting most of the SMs.
The Flash Decoding kernel splits the KV axis across `NUM_SPLITS`
programs and combines partial `(m, ℓ, acc)` tuples in a second small
kernel via log-sum-exp.

For GQA there's a second trick: each program processes `BLOCK_H` query
heads sharing one KV head, but Triton's `tl.dot` requires `M ≥ 16`.
We pad `BLOCK_H` to 16 by cyclic-replicating the in-group heads,
mask the duplicate stores at write-time, and recover full tensor-core
throughput while loading K and V exactly once per program.

Combined result: at Llama-3-70B with 32k cache, Flash Decoding takes
the same kernel from **3 153 μs** (the original AttnFuse) to **184 μs**
— a **17× speedup**, beating `flex_attention`'s 196 μs.

### 3. The Triton 3.1 bf16 cast workaround in the backward

The backward dQ matmul `dQ += dS @ K` hit a Triton 3.1 issue:
`tl.dot(dS_fp32, K_fp32)` silently produces wrong results when the
first operand is computed in registers (the standalone `tl.dot` works
fine). We tested `input_precision='ieee'` and `tf32x3` — both fail
identically with `|err| ≈ 3` in the gradient. Forcing the inputs to
bf16 before the dot recovers tensor-core correctness with only
mantissa-level precision loss (bf16 has fp32's exponent range so the
cast is lossless except for mantissa rounding).

The diagnostic path is documented inline in `codegen_backward.py` so
the next Triton release can be re-tested.

## How to read the code

If you want to understand one concrete example end-to-end:

1. **Look at the user code**: `examples/quickstart.ipynb`, cell 2
   (fused RoPE).
2. **Look at the IR**: `attnfuse/dsl/api.py::rope` (returns a `ScoreOp`
   with `rope=True`).
3. **Look at the compiler**: `attnfuse/compiler/codegen.py`, search
   for `ROPE_KIND == 1` to see the two rotation blocks (Q outside the
   loop, K inside).
4. **Look at the dispatch**: `attnfuse/runtime/dispatch.py`, search
   for `bundle.has_rope` to see how `cos` and `sin` get squeezed and
   passed to the kernel.
5. **Look at the benchmark**: `benchmarks/composition_bench.py`, which
   measures the fused-RoPE path against `flex_attention`'s preprocessed
   RoPE path.

## What's not in the artifact

- **Backward for cross-attention** with N_q ≠ N_kv. Causal /
  sliding-window backward requires N_q = N_kv (relative-position
  semantics); the dense backward works at any shape ratio.
- **Hopper-native codegen** (TMA descriptors + WGMMA + warpgroup
  cooperative copy). The Ampere-targeted `tl.dot` path runs on Hopper
  but does not exploit Hopper's biggest architectural wins. The H100
  results in §7.5 of the paper document this gap.
- **Multi-GPU sequence-parallel attention** (Ring Attention). The IR
  is structurally ready but the lowering pass to a sequence-parallel
  kernel is not built.
- **fp8 / int8 quantised attention**. Would be straightforward to add
  as another `tl.constexpr`-gated dtype path.

These are the "future work" items the paper enumerates.
