# AttnFuse — Design Notes

This document explains the IR levels, the compiler passes, and the codegen
contract. Read it once before extending the project; it answers most "why
is it shaped this way?" questions.

## 1. Two-level IR

We follow MLIR's progressive-lowering philosophy but keep the hierarchy as
shallow as possible — there are exactly two IR levels.

### 1.1 High-level IR (`attnfuse/ir/high_level.py`)

A small dataflow graph. Five node kinds:

| Node       | Role                                                      |
|------------|-----------------------------------------------------------|
| `TensorSym`| Symbolic Q / K / V leaf with shape and dtype              |
| `ScoreOp`  | The (Q · Kᵀ)·scale primitive (today only `SCALED_DOT`)    |
| `MaskOp`   | `FULL` / `CAUSAL` / `SLIDING_WINDOW`                      |
| `BiasOp`   | `ADDITIVE` (user tensor) or `ALIBI` (slope-table built)   |
| `NormOp`   | `SOFTMAX` (default) or `RELU`                             |
| `MatMulPV` | The final `probs @ V`. Always the root.                   |

The user's `@attention` function is run once on `TensorSym` placeholders.
Combinators are pure constructors; `MatMulPV` is captured via
`Expr.__matmul__`. The result is a `Graph` that hashes to a stable
`signature()` string used as the kernel-cache key.

### 1.2 Tiled IR (`attnfuse/ir/tiled.py`)

A `TiledKernel` is a flat record with everything codegen needs:
- `head_dim`, `dtype`
- `score_scale`
- `mask_kind` (+ optional `mask_window`)
- `bias_kind` (+ optional `bias_num_heads`)
- `norm_kind`
- `TileConfig`: `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`,
  `skip_full_mask_blocks`, `use_streaming_softmax`

It is intentionally *not* a tree. Once we know the user's spec is fusable,
the tile-loop schedule is fixed; flat data drives a templated kernel.

## 2. Passes

```
Graph (high-level)
  ├── fuse_score_softmax     — structural recogniser; rejects non-fusable graphs
  ├── choose_tile_config     — Ampere-tuned table; head_dim → tile sizes
  └── lower_to_tiled         — collapse mask/bias chain → flat TiledKernel
        │
        ▼
TiledKernel
  └── generate_triton_source — hand-written Triton kernel template
                               specialised by `tl.constexpr` flags
```

The fusion pass is *recognition* rather than *transformation*: every supported
attention spec already maps onto our single online-softmax loop, so all the
pass does is reject malformed graphs early. This keeps codegen straight-line
and easy to read.

## 3. Codegen contract

A single Triton kernel is emitted for every spec. Variation comes through
`tl.constexpr` arguments:

| Constexpr | Meaning                                         |
|-----------|-------------------------------------------------|
| `MASK_KIND` | 0 full, 1 causal, 2 sliding-window           |
| `BIAS_KIND` | 0 none, 1 ALiBi                              |
| `NORM_KIND` | 0 softmax, 1 ReLU-attention                  |
| `SKIP_EMPTY`| 1 → narrow `n_lo, n_hi` for causal/sw        |
| `BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `WINDOW` | Tile + shape literals  |

Triton specialises (and caches) per unique constexpr combination, so a
`dense + softmax` graph and a `causal + alibi + softmax` graph compile to
two different SASS binaries — both produced from the *same source string*.

Online softmax is the standard streaming-max recurrence (see comments in
`codegen.py`). The compute kernel keeps `m_i, l_i, acc` in fp32 even when
inputs are fp16/bf16, matching FlashAttention-2's numerical recipe.

## 4. Why this shape?

- **No autograd.** Forward only. Backward is a separate kernel and a
  separate research project; the proposal scopes the work to forward
  inference.
- **Self-attention only.** Cross-attention with mismatched K-seqlen would
  require splitting strides and is not a course-project priority.
- **No external bias tensors yet.** `BiasKind.ADDITIVE` is parsed by the
  IR but rejected by the lowering pass — wiring an arbitrary user tensor
  through dispatch + codegen is mechanical but adds a code path that
  nothing in the eval suite exercises.

## 5. RTX 3090 Ti notes

Ampere `sm_86`. The tile table in `compiler/tiling.py` was hand-picked
for this GPU; if you move to H100 or RTX 4090 you'll want to re-sweep
`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`. The kernel itself is
device-agnostic — only the *config* is hardware-specific.
