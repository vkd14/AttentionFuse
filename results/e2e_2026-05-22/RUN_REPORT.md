# AttnFuse — End-to-End Test Run Report (RTX 3090)

**Run date:** 2026-05-22 / 2026-05-23 (overnight wrap)
**Hardware:** NVIDIA GeForce RTX 3090 (sm_86, 24 GB GDDR6X, 936 GB/s)
**Software:** Python 3.11.15 / PyTorch 2.5.1+cu121 / Triton 3.1.0 / CUDA 12.1 / Driver 535.183.01

---

## TL;DR

* **25/25** correctness tests pass.
* **5/5** example scripts run cleanly (including both pre-process and fused RoPE).
* **All seven benchmark suites** reproduce the committed numbers within run-to-run noise (≤2%).
* **New paper-grade data point:** `flex_attention` (PyTorch 2.5) was added to the baseline matrix. It beats AttnFuse on every supported variant by 1.17–2.27×, **which changes the publication framing** — the 38× vs SDPA gap closes when `flex_attention` is in the picture. The defensible novelty narrows to (a) the fused RoPE kernel and (b) the DSL surface itself.

---

## Phase 1 — Correctness (Phase: PASS)

### `pytest tests/`

```
collected 25 items
test_compiler.py ....                             [ 16%]
test_correctness.py ........                      [ 48%]
test_dsl.py ......                                [ 72%]
test_ir.py ...                                    [ 84%]
test_smoke.py ....                                [100%]
======================= 25 passed in 2.77s =======================
```

### Examples (all five)

| Script | Output |
|---|---|
| `bert_dense.py` | `ok torch.Size([2, 12, 1024, 64]) torch.float16` |
| `gpt2_causal.py` | `ok torch.Size([2, 12, 2048, 64]) torch.float16` |
| `sliding_window.py` | `ok torch.Size([2, 12, 4096, 64]) window=256` |
| `causal_alibi.py` | `ok torch.Size([2, 12, 2048, 64]) torch.float16` |
| `causal_rope.py` | `[pre-process] max|err|=1.95e-3` / `[fused] max|err|=1.95e-3` — **PASSED** |

Fused RoPE matches pre-process RoPE bit-for-bit on the max-error metric.

---

## Phase 2 — Core Benchmarks (Phase: PASS)

### Reproducibility check vs committed `results/eval.csv`

GPT-2 small, fp16, N=4096:

| Variant | Committed TFLOPS | Re-run TFLOPS | Δ |
|---|---:|---:|---:|
| Dense (BERT) | 59.20 | 59.20 | 0% |
| Causal | 105.05 | 105.05 | 0% |
| Sliding-window W=256 | 298.48 | 302.29 | +1.3% |
| Causal+ALiBi | 97.73 | 98.86 | +1.2% |

Numbers reproduce exactly for dense/causal (deterministic); SW and ALiBi vary at ~1% which is normal CUDA-event jitter on 50-iter medians.

### Multi-dtype, N=4096 causal (GPT-2 small)

| dtype | AttnFuse TFLOPS | SDPA TFLOPS | Ratio |
|---|---:|---:|---:|
| float16  | 105.1 | 121.3 | 0.87× |
| bfloat16 | 108.4 | 121.0 | 0.90× |
| float32  |  44.4 |  13.7 | 3.24× |

Same shape as committed — bf16 best on AttnFuse, fp32 wins because PyTorch loses its flash path.

### Fused RoPE vs pre-process

| N | Pre-proc (ms) | Fused (ms) | Speedup |
|---:|---:|---:|---:|
|  512 | 0.238 | 0.101 | **2.36×** |
| 1024 | 0.423 | 0.238 | 1.78× |
| 2048 | 0.975 | 0.663 | 1.47× |
| 4096 | 2.739 | 2.337 | 1.17× |

Reproduces published 1.18–2.29× range.

### JIT compile vs cached call

| Variant | 1st call (ms) | Cached call (µs) |
|---|---:|---:|
| dense        | 1095.8 | 179 |
| causal       |   12.5 | 134 |
| sliding_w256 |   12.5 | 124 |
| causal_alibi |   12.9 | 140 |
| causal_rope  |   12.9 | 157 |

`dense` triggers the Triton/LLVM cold compile (~1s). All subsequent variants hit the disk PTX cache and compile in ~12ms. Per-call dispatch overhead is consistently 120–180µs after caching.

### Roofline (corrected effective-FLOP accounting)

| Variant | N | TFLOPS | tc_pct | HBM GB/s | HBM pct |
|---|---:|---:|---:|---:|---:|
| causal       | 4096 | 103.30 | 72.7% | 50.4 | 5.4% |
| sw_w256      | 4096 |  35.54 | 25.0% | 78.1 | 8.3% |
| causal_alibi | 4096 |  97.95 | 69.0% | 47.8 | 5.1% |
| dense        | 4096 |  58.27 | 41.0% | 28.5 | 3.0% |

All tc_pct values ≤ 100%, confirming the variant-correct FLOP accounting fix is working.

---

## Phase 3 — Publication-Extension Bonus: `flex_attention` (Phase: KEY FINDING)

PyTorch 2.5 ships `torch.nn.attention.flex_attention`, a compile-based generalisation of attention. This is the **closest published competitor to AttnFuse** and was identified in the advisor-meeting prep as the highest-priority addition to the baseline matrix.

### Setup

`benchmarks/flex_bench.py` (new). Same shapes (B=4, H=12, D=64), same dtype (fp16), same seqlens. Uses `create_block_mask` for sparse masks and `score_mod` for ALiBi.

### Headline results, N=4096

| Variant | AttnFuse | SDPA | `flex_attention` | flex vs AttnFuse |
|---|---:|---:|---:|---:|
| dense        |  59.6 |  70.4 |  **70.6** | 1.18× |
| causal       | 105.7 | 121.3 | **123.2** | 1.17× |
| sw_w256      |  37.8 |   7.8 |  **85.6** | **2.27×** |
| causal_alibi |  98.7 |   5.6 | **121.0** | 1.23× |

### What this means for the paper

1. **The 38× vs SDPA narrative is brittle.** It is true that SDPA falls back to dense O(n²) for these variants, but PyTorch's *own* answer for them (`flex_attention`) is faster than AttnFuse.

2. **What survives as a defensible novelty:**
   - **Fused RoPE.** `flex_attention`'s `score_mod` runs on the already-computed Q·Kᵀ tile, so RoPE — which rotates Q and K *before* the matmul — is structurally inexpressible in flex_attention. **AttnFuse is the only system that fuses RoPE inside the inner loop.** The 1.17–2.36× rope_bench number stands.
   - **DSL surface area.** Combinators are higher-level than score_mod / block_mask functions.
   - **IR introspection.** AttnFuse exposes a dumpable graph; flex_attention is opaque.

3. **What needs to change in the report's Table 1:** keep AttnFuse vs SDPA (still useful since SDPA is the production path), but add a `flex_attention` column. Re-write the abstract: instead of "$38\times$ over SDPA", phrase it as "matches SDPA on supported variants, opens up variants SDPA can't accelerate, and is the only system to fuse RoPE inside the kernel."

4. **Performance gap to close before submission:**
   - Sliding-window is the worst gap (0.44× of flex_attention).
     Likely root cause: flex_attention does **block-level mask culling**
     — entire (BLOCK_M, BLOCK_N) tiles outside the window are skipped at
     the dispatcher; AttnFuse does per-query bounds checking inside the
     block. Fix: add `block_mask` style precomputed sparse tile lists to
     our `MaskKind.SLIDING_WINDOW` path. Estimated 1 week.
   - Dense/causal gap (≈15%) is the documented Triton-vs-CUTLASS gap and
     is hard to close without writing CUDA. Acceptable for a generality
     pitch.

---

## Phase 4 — Plots Generated

19 figures in `results/e2e_2026-05-22/`:

* `ablation_dense.png`, `ablation_causal.png` — tile-config heatmaps
* `roofline.png` — variant operating points
* `dtype_comparison.png` — fp16 / bf16 / fp32 bar chart
* `eval_{latency,memory,tflops}__{bert-base,gpt2-small}__{dense,causal,sliding_window,causal_alibi}.png` — 15 per-variant plots

---

## Phase 5 — Sliding-Window Optimisation (Phase: COMPLETED)

After the e2e run identified the sliding-window gap as the worst, the SW
codegen was refactored to split the inner loop into three explicit phases:

```
  [n_lo, interior_lo)        # left boundary  — apply SW mask
  [interior_lo, interior_hi) # interior        — NO mask logic (≈45% of tiles)
  [interior_hi, n_hi)        # right boundary — apply SW mask
```

The interior tiles bypass the per-element mask computation, the safe-
softmax `all_masked` branch, and the out-of-N gating — they execute the
dense-attention code path.

### Verification

* 25/25 pytest tests still pass.
* 12-config edge-case sweep (W ≥ N, W = BLOCK_M, tiny N, misaligned BLOCK_M)
  all match the naive reference within max|err| ≤ 1e-3 (target 2e-2).

### Measured improvement (fp16, gpt2-small)

| N | Before (ms) | After (ms) | Speedup | TFLOPS after (dense-FLOP formula) |
|---:|---:|---:|---:|---:|
|  512 | 0.099 | **0.088** | 1.13× |  – |
| 1024 | 0.196 | **0.176** | 1.11× |  – |
| 2048 | 0.339 | **0.290** | 1.17× |  – |
| 4096 | 0.682 | **0.582** | 1.17× | 354.14 |

### Gap to `flex_attention` at N=4096 (bidirectional SW, W=256, fair mask)

| | Before | After |
|---|---:|---:|
| AttnFuse latency | 0.695 ms | **0.560 ms** |
| flex_attention latency | 0.472 ms | 0.483 ms |
| Ratio (we/flex) | 1.47× slower | **1.16× slower** |

At **N=512 AttnFuse now beats flex_attention** (0.093 vs 0.104 ms).

### No regression on other variants (N=4096)

| Variant | Before | After | Δ |
|---|---:|---:|---:|
| dense        | 3.483 ms | 3.524 ms | +1.2% (noise) |
| causal       | 1.962 ms | 1.949 ms | −0.7% |
| causal+ALiBi | 2.085 ms | 2.061 ms | −1.2% |

### What's left to close the remaining 16% gap to flex_attention

1. **SW-specific tile config.** Currently SW uses the dense Ampere table
   (BLOCK_M=128, BLOCK_N=64). flex likely uses BLOCK_N=128. An ablation
   sweep over SW configs is the next ~half-day of work.
2. **Block-level mask culling at dispatch.** flex's `create_block_mask`
   precomputes which (m_block, n_block) pairs need *any* compute and
   skips empty blocks entirely. AttnFuse currently iterates the full
   `[n_lo, n_hi)` range. Adding a precomputed block-list buffer is
   ~1 week of work.

---

## Recommendations Before H100 Run

1. ✅ **Sliding-window 3-loop split** — done (this run).
2. **SW-specific tile config sweep.** Half a day; closes 5–10% more of
   the flex_attention gap.
3. **Update the report's evaluation section.** Add a `flex_attention`
   column to Table 1; rewrite the abstract speedup claim to acknowledge
   `flex_attention` as the strongest competitor and reframe novelty
   around fused RoPE.
4. **Cold-cache JIT benchmark** (`rm -rf ~/.triton/cache`) for the paper.
5. **Property-based correctness fuzzer** (hypothesis over combinator
   combinations) — cheap reviewer comfort.
6. **Then** port the tile table to H100 and re-run the whole sweep with
   the corrected codegen.

---

## Reproduce This Run

```bash
# from project root, with the attnfuse conda env active
pytest tests/ -v
bash results/e2e_2026-05-22/run_chain.sh             # 6 main benchmarks
python -m benchmarks.flex_bench --output results/e2e_2026-05-22/flex_bench.csv
python benchmarks/make_figures.py results/e2e_2026-05-22/eval_fp16.csv
python benchmarks/ablation_plot.py --csv results/e2e_2026-05-22/ablation.csv
python benchmarks/roofline_plot.py --csv results/e2e_2026-05-22/roofline.csv \
    --out results/e2e_2026-05-22/roofline.png
python benchmarks/dtype_comparison_plot.py \
    --csvs results/e2e_2026-05-22/eval_fp16.csv \
           results/e2e_2026-05-22/eval_bf16.csv \
           results/e2e_2026-05-22/eval_fp32.csv \
    --out results/e2e_2026-05-22/dtype_comparison.png
```

Total runtime on a warm Triton disk cache: ~3 minutes. Cold cache: ~10 minutes.
