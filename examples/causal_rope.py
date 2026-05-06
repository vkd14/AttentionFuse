"""GPT-NeoX style: causal attention + RoPE positional embeddings.

Two usage modes:
  1. Pre-processing (existing): rotate Q/K in Python, pass rotated tensors to causal kernel.
  2. Fused (new): use af.rope() combinator so rotation happens inside the Triton kernel.

Both produce identical results; fused mode saves two host-side tensor operations
and keeps the original Q/K unchanged.
"""
import math
import torch
import attnfuse as af
from attnfuse.rope_utils import build_rope_cache, apply_rope


# ---- Mode 1: pre-processing approach ----------------------------------------

@af.attention
def causal_rope_preproc(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


# ---- Mode 2: fused RoPE kernel -----------------------------------------------

@af.attention
def causal_rope_fused(Q, K, V):
    s = af.rope(Q, K)          # rotation happens inside the Triton kernel
    s = af.causal(s)
    return af.softmax(s) @ V


def reference(Q, K, V, cos, sin):
    Qr = apply_rope(Q, cos, sin)
    Kr = apply_rope(K, cos, sin)
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.einsum("bhmd,bhnd->bhmn", Qr.float(), Kr.float()) * scale
    mask   = torch.tril(torch.ones(Q.shape[2], Q.shape[2], device=Q.device)).bool()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs  = torch.softmax(scores, dim=-1).to(Q.dtype)
    return torch.einsum("bhmn,bhnd->bhmd", probs, V)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    B, H, N, D = 2, 12, 1024, 64
    dtype = torch.float16
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    cos, sin = build_rope_cache(N, D, device="cuda", dtype=dtype)
    want = reference(Q, K, V, cos, sin)

    # Mode 1: pre-processing
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    out1  = causal_rope_preproc(Q_rot, K_rot, V)
    err1  = (out1.float() - want.float()).abs().max().item()
    print(f"[pre-process] max|err| = {err1:.4e}  (target < 2e-2)")
    assert err1 < 2e-2, f"pre-processing RoPE failed: {err1}"

    # Mode 2: fused
    out2 = causal_rope_fused(Q, K, V, cos=cos, sin=sin)
    err2 = (out2.float() - want.float()).abs().max().item()
    print(f"[fused kernel] max|err| = {err2:.4e}  (target < 2e-2)")
    assert err2 < 2e-2, f"fused RoPE failed: {err2}"

    print("causal + RoPE (both modes): PASSED")
