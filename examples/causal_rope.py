"""GPT-NeoX style: causal attention + RoPE positional embeddings.

Demonstrates DSL composability: RoPE pre-processing feeds directly into the
AttnFuse causal kernel.  The only code the user writes is 7 lines.
"""
import math
import torch
import attnfuse as af
from attnfuse.rope_utils import build_rope_cache, apply_rope


@af.attention
def causal_rope_attn(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def reference(Q, K, V, cos, sin):
    """Pure-PyTorch reference for correctness verification."""
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

    # Pre-rotate Q and K, then run AttnFuse causal kernel
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    got   = causal_rope_attn(Q_rot, K_rot, V)

    want = reference(Q, K, V, cos, sin)
    max_err = (got.float() - want.float()).abs().max().item()
    print(f"output shape : {got.shape}  dtype: {got.dtype}")
    print(f"max |err|    : {max_err:.4e}  (target < 2e-2)")
    assert max_err < 2e-2, f"RoPE + causal correctness check failed: {max_err}"
    print("causal + RoPE: PASSED")
