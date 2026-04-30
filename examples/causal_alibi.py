"""Causal + ALiBi positional bias (no learned position embeddings)."""
import os, torch
import attnfuse as af

NUM_HEADS = 12


@af.attention
def alibi_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    B, H, N, D = 2, NUM_HEADS, 2048, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    if os.environ.get("ATTNFUSE_DEBUG"):
        from attnfuse.ir.printer import format_graph
        print(format_graph(alibi_attn(Q, K, V, return_graph=True)))

    out = alibi_attn(Q, K, V)
    print("ok", out.shape, out.dtype)
