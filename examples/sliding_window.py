"""Mistral-style local (sliding-window) attention."""
import os, torch
import attnfuse as af

WINDOW = 256


@af.attention
def local_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.sliding_window(s, window_size=WINDOW)
    return af.softmax(s) @ V


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    B, H, N, D = 2, 12, 4096, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    if os.environ.get("ATTNFUSE_DEBUG"):
        from attnfuse.ir.printer import format_graph
        print(format_graph(local_attn(Q, K, V, return_graph=True)))

    out = local_attn(Q, K, V)
    print("ok", out.shape, out.dtype, "window=", WINDOW)
