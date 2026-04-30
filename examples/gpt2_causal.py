"""GPT-2 small causal attention via AttnFuse."""
import os, torch
import attnfuse as af


@af.attention
def gpt2_attn(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    B, H, N, D = 2, 12, 2048, 64    # GPT-2 small
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    if os.environ.get("ATTNFUSE_DEBUG"):
        from attnfuse.ir.printer import format_graph
        print(format_graph(gpt2_attn(Q, K, V, return_graph=True)))

    out = gpt2_attn(Q, K, V)
    print("ok", out.shape, out.dtype)
