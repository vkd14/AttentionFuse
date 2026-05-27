"""Backward-pass correctness tests.

For each supported variant + dtype + shape combination, runs forward then
backward through AttnFuse and compares the gradients against a naive
PyTorch reference via ``torch.autograd``. The reference is exact (no
recompute trick), so any deviation comes from the AttnFuse kernels.

Currently-supported scope (matches attnfuse/runtime/backward.py::can_backward):
  * Masks: dense, causal
  * Bias:  none, ALiBi
  * Norm:  softmax
  * No fused RoPE in backward yet (pre-process Q/K if you need RoPE +
    training; the standard apply_rope is autograd-tracked already)
  * MHA, GQA, MQA all supported (the dK/dV kernel inner-loops over a
    GROUP_SIZE Q-head range for each shared KV head)
"""
from __future__ import annotations

import pytest
import torch

import attnfuse as af
from attnfuse.reference.pytorch_naive import naive_attention
from attnfuse.runtime.dispatch import _alibi_slopes


pytestmark = pytest.mark.gpu
TOL_FP16 = 2e-2
TOL_BF16 = 4e-2
# The dQ matmul casts to bf16 inside the kernel to work around a Triton 3.1
# precision issue with fp32 tl.dot on the (M,N)@(N,D) reduction direction.
# So fp32 backward effectively has bf16-level precision, ~2e-2.
TOL_FP32 = 2e-2


def _tol(dtype):
    return {torch.float16: TOL_FP16,
            torch.bfloat16: TOL_BF16,
            torch.float32: TOL_FP32}[dtype]


def _expand_kv(t: torch.Tensor, group_size: int) -> torch.Tensor:
    B, H_kv, N, D = t.shape
    return (t.unsqueeze(2)
             .expand(B, H_kv, group_size, N, D)
             .reshape(B, H_kv * group_size, N, D)
             .contiguous())


def _check(O_ours, O_ref, gQ, gK, gV, gQr, gKr, gVr, tol, label):
    errs = {
        "O":  (O_ours - O_ref).abs().max().item(),
        "dQ": (gQ - gQr).abs().max().item(),
        "dK": (gK - gKr).abs().max().item(),
        "dV": (gV - gVr).abs().max().item(),
    }
    bad = [k for k, v in errs.items() if v > tol]
    assert not bad, f"{label}: errs={errs} > tol={tol}"


# --- dense -----------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [64, 256, 512])
def test_backward_dense(dtype, N):
    B, H, D = 2, 4, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    Qr, Kr, Vr = (t.detach().clone().requires_grad_() for t in (Q, K, V))
    O_ref = naive_attention(Qr, Kr, Vr)
    dO = torch.randn_like(O_ref)
    O_ref.backward(dO)

    @af.attention
    def dense(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V
    O = dense(Q, K, V); O.backward(dO)
    _check(O, O_ref, Q.grad, K.grad, V.grad, Qr.grad, Kr.grad, Vr.grad,
           _tol(dtype), f"dense N={N} {dtype}")


# --- causal ----------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [64, 256, 512])
def test_backward_causal(dtype, N):
    B, H, D = 2, 4, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    Qr, Kr, Vr = (t.detach().clone().requires_grad_() for t in (Q, K, V))
    O_ref = naive_attention(Qr, Kr, Vr, causal=True)
    dO = torch.randn_like(O_ref)
    O_ref.backward(dO)

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V
    O = causal(Q, K, V); O.backward(dO)
    _check(O, O_ref, Q.grad, K.grad, V.grad, Qr.grad, Kr.grad, Vr.grad,
           _tol(dtype), f"causal N={N} {dtype}")


# --- causal + ALiBi --------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [128, 256, 512])
def test_backward_causal_alibi(dtype, N):
    B, H, D = 2, 8, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    Qr, Kr, Vr = (t.detach().clone().requires_grad_() for t in (Q, K, V))
    slopes = _alibi_slopes(H, "cuda", str(dtype).replace("torch.", ""))
    O_ref = naive_attention(Qr, Kr, Vr, causal=True, alibi_slopes=slopes)
    dO = torch.randn_like(O_ref)
    O_ref.backward(dO)

    @af.attention
    def causal_alibi(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=H)
        s = af.causal(s)
        return af.softmax(s) @ V
    O = causal_alibi(Q, K, V); O.backward(dO)
    _check(O, O_ref, Q.grad, K.grad, V.grad, Qr.grad, Kr.grad, Vr.grad,
           _tol(dtype), f"causal+ALiBi N={N} {dtype}")


# --- GQA + causal ----------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_cfg", [(8, 2), (8, 1), (12, 4)],
                          ids=lambda c: f"H_q={c[0]}-H_kv={c[1]}")
def test_backward_gqa_causal(dtype, head_cfg):
    H_q, H_kv = head_cfg
    B, N, D = 1, 256, 64
    group = H_q // H_kv
    Q = torch.randn(B, H_q,  N, D, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype, requires_grad=True)
    # Reference: expand K, V to H_q heads, then run normal causal attention
    Kr_exp = _expand_kv(K.detach(), group).requires_grad_()
    Vr_exp = _expand_kv(V.detach(), group).requires_grad_()
    Qr = Q.detach().clone().requires_grad_()
    O_ref = naive_attention(Qr, Kr_exp, Vr_exp, causal=True)
    dO = torch.randn_like(O_ref)
    O_ref.backward(dO)

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V
    O = causal(Q, K, V); O.backward(dO)

    # Reduce expanded gradients back to GQA shape (sum across head group)
    dK_ref = Kr_exp.grad.view(B, H_kv, group, N, D).sum(dim=2)
    dV_ref = Vr_exp.grad.view(B, H_kv, group, N, D).sum(dim=2)
    _check(O, O_ref, Q.grad, K.grad, V.grad, Qr.grad, dK_ref, dV_ref,
           _tol(dtype) * 1.2,  # GQA reduction adds a bit of accumulation noise
           f"GQA causal H_q={H_q} H_kv={H_kv} {dtype}")
