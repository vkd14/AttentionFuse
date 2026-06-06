"""Hopper-spike dispatch wiring tests.

Two questions to answer:

  1. Numerics: when the spike is dispatched, does it produce the same
     result as the production AttnFuse causal kernel?

  2. Gating: does ``can_use_hopper_spike`` route the right calls and
     fall back for the right ones? The predicate is the only thing
     between "spike runs" and "production runs", and a buggy predicate
     either misses the win or routes through an unsupported variant.

These run on any CUDA GPU. On Ampere the eligibility predicate returns
False, so the spike path is never taken; numerics tests then just
re-verify the production kernel. On Hopper the spike path activates
for the qualifying cases and is the primary thing being tested.
"""
from __future__ import annotations

import pytest
import torch

import attnfuse as af
from attnfuse.reference.pytorch_naive import naive_attention
from attnfuse.rope_utils import build_rope_cache, apply_rope
from attnfuse.runtime.hopper_dispatch import (
    _IS_HOPPER, can_use_hopper_spike,
)


pytestmark = pytest.mark.gpu


@af.attention
def _causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


@af.attention
def _dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


@af.attention
def _sliding(Q, K, V):
    return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), 256)) @ V


@af.attention
def _rope_causal(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V


# ----------------------------------------------------------------------
# Eligibility predicate
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H_q,H_kv,N,D,dtype,expected",
    [
        # Eligible: causal, MHA, fp16, D=64, N=2048
        (1, 4, 4, 2048, 64,  torch.float16, True),
        # Eligible: causal, GQA group=4, fp16, D=128, N=4096
        (1, 8, 2, 4096, 128, torch.float16, True),
        # Eligible: causal, MHA, bf16, D=64, N=2048
        (1, 4, 4, 2048, 64,  torch.bfloat16, True),
        # Ineligible: N too small
        (1, 4, 4, 1024, 64,  torch.float16, False),
        # Ineligible: HEAD_DIM=256 not supported by spike yet
        (1, 4, 4, 2048, 256, torch.float16, False),
        # Ineligible: fp32
        (1, 4, 4, 2048, 64,  torch.float32, False),
    ],
)
def test_predicate_shape_dtype(B, H_q, H_kv, N, D, dtype, expected):
    Q = torch.empty(B, H_q, N, D, device="cuda", dtype=dtype)
    K = torch.empty(B, H_kv, N, D, device="cuda", dtype=dtype)
    causal_graph = _causal(Q, K, K, return_graph=True)
    got = can_use_hopper_spike(causal_graph, Q, K)
    assert got is (expected and _IS_HOPPER), \
        f"predicate got {got}, expected {expected and _IS_HOPPER} for " \
        f"B={B} H_q={H_q} H_kv={H_kv} N={N} D={D} dtype={dtype}"


def test_predicate_rejects_non_causal():
    """Dense / sliding-window must NOT be routed to the spike."""
    Q = torch.empty(1, 4, 2048, 64, device="cuda", dtype=torch.float16)
    K = torch.empty_like(Q)
    V = torch.empty_like(Q)
    assert can_use_hopper_spike(_dense(Q, K, V, return_graph=True),    Q, K) is False
    assert can_use_hopper_spike(_sliding(Q, K, V, return_graph=True),  Q, K) is False


def test_predicate_rejects_save_lse():
    """Backward path passes save_lse; spike must decline so production saves L."""
    Q = torch.empty(1, 4, 2048, 64, device="cuda", dtype=torch.float16)
    K = torch.empty_like(Q); V = torch.empty_like(Q)
    L = torch.empty(1, 4, 2048, device="cuda", dtype=torch.float32)
    assert can_use_hopper_spike(_causal(Q, K, V, return_graph=True), Q, K, save_lse=L) is False


# ----------------------------------------------------------------------
# Numerics parity through the @af.attention dispatch entry point
# ----------------------------------------------------------------------


@pytest.mark.parametrize("B,H_q,H_kv,N,D", [
    (1, 4, 4, 2048, 64),    # MHA, D=64
    (1, 8, 2, 4096, 64),    # GQA group=4, D=64
    (1, 4, 4, 2048, 128),   # MHA, D=128
    (1, 8, 2, 2048, 128),   # GQA group=4, D=128
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dispatch_numerics(B, H_q, H_kv, N, D, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H_q,  N, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)
    V = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)

    out = _causal(Q, K, V)
    # naive_attention assumes K,V already have H_q heads; expand for GQA.
    gs = H_q // H_kv
    K_full = K.repeat_interleave(gs, dim=1) if gs > 1 else K
    V_full = V.repeat_interleave(gs, dim=1) if gs > 1 else V
    ref = naive_attention(Q, K_full, V_full, causal=True).to(dtype)
    err = (out.float() - ref.float()).abs().max().item()
    # fp16 quantisation floor at this shape is ~2e-3; bf16 is ~8x looser
    # because of its 7-bit mantissa (vs fp16's 10).
    tol = 5e-3 if dtype is torch.float16 else 3e-2
    assert err < tol, f"err {err:.2e} too large for {dtype} at {Q.shape}"


# ----------------------------------------------------------------------
# RoPE+causal: Session 6 -- the structural-novelty path
# ----------------------------------------------------------------------


def test_predicate_accepts_rope_causal():
    """RoPE+causal must route to the spike (it's the headline case)."""
    Q = torch.empty(1, 4, 2048, 64, device="cuda", dtype=torch.float16)
    K = torch.empty_like(Q); V = torch.empty_like(Q)
    g = _rope_causal(Q, K, V, cos=torch.empty(2048, 64, device="cuda", dtype=torch.float16),
                              sin=torch.empty(2048, 64, device="cuda", dtype=torch.float16),
                              return_graph=True)
    assert can_use_hopper_spike(g, Q, K) is _IS_HOPPER


@pytest.mark.parametrize("B,H_q,H_kv,N,D", [
    (1, 4, 4, 2048, 64),    # MHA, D=64
    (1, 8, 2, 2048, 64),    # GQA group=4, D=64
    (1, 4, 4, 2048, 128),   # MHA, D=128
    (1, 8, 2, 2048, 128),   # GQA group=4, D=128
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dispatch_numerics_rope_causal(B, H_q, H_kv, N, D, dtype):
    """End-to-end: @af.attention rope_causal -> dispatch -> spike (on H100)."""
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H_q,  N, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)
    V = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)
    cos, sin = build_rope_cache(N, D, device="cuda", dtype=dtype)

    out = _rope_causal(Q, K, V, cos=cos, sin=sin)

    # Reference: rotate Q, K outside the kernel (the flex_attention path),
    # then standard causal attention. Expand GQA explicitly.
    gs = H_q // H_kv
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    K_full = K_rot.repeat_interleave(gs, dim=1) if gs > 1 else K_rot
    V_full = V.repeat_interleave(gs, dim=1)     if gs > 1 else V
    ref = naive_attention(Q_rot, K_full, V_full, causal=True).to(dtype)

    err = (out.float() - ref.float()).abs().max().item()
    tol = 5e-3 if dtype is torch.float16 else 3e-2
    assert err < tol, \
        f"err {err:.2e} too large for {dtype} at {Q.shape} (RoPE+causal)"
