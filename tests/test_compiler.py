"""Compiler-pass invariants (CPU-only)."""
import pytest
import torch

import attnfuse as af
from attnfuse.compiler.fusion import fuse_score_softmax, FusionError
from attnfuse.compiler.lowering import lower_to_tiled
from attnfuse.compiler.tiling import choose_tile_config
from attnfuse.compiler.codegen import kernel_constexprs, generate_triton_source


def _trace(fn, D=64, H=4, N=16):
    Q = torch.zeros(1, H, N, D)
    return fn(Q, Q.clone(), Q.clone(), return_graph=True)


def test_fusion_accepts_well_formed_graph():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V
    g = _trace(fn)
    fuse_score_softmax(g)  # must not raise


def test_tiling_picks_a_config_for_each_head_dim():
    for D in (32, 64, 96, 128, 256):
        @af.attention
        def fn(Q, K, V):
            return af.softmax(af.scaled_dot_product(Q, K)) @ V
        g = _trace(fn, D=D)
        cfg = choose_tile_config(g)
        assert cfg.BLOCK_M >= 32 and cfg.BLOCK_N >= 16


def test_codegen_constexprs_change_with_variant():
    @af.attention
    def dense(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    cd = kernel_constexprs(lower_to_tiled(_trace(dense)))
    cc = kernel_constexprs(lower_to_tiled(_trace(causal)))
    assert cd["MASK_KIND"] != cc["MASK_KIND"]


def test_triton_source_is_a_nonempty_string():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V
    src = generate_triton_source(lower_to_tiled(_trace(fn)))
    assert "def attnfuse_fwd_kernel" in src
