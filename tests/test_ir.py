"""IR walks, printer, lowering correctness on the CPU side."""
import torch

import attnfuse as af
from attnfuse.ir.high_level import MaskKind, BiasKind, NormKind
from attnfuse.ir.printer import format_graph, format_tiled
from attnfuse.compiler.lowering import lower_to_tiled


def _g():
    @af.attention
    def fn(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=4)
        s = af.causal(s)
        return af.softmax(s) @ V

    Q = torch.zeros(1, 4, 16, 64)
    return fn(Q, Q.clone(), Q.clone(), return_graph=True)


def test_format_graph_runs():
    s = format_graph(_g())
    assert "MatMulPV" in s
    assert "ScoreOp" in s


def test_lowering_collapses_to_tiled():
    k = lower_to_tiled(_g())
    assert k.head_dim == 64
    assert k.mask_kind is MaskKind.CAUSAL
    assert k.bias_kind is BiasKind.ALIBI
    assert k.norm_kind is NormKind.SOFTMAX
    assert abs(k.score_scale - 1.0 / (64 ** 0.5)) < 1e-6


def test_format_tiled_runs():
    out = format_tiled(lower_to_tiled(_g()))
    assert "BLOCK_M" in out
