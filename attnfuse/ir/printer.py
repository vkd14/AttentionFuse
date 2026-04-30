"""Pretty printers for both IR levels (used by ATTNFUSE_DEBUG=1)."""
from __future__ import annotations

from .high_level import (
    Graph, ScoreOp, MaskOp, BiasOp, NormOp, MatMulPV, TensorSym,
)
from .tiled import TiledKernel


def format_graph(g: Graph) -> str:
    """Return an indented textual rendering of the high-level IR."""
    lines: list[str] = []
    lines.append(f"# Graph(signature={g.signature()})")
    lines.append(f"  Q : {_fmt_sym(g.q)}")
    lines.append(f"  K : {_fmt_sym(g.k)}")
    lines.append(f"  V : {_fmt_sym(g.v)}")
    lines.append("  root:")
    lines.extend("    " + ln for ln in _fmt_node(g.root).splitlines())
    return "\n".join(lines)


def _fmt_sym(t: TensorSym) -> str:
    return f"{t.name}[{t.batch}x{t.num_heads}x{t.seqlen}x{t.head_dim} : {t.dtype}]"


def _fmt_node(node) -> str:
    if isinstance(node, TensorSym):
        return _fmt_sym(node)
    if isinstance(node, ScoreOp):
        return (
            f"ScoreOp({node.kind.value}, scale={node.scale})\n"
            f"  q = {_fmt_node(node.q)}\n"
            f"  k = {_fmt_node(node.k)}"
        )
    if isinstance(node, MaskOp):
        head = f"MaskOp({node.kind.value}"
        if node.window is not None:
            head += f", window={node.window}"
        head += ")"
        return f"{head}\n  scores = " + _indent_tail(_fmt_node(node.scores))
    if isinstance(node, BiasOp):
        head = f"BiasOp({node.kind.value}"
        if node.num_heads is not None:
            head += f", num_heads={node.num_heads}"
        head += ")"
        return f"{head}\n  scores = " + _indent_tail(_fmt_node(node.scores))
    if isinstance(node, NormOp):
        return f"NormOp({node.kind.value})\n  scores = " + _indent_tail(_fmt_node(node.scores))
    if isinstance(node, MatMulPV):
        return (
            "MatMulPV\n"
            f"  probs = {_indent_tail(_fmt_node(node.probs))}\n"
            f"  v     = {_fmt_node(node.v)}"
        )
    return repr(node)


def _indent_tail(s: str) -> str:
    """Indent every line after the first by 2 spaces (so child fields align)."""
    head, *rest = s.splitlines()
    if not rest:
        return head
    return head + "\n" + "\n".join("  " + ln for ln in rest)


def format_tiled(k: TiledKernel) -> str:
    cfg = k.config
    return (
        f"# TiledKernel(cache_key={k.cache_key})\n"
        f"  dtype           = {k.dtype}\n"
        f"  head_dim        = {k.head_dim}\n"
        f"  score_scale     = {k.score_scale}\n"
        f"  mask            = {k.mask_kind.value}"
        + (f" (window={k.mask_window})" if k.mask_window else "")
        + "\n"
        f"  bias            = {k.bias_kind.value if k.bias_kind else 'none'}"
        + (f" (heads={k.bias_num_heads})" if k.bias_num_heads else "")
        + "\n"
        f"  norm            = {k.norm_kind.value}\n"
        f"  BLOCK_M/BLOCK_N = {cfg.BLOCK_M}/{cfg.BLOCK_N}\n"
        f"  num_warps       = {cfg.num_warps}\n"
        f"  num_stages      = {cfg.num_stages}\n"
    )
