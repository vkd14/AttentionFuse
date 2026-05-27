"""High-level IR: a tiny dataflow graph of attention combinators.

Design goals
------------
* Pure dataclasses — easy to hash, easy to pretty-print, easy to lower.
* No Tensor objects: everything is symbolic.  The compiler is the only
  thing that ever touches real GPU memory.
* `Expr` overrides `__matmul__` so the user can write `probs @ V` inside
  their @attention function and have it captured into the IR.
"""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScoreKind(enum.Enum):
    SCALED_DOT = "scaled_dot"


class MaskKind(enum.Enum):
    FULL = "full"
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    BLOCK_SPARSE = "block_sparse"     # arbitrary user-supplied block mask


class BiasKind(enum.Enum):
    ADDITIVE = "additive"
    ALIBI = "alibi"


class NormKind(enum.Enum):
    SOFTMAX = "softmax"
    RELU = "relu"


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------


@dataclass
class Expr:
    """Base class for every IR node. Provides `@` capture for `probs @ V`."""

    def __matmul__(self, other: "TensorSym") -> "MatMulPV":
        if not isinstance(other, TensorSym):
            raise TypeError(
                "Right-hand side of `@` must be the symbolic V tensor; "
                f"got {type(other).__name__}"
            )
        return MatMulPV(probs=self, v=other)


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


@dataclass
class TensorSym(Expr):
    """A symbolic tensor of shape (batch, num_heads, seqlen, head_dim)."""
    name: str
    batch: int
    num_heads: int
    seqlen: int
    head_dim: int
    dtype: str  # e.g. "float16", "bfloat16", "float32"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.num_heads, self.seqlen, self.head_dim)


# ---------------------------------------------------------------------------
# Internal nodes
# ---------------------------------------------------------------------------


@dataclass
class ScoreOp(Expr):
    kind: ScoreKind
    q: TensorSym
    k: TensorSym
    scale: Optional[float] = None
    rope: bool = False   # True → fused RoPE applied to Q/K inside the Triton kernel  # None → 1/sqrt(head_dim)


@dataclass
class MaskOp(Expr):
    kind: MaskKind
    scores: Expr
    window: Optional[int] = None  # only for SLIDING_WINDOW


@dataclass
class BiasOp(Expr):
    kind: BiasKind
    scores: Expr
    bias: Optional[TensorSym] = None  # only for ADDITIVE
    num_heads: Optional[int] = None   # only for ALIBI


@dataclass
class NormOp(Expr):
    kind: NormKind
    scores: Expr


@dataclass
class MatMulPV(Expr):
    """Final `probs @ V` step. Always the root of a valid attention graph."""
    probs: Expr
    v: TensorSym


# ---------------------------------------------------------------------------
# Graph wrapper
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    q: TensorSym
    k: TensorSym
    v: TensorSym
    root: MatMulPV

    # Filled in by the inspector / compiler. Cached for hashability.
    _signature: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Inspection helpers (used by codegen and tests)
    # ------------------------------------------------------------------

    def walk(self):
        """Yield (node, parent) in pre-order from the root."""
        seen = set()

        def go(node, parent):
            if id(node) in seen:
                return
            seen.add(id(node))
            yield node, parent
            for child in _children(node):
                yield from go(child, node)

        yield from go(self.root, None)

    def collect_masks(self) -> list[MaskOp]:
        return [n for n, _ in self.walk() if isinstance(n, MaskOp)]

    def collect_biases(self) -> list[BiasOp]:
        return [n for n, _ in self.walk() if isinstance(n, BiasOp)]

    def norm(self) -> NormOp:
        norms = [n for n, _ in self.walk() if isinstance(n, NormOp)]
        if len(norms) != 1:
            raise ValueError(
                f"Graph must contain exactly one normalisation op; got {len(norms)}"
            )
        return norms[0]

    def score(self) -> ScoreOp:
        scores = [n for n, _ in self.walk() if isinstance(n, ScoreOp)]
        if len(scores) != 1:
            raise ValueError(
                f"Graph must contain exactly one score op; got {len(scores)}"
            )
        return scores[0]

    # ------------------------------------------------------------------
    # Stable signature -- used as the kernel-cache key
    # ------------------------------------------------------------------

    def signature(self) -> str:
        """Hash that uniquely identifies the *shape* of this graph (variant +
        head_dim + dtype). Two graphs with the same signature can share a
        compiled kernel; different seqlens/batches do NOT change the signature
        because Triton specialises on those at launch time."""
        if self._signature is not None:
            return self._signature

        parts: list[str] = []
        parts.append(f"dtype={self.q.dtype}")
        parts.append(f"head_dim={self.q.head_dim}")
        for node, _ in self.walk():
            if isinstance(node, ScoreOp):
                parts.append(f"score:{node.kind.value}:scale={node.scale}:rope={node.rope}")
            elif isinstance(node, MaskOp):
                parts.append(f"mask:{node.kind.value}:w={node.window}")
            elif isinstance(node, BiasOp):
                parts.append(f"bias:{node.kind.value}:h={node.num_heads}")
            elif isinstance(node, NormOp):
                parts.append(f"norm:{node.kind.value}")
            elif isinstance(node, MatMulPV):
                parts.append("matmul_pv")
        sig = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
        self._signature = sig
        return sig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _children(node):
    if isinstance(node, ScoreOp):
        return [node.q, node.k]
    if isinstance(node, MaskOp):
        return [node.scores]
    if isinstance(node, BiasOp):
        out = [node.scores]
        if node.bias is not None:
            out.append(node.bias)
        return out
    if isinstance(node, NormOp):
        return [node.scores]
    if isinstance(node, MatMulPV):
        return [node.probs, node.v]
    return []
