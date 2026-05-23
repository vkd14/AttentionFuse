"""Plot the composition-benchmark results.

Produces one bar chart with grouped bars: for each sequence length,
two bars (AttnFuse, flex_attention) per composition. The plot makes
the structural-novelty story visually obvious: variants where RoPE is
involved show a dramatic gap.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/composition_bench.csv")
    p.add_argument("--out", default="results/composition_bench.png")
    p.add_argument("--out-speedup", default="results/composition_speedup.png")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print(f"empty CSV {args.csv}"); return 1

    # Pivot: latency[(variant, N)][backend] = ms
    latency: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        v = r["variant"]
        N = int(r["seqlen"])
        lat = float(r["latency_ms"]) if r["latency_ms"] != "nan" else float("nan")
        latency[(v, N)][r["backend"]] = lat

    variants = ["causal", "causal_alibi", "causal_rope", "causal_rope_alibi"]
    pretty_v = {
        "causal":            "Causal",
        "causal_alibi":      "Causal + ALiBi",
        "causal_rope":       "Causal + RoPE",
        "causal_rope_alibi": "Causal + RoPE + ALiBi",
    }
    seqlens  = sorted({N for _, N in latency.keys()})

    # ---- speedup chart (the headline figure) ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_w = 0.20
    xs = np.arange(len(variants))
    for i, N in enumerate(seqlens):
        speedups = []
        for v in variants:
            af = latency[(v, N)].get("attnfuse")
            fl = latency[(v, N)].get("flex")
            if af and fl and not np.isnan(fl):
                speedups.append(fl / af)
            else:
                speedups.append(float("nan"))
        ax.bar(xs + (i - 1.5) * bar_w, speedups, bar_w,
               label=f"N = {N}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty_v[v] for v in variants],
                       rotation=15, ha="right")
    ax.set_ylabel("Speedup vs. flex_attention  (×)")
    ax.set_title("AttnFuse speedup over PyTorch flex_attention by composition\n"
                 "(RTX 3090, fp16, batch=4, 12 heads, head_dim=64)")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Highlight RoPE compositions
    for i, v in enumerate(variants):
        if "rope" in v:
            ax.axvspan(i - 0.4, i + 0.4, color="orange", alpha=0.08, zorder=0)
    ax.text(2.5, ax.get_ylim()[1] * 0.93,
            "RoPE compositions:\nflex_attention cannot fuse",
            ha="center", fontsize=9, style="italic",
            color="orange", weight="bold",
            bbox=dict(facecolor="white", edgecolor="orange",
                      alpha=0.9, boxstyle="round,pad=0.3"))
    fig.tight_layout()
    fig.savefig(args.out_speedup, dpi=140)
    print(f"wrote {args.out_speedup}")

    # ---- latency chart (the supporting figure) ----
    fig, axes = plt.subplots(1, len(seqlens), figsize=(14, 3.8),
                              sharey=False)
    for ax, N in zip(axes, seqlens):
        af_ms   = [latency[(v, N)].get("attnfuse", float("nan")) for v in variants]
        flex_ms = [latency[(v, N)].get("flex",     float("nan")) for v in variants]
        xs = np.arange(len(variants))
        ax.bar(xs - 0.2, af_ms,   0.4, label="AttnFuse", color="#1f77b4")
        ax.bar(xs + 0.2, flex_ms, 0.4, label="flex_attention", color="#ff7f0e")
        ax.set_xticks(xs)
        ax.set_xticklabels([pretty_v[v] for v in variants],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(f"N = {N}")
        ax.set_ylabel("Latency (ms)")
        ax.grid(axis="y", alpha=0.3)
        if N == seqlens[0]:
            ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Per-composition latency: AttnFuse vs flex_attention",
                 y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
