"""Render ablation CSV → heatmaps (one per variant).

Usage::  python benchmarks/ablation_plot.py [--csv results/ablation.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _heatmap(df_v: pd.DataFrame, variant: str, out_path: Path, title_suffix: str = "") -> None:
    # Use the best num_stages per (BLOCK_M, BLOCK_N, num_warps) triple
    best = (df_v[df_v["tflops"] != "oom"]
            .assign(tflops=lambda d: d["tflops"].astype(float))
            .sort_values("tflops", ascending=False)
            .drop_duplicates(["BLOCK_M", "BLOCK_N", "num_warps"]))

    # Pivot: rows = (BM, BN), cols = num_warps
    best["config"] = best.apply(lambda r: f"BM{int(r.BLOCK_M)}\nBN{int(r.BLOCK_N)}", axis=1)
    pivot = best.pivot_table(index="config", columns="num_warps", values="tflops", aggfunc="max")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=pivot.values.min() * 0.9, vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{w} warps" for w in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 80 else "white")

    plt.colorbar(im, ax=ax, label="TFLOPS")
    title = f"Tile ablation — {variant}{title_suffix}  (N=2048, best num_stages)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("num_warps")
    ax.set_ylabel("(BLOCK_M, BLOCK_N)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/ablation.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.csv).parent

    if "dtype" in df.columns:
        # Multi-dtype CSV: produce one heatmap per (variant, dtype) combination
        for (variant, dtype), grp in df.groupby(["variant", "dtype"]):
            suffix = f"  [{dtype}]"
            fname  = f"ablation_{variant}_{dtype.replace('float', 'f')}.png"
            _heatmap(grp, variant, out_dir / fname, title_suffix=suffix)
    else:
        for variant, grp in df.groupby("variant"):
            _heatmap(grp, variant, out_dir / f"ablation_{variant}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
