"""Generate dtype-comparison bar chart from merged evaluation CSVs.

Reads one or more eval CSVs (produced by bench_runner with different --dtype
flags), merges them, and plots AttnFuse TFLOPS across dtypes for each
(model, variant, seqlen) combination.

Usage::

    python benchmarks/dtype_comparison_plot.py \\
        --csvs results/eval.csv results/eval_bf16.csv results/eval_fp32.csv \\
        --out  results/dtype_comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_DTYPE_COLORS = {
    "float16":  "#1f77b4",
    "bfloat16": "#ff7f0e",
    "float32":  "#2ca02c",
}

_SEQLENS_SHOW = [512, 1024, 2048, 4096]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+",
                   default=["results/eval.csv",
                            "results/eval_bf16.csv",
                            "results/eval_fp32.csv"])
    p.add_argument("--out", default="results/dtype_comparison.png")
    args = p.parse_args()

    frames = []
    for path in args.csvs:
        if not Path(path).exists():
            print(f"  skipping missing {path}")
            continue
        df = pd.read_csv(path)
        # Infer dtype from filename if not present as a column
        if "dtype" not in df.columns:
            stem = Path(path).stem
            if "bf16" in stem or "bfloat16" in stem:
                df["dtype"] = "bfloat16"
            elif "fp32" in stem or "float32" in stem:
                df["dtype"] = "float32"
            else:
                df["dtype"] = "float16"
        frames.append(df)

    if not frames:
        print("No CSV files found. Run bench_runner.py first."); return 1

    df = pd.concat(frames, ignore_index=True)
    af = df[(df["baseline"] == "attnfuse") & (df["seqlen"].isin(_SEQLENS_SHOW))]

    combos = af[["model", "variant"]].drop_duplicates().values.tolist()
    if not combos:
        print("No AttnFuse data found."); return 1

    n_combos = len(combos)
    fig, axes = plt.subplots(1, n_combos, figsize=(5 * n_combos, 4), sharey=False)
    if n_combos == 1:
        axes = [axes]

    for ax, (model, variant) in zip(axes, combos):
        sub = af[(af["model"] == model) & (af["variant"] == variant)].copy()
        sub["tflops"] = pd.to_numeric(sub["tflops"], errors="coerce")

        dtypes  = sorted(sub["dtype"].unique())
        seqlens = sorted(sub["seqlen"].unique())
        x       = np.arange(len(seqlens))
        width   = 0.8 / max(len(dtypes), 1)

        for i, dtype in enumerate(dtypes):
            vals = []
            for sl in seqlens:
                row = sub[(sub["dtype"] == dtype) & (sub["seqlen"] == sl)]
                vals.append(row["tflops"].mean() if len(row) else float("nan"))
            offset = (i - (len(dtypes) - 1) / 2) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=dtype, color=_DTYPE_COLORS.get(dtype, "gray"), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(sl) for sl in seqlens], fontsize=9)
        ax.set_xlabel("Sequence length", fontsize=10)
        ax.set_ylabel("TFLOP/s", fontsize=10)
        ax.set_title(f"{model} / {variant}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AttnFuse throughput by dtype  (RTX 3090)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
