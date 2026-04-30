"""Render eval CSV → PNG/PDF figures used in the report.

Usage:  python benchmarks/make_figures.py results/eval.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    out_dir = Path(csv_path).parent

    for metric, ylabel, fname in (
        ("latency_ms", "Latency (ms, lower is better)", "eval_latency"),
        ("peak_mem_mb", "Peak GPU memory (MB)",         "eval_memory"),
        ("tflops",      "Throughput (TFLOP/s)",         "eval_tflops"),
    ):
        groups = df.groupby(["model", "variant"])
        for (model, variant), g in groups:
            fig, ax = plt.subplots(figsize=(6, 4))
            for baseline, gb in g.groupby("baseline"):
                gb = gb.sort_values("seqlen")
                ax.plot(gb["seqlen"], gb[metric], marker="o", label=baseline)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(g["seqlen"].unique()))
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("Sequence length")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model} / {variant}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = out_dir / f"{fname}__{model}__{variant}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: make_figures.py results/eval.csv", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
