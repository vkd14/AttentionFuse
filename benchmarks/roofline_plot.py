"""Generate a roofline model plot from results/roofline.csv.

The roofline model (Williams et al., 2009) characterises whether a kernel is
compute-bound or memory-bandwidth-bound by plotting its measured TFLOPS against
its arithmetic intensity (FLOP/byte) against the hardware roofline limits.

Usage::

    python benchmarks/roofline_plot.py [--csv results/roofline.csv]
                                       [--out results/roofline.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# RTX 3090 hardware limits
PEAK_TFLOPS = 142.0   # fp16 tensor-core peak
PEAK_HBM_GBS = 936.0  # GDDR6X HBM bandwidth (GB/s)
RIDGE_AI = PEAK_TFLOPS * 1e12 / (PEAK_HBM_GBS * 1e9)  # FLOP/byte at ridge point

MARKERS = {"dense": "o", "causal": "s", "sw_w256": "^", "causal_alibi": "D"}
COLORS  = {"dense": "#1f77b4", "causal": "#ff7f0e",
           "sw_w256": "#2ca02c", "causal_alibi": "#d62728"}
LABELS  = {"dense": "Dense", "causal": "Causal",
           "sw_w256": "Sliding-window (W=256)", "causal_alibi": "Causal + ALiBi"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/roofline.csv")
    p.add_argument("--out", default="results/roofline.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # Arithmetic intensity: FLOP/byte = TFLOPS*1e12 / (HBM_GBs*1e9)
    df["ai"] = df["tflops"] * 1e3 / df["hbm_gbs"]   # == tflops/hbm_gbs * 1000

    fig, ax = plt.subplots(figsize=(7, 5))

    # ---- Hardware roofline ----
    ai_range = np.logspace(1, 5, 400)
    # memory-bound slope: achievable TFLOPS = AI * HBM_bandwidth
    mem_roof  = ai_range * PEAK_HBM_GBS * 1e9 / 1e12   # in TFLOPS
    comp_roof = np.full_like(ai_range, PEAK_TFLOPS)
    roof      = np.minimum(mem_roof, comp_roof)

    ax.plot(ai_range, roof, "k-", linewidth=1.8, label="Hardware roofline (RTX 3090)", zorder=3)
    ax.axvline(RIDGE_AI, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(RIDGE_AI * 1.05, 2.0, f"Ridge\n{RIDGE_AI:.0f} FLOP/B",
            fontsize=8, color="gray", va="bottom")

    # ---- Operating points ----
    for variant, grp in df.groupby("variant"):
        grp = grp.sort_values("seqlen")
        ax.scatter(grp["ai"], grp["tflops"],
                   marker=MARKERS.get(variant, "o"),
                   color=COLORS.get(variant, "black"),
                   s=60, zorder=5, label=LABELS.get(variant, variant))
        # annotate each point with seqlen
        for _, row in grp.iterrows():
            ax.annotate(
                f"{int(row.seqlen)}",
                xy=(row.ai, row.tflops),
                xytext=(4, 2), textcoords="offset points",
                fontsize=7, color=COLORS.get(variant, "black"),
            )

    # ---- Region annotations ----
    ax.text(15, 90, "Memory-\nbandwidth\nbound", fontsize=8,
            ha="center", va="center", color="steelblue", alpha=0.7)
    ax.text(3000, 25, "Compute\nbound", fontsize=8,
            ha="center", va="center", color="darkorange", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1, 500)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Throughput (TFLOP/s)", fontsize=11)
    ax.set_title("AttnFuse Roofline — RTX 3090 (fp16)", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
