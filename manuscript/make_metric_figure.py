"""Figure 6 of the manuscript: the same bootstrap refits scored at two levels (global, per customer) with six metrics,
for all three datasets (review round 7: one figure for the metric x level decomposition).

Reads pipeline/results*/tables/matched_metric_summary.csv only; writes manuscript/figures/fig6_metric_level.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from figures import INK, INK2, SURF  # noqa: E402  (also applies the shared style)

DATA = {"IBM Telco": ROOT / "pipeline" / "results" / "tables",
        "Cell2Cell": ROOT / "pipeline" / "results_cell2cell" / "tables",
        "Iranian Churn": ROOT / "pipeline" / "results_iranian" / "tables"}
ROWS = [(5, "inclusion_ho", "Top-5 inclusion"), (5, "jaccard", "Top-5 Jaccard"), (5, "exact", "Exact top-5"),
        (3, "inclusion_ho", "Top-3 inclusion"), (3, "jaccard", "Top-3 Jaccard"), (3, "exact", "Exact top-3")]
LEVELS = [("global_mean", "Global ranking", "#0b0b0b", "s", 0.13),
          ("local_mean", "Per-customer explanation", "#2a78d6", "o", -0.13)]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.6), sharey=True)
    for ax, (label, T) in zip(axes, DATA.items()):
        s = pd.read_csv(T / "matched_metric_summary.csv")
        for i, (k, met, _) in enumerate(ROWS):
            y = len(ROWS) - 1 - i
            r = s[(s.k == k) & (s.metric == met)]
            for col, _, c, mk, dy in LEVELS:
                ax.plot([r[col].min(), r[col].max()], [y + dy, y + dy], color=c, lw=1.2, alpha=0.45, zorder=1)
                ax.scatter(r[col].mean(), y + dy, s=40, marker=mk, color=c, edgecolors=SURF, linewidths=1.0, zorder=3)
        if label == "IBM Telco":   # the single-model global figure that motivated the comparison
            ax.axvline(0.98, color=INK2, lw=0.8, ls="--")
            ax.text(0.965, 3.55, "98% reported\nfor one model", fontsize=6.5, color=INK2, ha="right", va="top")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, len(ROWS) - 0.3)
        ax.set_xlabel("Agreement across refits")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ROWS)), [n for _, _, n in reversed(ROWS)])
    handles = [Line2D([], [], ls="-", lw=1.2, marker=mk, color=c, markersize=6, label=f"{n} (mean; line = range over models)")
               for _, n, c, mk, _ in LEVELS]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.1), fontsize=8, labelcolor=INK)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig6_metric_level.png", dpi=600, bbox_inches="tight", facecolor=SURF)
    print("saved", out / "fig6_metric_level.png")


if __name__ == "__main__":
    main()
