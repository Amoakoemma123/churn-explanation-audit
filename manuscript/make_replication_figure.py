"""Figure 11 of the manuscript: global vs per-customer agreement, IBM Telco and Cell2Cell side by side.

Reads pipeline/results*/tables only; writes manuscript/figures/fig11_replication.png.
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
        "Cell2Cell": ROOT / "pipeline" / "results_cell2cell" / "tables"}
MODELS = ["LR", "DT", "RF", "XGB"]
NAME = {"LR": "Logistic Regression", "DT": "Decision Tree", "RF": "Random Forest", "XGB": "XGBoost"}
# one colour per measure (fixed order); marker shape repeats the identity for colour-blind and print readers
MEAS = [("Global top-5 inclusion", "#0b0b0b", "s"),
        ("Per-customer top-3, seed only", "#a8a7a1", "o"),
        ("Per-customer top-3, bootstrap", "#2a78d6", "o"),
        ("Per-customer top-3, other model class", "#eb6834", "D")]


def values(T):
    s = pd.read_csv(T / "stability_by_split.csv").groupby(["model", "condition"]).mean(numeric_only=True)
    h4 = pd.read_csv(T / "H4_cross_model.csv")
    ho = pd.read_csv(T / "matched_metric_summary.csv").set_index(["model", "k", "metric"])   # held-out inclusion
    out = {}
    for m in MODELS:
        between = h4[h4.pair.str.split("-").apply(lambda p: m in p)].between.mean()
        out[m] = [ho.loc[(m, 5, "inclusion_ho"), "global_mean"], s.loc[(m, "seed"), "local_top3_jaccard"],
                  s.loc[(m, "boot"), "local_top3_jaccard"], between]
    chance = pd.read_csv(T / "chance_baselines.csv").set_index("k").loc[3, "chance_jaccard"]
    return out, chance


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (label, T) in zip(axes, DATA.items()):
        v, chance = values(T)
        for i, m in enumerate(MODELS):
            y = len(MODELS) - 1 - i
            ax.plot([min(v[m]), max(v[m])], [y, y], color="#d6d5cf", lw=2, zorder=1)
            for k, ((name, c, mk), x) in enumerate(zip(MEAS, v[m])):
                if k == 1:   # seed floor as a hollow ring on top, so a coinciding marker stays visible
                    ax.scatter(x, y, s=52, marker=mk, facecolors="none", edgecolors=c, linewidths=1.8, zorder=4)
                else:
                    ax.scatter(x, y, s=46, marker=mk, color=c, edgecolors=SURF, linewidths=1.2, zorder=3)
        ax.axvline(chance, color=INK2, lw=0.8, ls=":")
        ax.text(chance + 0.01, len(MODELS) - 0.45, f"chance\n{chance:.2f}", fontsize=7, color=INK2, va="top")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlim(0, 1.03); ax.set_ylim(-0.6, len(MODELS) - 0.3)
        ax.set_xlabel("Agreement across refits")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(MODELS)), [NAME[m] for m in reversed(MODELS)])
    handles = [Line2D([], [], ls="", marker=mk, color=c, markersize=7, label=n,
                      markerfacecolor="none" if k == 1 else c, markeredgewidth=1.6 if k == 1 else 1)
               for k, (n, c, mk) in enumerate(MEAS)]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.13), fontsize=8,
               labelcolor=INK)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig11_replication.png", dpi=600, bbox_inches="tight", facecolor=SURF)
    print("saved", out / "fig11_replication.png")


if __name__ == "__main__":
    main()
