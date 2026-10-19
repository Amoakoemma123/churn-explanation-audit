"""Exploratory follow-up (not pre-registered): can per-customer explanations be made stable, or at least
flagged when they are not? Uses the existing bootstrap refits ("boot" condition); nothing is refitted.

  1. Averaging: average each customer's SHAP vector over m bootstrap refits (m = 1, 2, 5, 10, 25) and measure the
     top-3 Jaccard between averages built from DISJOINT groups of refits. m = 1 reproduces local_top3_jaccard.
     Because SHAP is additive in the model output, the average is the explanation of the bagged model that
     averages the refits' outputs (log-odds for LR/XGB, probability for DT/RF; exact for the tree models,
     approximate for LR, whose LinearSHAP background is each refit's own bootstrap sample).
  2. Accuracy of that bagged model (all 50 refits) against the mean single-refit test ROC-AUC.
  3. Flagging: estimate each customer's stability from the first r refits (r = 3, 5, 10) and compare it with the
     stability among the remaining refits: Spearman correlation, and ROC-AUC for picking out the customers whose
     held-out stability is below 0.5.

    python stability_aware.py            # IBM Telco      -> results/tables/X*_*.csv
    CHURN_DATASET=cell2cell python ...   # Cell2Cell      -> results_cell2cell/tables/X*_*.csv
    python stability_aware.py --figure   # both datasets  -> ../manuscript/figures/fig12_averaging.png
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import config as C

if "--quick" in sys.argv:
    C.quick_mode()

from analyze import ci95, pairwise_jaccard_within, topk_mask  # noqa: E402
from data import load  # noqa: E402

M_GROUP = (1, 2, 5, 10, 25)
R_FLAG = (3, 5, 10)
LOGIT_SPACE = {"LR", "XGB"}          # models whose SHAP is on the log-odds scale


def disjoint_average_jaccard(A, m, k=3):
    """A: R x N x F. Average A over disjoint groups of m refits; mean top-k Jaccard over group pairs, per customer."""
    g = A.shape[0] // m
    G = A[:g * m].reshape(g, m, *A.shape[1:]).mean(1)
    return pairwise_jaccard_within(G, k)


def bagged_auc(P, y, model):
    if model in LOGIT_SPACE:
        Pc = np.clip(P.astype(np.float64), 1e-7, 1 - 1e-7)
        score = np.log(Pc / (1 - Pc)).mean(0)
    else:
        score = P.mean(0)
    return roc_auc_score(y, score)


def analyse():
    _, y, _ = load()
    y = np.asarray(y)
    avg, acc, flag = [], [], []
    for s in C.OUTER_SEEDS:
        test = np.load(C.RAW / f"s{s}_split.npz")["test"]
        for m in C.MODELS:
            z = np.load(C.RAW / f"s{s}_{m}_boot.npz")
            A, P = z["A"].astype(np.float64), z["P_test"]
            for g in M_GROUP:
                if A.shape[0] // g >= 2:
                    avg.append(dict(split=s, model=m, m=g, local_top3_jaccard=disjoint_average_jaccard(A, g).mean()))
            acc.append(dict(split=s, model=m, single_auc_mean=np.mean([roc_auc_score(y[test], p) for p in P]),
                            bagged_auc=bagged_auc(P, y[test], m)))
            for r in (r for r in R_FLAG if A.shape[0] - r >= 2):   # the --quick run has only 4 refits
                est, truth = pairwise_jaccard_within(A[:r], 3), pairwise_jaccard_within(A[r:], 3)
                unstable = truth < 0.5
                flag.append(dict(split=s, model=m, r=r, spearman=stats.spearmanr(est, truth).statistic,
                                 auc_unstable=roc_auc_score(unstable, -est) if 0 < unstable.sum() < len(est) else np.nan,
                                 share_unstable=unstable.mean()))
    avg, acc, flag = pd.DataFrame(avg), pd.DataFrame(acc), pd.DataFrame(flag)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    avg.to_csv(C.TABLES / "X1_averaging.csv", index=False)
    acc.to_csv(C.TABLES / "X2_bagged_accuracy.csv", index=False)
    flag.to_csv(C.TABLES / "X3_flagging.csv", index=False)

    # summaries (mean and Nadeau-Bengio corrected 95% CI over outer splits)
    rows = []
    for (m, g), d in avg.groupby(["model", "m"]):
        lo, hi = ci95(d.local_top3_jaccard)
        rows.append(dict(model=m, m=g, mean=d.local_top3_jaccard.mean(), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(C.TABLES / "X1_averaging_summary.csv", index=False)
    acc.assign(diff=acc.bagged_auc - acc.single_auc_mean).groupby("model")[["single_auc_mean", "bagged_auc", "diff"]] \
        .mean().reset_index().to_csv(C.TABLES / "X2_bagged_accuracy_summary.csv", index=False)
    flag.groupby(["model", "r"])[["spearman", "auc_unstable", "share_unstable"]].mean().reset_index() \
        .to_csv(C.TABLES / "X3_flagging_summary.csv", index=False)
    print(pd.read_csv(C.TABLES / "X1_averaging_summary.csv").round(3).to_string(index=False))
    print(pd.read_csv(C.TABLES / "X2_bagged_accuracy_summary.csv").round(4).to_string(index=False))
    print(pd.read_csv(C.TABLES / "X3_flagging_summary.csv").round(3).to_string(index=False))


def figure():
    """Per-customer top-3 stability against the number of averaged refits, both datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import COL, INK2, NAME, SURF

    here = Path(__file__).resolve().parent
    sets = [("IBM Telco", here / "results" / "tables"), ("Cell2Cell", here / "results_cell2cell" / "tables")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (title, T) in zip(axes, sets):
        d = pd.read_csv(T / "X1_averaging_summary.csv")
        floor = pd.read_csv(T / "stability_summary.csv").set_index(["condition", "model"])
        for m in C.MODELS:
            e = d[d.model == m].sort_values("m")
            ax.errorbar(e.m, e["mean"], yerr=[e["mean"] - e.ci_low, e.ci_high - e["mean"]], color=COL[m], marker="o",
                        ms=4, lw=1.5, capsize=2, label=NAME[m])
            ax.axhline(floor.loc[("seed", m), "local_top3_jaccard_mean"], color=COL[m], ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(M_GROUP, [str(g) for g in M_GROUP]); ax.minorticks_off()
        ax.set_title(title); ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("Per-customer top-3 Jaccard")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.supxlabel("Bootstrap refits averaged per explanation (m)", fontsize=9, color=INK2, y=-0.04)
    fig.text(0.5, -0.12, "Dotted lines: seed-only stability of a single refit (same training data).", ha="center",
             fontsize=7.5, color=INK2)
    out = here.parent / "manuscript" / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig12_averaging.png", dpi=600, bbox_inches="tight", facecolor=SURF)
    print("saved", out / "fig12_averaging.png")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else analyse()
