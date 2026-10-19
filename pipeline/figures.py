"""Result figures, drawn only from results/tables (so they always match the tables)."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

import config as C  # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
COL = {"LR": "#2a78d6", "DT": "#eb6834", "RF": "#1baf7a", "XGB": "#6250d6"}
NAME = {"LR": "Logistic Regression", "DT": "Decision Tree", "RF": "Random Forest", "XGB": "XGBoost"}
MODELS = ["LR", "DT", "RF", "XGB"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
})


def _ci(x):
    """95% CI half-width with the Nadeau & Bengio (2003) correction for overlapping 80/20 splits."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * np.sqrt((1 / len(x) + 0.25) * x.var(ddof=1))


def _save(fig, out, name):
    fig.savefig(out / name, dpi=600, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_same_accuracy(T, out):
    perf = pd.read_csv(T / "performance_by_split.csv")
    st = pd.read_csv(T / "stability_by_split.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.45})
    y = np.arange(4)[::-1]
    for yi, m in zip(y, MODELS):
        v = perf[perf.model == m].roc_auc
        a1.errorbar(v.mean(), yi, xerr=_ci(v), fmt="o", ms=7, color=COL[m], capsize=3, lw=1.5)
        a1.text(v.mean(), yi + 0.25, f"{v.mean():.3f}", ha="center", fontsize=8, color=INK)
    lo = perf.roc_auc.min() - 0.02
    a1.set_xlim(np.floor(lo * 100) / 100, np.ceil((perf.roc_auc.max() + 0.01) * 100) / 100)
    a1.set_yticks(y, [NAME[m] for m in MODELS]); a1.set_ylim(-0.6, 3.6); a1.grid(axis="y", visible=False)
    a1.set_xlabel("Test ROC-AUC (mean, 95% CI over outer splits)"); a1.set_title("A. Predictive performance", loc="left")
    bw = 0.36
    for yi, m in zip(y, MODELS):
        for off, cond, c in ((bw / 2, "seed", "#c9c8c2"), (-bw / 2, "boot", COL[m])):
            v = st[(st.model == m) & (st.condition == cond)].local_top3_jaccard
            a2.barh(yi + off, v.mean(), height=bw - 0.04, color=c, xerr=_ci(v), error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
            a2.text(v.mean() + _ci(v) + 0.02, yi + off, f"{v.mean():.2f}", va="center", fontsize=8,
                    color=INK if cond == "boot" else INK2)
    a2.set_yticks(y, MODELS); a2.set_xlim(0, 1.15); a2.grid(axis="y", visible=False)
    a2.set_xlabel("Per-customer top-3 Jaccard"); a2.set_title("B. Explanation stability", loc="left")
    a2.legend(handles=[Patch(color="#c9c8c2", label="grey: seed only (same data)"),
                       Patch(color=COL["LR"], label="coloured: bootstrap refits")],
              loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=8)
    _save(fig, out, "R1_same_accuracy.png")


def fig_metric_sensitivity(T, out):
    st = pd.read_csv(T / "stability_by_split.csv").query("condition == 'boot'")
    metrics = [("global_top5_inclusion", "Top-5 inclusion\nfrequency"), ("global_spearman", "Global rank\ncorrelation"),
               ("global_top5_exact", "Exact top-5\nagreement"), ("local_top3_jaccard", "Per-customer\ntop-3 Jaccard")]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(metrics))
    for k, m in enumerate(MODELS):
        d = st[st.model == m]
        means = [d[c].mean() for c, _ in metrics]; errs = [_ci(d[c]) for c, _ in metrics]
        xx = x + (k - 1.5) * 0.06
        ax.plot(xx, means, "-", color=COL[m], lw=1.6, alpha=0.9)
        ax.errorbar(xx, means, yerr=errs, fmt="o", color=COL[m], ms=6, capsize=2, lw=1, label=NAME[m],
                    markeredgecolor=SURF)
    ax.set_xticks(x, [l for _, l in metrics]); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Agreement across refits (1 = identical)")
    if C.DATASET == "ibm":   # the published reference value is for IBM Telco only
        ax.axhline(0.98, color=MUTED, lw=0.9, ls="--")
        ax.text(-0.15, 0.995, "98% reported by El Attar & El-Hajj (2026)", fontsize=7.5, color=INK2, va="bottom")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    _save(fig, out, "R2_metric_sensitivity.png")


def fig_rank_heatmap(T, out):
    df = pd.read_csv(T / "global_ranks.csv", index_col=0)
    df = df.loc[df.mean(1).sort_values().index[:10]]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#0d3b7a", "#2a78d6", "#a9c9f0", "#f4f8fd"])
    ax.imshow(df.values, cmap=cmap, vmin=1, vmax=12, aspect="auto")
    ax.set_xticks(range(4), [NAME[m] for m in MODELS], fontsize=8); ax.set_yticks(range(len(df)), df.index, fontsize=8)
    ax.grid(False)
    for i in range(len(df)):
        for j in range(4):
            v = df.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color="white" if v <= 4 else INK,
                    fontweight="bold" if v == 1 else "normal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    _save(fig, out, "R3_rank_heatmap.png")


def fig_risk(T, out):
    loc = pd.read_csv(T / "customer_level_stability.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for m in MODELS:
        d = loc[loc.model == m].copy()
        d["bin"] = pd.qcut(d.dist, 10, labels=False, duplicates="drop")
        g = d.groupby("bin").agg(x=("dist", "median"), y=("jac3", "mean"))
        ax.plot(g.x, g.y, "-o", color=COL[m], lw=2, ms=5, label=NAME[m], markeredgecolor=SURF)
    ax.set_xlabel("Distance of predicted risk from the decision threshold\n(|logit p − logit threshold|, median of each decile)")
    ax.set_ylabel("Per-customer top-3 Jaccard")
    ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, "R4_stability_vs_risk.png")


SEG_LEVELS = {"Contract": [("Month-to-month", "Month-to-\nmonth"), ("One year", "One year"), ("Two year", "Two year")],
              "Tenure": [("0-12", "0–12\nmonths"), ("13-48", "13–48\nmonths"), ("49-72", "49–72\nmonths")]}
SEG_TITLE = {"Contract": "Contract type", "Tenure": "Tenure band"}


def fig_segments(T, out):
    seg = pd.read_csv(T / "H5_segments.csv")
    groups = [g for g in SEG_LEVELS if g in set(seg.segment)]
    order = [(g, lv) for g in groups for lv, _ in SEG_LEVELS[g] if ((seg.segment == g) & (seg.level == lv)).any()]
    labels = [dict(SEG_LEVELS[g])[lv] for g, lv in order]
    gi = np.array([groups.index(g) for g, _ in order])
    fig, ax = plt.subplots(figsize=(7.2 if len(groups) > 1 else 4.6, 3.1))
    x = np.arange(len(order)) + 0.6 * gi; w = 0.2
    for k, m in enumerate(MODELS):
        for xi, (s, l) in zip(x, order):
            r = seg[(seg.model == m) & (seg.segment == s) & (seg.level == l)].iloc[0]
            ax.bar(xi + (k - 1.5) * w, r.frac_below_0p5 * 100, width=w - 0.025, color=COL[m],
                   alpha=1.0 if r.reportable else 0.35, label=NAME[m] if xi == x[0] else None)
    ax.set_xticks(x, labels, fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_ylabel("% customers with unstable\ntop-3 reasons (Jaccard < 0.5)")
    ax.legend(ncol=4 if len(groups) > 1 else 2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ymax = ax.get_ylim()[1]
    for g in groups:
        ax.text(x[gi == groups.index(g)].mean(), -0.28 * ymax, SEG_TITLE[g], ha="center", fontsize=8.5,
                fontweight="bold", color=INK)
    _save(fig, out, "R5_segments.png")


def fig_lime(T, out):
    d = pd.read_csv(T / "RQ4_lime_by_split.csv")
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    y = np.arange(4)[::-1]; bw = 0.36
    for yi, m in zip(y, MODELS):
        g = d[d.model == m]
        for off, c, colr in ((bw / 2, "lime_self_top5", "#c9c8c2"), (-bw / 2, "lime_shap_top5", COL[m])):
            ax.barh(yi + off, g[c].mean(), height=bw - 0.04, color=colr, xerr=_ci(g[c]),
                    error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
            ax.text(g[c].mean() + _ci(g[c]) + 0.02, yi + off, f"{g[c].mean():.2f}", va="center", fontsize=8)
    ax.set_yticks(y, MODELS); ax.set_xlim(0, 1.1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Top-5 feature agreement (Jaccard)")
    ax.legend(handles=[Patch(color="#c9c8c2", label="grey: LIME vs LIME (different seeds)"),
                       Patch(color=COL["LR"], label="coloured: LIME vs SHAP")],
              loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=8)
    _save(fig, out, "R6_lime.png")


def fig_cross_model(T, out):
    d = pd.read_csv(T / "H4_cross_model.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(d))
    ax.bar(x - 0.18, d.within, width=0.34, color="#c9c8c2", label="within model (refit vs refit)")
    ax.bar(x + 0.18, d.between, width=0.34, color=COL["LR"], label="between models")
    for xi, a, b in zip(x, d.within, d.between):
        ax.text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.text(xi + 0.18, b + 0.01, f"{b:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, d.pair, fontsize=8); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Per-customer top-3 Jaccard")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, "R7_cross_model.png")


def make_all(T, out, feats):
    for f in (fig_same_accuracy, fig_metric_sensitivity, fig_rank_heatmap, fig_risk, fig_segments, fig_lime,
              fig_cross_model):
        f(T, out)
