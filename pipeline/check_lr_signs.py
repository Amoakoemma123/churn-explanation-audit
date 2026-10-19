"""Follow-up: why do Logistic Regression SHAP signs flip across bootstrap refits (sign agreement 0.96)?

Two possible mechanisms, checked separately on all outer splits:
  (a) the coefficient itself changes sign across refits (collinearity: tenure / TotalCharges / MonthlyCharges);
  (b) the coefficient is stable but the customer sits near the background mean, so the attribution
      coef * (x - mean) crosses zero when the bootstrap mean moves.
Writes results/tables/LR_sign_flips_by_feature.csv and LR_coefficients.csv.
"""
import json

import numpy as np
import pandas as pd

import sys

import config as C

if "--quick" in sys.argv:
    C.quick_mode()

from data import load, make_prep  # noqa: E402
from models import make_model  # noqa: E402

SPLITS = C.OUTER_SEEDS


def main():
    X, y, ids = load()
    feats = list(X.columns)
    flip_rows, coef_rows = [], []
    for s in SPLITS:
        sp = np.load(C.RAW / f"s{s}_split.npz")
        A = np.load(C.RAW / f"s{s}_LR_boot.npz")["A"].astype(float)           # R x N x F
        params = json.loads((C.RAW / f"s{s}_LR_tuning.json").read_text())["params"]
        cons = A.mean(0)
        top = np.argsort(-np.abs(cons), axis=1)[:, :3]
        for n in range(A.shape[1]):
            for f in top[n]:
                flips = (np.sign(A[:, n, f]) != np.sign(cons[n, f])).mean()
                flip_rows.append(dict(split=s, feature=feats[f], flip_rate=flips,
                                      rel_size=abs(cons[n, f]) / np.abs(cons[n]).sum()))
        # refit the same bootstrap samples to read coefficients
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        for r in range(C.B_BOOT):
            idx = np.random.default_rng(10_000 + r).integers(0, len(Xtr), len(Xtr))
            prep = make_prep(feats)
            Z = prep.fit_transform(Xtr.iloc[idx])
            m = make_model("LR", params, seed=r).fit(Z, ytr[idx])
            names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
            coef = dict(zip(names, m.coef_[0]))
            coef_rows.append(dict(split=s, refit=r, **{k: coef[k] for k in C.NUM}))

    flips = pd.DataFrame(flip_rows)
    by_feat = flips.groupby("feature").agg(instances=("flip_rate", "size"), mean_flip_rate=("flip_rate", "mean"),
                                           share_with_any_flip=("flip_rate", lambda v: (v > 0).mean()),
                                           mean_rel_size=("rel_size", "mean")).sort_values("mean_flip_rate", ascending=False)
    by_feat["share_of_all_flips"] = (flips.groupby("feature").flip_rate.sum() / flips.flip_rate.sum()).reindex(by_feat.index)
    by_feat.to_csv(C.TABLES / "LR_sign_flips_by_feature.csv")
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(C.TABLES / "LR_coefficients.csv", index=False)

    pd.set_option("display.width", 200)
    print(by_feat.round(3).to_string())
    print(f"\nStandardised LR coefficients across {len(SPLITS)} splits x {C.B_BOOT} bootstrap refits:")
    summ = pd.DataFrame({c: dict(mean=coefs[c].mean(), sd=coefs[c].std(), min=coefs[c].min(), max=coefs[c].max(),
                                 share_positive=(coefs[c] > 0).mean()) for c in C.NUM}).T
    print(summ.round(3).to_string())
    if C.DATASET == "ibm":
        print("\ncorr(tenure coef, TotalCharges coef) across refits:",
              round(np.corrcoef(coefs.tenure, coefs.TotalCharges)[0, 1], 3))
    plot(coefs, len(SPLITS))


def plot(coefs, n_splits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figures  # noqa: F401  (applies the shared style)
    if C.DATASET == "ibm":
        feats = ["tenure", "TotalCharges", "MonthlyCharges"]
        colors = {"tenure": "#2a78d6", "TotalCharges": "#1baf7a", "MonthlyCharges": "#eb6834"}
        lo, hi, xt, height = -3.4, 4.6, 3.35, 2.6
    else:   # the pre-specified collinear set (VIF > 10)
        feats = C.COLLINEAR
        colors = dict(zip(feats, ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#b8860b"]))
        v = coefs[feats].to_numpy()
        span = v.max() - v.min()
        lo, xt = v.min() - 0.08 * span, v.max() + 0.08 * span
        hi, height = xt + 0.45 * span, 0.75 * len(feats) + 0.4
    fig, ax = plt.subplots(figsize=(6.4, height))
    rng = np.random.default_rng(0)
    for i, c in enumerate(feats):
        v = coefs[c].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=9, color=colors[c], alpha=0.55, linewidths=0)
        ax.text(xt, i, f"{(v > 0).mean():.0%} positive", va="center", fontsize=8, color="#0b0b0b")
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(range(len(feats)), feats)
    ax.set_xlim(lo, hi); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Standardised LR coefficient ({n_splits} splits × {C.B_BOOT} bootstrap refits)")
    fig.savefig(C.FIGS / "R8_lr_coefficients.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
