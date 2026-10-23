"""Semi-synthetic ground truth (review round 16; post hoc, descriptive).

Stability is not validity: averaged explanations could be more self-consistent without being closer to the truth. Here
the real IBM Telco (or Iranian Churn) features are kept, and the churn label is replaced by one simulated from a known
model, whose SHAP values are the true per-customer attributions:
  - "linear" truth: logistic regression fitted once to all customers (true attribution = its interventional
    LinearSHAP in log-odds units, background = all customers);
  - "nonlinear" truth: XGBoost fitted once to all customers (true attribution = its TreeSHAP in log-odds units).
The simulated label is drawn once per truth (fixed seed). In each outer split, every model class is refitted R = 20
times on bootstrap samples of the training customers with the simulated label, using the hyperparameters tuned on the
real label, and explains the same 500 test customers. Per customer:
  - single: top-3 Jaccard between a refit's explanation and the true top 3 (mean over refits);
  - averaged: top-3 Jaccard between the average of 10 refits and the true top 3 (mean over the two groups of 10);
  - sign: share of refits whose sign matches the true sign, for the customer's true top-3 features;
  - full_fit: top-3 Jaccard to the truth of one model fitted to the whole training set (no bootstrap), the value that
    averaging bootstrap refits approaches; averaged20: the average of all 20 refits;
  - self: pairwise top-3 Jaccard among the refits, and between the two averages of 10 (stability, for reference).
Random forest and decision tree are explained in probability units, the truth in log-odds units; top-3 sets can
differ between the two scales, which slightly favours LR and XGBoost.

Final-review addition: GT_DRAWS=D repeats the whole check for D independent label draws (seeds 2026, 2027, ...;
draw 0 is the original one and keeps its file names), and GT_SPLITS=n uses only the first n outer splits (used for
the larger Cell2Cell sample, where RF TreeSHAP is slow).

    CHURN_DATASET=ibm|iranian|cell2cell [GT_DRAWS=5] [GT_SPLITS=3] python ground_truth.py
        (resumable; raw/gt_<truth>[_d<draw>]_s<seed>_<model>.npz)
    -> <results>/tables/RB_ground_truth_by_split.csv, RB_ground_truth.csv (draw 0 only, as reported before)
       and, with GT_DRAWS > 1, RB_ground_truth_draws.csv (pooled over draws; sd_* = SD of the draw means)
"""
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import config as C
from analyze import ci95, pairwise_jaccard_within, topk_mask
from data import aggregate, group_index, load, make_prep, outer_split
from models import fit_and_explain, make_model, shap_values

R = 20
TRUTHS = {"linear": "LR", "nonlinear": "XGB"}
DRAWS = int(os.environ.get("GT_DRAWS", "1").strip())
SPLITS = C.OUTER_SEEDS[:int(os.environ.get("GT_SPLITS", len(C.OUTER_SEEDS)))]


def jac_to(A, T, k=3):
    """Per-customer top-k Jaccard between each explanation in A (R x N x F) and the truth T (N x F), mean over R."""
    ma, mt = topk_mask(A, k), topk_mask(T[None], k)[0]
    inter = (ma * mt[None]).sum(-1)
    return (inter / (2 * k - inter)).mean(0)


def truth_model(name, X, y):
    """Fit the truth on all customers; return its logit function's attributions for all customers and P(churn)."""
    s0 = C.OUTER_SEEDS[0]
    params = json.loads((C.RAW / f"s{s0}_{name}_tuning.json").read_text())["params"]
    cols = list(X.columns)
    prep = make_prep(cols)
    Z = prep.fit_transform(X)
    model = make_model(name, params, seed=0).fit(Z, y)
    p = model.predict_proba(Z)[:, 1]
    A = aggregate(np.asarray(shap_values(name, model, Z, Z)), group_index(prep, cols), len(cols))
    return A.astype(np.float64), p


def main():
    X, y, ids = load()
    rows = []
    for truth, tname in TRUTHS.items():
        T_all, p_true = truth_model(tname, X, y)
        for draw in range(DRAWS):
            tag = "" if draw == 0 else f"_d{draw}"
            y_sim = (np.random.default_rng(2026 + draw).random(len(y)) < p_true).astype(int)
            print(f"{truth} truth ({tname}) draw {draw}: simulated churn rate {y_sim.mean():.3f} (real {y.mean():.3f}); "
                  f"oracle ROC-AUC {roc_auc_score(y_sim, p_true):.3f}", flush=True)
            for s in SPLITS:
                sp = outer_split(X, y, ids, s)
                Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y_sim[sp["train"]]
                Xte, Xexp = X.iloc[sp["test"]].reset_index(drop=True), X.iloc[sp["explain"]].reset_index(drop=True)
                yte, T = y_sim[sp["test"]], T_all[sp["explain"]]
                oracle = roc_auc_score(yte, p_true[sp["test"]])
                for m in C.MODELS:
                    out = C.RAW / f"gt_{truth}{tag}_s{s}_{m}.npz"
                    if not out.exists():
                        params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
                        res = Parallel(n_jobs=C.N_JOBS)(delayed(fit_and_explain)(m, params, 700 + r, True, Xtr, ytr, Xte, Xexp)
                                                        for r in range(R))
                        np.savez_compressed(out, P_test=np.stack([q[0] for q in res]), A=np.stack([q[2] for q in res]))
                    full = C.RAW / f"gt_{truth}{tag}_s{s}_{m}_full.npz"
                    if not full.exists():
                        params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
                        q = fit_and_explain(m, params, 0, False, Xtr, ytr, Xte, Xexp)
                        np.savez_compressed(full, P_test=q[0], A=q[2])
                    F = np.load(full)["A"].astype(np.float64)
                    d = np.load(out)
                    A, P = d["A"].astype(np.float64), d["P_test"].astype(np.float64)
                    single = jac_to(A, T)
                    G = [A[g * 10:(g + 1) * 10].mean(0) for g in (0, 1)]
                    avg = np.mean([jac_to(G[g][None], T) for g in (0, 1)], axis=0)
                    ttop = np.argsort(-np.abs(T), axis=1)[:, :3]
                    n = np.arange(len(T))[:, None]
                    sign = (np.sign(A[:, n, ttop]) == np.sign(T[n, ttop])[None]).mean()
                    rows.append(dict(truth=truth, draw=draw, split=s, model=m, single=single.mean(), averaged=avg.mean(),
                                     gain=avg.mean() - single.mean(), sign_true_top3=sign,
                                     full_fit=jac_to(F[None], T).mean(), averaged20=jac_to(A.mean(0)[None], T).mean(),
                                     self_single=pairwise_jaccard_within(A, 3).mean(),
                                     self_avg=jac_to(G[0][None], G[1]).mean(),
                                     auc_single=np.mean([roc_auc_score(yte, p) for p in P]),
                                     auc_bag=roc_auc_score(yte, P.mean(0)), auc_oracle=oracle))
                    print(f"{truth} draw {draw} split {s} {m} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "RB_ground_truth_by_split.csv", index=False)
    if DRAWS > 1:
        pooled = []
        for (truth, m), g in df.groupby(["truth", "model"], sort=False):
            dm = g.groupby("draw").mean(numeric_only=True)
            r = dict(truth=truth, model=m, draws=DRAWS, splits=len(SPLITS))
            for c in ("single", "averaged", "averaged20", "full_fit", "gain", "self_single", "self_avg",
                      "sign_true_top3", "auc_single", "auc_bag", "auc_oracle"):
                r[c], r["sd_" + c] = dm[c].mean(), dm[c].std(ddof=1)
            r["avg20_minus_full"] = (dm.averaged20 - dm.full_fit).mean()
            r["gain_lo"], r["gain_hi"] = ci95(g.groupby("split").gain.mean())   # draws averaged within split
            r["draws_gain_pos"] = int((dm.gain > 0).sum())
            pooled.append(r)
        pd.DataFrame(pooled).to_csv(C.TABLES / "RB_ground_truth_draws.csv", index=False)
        print(pd.DataFrame(pooled).round(3).to_string(index=False))
        df = df[df.draw == 0]
    summ = []
    for (truth, m), g in df.groupby(["truth", "model"], sort=False):
        r = g.mean(numeric_only=True).drop(["split", "draw"]).to_dict()
        r.update(truth=truth, model=m)
        r["gain_lo"], r["gain_hi"] = ci95(g.gain)
        summ.append(r)
    out = pd.DataFrame(summ)
    out = out[["truth", "model"] + [c for c in out.columns if c not in ("truth", "model")]]
    out.to_csv(C.TABLES / "RB_ground_truth.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
