"""Averaging across disjoint training populations (review round 15; post hoc, descriptive).

The cross-split check of robustness.py compares training sets that share about 80% of their customers. Here each
outer split's training set is divided into two stratified halves that share NO customer. In each half, 10 bootstrap
refits are fitted with the split's tuned hyperparameters (tuned on the full training set, so leaf sizes are relatively
larger on a half) and explain the same 500 test customers. Per split and model:
  - single, across halves: top-3 Jaccard between refits of half 1 and refits of half 2 (10 x 10 pairs);
  - averaged, across halves: top-3 Jaccard between the SHAP average of the 10 refits of each half (one pair);
  - single, within a half: pairwise top-3 Jaccard among the refits of the same half (reference);
  - test ROC-AUC of single refits and of the bagged model of the 10 refits of a half.

Final-review addition, HALVES_RETUNE=1: each half is treated as a complete study of its own. Hyperparameters are
re-tuned on the half by the same grid search (minimum leaf sizes halved, because the half has half the customers),
so nothing tuned on the full training set reaches the halves. On Cell2Cell each half holds about 20,400 customers,
more than the whole IBM Telco training set.

    CHURN_DATASET=ibm|cell2cell|iranian [HALVES_RETUNE=1] python disjoint_halves.py
        (resumable; raw/s<seed>_<model>_halves[_rt].npz, raw/s<seed>_<model>_half<h>_tuning.json)
    -> <results>/tables/RB_halves_by_split.csv, RB_halves.csv (or RB_halves_retuned_by_split.csv, RB_halves_retuned.csv)
"""
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
from analyze import ci95, pairwise_jaccard_within, topk_mask
from data import load, outer_split
import models
from models import fit_and_explain, tune
from stability_aware import bagged_auc

R_HALF = 10
RETUNE = os.environ.get("HALVES_RETUNE", "").strip() == "1"
SUFFIX = "_rt" if RETUNE else ""
if RETUNE:
    for g in models.GRIDS.values():
        if "min_samples_leaf" in g:
            g["min_samples_leaf"] = [max(1, round(v / 2)) for v in g["min_samples_leaf"]]


def half_params(s, m, h, Xh, yh):
    """Tuned hyperparameters for one half (cached), or the split's full-training-set ones without HALVES_RETUNE."""
    if not RETUNE:
        return json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
    f = C.RAW / f"s{s}_{m}_half{h}_tuning.json"
    if not f.exists():
        params, auc, _, _ = tune(m, Xh, yh, s, C.N_JOBS)
        f.write_text(json.dumps(dict(params=params, cv_auc=auc)))
    return json.loads(f.read_text())["params"]


def jac_between(A, B, k=3):
    """Mean over all pairs (a in A, b in B) of the per-customer top-k Jaccard; returns per-customer means."""
    ma, mb = topk_mask(A, k), topk_mask(B, k)
    inter = np.einsum("rnf,snf->rsn", ma.astype(np.float64), mb.astype(np.float64))
    return (inter / (2 * k - inter)).mean((0, 1))


def main():
    X, y, ids = load()
    rows = []
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        Xte, Xexp = X.iloc[sp["test"]].reset_index(drop=True), X.iloc[sp["explain"]].reset_index(drop=True)
        yte = y[sp["test"]]
        halves = [h for _, h in StratifiedKFold(2, shuffle=True, random_state=s).split(Xtr, ytr)]
        assert not set(halves[0]) & set(halves[1])
        for m in C.MODELS:
            out = C.RAW / f"s{s}_{m}_halves{SUFFIX}.npz"
            if not out.exists():
                Xh = [Xtr.iloc[halves[h]].reset_index(drop=True) for h in (0, 1)]
                params = [half_params(s, m, h, Xh[h], ytr[halves[h]]) for h in (0, 1)]
                jobs = [delayed(fit_and_explain)(m, params[h], 500 + 100 * h + r, True, Xh[h], ytr[halves[h]], Xte, Xexp)
                        for h in (0, 1) for r in range(R_HALF)]
                res = Parallel(n_jobs=C.N_JOBS)(jobs)
                P = np.stack([q[0] for q in res]).reshape(2, R_HALF, -1)
                A = np.stack([q[2] for q in res]).reshape(2, R_HALF, *res[0][2].shape)
                np.savez_compressed(out, P_test=P, A=A)
                print(f"split {s} {m} halves{SUFFIX} done", flush=True)
            d = np.load(out)
            P, A = d["P_test"].astype(np.float64), d["A"].astype(np.float64)
            single_x = jac_between(A[0], A[1])
            avg_x = jac_between(A[0].mean(0)[None], A[1].mean(0)[None])
            within = np.mean([pairwise_jaccard_within(A[h], 3) for h in (0, 1)], axis=0)
            rows.append(dict(split=s, model=m, single_within=within.mean(), single_across=single_x.mean(),
                             avg10_across=avg_x.mean(), gain=avg_x.mean() - single_x.mean(),
                             unstable_single_across=(single_x < 0.5).mean(), unstable_avg10_across=(avg_x < 0.5).mean(),
                             auc_single=np.mean([roc_auc_score(yte, P[h, r]) for h in (0, 1) for r in range(R_HALF)]),
                             auc_bag10=np.mean([bagged_auc(P[h], yte, m) for h in (0, 1)])))
    df = pd.DataFrame(rows)
    df["auc_gain"] = df.auc_bag10 - df.auc_single
    df.to_csv(C.TABLES / f"RB_halves{'_retuned' if RETUNE else ''}_by_split.csv", index=False)
    summ = []
    for m, g in df.groupby("model", sort=False):
        r = g.mean(numeric_only=True).drop("split").to_dict()
        r["model"] = m
        r["gain_lo"], r["gain_hi"] = ci95(g.gain)
        r["auc_gain_lo"], r["auc_gain_hi"] = ci95(g.auc_gain)
        summ.append(r)
    out = pd.DataFrame(summ)
    out = out[["model"] + [c for c in out.columns if c != "model"]]
    out.to_csv(C.TABLES / f"RB_halves{'_retuned' if RETUNE else ''}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
