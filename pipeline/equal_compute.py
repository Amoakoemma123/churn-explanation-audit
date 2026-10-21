"""Equal-compute comparison for the averaging remedy (review round 11, post hoc).

Averaging SHAP over m = 10 bootstrap refits costs ten fits. The equal-compute alternative is ONE model with ten
times the capacity: random forest with 10 x 200 trees; XGBoost with 10 x n_estimators at a tenth of the learning
rate. Each large model is fitted on the bootstrap sample of refit r (r = 0..4, the same samples as the bootstrap
condition), so pairs of large models differ in their training sample exactly as pairs of refits do.

Per split and model: top-3 Jaccard between the 5 large models (10 pairs) and their mean test ROC-AUC, next to
(a) single refits, (b) explanations averaged over disjoint groups of 10 refits and (c) the bagged model of 10 refits.
IBM Telco and Iranian Churn only (a 2,000-tree random forest with TreeSHAP is too slow on Cell2Cell).

    CHURN_DATASET=ibm|iranian python equal_compute.py      -> <results>/tables/RB_equal_compute*.csv
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import config as C
import models
from analyze import pairwise_jaccard_within
from data import load, outer_split
from stability_aware import bagged_auc, disjoint_average_jaccard

SCALE = 10
N_LARGE = 5


def _large_fit(name, params, r, Xtr, ytr, Xte, Xexp):
    """Runs in a worker process: patch the model factory there, then fit and explain as for any refit.
    Workers are reused, so keep the original factory once; otherwise a second patch would scale the model twice."""
    if not hasattr(models, "_base_make_model"):
        models._base_make_model = models.make_model
    base = models._base_make_model

    def big(name_, params_, seed):
        if name_ == "RF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200 * SCALE, random_state=seed, n_jobs=1, **params_)
        p = dict(params_, n_estimators=params_["n_estimators"] * SCALE, learning_rate=params_["learning_rate"] / SCALE)
        return base(name_, p, seed)

    models.make_model = big
    return models.fit_and_explain(name, params, r, True, Xtr, ytr, Xte, Xexp)


def main():
    assert C.DATASET in ("ibm", "iranian"), "equal-compute check runs on IBM Telco and Iranian Churn only"
    X, y, ids = load()
    y = np.asarray(y)
    rows = []
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        Xte, Xexp = X.iloc[sp["test"]].reset_index(drop=True), X.iloc[sp["explain"]].reset_index(drop=True)
        for m in ("RF", "XGB"):
            out = C.RAW / f"s{s}_{m}_large{SCALE}.npz"
            if not out.exists():
                params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
                res = Parallel(n_jobs=3)(delayed(_large_fit)(m, params, r, Xtr, ytr, Xte, Xexp) for r in range(N_LARGE))
                np.savez_compressed(out, P_test=np.stack([q[0] for q in res]), A=np.stack([q[2] for q in res]))
            z = np.load(out)
            b = np.load(C.RAW / f"s{s}_{m}_boot.npz")
            A, P = b["A"].astype(np.float64), b["P_test"]
            yt = y[sp["test"]]
            rows.append(dict(split=s, model=m,
                             large_jac3=pairwise_jaccard_within(z["A"].astype(np.float64), 3).mean(),
                             single_jac3=pairwise_jaccard_within(A, 3).mean(),
                             avg10_jac3=disjoint_average_jaccard(A, 10, 3).mean(),
                             large_auc=np.mean([roc_auc_score(yt, p) for p in z["P_test"]]),
                             single_auc=np.mean([roc_auc_score(yt, p) for p in P]),
                             bag10_auc=np.mean([bagged_auc(P[g * 10:(g + 1) * 10], yt, m) for g in range(len(P) // 10)])))
        print(f"split {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "RB_equal_compute.csv", index=False)
    summ = df.groupby("model").mean(numeric_only=True).drop(columns="split")
    summ.to_csv(C.TABLES / "RB_equal_compute_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
