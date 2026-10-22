"""Sign agreement beyond the consensus top 3, and stability away from the base rate (review round 16; post hoc).

1. H3 measures sign agreement only for each customer's three consensus features, which have large |SHAP| and so
   rarely flip. Here the same measure is computed for consensus ranks 1-3, 4-6 and 7-10 and for all features
   (sign of each refit's SHAP value against the sign of the customer's mean SHAP value; a zero counts as a mismatch,
   as in analyze.sign_vs_membership).
2. Weak models (Cell2Cell) predict many customers near the base rate. Per-customer top-3 agreement is tabulated by
   quartile of the distance of the predicted risk from the base rate, on the logit scale.

    CHURN_DATASET=ibm|cell2cell|iranian python sign_ranks.py
    -> <results>/tables/RB_sign_ranks.csv, RB_baserate_quartiles.csv
"""
import json

import numpy as np
import pandas as pd

import config as C
from data import load

MODELS = ["LR", "DT", "RF", "XGB"]
BANDS = {"1-3": (0, 3), "4-6": (3, 6), "7-10": (6, 10), "all": (0, None)}


def sign_by_rank(A):
    cons = A.mean(0)                                            # N x F
    order = np.argsort(-np.abs(cons), axis=1)                   # N x F, consensus ranks
    n = np.arange(A.shape[1])[:, None]
    out = {}
    for name, (a, b) in BANDS.items():
        idx = order[:, a:b]
        out[name] = (np.sign(A[:, n, idx]) == np.sign(cons[n, idx])[None]).mean()
    return out


def main():
    X, y, ids = load()
    base = y.mean()
    rows = []
    for s in C.OUTER_SEEDS:
        for m in MODELS:
            A = np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64)
            rows.append(dict(split=s, model=m, **sign_by_rank(A)))
    t = pd.DataFrame(rows)
    out = t.groupby("model", sort=False).mean(numeric_only=True).drop(columns="split")
    out.to_csv(C.TABLES / "RB_sign_ranks.csv")
    print(out.round(3).to_string())

    loc = pd.read_csv(C.TABLES / "customer_level_stability.csv")
    lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
    loc["d_base"] = np.abs(lg(loc.p_mean) - lg(base))
    loc["q"] = loc.groupby(["split", "model"]).d_base.transform(lambda v: pd.qcut(v, 4, labels=False))
    q = loc.groupby(["model", "q"]).jac3.mean().unstack()
    q.columns = [f"q{c + 1}" for c in q.columns]
    q["q4_minus_q1"] = q.q4 - q.q1
    q["all"] = loc.groupby("model").jac3.mean()
    q = q.reindex(MODELS)
    q.to_csv(C.TABLES / "RB_baserate_quartiles.csv")
    print(f"base rate {base:.3f}\n" + q.round(3).to_string())


if __name__ == "__main__":
    main()
