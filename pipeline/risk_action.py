"""Review round 15 (post hoc, descriptive): why are the actions of high-risk customers more stable, and the mean
number of shared top-3 features between two bootstrap refits (computed directly, not derived from the Jaccard index).

Per split and model: Spearman correlation between a customer's predicted risk (bagged model of the 50 refits) and
(a) the pairwise agreement of the action implied by the main reason, (b) the share of the customer's positive SHAP
mass on the largest positive feature (dominance of the main reason). Mean top-3 intersection over refit pairs.

    CHURN_DATASET=ibm|cell2cell|iranian python risk_action.py   -> <results>/tables/RB_risk_action.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from analyze import topk_mask
from robustness import action_of, actions, pair_agree


def main():
    rows = []
    for s in C.OUTER_SEEDS:
        for m in C.MODELS:
            d = np.load(C.RAW / f"s{s}_{m}_boot.npz")
            A = d["A"].astype(np.float64)
            P = np.clip(d["P_exp"].astype(np.float64), 1e-6, 1 - 1e-6)
            risk = P.mean(0)
            agree = pair_agree(actions(A, action_of(d["features"])))
            pos = np.where(A > 0, A, 0).mean(0)                     # mean positive SHAP per customer and feature
            dom = pos.max(-1) / np.maximum(pos.sum(-1), 1e-12)
            M = topk_mask(A, 3)
            inter = np.mean([(M[i] * M[j]).sum(-1).mean() for i, j in combinations(range(len(M)), 2)])
            rows.append(dict(split=s, model=m, rho_risk_action=stats.spearmanr(risk, agree)[0],
                             rho_risk_dominance=stats.spearmanr(risk, dom)[0], mean_top3_shared=inter))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_risk_action.csv", index=False)
    print(out.groupby("model", sort=False).mean(numeric_only=True).drop(columns="split").round(3).to_string())


if __name__ == "__main__":
    main()
