"""Profit and decision impact of explanation instability (post hoc, descriptive; requested in review).

A retention campaign contacts the fraction eta of test customers with the highest predicted risk. Following the
profit framework of Verbeke et al. (2012), a contacted churner accepts the offer with probability gamma and is
retained (gain CLV, cost incentive delta + contact c); a contacted churner who declines costs c; a contacted
non-churner takes the incentive (cost delta + c). Campaign profit per test customer at eta is the realised sum over
the contacted customers divided by n_test; the maximum profit (MP) is its maximum over eta. Settings are illustrative:
CLV = 200, delta = 10, c = 1, gamma = 0.3, with a sensitivity grid over CLV and gamma.

No model is refitted: every bootstrap refit already stored its test-set predictions (P_test) and the predictions and
SHAP values of the 500 explained customers (P_exp, A). Per split and model:
  1. Targeting value: MP of each single refit (mean over 50) and of the bagged models of 10 and 50 refits.
  2. Targeting stability, for the bagged model's profit-maximising campaign size and for a fixed budget (top 10%
     by risk): contacted sets of single refits, pairwise Jaccard
     (and between bagged models of disjoint groups of 10 refits).
  3. Action stability where it matters: explained customers whom the bagged campaign contacts; the retention action
     of each customer's main reason (largest positive SHAP; mapping as in robustness.py) across single refits and
     across averages of 10 refits; share of contacted customers with an unstable action (pairwise agreement < 0.8)
     and their share of the campaign's expected retained value (sum of p * gamma * CLV over contacted customers).

    CHURN_DATASET=ibm|cell2cell|iranian python profit.py   -> <results>/tables/PR_profit*.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd

import config as C
from analyze import ci95
from data import load, outer_split
from robustness import action_of, actions, pair_agree
from stability_aware import LOGIT_SPACE

MODELS = C.MODELS
CLV, DELTA, COST, GAMMA = 200.0, 10.0, 1.0, 0.3
GRID = [(clv, g) for clv in (100.0, 200.0, 500.0) for g in (0.1, 0.3, 0.5)]
M_AVG = 10
BUDGET = 0.10


def gain(y, clv=CLV, gamma=GAMMA):
    """Realised profit of contacting each customer."""
    return np.where(y == 1, gamma * (clv - DELTA - COST) - (1 - gamma) * COST, -(DELTA + COST))


def profit_curve(score, y, clv=CLV, gamma=GAMMA):
    """Profit per test customer when contacting the top 1..n customers by score."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(gain(y[order], clv, gamma)) / len(y)


def mp(score, y, clv=CLV, gamma=GAMMA):
    c = profit_curve(score, y, clv, gamma)
    i = int(np.argmax(c))
    return max(c[i], 0.0), (i + 1) if c[i] > 0 else 0


def bag_score(P, model):
    if model in LOGIT_SPACE:
        Pc = np.clip(P.astype(np.float64), 1e-7, 1 - 1e-7)
        return np.log(Pc / (1 - Pc)).mean(0)
    return P.astype(np.float64).mean(0)


def to_prob(s, model):
    return 1 / (1 + np.exp(-s)) if model in LOGIT_SPACE else s


def top_mask(score, n):
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:n]] = True
    return m


def set_jaccard(M):
    """R x N boolean contacted sets -> mean pairwise Jaccard."""
    M = M.astype(np.float64)
    inter = M @ M.T
    size = M.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(len(M), 1)
    return float(np.mean(inter[iu] / np.maximum(union[iu], 1)))


def main():
    X, y, ids = load()
    rows, sens = [], []
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        yt, ye = y[sp["test"]], y[sp["explain"]]
        for m in MODELS:
            d = np.load(C.RAW / f"s{s}_{m}_boot.npz")
            P, Pe, A = d["P_test"], d["P_exp"], d["A"].astype(np.float64)
            R = P.shape[0]
            single = [mp(P[r].astype(np.float64), yt) for r in range(R)]
            bag50 = bag_score(P, m)
            mp50, n50 = mp(bag50, yt)
            groups = [bag_score(P[g * M_AVG:(g + 1) * M_AVG], m) for g in range(R // M_AVG)]
            mp10 = np.mean([mp(g, yt)[0] for g in groups])
            act = action_of(d["features"])
            L = actions(A, act)
            agree1 = pair_agree(L)
            Ag = A[:(R // M_AVG) * M_AVG].reshape(R // M_AVG, M_AVG, *A.shape[1:]).mean(1)
            agree10 = pair_agree(actions(Ag, act))
            unstable = agree1 < 0.8
            bag_e = bag_score(Pe, m)
            value = to_prob(bag_e, m) * GAMMA * CLV
            # two campaigns: the bagged model's profit-maximising size, and a fixed budget (top 10% by risk)
            for camp, n in (("max-profit", max(n50, 1)), ("top-10%", int(round(BUDGET * len(yt))))):
                curves = [profit_curve(P[r].astype(np.float64), yt) for r in range(R)]
                fixed_single = np.mean([c[n - 1] for c in curves])
                fixed_bag50 = profit_curve(bag50, yt)[n - 1]
                fixed_bag10 = np.mean([profit_curve(g, yt)[n - 1] for g in groups])
                j_single = set_jaccard(np.stack([top_mask(P[r], n) for r in range(R)]))
                j_bag10 = set_jaccard(np.stack([top_mask(g, n) for g in groups]))
                cut = np.sort(bag50)[::-1][n - 1]       # contacted explained customers: at or above the cut-off
                hit = bag_e >= cut
                rows.append(dict(
                    split=s, model=m, campaign=camp, eta=n / len(yt),
                    mp_single=np.mean([v for v, _ in single]) if camp == "max-profit" else fixed_single,
                    mp_bag10=mp10 if camp == "max-profit" else fixed_bag10,
                    mp_bag50=mp50 if camp == "max-profit" else fixed_bag50,
                    target_j_single=j_single, target_j_bag10=j_bag10, n_contacted_exp=int(hit.sum()),
                    churn_rate_contacted=ye[hit].mean() if hit.any() else np.nan,
                    act_agree_contacted=agree1[hit].mean(), act_agree_other=agree1[~hit].mean(),
                    unstable_contacted=unstable[hit].mean(), unstable_other=unstable[~hit].mean(),
                    value_share_unstable=value[hit & unstable].sum() / max(value[hit].sum(), 1e-12),
                    act_agree10_contacted=agree10[hit].mean(), unstable10_contacted=(agree10[hit] < 0.8).mean()))
            for clv, g in GRID:
                sens.append(dict(split=s, model=m, clv=clv, gamma=g,
                                 mp_single=np.mean([mp(P[r].astype(np.float64), yt, clv, g)[0] for r in range(R)]),
                                 mp_bag50=mp(bag50, yt, clv, g)[0], eta_bag50=mp(bag50, yt, clv, g)[1] / len(yt)))
        print(f"split {s} done", flush=True)
    df = pd.DataFrame(rows)
    df["mp_gain50"] = df.mp_bag50 - df.mp_single
    df["mp_gain10"] = df.mp_bag10 - df.mp_single
    df.to_csv(C.TABLES / "PR_profit_by_split.csv", index=False)
    summ = []
    for (camp, m), g in df.groupby(["campaign", "model"], sort=False):
        r = g.mean(numeric_only=True).drop("split").to_dict()
        r["campaign"], r["model"] = camp, m
        for col in ("mp_gain10", "mp_gain50"):
            r[f"{col}_lo"], r[f"{col}_hi"] = ci95(g[col])
        summ.append(r)
    out = pd.DataFrame(summ)[["campaign", "model"] + [c for c in pd.DataFrame(summ).columns if c not in ("campaign", "model")]]
    out.to_csv(C.TABLES / "PR_profit.csv", index=False)
    sd = pd.DataFrame(sens)
    sd["gain"] = sd.mp_bag50 - sd.mp_single
    sd.groupby(["model", "clv", "gamma"], sort=False).mean(numeric_only=True).drop(columns="split").reset_index() \
      .to_csv(C.TABLES / "PR_profit_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
