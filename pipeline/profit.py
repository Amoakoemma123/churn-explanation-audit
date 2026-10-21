"""Decision impact of explanation instability in a simulated retention campaign (post hoc, descriptive).

Profit framework of Verbeke et al. (2012): a contacted churner accepts the incentive with probability gamma and is
retained (gain CLV - delta - c); a contacted churner who declines costs c; a contacted non-churner takes the incentive
(cost delta + c). Illustrative settings CLV = 200, delta = 10, c = 1, gamma = 0.3, with a sensitivity grid over CLV
and gamma. No model is refitted: every bootstrap refit stored its test-set predictions (P_test) and the predictions
and SHAP values of the 500 explained customers (P_exp, A).

Two campaigns, both chosen without looking at test outcomes:
  top-10%     the 10% of test customers with the highest predicted risk (primary; a fixed retention budget);
  break-even  every customer whose predicted churn probability exceeds the break-even risk
              p* = (delta + c) / (gamma (CLV - delta - c) - (1 - gamma) c + delta + c), 0.164 at the main settings.

Per split, model and campaign:
  1. Profit per test customer of single refits (mean over 50) and of bagged models of 10 refits (5 disjoint groups).
  2. Contacted-set overlap: pairwise Jaccard of the contacted customers (single refits; bagged models of 10).
  3. Exposure: among the explained customers that the bagged campaign (50 refits) contacts, the share whose retention
     action (main reason = largest positive SHAP, mapping in robustness.py) is unstable (pairwise agreement below
     tau = 0.8; also 0.7 and 0.9), and their share of the campaign's expected retained value (p * gamma * CLV).
  4. Offer-dependent acceptance: a churner offered an action other than the reference action accepts with
     probability gamma (1 - kappa), kappa in {0.1, 0.25, 0.5}. The reference is the action of the SHAP values averaged
     over the held-out refits 25-49; offers come from single refits 0-24 or from averages of refits 0-9 and 10-19.
     Profit lost per test customer = kappa gamma (CLV - delta) x (contacted churners per test customer) x mismatch
     rate among contacted explained churners; reported also as a share of the single-refit campaign profit.
Across outer splits (training sets sharing about 80% of customers): for customers explained in two splits and contacted
in both, action agreement between single refits of the two splits and between averages of 10 refits.

    CHURN_DATASET=ibm|cell2cell|iranian python profit.py   -> <results>/tables/PR_*.csv
"""
from collections import defaultdict

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
KAPPAS = (0.1, 0.25, 0.5)
TAUS = (0.7, 0.8, 0.9)
M_AVG = 10
BUDGET = 0.10
HALF = 25


def breakeven(clv=CLV, gamma=GAMMA):
    return (DELTA + COST) / (gamma * (clv - DELTA - COST) - (1 - gamma) * COST + DELTA + COST)


def gain(y, clv=CLV, gamma=GAMMA):
    """Realised profit of contacting each customer."""
    return np.where(y == 1, gamma * (clv - DELTA - COST) - (1 - gamma) * COST, -(DELTA + COST))


def bag_score(P, model):
    """Bagged score, averaged on the scale the model is explained on (log-odds for LR and XGB)."""
    if model in LOGIT_SPACE:
        Pc = np.clip(P.astype(np.float64), 1e-7, 1 - 1e-7)
        return np.log(Pc / (1 - Pc)).mean(0)
    return P.astype(np.float64).mean(0)


def to_prob(s, model):
    return 1 / (1 + np.exp(-s)) if model in LOGIT_SPACE else s


def contact(prob, campaign, clv=CLV, gamma=GAMMA):
    """Boolean contact mask from predicted churn probabilities."""
    if campaign == "top-10%":
        m = np.zeros(len(prob), bool)
        m[np.argsort(-prob, kind="stable")[:int(round(BUDGET * len(prob)))]] = True
        return m
    return prob >= breakeven(clv, gamma)


def profit(mask, y, clv=CLV, gamma=GAMMA):
    return gain(y[mask], clv, gamma).sum() / len(y)


def set_jaccard(M):
    """R x N boolean contacted sets -> mean pairwise Jaccard."""
    M = M.astype(np.float64)
    inter = M @ M.T
    size = M.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(len(M), 1)
    return float(np.mean(inter[iu] / np.maximum(union[iu], 1)))


def label_freq(L, labels):
    """R x N labels -> N x len(labels) relative frequencies."""
    return np.stack([(L == a).mean(0) for a in labels], axis=1)


CAMPAIGNS = ("top-10%", "break-even")


def main():
    X, y, ids = load()
    rows, sens, keep = [], [], defaultdict(dict)
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        yt, ye = y[sp["test"]], y[sp["explain"]]
        for m in MODELS:
            d = np.load(C.RAW / f"s{s}_{m}_boot.npz")
            P, Pe, A = d["P_test"].astype(np.float64), d["P_exp"], d["A"].astype(np.float64)
            R = P.shape[0]
            G = R // M_AVG
            groups = [to_prob(bag_score(P[g * M_AVG:(g + 1) * M_AVG], m), m) for g in range(G)]
            prob50 = to_prob(bag_score(P, m), m)
            prob_e = to_prob(bag_score(Pe, m), m)
            act = action_of(d["features"])
            labels = np.unique(act)
            L = actions(A, act)
            agree1 = pair_agree(L)
            L10 = actions(A[:G * M_AVG].reshape(G, M_AVG, *A.shape[1:]).mean(1), act)
            agree10 = pair_agree(L10)
            ref = actions(A[HALF:].mean(0)[None], act)[0]                     # held-out reference action
            mis1 = (L[:HALF] != ref).mean(0)                                   # per customer, over refits 0-24
            mis10 = np.mean([actions(A[g * M_AVG:(g + 1) * M_AVG].mean(0)[None], act)[0] != ref
                             for g in range(HALF // M_AVG)], axis=0)
            value = prob_e * GAMMA * CLV
            for camp in CAMPAIGNS:
                masks = np.stack([contact(P[r], camp) for r in range(R)])
                gmasks = np.stack([contact(g, camp) for g in groups])
                p_single = np.mean([profit(mk, yt) for mk in masks])
                p_bag10 = np.mean([profit(mk, yt) for mk in gmasks])
                c50 = contact(prob50, camp)
                cut = prob50[c50].min() if c50.any() else np.inf
                hit = prob_e >= cut                                            # contacted explained customers
                churn_hit = hit & (ye == 1)
                churners_per_cust = yt[c50].sum() / len(yt)
                r = dict(split=s, model=m, campaign=camp, contacted=c50.mean(), profit_single=p_single,
                         profit_bag10=p_bag10, gain10=p_bag10 - p_single,
                         target_j_single=set_jaccard(masks), target_j_bag10=set_jaccard(gmasks),
                         n_contacted_exp=int(hit.sum()), n_churners_contacted_exp=int(churn_hit.sum()),
                         unstable_other=(agree1[~hit] < 0.8).mean() if (~hit).any() else np.nan,
                         value_exposed=value[hit & (agree1 < 0.8)].sum() / max(value[hit].sum(), 1e-12),
                         mismatch_single=mis1[churn_hit].mean() if churn_hit.any() else np.nan,
                         mismatch_bag10=mis10[churn_hit].mean() if churn_hit.any() else np.nan)
                for t in TAUS:
                    r[f"unstable_{t}"] = (agree1[hit] < t).mean()
                    r[f"unstable10_{t}"] = (agree10[hit] < t).mean()
                for k in KAPPAS:
                    unit = k * GAMMA * (CLV - DELTA) * churners_per_cust
                    r[f"loss_single_k{k}"] = unit * r["mismatch_single"]
                    r[f"loss_bag10_k{k}"] = unit * r["mismatch_bag10"]
                    r[f"loss_share_single_k{k}"] = r[f"loss_single_k{k}"] / p_single if p_single > 0 else np.nan
                    r[f"loss_share_bag10_k{k}"] = r[f"loss_bag10_k{k}"] / p_bag10 if p_bag10 > 0 else np.nan
                rows.append(r)
                keep[(s, m)][camp] = hit
                for clv, g in GRID:
                    ms = [contact(P[i], camp, clv, g) for i in range(R)]
                    gs = [contact(x, camp, clv, g) for x in groups]
                    ps = np.mean([profit(mk, yt, clv, g) for mk in ms])
                    pb = np.mean([profit(mk, yt, clv, g) for mk in gs])
                    sens.append(dict(split=s, model=m, campaign=camp, clv=clv, gamma=g, profit_single=ps,
                                     profit_bag10=pb, gain10=pb - ps, contacted=np.mean([x.mean() for x in gs])))
            keep[(s, m)].update(ids=sp["explain"], F=label_freq(L, labels), F10=label_freq(L10, labels), labels=labels)
        print(f"split {s} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "PR_profit_by_split.csv", index=False)
    summ = []
    for (camp, m), g in df.groupby(["campaign", "model"], sort=False):
        r = g.mean(numeric_only=True).drop("split").to_dict()
        r["campaign"], r["model"] = camp, m
        r["gain10_lo"], r["gain10_hi"] = ci95(g.gain10)
        summ.append(r)
    out = pd.DataFrame(summ)
    out = out[["campaign", "model"] + [c for c in out.columns if c not in ("campaign", "model")]]
    out.to_csv(C.TABLES / "PR_profit.csv", index=False)
    sd = pd.DataFrame(sens)
    sd.groupby(["campaign", "model", "clv", "gamma"], sort=False).mean(numeric_only=True).drop(columns="split") \
      .reset_index().to_csv(C.TABLES / "PR_profit_sensitivity.csv", index=False)

    # ---- across outer splits: same customer, contacted in both splits
    cross = []
    seeds = list(C.OUTER_SEEDS)
    for m in MODELS:
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                ka, kb = keep[(a, m)], keep[(b, m)]
                common, ia, ib = np.intersect1d(ka["ids"], kb["ids"], return_indices=True)
                if not len(common):
                    continue
                assert list(ka["labels"]) == list(kb["labels"])
                ag1 = (ka["F"][ia] * kb["F"][ib]).sum(1)                      # agreement over all refit pairs
                ag10 = (ka["F10"][ia] * kb["F10"][ib]).sum(1)
                for camp in CAMPAIGNS + ("all",):
                    sel = np.ones(len(common), bool) if camp == "all" else ka[camp][ia] & kb[camp][ib]
                    if sel.any():
                        cross.append(dict(model=m, s1=a, s2=b, campaign=camp, n=int(sel.sum()),
                                          agree_single=ag1[sel].mean(), agree_bag10=ag10[sel].mean(),
                                          unstable_single=(ag1[sel] < 0.8).mean(), unstable_bag10=(ag10[sel] < 0.8).mean()))
    cr = pd.DataFrame(cross)
    cr.to_csv(C.TABLES / "PR_crosssplit_pairs.csv", index=False)
    w = lambda g, c: np.average(g[c], weights=g.n)
    cs = pd.DataFrame([dict(campaign=camp, model=m, n=int(g.n.sum()),
                            **{c: w(g, c) for c in ("agree_single", "agree_bag10", "unstable_single", "unstable_bag10")})
                       for (camp, m), g in cr.groupby(["campaign", "model"], sort=False)])
    cs.to_csv(C.TABLES / "PR_crosssplit.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["campaign", "model", "contacted", "profit_single", "profit_bag10", "gain10", "gain10_lo", "gain10_hi",
               "target_j_single", "target_j_bag10", "unstable_0.8", "unstable10_0.8", "value_exposed",
               "mismatch_single", "mismatch_bag10", "loss_share_single_k0.25", "loss_share_bag10_k0.25"]].round(3).to_string())
    print(cs.round(3).to_string())


if __name__ == "__main__":
    main()
