"""Robustness checks requested in peer review (post hoc, not pre-registered; no reported test is changed).

    python robustness.py ties      # near-ties, magnitude overlap, personalisation, chance correction, retention actions
    python robustness.py sub80     # 50 refits on 80% subsamples without replacement (writes raw/*_sub80.npz)
    python robustness.py common    # all four models explained by one explainer on one scale (3 splits, 10 refits)
    python robustness.py k5        # IBM Telco: all 32 tests recomputed on the first five outer splits only
    python robustness.py crosssplit  # averaged explanations across outer splits (different training customers)
    python robustness.py example     # IBM Telco: one customer's top-3 reasons across refits and models
    python robustness.py bagfloor    # seed-only floor of an explanation averaged over 10 refits
    python robustness.py nostatus    # Iranian Churn: bootstrap refits without Status (possible label leakage)

The dataset is chosen with CHURN_DATASET as elsewhere; every check writes results*/tables/RB_*.csv.
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import config as C
from analyze import pairwise_jaccard_within, sign_vs_membership, topk_mask
from data import load, outer_split

MODELS = C.MODELS

# ---------------------------------------------------------------- retention actions (illustrative mapping)
# The main reason for a customer's risk is the feature with the largest POSITIVE SHAP value; it is mapped to the
# retention action a team would plausibly take. Features that suggest no action (demographics) map to "none".
ACTIONS = {
    "ibm": {"Contract": "contract offer", "tenure": "loyalty reward",
            "MonthlyCharges": "price discount", "TotalCharges": "price discount",
            "PaymentMethod": "billing change", "PaperlessBilling": "billing change",
            **{f: "service bundle" for f in ("InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                             "TechSupport", "StreamingTV", "StreamingMovies", "PhoneService",
                                             "MultipleLines")},
            **{f: "none" for f in ("gender", "SeniorCitizen", "Partner", "Dependents")}},
    "iranian": {"CallFailure": "network quality", "Complains": "complaint follow-up",
                "SubscriptionLength": "loyalty reward", "ChargeAmount": "price discount",
                "CustomerValue": "price discount", "TariffPlan": "plan change", "Status": "reactivation offer",
                **{f: "usage bundle" for f in ("SecondsOfUse", "FrequencyOfUse", "FrequencyOfSMS",
                                               "DistinctCalledNumbers")},
                "AgeGroup": "none"},
}


def c2c_action(f):
    if f.startswith("Handset") or f in ("CurrentEquipmentDays", "NewCellphoneUser", "NotNewCellphoneUser"):
        return "handset upgrade"
    if f in ("MonthlyRevenue", "TotalRecurringCharge", "OverageMinutes", "PercChangeRevenues",
             "AdjustmentsToCreditRating", "CreditRating"):
        return "price plan"
    if f in ("DroppedCalls", "BlockedCalls", "DroppedBlockedCalls", "UnansweredCalls"):
        return "network quality"
    if f in ("CustomerCareCalls", "RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam",
             "DirectorAssistedCalls"):
        return "care contact"
    if f == "MonthsInService":
        return "loyalty reward"
    if f in ("MonthlyMinutes", "PercChangeMinutes", "RoamingCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls",
             "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut", "CallForwardingCalls", "CallWaitingCalls",
             "UniqueSubs", "ActiveSubs"):
        return "usage plan"
    return "none"


def action_of(feats):
    table = ACTIONS.get(C.DATASET)
    return np.array([table[f] if table else c2c_action(f) for f in feats] + ["none"])   # last: no positive SHAP


def actions(A, act):
    """R x N x F SHAP -> R x N action labels from the feature with the largest positive SHAP value."""
    pos = np.where(A > 0, A, -np.inf)
    top = pos.argmax(-1)
    top[np.isneginf(pos.max(-1))] = len(act) - 1
    return act[top]


def pair_agree(L):
    """R x N labels -> per customer, share of refit pairs with the same label."""
    R = L.shape[0]
    return np.mean([(L[i] == L[j]) for i, j in combinations(range(R), 2)], axis=0)


def mass_overlap(A):
    """Magnitude-weighted agreement: sum over features of min(p_r, p_s), p = |SHAP| share; per customer."""
    P = np.abs(A) / np.maximum(np.abs(A).sum(-1, keepdims=True), 1e-12)
    return np.mean([np.minimum(P[i], P[j]).sum(-1) for i, j in combinations(range(A.shape[0]), 2)], axis=0)


def ties():
    feats = list(np.load(C.RAW / f"s{C.OUTER_SEEDS[0]}_LR_boot.npz")["features"])
    F, act = len(feats), action_of(feats)
    rows, quint, between = [], [], []
    for s in C.OUTER_SEEDS:
        boot = {m: np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64) for m in MODELS}
        for m in MODELS:
            A = boot[m]
            seed = np.load(C.RAW / f"s{s}_{m}_seed.npz")["A"].astype(np.float64)
            jac = pairwise_jaccard_within(A, 3)
            srt = -np.sort(-np.abs(A), axis=-1)
            gap = ((srt[..., 2] - srt[..., 3]) / np.maximum(srt.sum(-1), 1e-12)).mean(0)   # 3rd vs 4th, per customer
            M, G = topk_mask(A, 3), topk_mask(np.abs(A).mean(1, keepdims=True), 3)        # customer vs global top 3
            inter = (M * G).sum(-1)
            sign_ok, member = sign_vs_membership(A, 3)
            L, Lavg = actions(A, act), None
            g = A.shape[0] // 10
            Lavg = actions(A[:g * 10].reshape(g, 10, *A.shape[1:]).mean(1), act)
            top1 = np.abs(A).argmax(-1)
            rows.append(dict(split=s, model=m, jac3=jac.mean(), mass_boot=mass_overlap(A).mean(),
                             mass_seed=mass_overlap(seed).mean(), rho_gap=stats.spearmanr(gap, jac).statistic,
                             personal_jac=(inter / (6 - inter)).mean(),
                             sign_cc=((sign_ok - 0.5) / 0.5).mean(), member_cc=((member - 3 / F) / (1 - 3 / F)).mean(),
                             top1_agree=pair_agree(top1).mean(), action_agree=pair_agree(L).mean(),
                             action_unstable=(pair_agree(L) < 0.8).mean(), action_agree_avg10=pair_agree(Lavg).mean(),
                             n_actions=len(set(L.ravel()))))
            q = pd.qcut(gap, 5, labels=False, duplicates="drop")
            for k in range(int(q.max()) + 1):
                quint.append(dict(split=s, model=m, quintile=k + 1, jac3=jac[q == k].mean()))
        for a, b in combinations(MODELS, 2):
            La, Lb = actions(boot[a], act), actions(boot[b], act)
            between.append(dict(split=s, pair=f"{a}-{b}", action_between=(La == Lb).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_ties.csv", index=False)
    pd.DataFrame(quint).to_csv(C.TABLES / "RB_gap_quintiles.csv", index=False)
    pd.DataFrame(between).to_csv(C.TABLES / "RB_actions_between.csv", index=False)
    print(out.groupby("model").mean(numeric_only=True).drop(columns="split").round(3).to_string())
    print(pd.DataFrame(quint).groupby(["model", "quintile"]).jac3.mean().unstack().round(3).to_string())
    print(pd.DataFrame(between).groupby("pair").action_between.mean().round(3).to_string())


# ---------------------------------------------------------------- 80% subsampling instead of the bootstrap
def _fit_sub(name, params, r, Xtr, ytr, Xte, Xexp):
    from models import aggregate, group_index, make_model, make_prep, shap_values
    idx = np.random.default_rng(20_000 + r).choice(len(Xtr), int(0.8 * len(Xtr)), replace=False)
    cols = list(Xtr.columns)
    prep = make_prep(cols)
    Zfit = prep.fit_transform(Xtr.iloc[idx])
    model = make_model(name, params, seed=r).fit(Zfit, ytr[idx])
    Zexp = prep.transform(Xexp)
    A = aggregate(np.asarray(shap_values(name, model, Zfit, Zexp)), group_index(prep, cols), len(cols))
    return model.predict_proba(prep.transform(Xte))[:, 1].astype(np.float32), A.astype(np.float32)


def sub80():
    import json
    X, y, ids = load()
    rows = []
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        Xte, Xexp = X.iloc[sp["test"]].reset_index(drop=True), X.iloc[sp["explain"]].reset_index(drop=True)
        for m in MODELS:
            out = C.RAW / f"s{s}_{m}_sub80.npz"
            if not out.exists():
                params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
                res = Parallel(n_jobs=C.N_JOBS)(delayed(_fit_sub)(m, params, r, Xtr, ytr, Xte, Xexp)
                                                for r in range(C.B_BOOT))
                np.savez_compressed(out, P_test=np.stack([q[0] for q in res]), A=np.stack([q[1] for q in res]))
                print(f"split {s} {m} sub80 done", flush=True)
            A = np.load(out)["A"].astype(np.float64)
            boot = np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64)
            rows.append(dict(split=s, model=m, jac3_sub80=pairwise_jaccard_within(A, 3).mean(),
                             jac3_boot=pairwise_jaccard_within(boot, 3).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_sub80.csv", index=False)
    print(out.groupby("model").mean(numeric_only=True).drop(columns="split").round(3).to_string())


# ---------------------------------------------------------------- one explainer, one scale, one value function
# Reported in the paper: IBM Telco on all 10 splits, Iranian Churn on the first 3 (10 refits x 100 customers each);
# Cell2Cell was not run. The full Iranian and Cell2Cell runs need several GB of free memory.
COMMON_SPLITS = tuple(C.OUTER_SEEDS) if C.DATASET == "ibm" else tuple(C.OUTER_SEEDS[:3])
COMMON_R, COMMON_N, COMMON_BG = 10, 100, 50


def _codes(X, cats):
    Z = X.copy()
    for c, levels in cats.items():
        Z[c] = pd.Categorical(Z[c], categories=levels).codes
    return Z.to_numpy(dtype=float)


def _decode(Z, cols, cats):
    df = pd.DataFrame(Z, columns=cols)
    for c, levels in cats.items():
        df[c] = np.asarray(levels, dtype=object)[np.clip(np.rint(df[c].to_numpy()).astype(int), 0, len(levels) - 1)]
    return df


def _common_one(name, params, r, Xtr, ytr, Xexp, bg, cats, seed):
    """Refit exactly as in the main experiment (same bootstrap sample), then explain the PROBABILITY output with
    interventional permutation SHAP against a background shared by all models and refits."""
    import shap
    from models import make_model, make_prep
    cols = list(Xtr.columns)
    idx = np.random.default_rng(10_000 + r).integers(0, len(Xtr), len(Xtr))
    prep = make_prep(cols)
    model = make_model(name, params, seed=r).fit(prep.fit_transform(Xtr.iloc[idx]), ytr[idx])
    f = lambda Z: model.predict_proba(prep.transform(_decode(Z, cols, cats)))[:, 1]
    ex = shap.explainers.Permutation(f, shap.maskers.Independent(bg, max_samples=len(bg)), seed=seed)
    return ex(_codes(Xexp, cats), max_evals=10 * (2 * len(cols) + 1), silent=True).values.astype(np.float32)


def common():
    import json
    X, y, ids = load()
    cats = {c: sorted(X[c].unique()) for c in X.columns if X[c].dtype == object}
    jobs, keys = [], []
    for s in COMMON_SPLITS:
        sp = outer_split(X, y, ids, s)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        Xexp = X.iloc[sp["explain"][:COMMON_N]].reset_index(drop=True)
        bg = _codes(Xtr.sample(COMMON_BG, random_state=s), cats)
        for m in MODELS:
            params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
            for r in range(COMMON_R):
                jobs.append(delayed(_common_one)(m, params, r, Xtr, ytr, Xexp, bg, cats, 0)); keys.append((s, m, r, 0))
            jobs.append(delayed(_common_one)(m, params, 0, Xtr, ytr, Xexp, bg, cats, 1)); keys.append((s, m, 0, 1))
    res = dict(zip(keys, Parallel(n_jobs=C.N_JOBS, verbose=5)(jobs)))
    np.savez_compressed(C.RAW / "common_explainer.npz", **{f"s{s}_{m}_r{r}_e{e}": v for (s, m, r, e), v in res.items()})
    rows = []
    for s in COMMON_SPLITS:
        A = {m: np.stack([res[(s, m, r, 0)] for r in range(COMMON_R)]) for m in MODELS}
        native = {m: np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"][:COMMON_R, :COMMON_N].astype(np.float64) for m in MODELS}
        for m in MODELS:
            noise = topk_mask(np.stack([res[(s, m, 0, 0)], res[(s, m, 0, 1)]]), 3)
            inter = (noise[0] * noise[1]).sum(-1)
            rows.append(dict(split=s, kind="within", pair=m, common=pairwise_jaccard_within(A[m], 3).mean(),
                             native=pairwise_jaccard_within(native[m], 3).mean(),
                             explainer_floor=(inter / (6 - inter)).mean()))
        for a, b in combinations(MODELS, 2):
            j = lambda D: np.mean([((topk_mask(D[a][r], 3) * topk_mask(D[b][r], 3)).sum(-1) /
                                    (6 - (topk_mask(D[a][r], 3) * topk_mask(D[b][r], 3)).sum(-1))).mean()
                                   for r in range(COMMON_R)])
            rows.append(dict(split=s, kind="between", pair=f"{a}-{b}", common=j(A), native=j(native)))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_common_explainer.csv", index=False)
    print(out.groupby(["kind", "pair"]).mean(numeric_only=True).drop(columns="split").round(3).to_string())


# ---------------------------------------------------------------- remedy across training populations
def crosssplit():
    """Customers explained in two outer splits were explained by models trained on different training sets (and
    tuned separately). Compare single refits with explanations averaged over 10 refits (and over all 50) across
    splits: this varies the training population, not only the bootstrap draw."""
    sp = {s: np.load(C.RAW / f"s{s}_split.npz")["explain"] for s in C.OUTER_SEEDS}
    rows = []
    for m in MODELS:
        A = {s: np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64) for s in C.OUTER_SEEDS}
        for s1, s2 in combinations(C.OUTER_SEEDS, 2):
            common_ids, i1, i2 = np.intersect1d(sp[s1], sp[s2], return_indices=True)
            if len(common_ids) == 0:
                continue
            a, b = A[s1][:, i1], A[s2][:, i2]                      # R x n x F for the shared customers
            jac = lambda X, Y: ((topk_mask(X, 3) * topk_mask(Y, 3)).sum(-1) / (6 - (topk_mask(X, 3) * topk_mask(Y, 3)).sum(-1)))
            single = np.mean([jac(a[r], b[r2]) for r in range(0, 50, 5) for r2 in range(0, 50, 5)], axis=0)
            g1, g2 = a.reshape(5, 10, *a.shape[1:]).mean(1), b.reshape(5, 10, *b.shape[1:]).mean(1)
            avg10 = np.mean([jac(g1[i], g2[j]) for i in range(5) for j in range(5)], axis=0)
            avg50 = jac(a.mean(0), b.mean(0))
            rows.append(dict(model=m, s1=s1, s2=s2, n=len(common_ids), single=single.mean(), avg10=avg10.mean(),
                             avg50=avg50.mean(), w=len(common_ids)))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_crosssplit.csv", index=False)
    summ = out.groupby("model").apply(lambda d: pd.Series({c: np.average(d[c], weights=d.w) for c in ("single", "avg10", "avg50")}
                                                          | {"customer_pairs": d.n.sum(), "split_pairs": len(d)}))
    summ.to_csv(C.TABLES / "RB_crosssplit_summary.csv")
    print(summ.round(3).to_string())


def crosssplit_ci():
    """Corrected 95% CIs for the cross-split gains (review round 6), from RB_crosssplit.csv (no refitting).
    The 45 split pairs are not independent, so each outer split s gets one value: the customer-weighted mean gain
    over the 9 pairs that contain s. The K = 10 per-split values get the Nadeau-Bengio corrected CI (analyze.ci95);
    each pair enters two splits' values, so the interval is approximate."""
    from analyze import ci95
    out = pd.read_csv(C.TABLES / "RB_crosssplit.csv")
    rows = []
    for m, d in out.groupby("model"):
        for gain in ("avg10", "avg50"):
            vals = []
            for s in C.OUTER_SEEDS:
                e = d[(d.s1 == s) | (d.s2 == s)]
                vals.append(np.average(e[gain] - e.single, weights=e.w))
            lo, hi = ci95(vals)
            rows.append(dict(model=m, comparison=f"{gain}-single", gain_mean=np.mean(vals), ci_lo=lo, ci_hi=hi,
                             K=len(vals), customer_pairs=int(d.n.sum())))
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "RB_crosssplit_ci.csv", index=False)
    print(res.round(3).to_string(index=False))


def example():
    """Worked example: one IBM customer explained in split 101, top-3 reasons in five refits of each model."""
    feats = list(np.load(C.RAW / f"s{C.OUTER_SEEDS[0]}_LR_boot.npz")["features"])
    s = C.OUTER_SEEDS[0]
    A = {m: np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64) for m in MODELS}
    jac = {m: pairwise_jaccard_within(A[m], 3) for m in MODELS}
    # a customer close to the median instability of all four models (a typical, not an extreme, case)
    score = sum(np.abs(jac[m] - np.median(jac[m])) for m in MODELS)
    n = int(np.argmin(score))
    rows = []
    for m in MODELS:
        tops = [tuple(sorted(feats[f] for f in np.argsort(-np.abs(A[m][r, n]))[:3])) for r in range(A[m].shape[0])]
        vc = pd.Series(tops).value_counts()
        for k, (sset, cnt) in enumerate(vc.head(3).items()):
            rows.append(dict(model=m, rank=k + 1, top3=", ".join(sset), refits=int(cnt), jac3=jac[m][n],
                             n_distinct=len(vc), customer=n))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_example.csv", index=False)
    print(out.to_string(index=False))


# ---------------------------------------------------------------- bagged model's own seed floor
def bagfloor():
    """Seed-only floor of the averaged explanation: the 20 seed-only refits (same data) split into two groups of 10,
    averaged, and compared. The averaged bootstrap explanation (m = 10) should be read against this floor."""
    rows = []
    for s in C.OUTER_SEEDS:
        for m in MODELS:
            A = np.load(C.RAW / f"s{s}_{m}_seed.npz")["A"].astype(np.float64)
            g = A[:20].reshape(2, 10, *A.shape[1:]).mean(1)
            rows.append(dict(split=s, model=m, bag_seed_floor=pairwise_jaccard_within(g, 3).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_bagfloor.csv", index=False)
    print(out.groupby("model").bag_seed_floor.mean().round(3).to_string())


# ---------------------------------------------------------------- Iranian Churn without Status (possible leakage)
def nostatus():
    """Bootstrap refits without `Status` (Iranian Churn only): accuracy and per-customer stability."""
    import json
    from sklearn.metrics import roc_auc_score
    from models import fit_and_explain
    assert C.DATASET == "iranian"
    X, y, ids = load()
    cols = [c for c in X.columns if c != "Status"]
    rows = []
    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True)[cols], y[sp["train"]]
        Xte = X.iloc[sp["test"]].reset_index(drop=True)[cols]
        Xexp = X.iloc[sp["explain"]].reset_index(drop=True)[cols]
        for m in MODELS:
            out = C.RAW / f"s{s}_{m}_boot_noStatus.npz"
            if not out.exists():
                params = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())["params"]
                res = Parallel(n_jobs=3)(delayed(fit_and_explain)(m, params, r, True, Xtr, ytr, Xte, Xexp)
                                         for r in range(C.B_BOOT))
                np.savez_compressed(out, P_test=np.stack([q[0] for q in res]), A=np.stack([q[2] for q in res]))
            z = np.load(out)
            w = np.load(C.RAW / f"s{s}_{m}_boot.npz")             # same bootstrap refits with Status, for a like-for-like AUC
            rows.append(dict(split=s, model=m, jac3=pairwise_jaccard_within(z["A"].astype(np.float64), 3).mean(),
                             auc=np.mean([roc_auc_score(y[sp["test"]], p) for p in z["P_test"]]),
                             auc_with=np.mean([roc_auc_score(y[sp["test"]], p) for p in w["P_test"]])))
        print(f"split {s} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "RB_nostatus.csv", index=False)
    print(out.groupby("model")[["auc", "jac3"]].mean().round(3).to_string())


# ---------------------------------------------------------------- IBM Telco: all tests with K = 5 (the original plan)
def k5():
    import analyze
    C.OUTER_SEEDS = C.OUTER_SEEDS[:5]
    C.TABLES, C.FIGS = C.RESULTS / "tables_k5", C.RESULTS / "figures_k5"
    analyze.main()
    k5t = pd.read_csv(C.TABLES / "hypothesis_tests_holm.csv")
    k10 = pd.read_csv(C.RESULTS / "tables" / "hypothesis_tests_holm.csv")
    cmp = k10[["hypothesis", "test", "p_holm", "reject_0.05"]].merge(
        k5t[["test", "p_holm", "reject_0.05"]], on="test", suffixes=("_k10", "_k5"))
    cmp.to_csv(C.RESULTS / "tables" / "RB_k5_vs_k10.csv", index=False)
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    {"ties": ties, "sub80": sub80, "common": common, "k5": k5, "crosssplit": crosssplit, "crosssplit_ci": crosssplit_ci, "example": example,
     "bagfloor": bagfloor, "nostatus": nostatus}[sys.argv[1]]()
