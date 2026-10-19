"""Pilot: does SHAP explanation instability exist on IBM Telco churn, and is it big enough to study?

Design
  - One fixed stratified test split (20%). Explanations are computed for a fixed set of 500 test customers.
  - R=30 refits per model under three conditions:
      seed      : same training data, only the model's random_state changes
      boot      : bootstrap resample of the training pool + new random_state
      boot_noTC : as boot, but TotalCharges dropped (TotalCharges ~ tenure x MonthlyCharges)
  - SHAP values on one-hot columns are summed back to the 19 original features (valid by additivity).
  - Global stability : pairwise Spearman of mean|SHAP| rankings, top-5 Jaccard, rank range per feature.
  - Local stability  : per customer, top-3 Jaccard across refits and sign agreement with the consensus.
  - Cross-model      : agreement between models (Rashomon check).
  - LIME check       : LIME self-noise (repeat with new seeds) vs LIME-SHAP agreement, XGBoost, 30 customers.

Usage: python pilot_stability.py   (writes to ./results)
"""
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "Telco-Customer-Churn.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

R = 30
N_EXPLAIN = 500
SPLIT_SEED = 42
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]


def load():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan))
    # The 11 blanks are all tenure==0 customers who have not been billed yet
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    X = df.drop(columns=["customerID", "Churn"])
    X["SeniorCitizen"] = X["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return X, y


def make_model(name, seed):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=5000)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                      random_state=seed, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=1, eval_metric="logloss")
    raise ValueError(name)


def make_prep(cols):
    num = [c for c in NUM if c in cols]
    cat = [c for c in cols if c not in num]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


def group_index(prep, cols):
    """Map each transformed column to its original feature."""
    names = prep.get_feature_names_out()
    owner = []
    for n in names:
        raw = n.split("__", 1)[1]
        owner.append(next(c for c in sorted(cols, key=len, reverse=True)
                          if raw == c or raw.startswith(c + "_")))
    return np.array([cols.index(o) for o in owner])


def aggregate(sv, gidx, nfeat):
    out = np.zeros((sv.shape[0], nfeat))
    for j, g in enumerate(gidx):
        out[:, g] += sv[:, j]
    return out


def one_run(name, cond, r, Xtr, ytr, Xte, yte, Xexp):
    cols = list(Xtr.columns)
    rng = np.random.default_rng(1000 + r)
    if cond.startswith("boot"):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        Xfit, yfit = Xtr.iloc[idx], ytr[idx]
    else:
        Xfit, yfit = Xtr, ytr
    prep = make_prep(cols)
    Zfit = prep.fit_transform(Xfit)
    model = make_model(name, seed=r).fit(Zfit, yfit)
    p = model.predict_proba(prep.transform(Xte))[:, 1]
    Zexp = prep.transform(Xexp)
    if name == "LR":
        # Full fitting set as background, so the reference point is not an extra source of noise
        sv = shap.LinearExplainer(model, Zfit).shap_values(Zexp)
    elif name == "XGB":
        # XGBoost's native TreeSHAP (shap 0.49 cannot parse xgboost>=3.1 base_score); last column is the bias
        import xgboost as xgb
        sv = model.get_booster().predict(xgb.DMatrix(Zexp), pred_contribs=True)[:, :-1]
    else:
        sv = shap.TreeExplainer(model).shap_values(Zexp)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
    A = aggregate(np.asarray(sv), group_index(prep, cols), len(cols))
    return dict(auc=roc_auc_score(yte, p), prauc=average_precision_score(yte, p), A=A)


# ---------------------------------------------------------------- metrics
def ranks_desc(v):
    """Rank 1 = most important."""
    return (-v).argsort(axis=-1).argsort(axis=-1) + 1


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def global_stability(G, feats, k=5):
    pairs = list(combinations(range(len(G)), 2))
    rho = [spearmanr(G[i], G[j])[0] for i, j in pairs]
    topk = [np.argsort(-g)[:k] for g in G]
    jac = [jaccard(topk[i], topk[j]) for i, j in pairs]
    rk = ranks_desc(G)
    rank_tbl = pd.DataFrame({"feature": feats, "median_rank": np.median(rk, 0),
                             "min_rank": rk.min(0), "max_rank": rk.max(0),
                             "mean_abs_shap": G.mean(0)}).sort_values("median_rank")
    return dict(spearman_mean=float(np.mean(rho)), spearman_min=float(np.min(rho)),
                top5_jaccard=float(np.mean(jac)),
                top5_identical=float(np.mean([j == 1 for j in jac]))), rank_tbl


def local_stability(A, k=3):
    """A: runs x customers x features."""
    R_, N, _ = A.shape
    top = np.argsort(-np.abs(A), axis=2)[:, :, :k]
    jac = np.zeros(N)
    pairs = list(combinations(range(R_), 2))
    for i, j in pairs:
        jac += np.array([jaccard(top[i, n], top[j, n]) for n in range(N)])
    jac /= len(pairs)
    cons = A.mean(0)
    ctop = np.argsort(-np.abs(cons), axis=1)[:, :k]
    sign_ok = []
    for n in range(N):
        s = np.sign(cons[n, ctop[n]])
        sign_ok.append((np.sign(A[:, n, ctop[n]]) == s).mean())
    return dict(top3_jaccard_mean=float(jac.mean()),
                top3_jaccard_p10=float(np.percentile(jac, 10)),
                frac_customers_top3_jac_below_0p5=float((jac < 0.5).mean()),
                sign_agreement_top3=float(np.mean(sign_ok))), jac


def lime_check(Xtr, ytr, Xexp_df, shap_A, cols, n_cust=30, n_rep=5):
    from lime.lime_tabular import LimeTabularExplainer
    prep = make_prep(cols)
    model = make_model("XGB", 0).fit(prep.fit_transform(Xtr), ytr)
    cat_cols = [c for c in cols if c not in NUM]
    cats = {c: sorted(Xtr[c].unique()) for c in cat_cols}

    def encode(df):
        e = df.copy()
        for c in cat_cols:
            e[c] = e[c].map({v: i for i, v in enumerate(cats[c])})
        return e.to_numpy(dtype=float)

    def predict(arr):
        df = pd.DataFrame(arr, columns=cols)
        for c in cat_cols:
            df[c] = [cats[c][int(round(v))] for v in df[c]]
        for c in NUM:
            df[c] = df[c].astype(float)
        return model.predict_proba(prep.transform(df))

    Etr = encode(Xtr)
    Eexp = encode(Xexp_df.iloc[:n_cust])
    cat_idx = [cols.index(c) for c in cat_cols]
    tops = np.zeros((n_rep, n_cust, 5), dtype=int)
    for rep in range(n_rep):
        ex = LimeTabularExplainer(Etr, feature_names=cols, categorical_features=cat_idx,
                                  class_names=["No", "Yes"], discretize_continuous=True,
                                  random_state=rep)
        for n in range(n_cust):
            e = ex.explain_instance(Eexp[n], predict, num_features=len(cols), labels=(1,))
            w = np.zeros(len(cols))
            for f, val in e.as_map()[1]:
                w[f] = val
            tops[rep, n] = np.argsort(-np.abs(w))[:5]
    self_j = np.mean([jaccard(tops[i, n], tops[j, n])
                      for i, j in combinations(range(n_rep), 2) for n in range(n_cust)])
    shap_top = np.argsort(-np.abs(shap_A[:n_cust]), axis=1)[:, :5]
    cross_j = np.mean([jaccard(tops[r, n], shap_top[n]) for r in range(n_rep) for n in range(n_cust)])
    return dict(lime_self_top5_jaccard=float(self_j), lime_vs_shap_top5_jaccard=float(cross_j),
                n_customers=n_cust, n_repeats=n_rep)


def main():
    t0 = time.time()
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    _, Xexp = train_test_split(Xte, test_size=N_EXPLAIN, stratify=yte, random_state=SPLIT_SEED)
    Xexp = Xexp.reset_index(drop=True)

    tc = X["TotalCharges"]
    prod = X["tenure"] * X["MonthlyCharges"]
    diag = dict(n=len(X), churn_rate=float(y.mean()),
                corr_TotalCharges_vs_tenure_x_Monthly=float(np.corrcoef(tc, prod)[0, 1]),
                corr_TotalCharges_vs_tenure=float(np.corrcoef(tc, X["tenure"])[0, 1]))
    print(diag)

    models = ["LR", "DT", "RF", "XGB"]
    conds = ["seed", "boot", "boot_noTC"]
    summary, all_A, local_jac = [], {}, {}
    for cond in conds:
        drop = ["TotalCharges"] if cond == "boot_noTC" else []
        Xtr_c, Xte_c, Xexp_c = (d.drop(columns=drop) for d in (Xtr, Xte, Xexp))
        feats = list(Xtr_c.columns)
        for m in models:
            t = time.time()
            runs = Parallel(n_jobs=8)(delayed(one_run)(m, cond, r, Xtr_c, ytr, Xte_c, yte, Xexp_c)
                                      for r in range(R))
            A = np.stack([q["A"] for q in runs])
            G = np.abs(A).mean(1)
            gs, rank_tbl = global_stability(G, feats)
            ls, jac = local_stability(A)
            aucs = [q["auc"] for q in runs]
            pr = [q["prauc"] for q in runs]
            row = dict(model=m, condition=cond, auc_mean=np.mean(aucs), auc_sd=np.std(aucs),
                       prauc_mean=np.mean(pr), prauc_sd=np.std(pr), **gs, **ls)
            if "TotalCharges" in feats:
                share = G[:, feats.index("tenure")] / (G[:, feats.index("tenure")] + G[:, feats.index("TotalCharges")])
                row["tenure_share_vs_TC_mean"] = float(share.mean())
                row["tenure_share_vs_TC_sd"] = float(share.std())
            summary.append(row)
            rank_tbl.to_csv(OUT / f"ranks_{m}_{cond}.csv", index=False)
            all_A[(m, cond)] = A
            local_jac[(m, cond)] = jac
            np.save(OUT / f"shap_{m}_{cond}.npy", A)
            print(f"{cond:9s} {m:3s} AUC {row['auc_mean']:.4f}±{row['auc_sd']:.4f} "
                  f"rho {gs['spearman_mean']:.3f} top5J {gs['top5_jaccard']:.3f} "
                  f"loc-top3J {ls['top3_jaccard_mean']:.3f} sign {ls['sign_agreement_top3']:.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)

    # Cross-model agreement (Rashomon check), boot condition, mean explanation over refits
    feats = list(Xtr.columns)
    cross = []
    for a, b in combinations(models, 2):
        Aa, Ab = all_A[(a, "boot")].mean(0), all_A[(b, "boot")].mean(0)
        ga, gb = np.abs(all_A[(a, "boot")]).mean((0, 1)), np.abs(all_A[(b, "boot")]).mean((0, 1))
        ta, tb = np.argsort(-np.abs(Aa), 1)[:, :3], np.argsort(-np.abs(Ab), 1)[:, :3]
        cross.append(dict(pair=f"{a}-{b}", global_spearman=float(spearmanr(ga, gb)[0]),
                          global_top5_jaccard=jaccard(np.argsort(-ga)[:5], np.argsort(-gb)[:5]),
                          local_top3_jaccard=float(np.mean([jaccard(ta[n], tb[n]) for n in range(len(ta))]))))
    pd.DataFrame(cross).to_csv(OUT / "cross_model.csv", index=False)
    print(pd.DataFrame(cross).to_string(index=False))

    # LIME self-noise vs SHAP agreement (XGB, full training pool, seed run 0)
    run0 = one_run("XGB", "seed", 0, Xtr, ytr, Xte, yte, Xexp)
    lime_res = lime_check(Xtr, ytr, Xexp, run0["A"], feats)
    print(lime_res)

    json.dump(dict(diagnostics=diag, lime=lime_res, R=R, n_explain=N_EXPLAIN,
                   runtime_min=(time.time() - t0) / 60),
              open(OUT / "pilot_meta.json", "w"), indent=2)
    np.save(OUT / "local_jac_boot.npy", np.stack([local_jac[(m, "boot")] for m in models]))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
