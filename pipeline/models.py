"""Model factory, hyperparameter grids, tuning, and the two explainers."""
import itertools
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import config as C
from data import aggregate, group_index, make_prep

warnings.filterwarnings("ignore")

GRIDS = {
    "LR": {"C": [0.01, 0.1, 1.0, 10.0]},
    "DT": {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 20, 50]},
    "RF": {"min_samples_leaf": [3, 5, 10], "max_features": ["sqrt", 0.33]},
    "XGB": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]},
}
if C.DATASET == "cell2cell":
    # Cell2Cell trains on 40,837 customers instead of 5,634 (x7.25): minimum leaf sizes are scaled by the same
    # factor so that leaves hold a comparable share of the data (and RF TreeSHAP stays feasible; replication plan).
    GRIDS["DT"] = {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [72, 145, 362]}
    GRIDS["RF"] = {"min_samples_leaf": [22, 36, 72], "max_features": ["sqrt", 0.33]}


def make_model(name, params, seed):
    if name == "LR":
        return LogisticRegression(max_iter=5000, **params)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1, **params)
    if name == "XGB":
        return XGBClassifier_(seed, params)
    raise ValueError(name)


def XGBClassifier_(seed, params):
    return xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1,
                             eval_metric="logloss", **params)


def _cv_one(name, params, X, y, cols, seed):
    """Out-of-fold probabilities for one configuration."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed).split(X, y):
        prep = make_prep(cols)
        m = make_model(name, params, seed=0).fit(prep.fit_transform(X.iloc[tr]), y[tr])
        oof[va] = m.predict_proba(prep.transform(X.iloc[va]))[:, 1]
    return params, roc_auc_score(y, oof), oof


def tune(name, X, y, seed, n_jobs):
    """Grid search by stratified CV ROC-AUC on the training set only.
    Returns best params, CV AUC, and an F1-maximising threshold chosen on the out-of-fold predictions."""
    from joblib import Parallel, delayed
    cols = list(X.columns)
    keys = list(GRIDS[name])
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRIDS[name].values())]
    res = Parallel(n_jobs=n_jobs)(delayed(_cv_one)(name, p, X, y, cols, seed) for p in combos)
    best_params, best_auc, oof = max(res, key=lambda r: r[1])
    grid = np.linspace(0.05, 0.95, 91)
    f1 = []
    for t in grid:
        pred = oof >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    thr = float(grid[int(np.argmax(f1))])
    table = pd.DataFrame([{**p, "cv_auc": a} for p, a, _ in res])
    return best_params, float(best_auc), thr, table


def shap_values(name, model, Zfit, Zexp):
    """SHAP values: log-odds units for LR and XGB, probability units for the sklearn trees (ranks, top-k sets
    and signs, which all stability measures use, do not depend on the unit)."""
    if name == "LR":
        # full fitting set as background: a random background adds noise even to a deterministic model
        return shap.LinearExplainer(model, Zfit).shap_values(Zexp)
    if name == "XGB":
        # shap 0.49 cannot parse xgboost>=3.1 base_score; native TreeSHAP gives the same values
        return model.get_booster().predict(xgb.DMatrix(Zexp), pred_contribs=True)[:, :-1]
    # sklearn trees: model_output="raw" is the predicted probability for these classifiers
    sv = shap.TreeExplainer(model, model_output="raw").shap_values(Zexp)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def fit_and_explain(name, params, seed, boot, Xtr, ytr, Xte, Xexp):
    """One refit: returns test probabilities, explained-customer probabilities and SHAP (n_explain x n_features)."""
    cols = list(Xtr.columns)
    if boot:
        idx = np.random.default_rng(10_000 + seed).integers(0, len(Xtr), len(Xtr))
        Xfit, yfit = Xtr.iloc[idx], ytr[idx]
    else:
        Xfit, yfit = Xtr, ytr
    prep = make_prep(cols)
    Zfit = prep.fit_transform(Xfit)
    model = make_model(name, params, seed=seed).fit(Zfit, yfit)
    Zexp = prep.transform(Xexp)
    p_test = model.predict_proba(prep.transform(Xte))[:, 1]
    p_exp = model.predict_proba(Zexp)[:, 1]
    A = aggregate(np.asarray(shap_values(name, model, Zfit, Zexp)), group_index(prep, cols), len(cols))
    return p_test.astype(np.float32), p_exp.astype(np.float32), A.astype(np.float32)


# ---------------------------------------------------------------- LIME
def lime_explain(name, params, Xtr, ytr, Xexp, seeds, n_samples):
    """LIME weights (len(seeds) x n_customers x n_features) for the seed-0 full-data model,
    plus SHAP for the same model, so LIME-LIME and LIME-SHAP agreement use one fitted model."""
    from lime.lime_tabular import LimeTabularExplainer
    cols = list(Xtr.columns)
    num = [c for c in C.NUM if c in cols]
    if C.IMPUTE:
        # LIME's sampler cannot handle missing values. Filling with the training medians first leaves the
        # model unchanged, because the preprocessor would impute exactly these medians anyway.
        med = Xtr[num].median()
        Xtr, Xexp = Xtr.fillna(med), Xexp.fillna(med)
    prep = make_prep(cols)
    Zfit = prep.fit_transform(Xtr)
    model = make_model(name, params, seed=0).fit(Zfit, ytr)
    cat_cols = [c for c in cols if c not in num]
    cats = {c: sorted(Xtr[c].unique()) for c in cat_cols}

    def encode(df):
        e = df.copy()
        for c in cat_cols:
            e[c] = e[c].map({v: i for i, v in enumerate(cats[c])})
        return e.to_numpy(dtype=float)

    def predict(arr):
        df = pd.DataFrame(arr, columns=cols)
        for c in cat_cols:
            df[c] = np.asarray(cats[c], dtype=object)[np.clip(np.rint(df[c].to_numpy()).astype(int), 0, len(cats[c]) - 1)]
        for c in num:
            df[c] = df[c].astype(float)
        return model.predict_proba(prep.transform(df))

    Etr, Eexp = encode(Xtr), encode(Xexp)
    cat_idx = [cols.index(c) for c in cat_cols]
    W = np.zeros((len(seeds), len(Xexp), len(cols)), dtype=np.float32)
    for si, s in enumerate(seeds):
        ex = LimeTabularExplainer(Etr, feature_names=cols, categorical_features=cat_idx,
                                  class_names=["No", "Yes"], discretize_continuous=True, random_state=s)
        for n in range(len(Xexp)):
            e = ex.explain_instance(Eexp[n], predict, num_features=len(cols), labels=(1,), num_samples=n_samples)
            for f, val in e.as_map()[1]:
                W[si, n, f] = val
    S = aggregate(np.asarray(shap_values(name, model, Zfit, prep.transform(Xexp))), group_index(prep, cols), len(cols))
    return W, S.astype(np.float32)
