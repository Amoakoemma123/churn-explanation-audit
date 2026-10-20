"""Data loading, outer splits and the leakage-safe preprocessor."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def load():
    raw = C.DATA.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == C.DATA_SHA256, "dataset checksum mismatch"
    df = pd.read_csv(C.DATA)
    if C.DATASET == "cell2cell":
        return _load_cell2cell(df)
    if C.DATASET == "iranian":
        return _load_iranian(df)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan))
    # The 11 blanks are all tenure==0 customers who have not been billed yet
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    ids = df["customerID"].to_numpy()
    X = df.drop(columns=["customerID", "Churn"])
    X["SeniorCitizen"] = X["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return X, y, ids


def _load_cell2cell(df):
    """Cell2Cell training file (the holdout file has no labels). ServiceArea (747 levels) is dropped;
    missing numeric values are imputed inside the preprocessor, from training data only."""
    assert len(df) == 51047 and df["CustomerID"].is_unique
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    ids = df["CustomerID"].to_numpy()
    X = df.drop(columns=["CustomerID", "Churn", "ServiceArea"])
    assert sorted(X.select_dtypes("number").columns) == sorted(C.NUM)
    for c in X.columns:
        if c not in C.NUM:
            X[c] = X[c].astype(str)
    return X, y, ids


def _load_iranian(df):
    """UCI Iranian Churn (plan Section 2): names without spaces, Age dropped (one value per AgeGroup),
    exact duplicate rows removed so that no record can fall in both parts of a split. All predictors are numeric."""
    df.columns = ["".join(w.capitalize() if w.islower() else w for w in c.split()) for c in df.columns]
    assert len(df) == 3150 and (df.groupby("AgeGroup").Age.nunique() == 1).all()
    df = df.drop(columns="Age").drop_duplicates().reset_index(drop=True)
    assert len(df) == 2850
    y = df["Churn"].astype(int).to_numpy()
    X = df.drop(columns="Churn").astype(float)
    assert list(X.columns) == C.NUM
    return X, y, np.arange(len(df))


def outer_split(X, y, ids, seed):
    """Stratified 80/20 split plus a fixed stratified explanation set drawn from the test part."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    _, ex = train_test_split(te, test_size=C.N_EXPLAIN, stratify=y[te], random_state=seed)
    return dict(train=tr, test=te, explain=ex)


def make_prep(cols):
    num = [c for c in C.NUM if c in cols]
    cat = [c for c in cols if c not in num]
    scale = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if C.IMPUTE else StandardScaler()
    return ColumnTransformer([("num", scale, num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


def group_index(prep, cols):
    """Map every transformed column back to its original feature index."""
    owner = []
    for n in prep.get_feature_names_out():
        raw = n.split("__", 1)[1]
        owner.append(next(c for c in sorted(cols, key=len, reverse=True)
                          if raw == c or raw.startswith(c + "_")))
    return np.array([cols.index(o) for o in owner])


def aggregate(sv, gidx, nfeat):
    """Sum one-hot SHAP columns back to original features (valid by additivity)."""
    out = np.zeros((sv.shape[0], nfeat), dtype=np.float64)
    np.add.at(out.T, gidx, sv.T)
    return out
