"""Pilot, part 2: customer-segment analysis (RQ4), boot condition.

Segments (fixed before looking at results): Contract (3 levels) and tenure band (0-12, 13-48, 49-72 months).
  - Performance per segment: 30 bootstrap refits (same seeds as pilot_stability.py), ROC-AUC / PR-AUC on the test set.
  - Attribution per segment: mean |SHAP| ranking within each segment, from the saved shap_<model>_boot.npy arrays.
  - Local stability per segment: per-customer top-3 Jaccard across refits, grouped by segment.

Caveat built in: SHAP is measured against a GLOBAL baseline, so the feature that defines a segment
(e.g. Contract inside "Two year") carries a large attribution by construction. Rankings are therefore
reported both with and without the segmenting feature.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pilot_stability import N_EXPLAIN, OUT, R, SPLIT_SEED, load, make_model, make_prep

MODELS = ["LR", "DT", "RF", "XGB"]


def tenure_band(t):
    return pd.cut(t, [-1, 12, 48, 72], labels=["0-12", "13-48", "49-72"]).astype(str)


def predict_run(name, r, Xtr, ytr, Xte):
    rng = np.random.default_rng(1000 + r)  # same bootstrap as pilot_stability.one_run
    idx = rng.integers(0, len(Xtr), len(Xtr))
    prep = make_prep(list(Xtr.columns))
    model = make_model(name, seed=r).fit(prep.fit_transform(Xtr.iloc[idx]), ytr[idx])
    return model.predict_proba(prep.transform(Xte))[:, 1]


def main():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    _, Xexp = train_test_split(Xte, test_size=N_EXPLAIN, stratify=yte, random_state=SPLIT_SEED)
    Xexp = Xexp.reset_index(drop=True)
    feats = list(Xtr.columns)

    seg_te = {"Contract": Xte["Contract"].to_numpy(), "tenure": tenure_band(Xte["tenure"]).to_numpy()}
    seg_ex = {"Contract": Xexp["Contract"].to_numpy(), "tenure": tenure_band(Xexp["tenure"]).to_numpy()}
    seg_feat = {"Contract": ["Contract"], "tenure": ["tenure", "TotalCharges"]}
    jac_all = np.load(OUT / "local_jac_boot.npy")  # models x customers

    perf, attr, stab = [], [], []
    for mi, m in enumerate(MODELS):
        P = np.stack(Parallel(n_jobs=8)(delayed(predict_run)(m, r, Xtr, ytr, Xte) for r in range(R)))
        A = np.load(OUT / f"shap_{m}_boot.npy")  # runs x customers x features
        for sname in seg_te:
            for level in sorted(set(seg_te[sname])):
                mt = seg_te[sname] == level
                aucs = [roc_auc_score(yte[mt], p[mt]) for p in P]
                prs = [average_precision_score(yte[mt], p[mt]) for p in P]
                perf.append(dict(model=m, segment=sname, level=level, n_test=int(mt.sum()),
                                 churn_rate=float(yte[mt].mean()), auc_mean=np.mean(aucs), auc_sd=np.std(aucs),
                                 prauc_mean=np.mean(prs), prauc_sd=np.std(prs)))
                me = seg_ex[sname] == level
                g = np.abs(A[:, me, :]).mean((0, 1))
                order = [feats[i] for i in np.argsort(-g)]
                order_wo = [f for f in order if f not in seg_feat[sname]]
                attr.append(dict(model=m, segment=sname, level=level, n_explain=int(me.sum()),
                                 top5_all=", ".join(order[:5]), top5_excl_segment_feature=", ".join(order_wo[:5])))
                stab.append(dict(model=m, segment=sname, level=level, n_explain=int(me.sum()),
                                 local_top3_jaccard=float(jac_all[mi, me].mean()),
                                 frac_below_0p5=float((jac_all[mi, me] < 0.5).mean())))
        print(m, "done", flush=True)

    perf, attr, stab = pd.DataFrame(perf), pd.DataFrame(attr), pd.DataFrame(stab)
    perf.to_csv(OUT / "segment_performance.csv", index=False)
    attr.to_csv(OUT / "segment_attribution.csv", index=False)
    stab.to_csv(OUT / "segment_stability.csv", index=False)

    # Do segments rank features differently? Spearman between segment-level importance vectors,
    # excluding the segmenting feature(s).
    rows = []
    for m in MODELS:
        A = np.abs(np.load(OUT / f"shap_{m}_boot.npy")).mean(0)
        for sname, levels in (("Contract", ["Month-to-month", "One year", "Two year"]),
                              ("tenure", ["0-12", "13-48", "49-72"])):
            keep = [i for i, f in enumerate(feats) if f not in seg_feat[sname]]
            vec = {lv: A[seg_ex[sname] == lv][:, keep].mean(0) for lv in levels}
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    rows.append(dict(model=m, segment=sname, pair=f"{levels[i]} vs {levels[j]}",
                                     spearman=float(spearmanr(vec[levels[i]], vec[levels[j]])[0])))
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "segment_rank_agreement.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    for name, df in (("PERFORMANCE", perf), ("STABILITY", stab), ("ATTRIBUTION", attr), ("RANK AGREEMENT", cross)):
        print(f"\n== {name} ==\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
