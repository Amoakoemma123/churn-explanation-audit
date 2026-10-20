"""Stage 2: turn raw arrays into tables, hypothesis tests and figures.

    python analyze.py           # reads results/raw, writes results/tables + results/figures
    python analyze.py --quick   # same for the smoke-test run

Unit of analysis for every test is the outer split (K repeated random 80/20 splits); customer-level
quantities are averaged within a split first. Because the splits share customers they are not independent,
so every test and CI uses the corrected resampled t-test of Nadeau & Bengio (2003).
"""
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

if "--quick" in sys.argv:
    C.quick_mode()

from data import load  # noqa: E402

MODELS = C.MODELS
EQUIV = ["LR", "RF", "XGB"]       # models expected inside the H1 margin (DT expected outside)


# ---------------------------------------------------------------- loading
def load_raw():
    X, y, ids = load()
    feats = list(X.columns)
    splits, runs, lime, tuning = {}, {}, {}, {}
    for s in C.OUTER_SEEDS:
        splits[s] = dict(np.load(C.RAW / f"s{s}_split.npz"))
        for m in MODELS:
            tuning[(s, m)] = json.loads((C.RAW / f"s{s}_{m}_tuning.json").read_text())
            for cond in C.CONDITIONS:
                z = np.load(C.RAW / f"s{s}_{m}_{cond}.npz")
                runs[(s, m, cond)] = dict(P_test=z["P_test"], P_exp=z["P_exp"], A=z["A"].astype(np.float64),
                                          features=list(z["features"]))
            z = np.load(C.RAW / f"s{s}_{m}_lime.npz")
            lime[(s, m)] = dict(W=z["W"], S=z["S"])
    return X, y, ids, feats, splits, runs, lime, tuning


# ---------------------------------------------------------------- metrics
def topk_mask(A, k):
    """A: (..., F) attributions -> boolean mask of the k largest |A| along the last axis."""
    idx = np.argsort(-np.abs(A), axis=-1)[..., :k]
    M = np.zeros(A.shape, dtype=np.float32)
    np.put_along_axis(M, idx, 1.0, axis=-1)
    return M


def pairwise_jaccard_within(A, k):
    """A: R x N x F. Mean Jaccard of top-k sets over all refit pairs, per customer (N,)."""
    M = topk_mask(A, k)
    inter = np.einsum("rnf,snf->rsn", M, M)
    J = inter / (2 * k - inter)
    iu = np.triu_indices(A.shape[0], 1)
    return J[iu].mean(0)


def pairwise_jaccard_between(A, B, k):
    """Mean Jaccard between every refit of model A and every refit of model B, per customer."""
    Ma, Mb = topk_mask(A, k), topk_mask(B, k)
    inter = np.einsum("rnf,snf->rsn", Ma, Mb)
    return (inter / (2 * k - inter)).mean((0, 1))


def global_measures(A, k=5):
    G = np.abs(A).mean(1)                                  # R x F
    R = G.shape[0]
    rk = stats.rankdata(-G, axis=1)
    pairs = list(combinations(range(R), 2))
    rho = np.mean([stats.spearmanr(G[i], G[j])[0] for i, j in pairs])
    tops = [frozenset(np.argsort(-g)[:k]) for g in G]
    jac = np.mean([len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i, j in pairs])
    exact = np.mean([tops[i] == tops[j] for i, j in pairs])
    cons = np.argsort(-G.mean(0))[:k]
    incl = np.mean([[c in t for t in tops] for c in cons])
    top3 = [frozenset(np.argsort(-g)[:3]) for g in G]
    jac3 = np.mean([len(top3[i] & top3[j]) / len(top3[i] | top3[j]) for i, j in pairs])
    return dict(global_spearman=rho, global_top5_jaccard=jac, global_top5_exact=exact,
                global_top5_inclusion=incl, global_top3_jaccard=jac3, median_rank=np.median(rk, 0))


def sign_vs_membership(A, k=3):
    """Per customer: for the consensus top-k features, P(sign = consensus sign) and P(still in the refit's top-k)."""
    cons = A.mean(0)                                        # N x F
    ctop = np.argsort(-np.abs(cons), axis=1)[:, :k]          # N x k
    M = topk_mask(A, k)                                     # R x N x F
    n = np.arange(A.shape[1])[:, None]
    sign_ok = (np.sign(A[:, n, ctop]) == np.sign(cons[n, ctop])[None]).mean((0, 2))
    member = M[:, n, ctop].mean((0, 2))
    return sign_ok, member


def chance_jaccard(F, k, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((n, F)).argsort(1)[:, :k]
    b = rng.random((n, F)).argsort(1)[:, :k]
    inter = np.array([len(set(x) & set(y)) for x, y in zip(a[:5000], b[:5000])])
    return float(np.mean(inter / (2 * k - inter)))


def corrected_se(x):
    """Standard error of a mean over K repeated random train/test splits, with the Nadeau & Bengio (2003)
    correction: the splits share customers, so Var is inflated by (1/K + n_test/n_train) instead of 1/K."""
    x = np.asarray(x, float)
    rho = C.TEST_SIZE / (1 - C.TEST_SIZE)
    return np.sqrt((1 / len(x) + rho) * x.var(ddof=1))


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * corrected_se(x)
    return (x.mean() - h, x.mean() + h)


def one_sample(d, alternative):
    """Corrected resampled t-test of mean(d) against 0 (n = K outer splits)."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, np.nan
    t = d.mean() / corrected_se(d)
    df = len(d) - 1
    p = {"less": stats.t.cdf(t, df), "greater": stats.t.sf(t, df), "two-sided": 2 * stats.t.sf(abs(t), df)}[alternative]
    return float(t), float(p)


def paired(a, b, alternative):
    """Corrected resampled paired t-test across outer splits. Returns t, p, mean diff."""
    d = np.asarray(a, float) - np.asarray(b, float)
    t, p = one_sample(d, alternative)
    return t, p, float(d.mean())


def tost(d, margin):
    """Two one-sided tests (Schuirmann, 1987) for equivalence within +-margin, corrected SE."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, (np.nan, np.nan)
    se = corrected_se(d)
    df = len(d) - 1
    p_low = 1 - stats.t.cdf((d.mean() + margin) / se, df)
    p_up = stats.t.cdf((d.mean() - margin) / se, df)
    h = stats.t.ppf(0.95, df) * se
    return float(max(p_low, p_up)), (float(d.mean() - h), float(d.mean() + h))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def auc_metrics(y, p, thr):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    pred = p >= thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12))


def segment_labels(X, idx):
    """Pre-specified customer groups for the rows idx: tenure band, plus contract type where the data have one."""
    out = dict(tenure_band=pd.cut(X[C.TENURE].to_numpy()[idx], C.TENURE_BINS, labels=C.TENURE_LABELS).astype(str))
    if "Contract" in X.columns:
        out = dict(contract=X["Contract"].to_numpy()[idx], **out)
    return out


def lr_flips(A, k=3):
    """For each customer's k consensus features: (feature index, share of refits whose sign differs from consensus)."""
    cons = A.mean(0)
    top = np.argsort(-np.abs(cons), axis=1)[:, :k]
    n = np.arange(A.shape[1])[:, None]
    rate = (np.sign(A[:, n, top]) != np.sign(cons[n, top])[None]).mean(0)
    return pd.DataFrame(dict(feature=top.ravel(), flip_rate=rate.ravel()))


# ---------------------------------------------------------------- analysis
def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)
    C.FIGS.mkdir(parents=True, exist_ok=True)
    X, y, ids, feats, splits, runs, lime, tuning = load_raw()
    F = len(feats)
    S = C.OUTER_SEEDS
    tests = []   # (hypothesis, description, p) for Holm

    # ---------------- RQ1 / H1: performance
    perf = []
    for s in S:
        yte = y[splits[s]["test"]]
        for m in MODELS:
            thr = tuning[(s, m)]["threshold"]
            row = dict(split=s, model=m, **auc_metrics(yte, runs[(s, m, "seed")]["P_test"][0], thr),
                       cv_auc=tuning[(s, m)]["cv_auc"], threshold=thr, params=json.dumps(tuning[(s, m)]["params"]))
            row["boot_auc_mean"] = np.mean([auc_metrics(yte, p, thr)["roc_auc"] for p in runs[(s, m, "boot")]["P_test"]])
            row["noTC_auc_mean"] = np.mean([auc_metrics(yte, p, thr)["roc_auc"] for p in runs[(s, m, C.ABLATION)]["P_test"]])
            perf.append(row)
    perf = pd.DataFrame(perf)
    perf.to_csv(C.TABLES / "performance_by_split.csv", index=False)
    summ = []
    for m in MODELS:
        d = perf[perf.model == m]
        r = dict(model=m)
        for c in ("roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "boot_auc_mean"):
            lo, hi = ci95(d[c]); r[c] = d[c].mean(); r[c + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        summ.append(r)
    pd.DataFrame(summ).to_csv(C.TABLES / "performance_summary.csv", index=False)

    h1 = []
    for a, b in combinations(MODELS, 2):
        d = perf[perf.model == a].roc_auc.to_numpy() - perf[perf.model == b].roc_auc.to_numpy()
        p, ci = tost(d, C.AUC_MARGIN)
        h1.append(dict(pair=f"{a}-{b}", mean_diff=d.mean(), ci90_low=ci[0], ci90_high=ci[1], tost_p=p,
                       equivalent=bool(p < 0.05)))
        if a in EQUIV and b in EQUIV:
            tests.append(("H1", f"AUC equivalence {a}-{b} (TOST, +-{C.AUC_MARGIN})", p))
    pd.DataFrame(h1).to_csv(C.TABLES / "H1_equivalence.csv", index=False)

    # ---------------- RQ2 / RQ6: stability per split, model, condition
    stab, local_rows = [], []
    chance = {k: chance_jaccard(F, k) for k in C.TOP_K}
    for s in S:
        ex = splits[s]["explain"]
        for m in MODELS:
            thr = tuning[(s, m)]["threshold"]
            for cond in C.CONDITIONS:
                A = runs[(s, m, cond)]["A"]
                g = global_measures(A)
                jl = {k: pairwise_jaccard_within(A, k) for k in C.TOP_K}
                sg, mem = sign_vs_membership(A)
                row = dict(split=s, model=m, condition=cond,
                           **{k_: v for k_, v in g.items() if k_ != "median_rank"},
                           **{f"local_top{k}_jaccard": jl[k].mean() for k in C.TOP_K},
                           frac_customers_top3_below_0p5=(jl[3] < 0.5).mean(),
                           sign_agreement_top3=sg.mean(), membership_top3=mem.mean())
                stab.append(row)
                if cond == "boot":
                    p = runs[(s, m, cond)]["P_exp"].mean(0).astype(float)
                    logit = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
                    local_rows.append(pd.DataFrame(dict(
                        split=s, model=m, customer=ids[ex], jac3=jl[3], sign=sg, member=mem, p_mean=p,
                        dist=np.abs(logit(p) - logit(np.full_like(p, thr))),
                        **segment_labels(X, ex), churn=y[ex])))
    stab = pd.DataFrame(stab)
    stab.to_csv(C.TABLES / "stability_by_split.csv", index=False)
    local = pd.concat(local_rows, ignore_index=True)
    local.to_csv(C.TABLES / "customer_level_stability.csv", index=False)

    agg_cols = [c for c in stab.columns if c not in ("split", "model", "condition")]
    stab_sum = stab.groupby(["condition", "model"])[agg_cols].agg(["mean", "std"])
    stab_sum.columns = [f"{a}_{b}" for a, b in stab_sum.columns]
    stab_sum.reset_index().to_csv(C.TABLES / "stability_summary.csv", index=False)
    pd.DataFrame([dict(k=k, chance_jaccard=v) for k, v in chance.items()]).to_csv(C.TABLES / "chance_baselines.csv", index=False)

    def col(m, cond, c):
        return stab[(stab.model == m) & (stab.condition == cond)].sort_values("split")[c].to_numpy()

    # ---------------- H2: refit instability beyond the seed floor; local below global
    h2 = []
    for m in MODELS:
        t1, p1, d1 = paired(col(m, "boot", "local_top3_jaccard"), col(m, "seed", "local_top3_jaccard"), "less")
        t2, p2, d2 = paired(col(m, "boot", "local_top3_jaccard"), col(m, "boot", "global_top3_jaccard"), "less")
        h2.append(dict(model=m, boot_minus_seed=d1, p_boot_lt_seed=p1, local_minus_global=d2, p_local_lt_global=p2))
        tests.append(("H2a", f"{m}: local top-3 Jaccard boot < seed floor", p1))
        tests.append(("H2b", f"{m}: local top-3 Jaccard < global top-3 Jaccard", p2))
    pd.DataFrame(h2).to_csv(C.TABLES / "H2_refit_stability.csv", index=False)

    # ---------------- H3: signs more stable than ranks
    h3 = []
    for m in MODELS:
        t, p, d = paired(col(m, "boot", "sign_agreement_top3"), col(m, "boot", "membership_top3"), "greater")
        h3.append(dict(model=m, sign_agreement=col(m, "boot", "sign_agreement_top3").mean(),
                       top3_membership=col(m, "boot", "membership_top3").mean(), diff=d, p=p))
        tests.append(("H3", f"{m}: sign agreement > top-3 membership", p))
    pd.DataFrame(h3).to_csv(C.TABLES / "H3_sign_vs_rank.csv", index=False)

    # ---------------- H4: cross-model vs within-model agreement (same bootstrap samples per refit index)
    h4, cross_global = [], []
    for a, b in combinations(MODELS, 2):
        within, between = [], []
        for s in S:
            Aa, Ab = runs[(s, a, "boot")]["A"], runs[(s, b, "boot")]["A"]
            within.append(0.5 * (pairwise_jaccard_within(Aa, 3).mean() + pairwise_jaccard_within(Ab, 3).mean()))
            between.append(pairwise_jaccard_between(Aa, Ab, 3).mean())
            ga, gb = np.abs(Aa).mean((0, 1)), np.abs(Ab).mean((0, 1))
            cross_global.append(dict(split=s, pair=f"{a}-{b}", global_spearman=stats.spearmanr(ga, gb)[0],
                                     top1_a=feats[int(np.argmax(ga))], top1_b=feats[int(np.argmax(gb))]))
        t, p, d = paired(between, within, "less")
        h4.append(dict(pair=f"{a}-{b}", within=np.mean(within), between=np.mean(between), diff=d, p=p))
        if a in EQUIV and b in EQUIV:
            tests.append(("H4", f"{a}-{b}: cross-model < within-model top-3 Jaccard", p))
    pd.DataFrame(h4).to_csv(C.TABLES / "H4_cross_model.csv", index=False)
    pd.DataFrame(cross_global).to_csv(C.TABLES / "cross_model_global.csv", index=False)

    # ---------------- RQ4: LIME self-agreement vs LIME-SHAP agreement
    rq4 = []
    for s in S:
        for m in MODELS:
            W, Sh = lime[(s, m)]["W"].astype(float), lime[(s, m)]["S"].astype(float)
            self_j = pairwise_jaccard_within(W, 5).mean()
            Ms = topk_mask(Sh, 5)[None]
            Mw = topk_mask(W, 5)
            inter = (Mw * Ms).sum(-1)
            cross_j = (inter / (10 - inter)).mean()
            rq4.append(dict(split=s, model=m, lime_self_top5=self_j, lime_shap_top5=cross_j))
    rq4 = pd.DataFrame(rq4)
    rq4.to_csv(C.TABLES / "RQ4_lime_by_split.csv", index=False)
    rq4.groupby("model")[["lime_self_top5", "lime_shap_top5"]].agg(["mean", "std"]).to_csv(C.TABLES / "RQ4_lime_summary.csv")

    # ---------------- H5: where explanations fail (risk distance and segments)
    h5 = []
    for m in MODELS:
        rhos = []
        for s in S:
            d = local[(local.model == m) & (local.split == s)]
            rhos.append(stats.spearmanr(d.dist, d.jac3)[0])
        t, p = one_sample(rhos, "greater")
        h5.append(dict(model=m, mean_spearman_dist_vs_stability=np.mean(rhos), p=float(p)))
        tests.append(("H5a", f"{m}: stability rises with distance from threshold", float(p)))
    pd.DataFrame(h5).to_csv(C.TABLES / "H5_risk_distance.csv", index=False)

    seg_rows = []
    for (m, seg_name, seg_col) in [(m, n, c) for m in MODELS for n, c in C.SEGMENTS]:
        for s in S:
            d = local[(local.model == m) & (local.split == s)]
            yte = y[splits[s]["test"]]
            seg_te = segment_labels(X, splits[s]["test"])[seg_col]
            for lv, g in d.groupby(seg_col):
                seg_rows.append(dict(model=m, segment=seg_name, level=lv, split=s, n_explain=len(g),
                                     test_churners=int(yte[seg_te == lv].sum()),
                                     jac3=g.jac3.mean(), frac_below_0p5=(g.jac3 < 0.5).mean()))
    seg = pd.DataFrame(seg_rows)
    seg_sum = seg.groupby(["model", "segment", "level"]).agg(
        n_explain=("n_explain", "mean"), test_churners=("test_churners", "mean"),
        jac3=("jac3", "mean"), jac3_sd=("jac3", "std"), frac_below_0p5=("frac_below_0p5", "mean")).reset_index()
    seg_sum["reportable"] = seg_sum.test_churners >= C.MIN_CHURNERS
    seg_sum.to_csv(C.TABLES / "H5_segments.csv", index=False)
    # pre-specified contrasts: One year vs Month-to-month, 13-48 vs 0-12 (two-sided not assumed: H5 predicts lower stability)
    for m in MODELS:
        for seg_name, a, b in C.CONTRASTS:
            va = seg[(seg.model == m) & (seg.segment == seg_name) & (seg.level == a)].sort_values("split").jac3
            vb = seg[(seg.model == m) & (seg.segment == seg_name) & (seg.level == b)].sort_values("split").jac3
            t, p, dd = paired(va, vb, "less")
            tests.append(("H5b", f"{m}: {a} less stable than {b}", p))

    # ---------------- H6: collinearity ablation
    h6 = []
    for m in MODELS:
        t, p, d = paired(col(m, C.ABLATION, "local_top3_jaccard"), col(m, "boot", "local_top3_jaccard"), "greater")
        auc_d = (perf[perf.model == m].noTC_auc_mean - perf[perf.model == m].boot_auc_mean).to_numpy()
        h6.append(dict(model=m, jac_noTC=col(m, C.ABLATION, "local_top3_jaccard").mean(),
                       jac_full=col(m, "boot", "local_top3_jaccard").mean(), diff=d, p=p,
                       auc_change=auc_d.mean(), auc_change_ci=str(tuple(round(v, 4) for v in ci95(auc_d)))))
        if m in ("LR", "DT"):
            tests.append(("H6", f"{m}: dropping {C.DROP[0]} raises local stability", p))
    pd.DataFrame(h6).to_csv(C.TABLES / "H6_collinearity.csv", index=False)

    # ---------------- H7 (Cell2Cell and Iranian Churn, confirmatory): LR sign flips concentrate in collinear features
    if hasattr(C, "COLLINEAR"):
        h7 = []
        coll = [feats.index(f) for f in C.COLLINEAR]
        for s in S:
            fl = lr_flips(runs[(s, "LR", "boot")]["A"])
            hit = np.isin(fl.feature, coll)
            total = fl.flip_rate.sum()
            h7.append(dict(split=s, flip_share=fl.flip_rate[hit].sum() / total if total > 0 else np.nan,
                           instance_share=hit.mean(), total_flip_rate=total))
        h7 = pd.DataFrame(h7)
        h7["diff"] = h7.flip_share - h7.instance_share
        h7.to_csv(C.TABLES / "H7_collinear_flips.csv", index=False)
        d7 = h7["diff"].dropna()
        p7 = one_sample(d7, "greater")[1] if len(d7) >= 3 else np.nan
        tests.append(("H7", "LR: sign flips concentrate in VIF>10 features", p7))

    # ---------------- Global ranks per model (for the heatmap)
    ranks = {}
    for m in MODELS:
        G = np.mean([np.abs(runs[(s, m, "boot")]["A"]).mean((0, 1)) for s in S], axis=0)
        ranks[m] = stats.rankdata(-G).astype(int)
    pd.DataFrame(ranks, index=feats).to_csv(C.TABLES / "global_ranks.csv")

    # ---------------- Holm
    t = pd.DataFrame(tests, columns=["hypothesis", "test", "p"])
    t["p_holm"] = holm(t.p.fillna(1.0))
    t["reject_0.05"] = t.p_holm < 0.05
    t.to_csv(C.TABLES / "hypothesis_tests_holm.csv", index=False)

    import figures
    figures.make_all(C.TABLES, C.FIGS, feats)
    print(pd.read_csv(C.TABLES / "performance_summary.csv").round(3).to_string(index=False))
    print(stab_sum.reset_index()[["condition", "model", "local_top3_jaccard_mean", "global_top5_inclusion_mean",
                                   "global_top5_exact_mean", "sign_agreement_top3_mean"]].round(3).to_string(index=False))
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
