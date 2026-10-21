"""Matched-metric comparison of global and per-customer stability (review round 6, item 1).

The headline contrast in the manuscript puts a lenient GLOBAL metric (top-5 inclusion) next to a strict LOCAL
metric (top-3 Jaccard). This script applies the SAME metrics to both levels, from the saved raw SHAP values of the
bootstrap condition (no refitting):

  inclusion@k  : share of the consensus top-k features that are still in a refit's top-k
                 (global consensus = top-k of mean |SHAP| over refits and customers;
                  per-customer consensus = top-k of that customer's mean |SHAP| over refits)
  jaccard@k    : mean pairwise Jaccard of the top-k sets over all refit pairs
  exact@k      : share of refit pairs whose top-k sets are identical
  inclusion_ho@k: held-out inclusion (review round 8): the consensus is built from one half of the refits and
                 scored on the other half (and vice versa), so a refit never votes for the consensus it is scored on

decomposition.csv splits the headline contrast H = global inclusion@5 - per-customer jaccard@3 (and the k = 3
analogue) into a metric and a level part along both paths (level first / metric first) and their average, which
does not depend on the order (the two-factor Shapley split). Held-out inclusion is used throughout.

Per-customer values are averaged over the 500 explained customers. Each split gives one number per model; the
K = 10 splits are summarised with Nadeau-Bengio corrected 95% CIs (analyze.ci95). gap = global - per-customer.

Run once per dataset:  CHURN_DATASET=ibm|cell2cell|iranian  python matched_metric.py
Output: <results>/tables/matched_metric_splits.csv and matched_metric_summary.csv
      python matched_metric.py subsets   (global stability from n random customers)
"""
import numpy as np
import pandas as pd

import config as C
from analyze import ci95, topk_mask

KS = (3, 5)


def pair_stats(M, k):
    """M: R x ... x F top-k masks -> (mean pairwise Jaccard, share of identical pairs) along the leading axis."""
    inter = np.einsum("r...f,s...f->rs...", M, M)
    iu = np.triu_indices(M.shape[0], 1)
    inter = inter[iu]                                   # pairs x ...
    return (inter / (2 * k - inter)).mean(0), (inter == k).mean(0)


def measures(A, k):
    absA = np.abs(A)                                    # R x N x F
    # global level: one importance vector per refit
    G = absA.mean(1)                                    # R x F
    Mg = topk_mask(G, k)
    cons_g = np.argsort(-G.mean(0))[:k]
    g_incl = Mg[:, cons_g].mean()
    g_jac, g_exact = pair_stats(Mg, k)
    half = A.shape[0] // 2
    h = [slice(0, half), slice(half, 2 * half)]
    g_ho = np.mean([Mg[h[1 - i]][:, np.argsort(-G[h[i]].mean(0))[:k]].mean() for i in (0, 1)])
    # per-customer level: the same three metrics, one consensus per customer
    Ml = topk_mask(absA, k)                             # R x N x F
    cons_l = np.argsort(-absA.mean(0), axis=1)[:, :k]   # N x k
    n = np.arange(A.shape[1])[:, None]
    l_incl_c = Ml[:, n, cons_l].mean((0, 2))            # N
    l_jac_c, l_exact_c = pair_stats(Ml, k)              # N, N
    l_ho_c = np.mean([Ml[h[1 - i]][:, n, np.argsort(-absA[h[i]].mean(0), axis=1)[:, :k]].mean((0, 2)) for i in (0, 1)], 0)
    return {f"global_inclusion@{k}": g_incl, f"local_inclusion@{k}": l_incl_c.mean(),
            f"global_jaccard@{k}": g_jac, f"local_jaccard@{k}": l_jac_c.mean(),
            f"global_inclusion_ho@{k}": g_ho, f"local_inclusion_ho@{k}": l_ho_c.mean(),
            f"global_exact@{k}": g_exact, f"local_exact@{k}": l_exact_c.mean(),
            # share of customers whose own reasons are less stable than the global summary, same metric
            f"share_below_global_inclusion@{k}": (l_incl_c < g_incl).mean()}


def shares(gi, li, gj, lj):
    """Two-factor split of H = gi - lj (level: global/local; metric: inclusion/jaccard@3), as shares of H."""
    H = gi - lj
    level_first, metric_first = (li - lj) / H, (gi - gj) / H        # metric share along each path
    return H, level_first, metric_first, (level_first + metric_first) / 2, level_first - metric_first


def decomposition(df, B=2000, seed=0):
    """Metric/level split of the headline contrast, with the interaction (difference between the two paths, as a
    share of H) and 95% percentile intervals from resampling the K outer splits (approximate: the splits overlap)."""
    rng = np.random.default_rng(seed)
    out = []
    for m, g in df.groupby("model", sort=False):
        for k in KS:   # lenient global (held-out inclusion@k) against strict per-customer (jaccard@3)
            cols = [f"global_inclusion_ho@{k}", f"local_inclusion_ho@{k}", "global_jaccard@3", "local_jaccard@3"]
            v = g[cols].to_numpy()
            H, lf, mf, sh, inter = shares(*v.mean(0))
            boot = np.array([shares(*v[rng.integers(0, len(v), len(v))].mean(0))[3] for _ in range(B)])
            out.append(dict(dataset=C.DATASET, model=m, k_inclusion=k, contrast=H, metric_share_level_first=lf,
                            metric_share_metric_first=mf, metric_share=sh, interaction=inter,
                            share_ci_lo=np.percentile(boot, 2.5), share_ci_hi=np.percentile(boot, 97.5)))
    return pd.DataFrame(out)


def subsets(ns=(1, 5, 25, 100, 500), draws=20, seed=0):
    """How much of the global level's stability is averaging over customers: global top-3 Jaccard across refits
    when the global ranking is built from n random explained customers instead of all 500 (n = 1 is one customer's
    own explanation). Output: <results>/tables/matched_metric_subsets.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in C.OUTER_SEEDS:
        for m in C.MODELS:
            absA = np.abs(np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64))
            for n in ns:
                picks = [np.arange(absA.shape[1])] if n >= absA.shape[1] else                     [rng.choice(absA.shape[1], n, replace=False) for _ in range(draws)]
                jac = np.mean([pair_stats(topk_mask(absA[:, p].mean(1), 3), 3)[0] for p in picks])
                rows.append(dict(dataset=C.DATASET, split=s, model=m, n_customers=n, top3_jaccard=jac))
    out = pd.DataFrame(rows)
    summ = out.groupby(["model", "n_customers"], sort=False).top3_jaccard.mean().reset_index()
    summ.to_csv(C.TABLES / "matched_metric_subsets.csv", index=False)
    print(summ.pivot(index="model", columns="n_customers", values="top3_jaccard").round(3).to_string())


def main():
    rows = []
    for s in C.OUTER_SEEDS:
        for m in C.MODELS:
            A = np.load(C.RAW / f"s{s}_{m}_boot.npz")["A"].astype(np.float64)
            r = dict(dataset=C.DATASET, split=s, model=m, n_features=A.shape[2])
            for k in KS:
                r.update(measures(A, k))
            rows.append(r)
            print(s, m, {key: round(float(v), 3) for key, v in r.items() if "@5" in key})
    df = pd.DataFrame(rows)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / "matched_metric_splits.csv", index=False)

    out = []
    for m, g in df.groupby("model", sort=False):
        for k in KS:
            for metric in ("inclusion", "inclusion_ho", "jaccard", "exact"):
                gl, lo = g[f"global_{metric}@{k}"].values, g[f"local_{metric}@{k}"].values
                gap = gl - lo
                lo_ci, gap_ci = ci95(lo), ci95(gap)
                out.append(dict(dataset=C.DATASET, model=m, k=k, metric=metric, K=len(g),
                                global_mean=gl.mean(), local_mean=lo.mean(),
                                local_ci_lo=lo_ci[0], local_ci_hi=lo_ci[1],
                                gap_mean=gap.mean(), gap_ci_lo=gap_ci[0], gap_ci_hi=gap_ci[1]))
    summ = pd.DataFrame(out)
    share = df.groupby("model", sort=False)[[f"share_below_global_inclusion@{k}" for k in KS]].mean()
    summ.to_csv(C.TABLES / "matched_metric_summary.csv", index=False)
    dec = decomposition(df)
    dec.to_csv(C.TABLES / "matched_metric_decomposition.csv", index=False)
    print(dec.round(3).to_string(index=False))
    share.to_csv(C.TABLES / "matched_metric_share_below.csv")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(share.round(3))


if __name__ == "__main__":
    import sys
    subsets() if sys.argv[1:] == ["subsets"] else main()
