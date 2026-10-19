"""Stage 1: compute and save everything expensive. Resumable: finished units are skipped.

    python run_experiments.py           # full run (proposal Section 9.2)
    python run_experiments.py --quick   # smoke test, writes to results_quick/

Per outer split s, model m, condition c it writes raw/s{s}_{m}_{c}.npz with
  P_test (R x n_test), P_exp (R x n_explain), A (R x n_explain x n_features) SHAP,
and raw/s{s}_{m}_lime.npz with LIME weights (seeds x customers x features) + SHAP of the same model.
"""
import json
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import config as C

if "--quick" in sys.argv:
    C.quick_mode()

from data import load, outer_split  # noqa: E402
from models import fit_and_explain, lime_explain, tune  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    C.RAW.mkdir(parents=True, exist_ok=True)
    X, y, ids = load()
    meta = dict(outer_seeds=C.OUTER_SEEDS, B_boot=C.B_BOOT, B_seed=C.B_SEED, n_explain=C.N_EXPLAIN,
                lime=dict(customers=C.LIME_CUSTOMERS, seeds=C.LIME_SEEDS, samples=C.LIME_SAMPLES),
                features=list(X.columns))
    (C.RAW / "meta.json").write_text(json.dumps(meta, indent=2))
    t_all = time.time()

    for s in C.OUTER_SEEDS:
        sp = outer_split(X, y, ids, s)
        np.savez(C.RAW / f"s{s}_split.npz", **sp)
        Xtr, ytr = X.iloc[sp["train"]].reset_index(drop=True), y[sp["train"]]
        Xte, Xexp = X.iloc[sp["test"]].reset_index(drop=True), X.iloc[sp["explain"]].reset_index(drop=True)

        for m in C.MODELS:
            tfile = C.RAW / f"s{s}_{m}_tuning.json"
            if tfile.exists():
                tuned = json.loads(tfile.read_text())
            else:
                t = time.time()
                params, cv_auc, thr, table = tune(m, Xtr, ytr, seed=s, n_jobs=C.N_JOBS)
                tuned = dict(params=params, cv_auc=cv_auc, threshold=thr)
                tfile.write_text(json.dumps(tuned, indent=2))
                table.to_csv(C.RAW / f"s{s}_{m}_grid.csv", index=False)
                log(f"split {s} {m}: tuned {params} cv_auc={cv_auc:.4f} thr={thr:.2f} ({time.time() - t:.0f}s)")

            for cond, (boot, drop) in C.CONDITIONS.items():
                out = C.RAW / f"s{s}_{m}_{cond}.npz"
                if out.exists():
                    continue
                t = time.time()
                R = C.B_BOOT if boot else C.B_SEED
                cols = [c for c in X.columns if c not in drop]
                runs = Parallel(n_jobs=C.N_JOBS)(
                    delayed(fit_and_explain)(m, tuned["params"], r, boot, Xtr[cols], ytr, Xte[cols], Xexp[cols])
                    for r in range(R))
                np.savez_compressed(out, P_test=np.stack([q[0] for q in runs]),
                                    P_exp=np.stack([q[1] for q in runs]),
                                    A=np.stack([q[2] for q in runs]), features=np.array(cols))
                log(f"split {s} {m} {cond}: {R} refits ({time.time() - t:.0f}s)")

            lfile = C.RAW / f"s{s}_{m}_lime.npz"
            if not lfile.exists():
                t = time.time()
                Xl = Xexp.iloc[:C.LIME_CUSTOMERS]
                chunks = np.array_split(np.arange(len(Xl)), min(C.N_JOBS, len(Xl)))
                res = Parallel(n_jobs=C.N_JOBS)(
                    delayed(lime_explain)(m, tuned["params"], Xtr, ytr, Xl.iloc[ch].reset_index(drop=True),
                                          list(range(C.LIME_SEEDS)), C.LIME_SAMPLES) for ch in chunks)
                np.savez_compressed(lfile, W=np.concatenate([r[0] for r in res], axis=1),
                                    S=np.concatenate([r[1] for r in res], axis=0))
                log(f"split {s} {m} LIME: {len(Xl)} customers x {C.LIME_SEEDS} seeds ({time.time() - t:.0f}s)")

    log(f"all done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
