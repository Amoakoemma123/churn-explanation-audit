# Stable Rankings, Unstable Reasons

Code and result tables for *Stable Rankings, Unstable Reasons: A Two-Dataset Audit of Per-Customer SHAP and LIME
Explanations for Telecom Churn Prediction*.

The audit asks whether the SHAP and LIME reasons given for an individual customer's churn risk survive retraining on
resampled data, a change of model class (logistic regression, decision tree, random forest, XGBoost) and a change of
explainer. It runs on IBM Telco (primary analysis) and on Cell2Cell (a replication whose plan, in
`replication/REPLICATION_PLAN.md`, was frozen with a SHA-256 checksum before any Cell2Cell result was computed;
see `replication/plan_frozen.txt`).

## Reproduce

```
pip install -r requirements.txt
python get_data.py                      # downloads both datasets into data/ and checks their checksums
cd pipeline
python run_experiments.py               # IBM Telco: tuning, refits, SHAP, LIME (~1 h on 8 cores; resumable)
python check_lr_signs.py
python analyze.py                       # tables, hypothesis tests and figures -> results/
```

For the Cell2Cell replication, set `CHURN_DATASET=cell2cell` and run the same three commands (~13 h; output in
`results_cell2cell/`). On Windows cmd, write `set "CHURN_DATASET=cell2cell"` with the quotes. Then
`python ../manuscript/make_replication_figure.py` draws the two-dataset comparison figure.
`python run_experiments.py --quick && python analyze.py --quick` is a short smoke test.

Design, settings and output files are described in `pipeline/README.md`; every constant is in `pipeline/config.py`.

## What is included

- `pipeline/results/tables/` and `pipeline/results_cell2cell/tables/`: every result table behind the paper's numbers.
  Raw per-refit SHAP arrays (about 600 MB) are not included; `run_experiments.py` regenerates them.
- `pilot/`: the preliminary study that motivated two of the hypotheses (its outer split, seed 42, is excluded from the
  main analysis).
- No data. IBM Telco is Apache-2.0 and Cell2Cell's licence is not stated, so both are downloaded by `get_data.py`;
  for the same reason the per-customer Cell2Cell table is left out.

## Licence

MIT (see `LICENSE`). The datasets keep their own terms.
