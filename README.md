# Unstable Reasons, Stable Directions

Code and result tables for *Unstable Reasons, Stable Directions: A Three-Dataset Audit of SHAP and LIME Explanations
for Telecom Churn Prediction*.

The audit asks whether the SHAP and LIME reasons given for an individual customer's churn risk survive retraining on
resampled data, a change of model class (logistic regression, decision tree, random forest, XGBoost) and a change of
explainer. It runs on IBM Telco (primary analysis), on Cell2Cell (a replication whose plan, in
`replication/REPLICATION_PLAN.md`, was frozen with a SHA-256 checksum before any Cell2Cell result was computed;
see `replication/plan_frozen.txt`) and on Iranian Churn (a planned test of the averaging and flagging remedies,
plan in `replication_iranian/`).

## Reproduce

```
pip install -r requirements.txt
python get_data.py                      # downloads both datasets into data/ and checks their checksums
cd pipeline
python run_experiments.py               # IBM Telco: tuning, refits, SHAP, LIME (~1 h on 8 cores; resumable)
python check_lr_signs.py
python analyze.py                       # tables, hypothesis tests and figures -> results/
python stability_aware.py               # exploratory: averaging over refits and flagging unstable customers
python robustness.py ties               # robustness checks (also: sub80, common, k5, crosssplit, ...)
python matched_metric.py                # the same stability metrics at the global and per-customer level
python equal_compute.py                 # averaging vs one model of equal computing cost (IBM, Iranian)
python profit.py                        # simulated retention campaign: profit and stability of actions
```

For the Cell2Cell replication, set `CHURN_DATASET=cell2cell` and run the same commands (~13 h; output in
`results_cell2cell/`); for the confirmatory third study, `CHURN_DATASET=iranian` (~30 min; plan in
`replication_iranian/`). On Windows cmd, write `set "CHURN_DATASET=cell2cell"` with the quotes. Then
`python ../manuscript/make_replication_figure.py` and `python stability_aware.py --figure` draw the two-dataset figures.
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
