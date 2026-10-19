# Churn explanation-stability pipeline

This pipeline produces every result in the paper. All settings live in `config.py`.

```
python run_experiments.py      # stage 1: tuning, refits, SHAP, LIME -> results/raw   (~60 min for 10 splits, resumable)
python check_lr_signs.py       # follow-up: why LR signs flip (exploratory, all splits)
python analyze.py              # stage 2: tables, hypothesis tests, figures -> results/tables, results/figures
python stability_aware.py      # exploratory follow-up: averaging SHAP over refits, flagging unstable customers -> results/tables/X*
python run_experiments.py --quick && python analyze.py --quick   # 1-minute smoke test -> results_quick/
```

## Design

| Setting | Value |
|---|---|
| Outer splits | 10 stratified 80/20 splits, seeds 101–110 (pilot seed 42 excluded). The plan was 5; 106–110 were added after the first analysis to settle borderline tests (a documented deviation) |
| Tuning | 5-fold stratified CV ROC-AUC on the training part only; F1 threshold chosen on out-of-fold predictions |
| Refits per model and split | `seed`: 20 refits on the same data, new random seed (the floor); `boot`: 50 bootstrap refits; `boot_noTC`: 50 bootstrap refits without `TotalCharges` |
| Explained customers | 500 stratified test customers per split; SHAP from every refit |
| LIME | 50 of those customers × 10 seeds × 5,000 samples, on the full-data model, with SHAP of the same model |
| Statistics | Unit = outer split (K = 10). Corrected resampled t-tests (Nadeau & Bengio, 2003), because the splits overlap; TOST for equivalence; Holm correction over all 32 tests in `hypothesis_tests_holm.csv` |

Refit *r* uses the same bootstrap sample for every model. Cross-model comparisons therefore isolate the model class.

## SHAP scale

- LR and XGBoost: log-odds.
- RF and DT: probability. Exact TreeSHAP cannot give log-odds for averaged sklearn trees.

All stability measures use top-k sets, ranks and signs, which do not depend on scale.

## Implementation notes

- SHAP 0.49.1 cannot read XGBoost ≥ 3.1 models, so XGBoost uses its native `pred_contribs` (the same TreeSHAP algorithm).
- The LinearSHAP background is the full fitting set. A random background sample makes a deterministic model look unstable (found in the pilot).
- The dataset SHA-256 is checked on load.

## Outputs (`results/tables`)

| File | Content |
|---|---|
| `performance_*` | RQ1: test metrics per split and summary with 95% CIs |
| `H1_equivalence` | TOST for AUC equivalence (±0.01) |
| `stability_*` | RQ2/RQ6: all stability metrics per split/model/condition |
| `H2`–`H6` CSVs | One file per hypothesis |
| `RQ4_lime_*` | LIME self-agreement vs LIME–SHAP agreement |
| `customer_level_stability` | Per-customer stability, predicted risk and segments (for H5) |
| `hypothesis_tests_holm` | Every test with raw and Holm-adjusted p |
| `chance_baselines` | Expected Jaccard for random top-k sets |
