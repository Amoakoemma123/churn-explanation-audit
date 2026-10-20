# Confirmatory study on a third dataset: Iranian Churn (plan)

Written on 20 October 2026, before any model was fitted on these data. At this point the IBM Telco results, the
Cell2Cell replication and the exploratory averaging and flagging analysis (manuscript Sections 3.11 and 4.9) were
known. The only facts computed from the new data are those stated in Section 2: file checksum, row counts,
duplicates, churn rate, variance inflation factors of the predictors (without the outcome) and tenure band sizes.
This plan is frozen by recording its SHA-256 checksum and time stamp in `plan_frozen.txt`. It will not be amended;
any deviation will be reported as such.

## 1. Purpose

1. **Confirm the remedy.** The averaging and flagging results were exploratory: they were found on the same refits that
   established the instability. This study tests them prospectively, with every setting fixed below.
2. **Third test of the audit.** Repeat the core audit on real customer records with a clear licence and much stronger
   predictors than Cell2Cell.

## 2. Data

- **Source:** Iranian Churn [Dataset] (2020), UCI Machine Learning Repository, https://doi.org/10.24432/C5JW3Z,
  licence CC BY 4.0. File `data_raw/iranian_churn.csv` from https://archive.ics.uci.edu/static/public/563/data.csv,
  SHA-256 `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- **Content:** 3,150 customers of an Iranian telecom operator; predictors aggregated over the first 9 months, churn
  status at the end of month 12 (a 3-month planning gap). No missing values.
- **Preprocessing (fixed now):**
  - Column names are written without spaces (e.g., `Subscription  Length` → `SubscriptionLength`).
  - `Age` is dropped: it takes exactly one value per `AgeGroup` (15, 25, 30, 45, 55), so it duplicates `AgeGroup`.
  - The 300 rows that are exact duplicates of another row (all columns including `Churn`) are removed, so that no
    customer record can appear in both the training and the test part of a split. This leaves **2,850 customers,
    15.65% churners**. Fourteen groups of rows with identical predictors but different churn labels are kept.
  - The 11 remaining predictors are all numeric (binary and ordinal variables keep their numeric codes) and are
    standardised on the training data only; there are no categorical columns to one-hot encode.

## 3. Design (unchanged from IBM Telco unless stated)

- 10 stratified 80/20 outer splits, seeds 101–110 (2,280 training and 570 test customers); 500 stratified test
  customers explained per split.
- Logistic regression, decision tree, random forest and XGBoost, tuned by stratified 5-fold CV ROC-AUC on the training
  part with the **IBM Telco hyperparameter grids unchanged** (the training set is 0.40 times the IBM size; the IBM
  leaf sizes of 3–50 remain small relative to it, so no rescaling is applied).
- Conditions: `seed` (20 refits, same data), `boot` (50 bootstrap refits), and the collinearity ablation `boot_noCV`
  (50 bootstrap refits without `CustomerValue`).
- SHAP and LIME exactly as for IBM Telco (LinearSHAP with the fitting set as background; TreeSHAP; XGBoost
  `pred_contribs`; LIME 50 customers × 10 seeds × 5,000 samples).
- Stability measures, chance baselines and the Nadeau–Bengio corrected resampled t-test as in the paper.

## 4. Collinearity and segments (fixed from the predictors alone)

- VIF of each predictor against all others: `CustomerValue` 50.0, `FrequencyOfSMS` 39.7, `SecondsOfUse` 21.6,
  `FrequencyOfUse` 18.3; all others ≤ 3.2. `CustomerValue` ("the calculated value of the customer") has the highest
  VIF and is the analogue of `TotalCharges`, so it is the feature dropped in the ablation.
- Collinear set for T-H7 (VIF > 10): `CustomerValue`, `FrequencyOfSMS`, `SecondsOfUse`, `FrequencyOfUse`.
- Tenure bands use the IBM cut points on `SubscriptionLength` (3–47 months): 0–12 months has 132 customers (about 4 test
  churners per split) and 13–48 months 2,718; no customer exceeds 48 months. Because the 0–12 band has far fewer than
  30 test churners, the tenure contrast (H5b) is **not tested**; segment results are descriptive only.

## 5. Confirmatory hypotheses (one Holm family, α = 0.05)

Core audit, restated from the IBM analysis (same tests as in Appendices B and C):

| ID | Prediction | Tests |
|---|---|---|
| T-H1 | LR, RF and XGBoost ROC-AUC equivalent within ±0.01 (TOST) | 3 |
| T-H2a | Bootstrap per-customer top-3 agreement below the seed-only floor | 4 |
| T-H2b | Per-customer top-3 agreement below global top-3 agreement | 4 |
| T-H3 | Signs more stable than top-3 membership | 4 |
| T-H4 | Between-model agreement below within-model agreement (LR–RF, LR–XGB, RF–XGB) | 3 |
| T-H5a | Stability rises with the distance of predicted risk from the threshold | 4 |
| T-H6 | Dropping `CustomerValue` raises LR and DT per-customer stability | 2 |
| T-H7 | LR sign flips concentrate in the VIF > 10 set | 1 |

Remedy (the purpose of this study; settings fixed from the exploratory analysis):

| ID | Prediction | Test per model (4 models each) |
|---|---|---|
| T-R1 | Averaging SHAP over m = 10 bootstrap refits raises per-customer top-3 agreement above single refits (m = 1) | Paired one-sided corrected t-test over the 10 splits |
| T-R2 | The bagged model of all 50 refits is not less accurate than a single refit: ROC-AUC difference > −0.01 | One-sided corrected t-test of (difference + 0.01) > 0 |
| T-R3 | Stability estimated from the first 5 refits identifies customers whose stability among the other 45 refits is below 0.5 with ROC-AUC > 0.70 | One-sided corrected t-test of (ROC-AUC − 0.70) > 0; splits with fewer than 10 such customers among the 500 are excluded, and if fewer than 3 splits remain the test is not evaluable and counts as not significant |

In total 37 tests (25 core, 12 remedy), adjusted together with the Holm procedure. A hypothesis is **supported** if all
its tests are significant, **partly supported** if some are, and **not supported** if none are. Averaging and flagging
use exactly the definitions of manuscript Section 3.11 (disjoint groups of refits in refit order; the first r refits
for the flag).

Descriptive only: the full averaging curve (m = 1, 2, 5, 10, 25), flagging with r = 3 and 10, LIME, tenure segments,
global ranks.

## 6. Code

The existing pipeline is run with `CHURN_DATASET=iranian` (results in `pipeline/results_iranian/`):
`run_experiments.py` → `analyze.py` → `stability_aware.py`. Code changes are limited to loading this dataset, its
settings in `config.py`, the remedy tests T-R1 to T-R3 and the combined Holm family; no metric, threshold or test
defined in the paper is changed. The IBM and Cell2Cell results must remain byte-identical after these changes.
