# Replication Plan: Cell2Cell

Written 18 October 2026, before any Cell2Cell result was computed. The plan's SHA-256 checksum is recorded in `replication/plan_frozen.txt`; any later change to this file is listed in the "Amendments" section at the end, with its reason.

## 1. Purpose

This replication re-runs the IBM Telco explanation-stability audit on a dataset of **real** telecom customers. It tests whether the main findings hold outside a fictional sample dataset.

The IBM exploratory finding was that a collinear price feature makes Logistic Regression explanation signs arbitrary. Here it is turned into a **confirmatory** hypothesis (H7).

## 2. Data

- **File:** `cell2celltrain.csv`, from Kaggle `jpacse/datasets-for-churn-telecom` (version 2, downloaded 18 October 2026).
- **Origin:** the Teradata Center for Customer Relationship Management at Duke University.
- **Licence:** listed as "Unknown" on Kaggle. Only the derived results will be published; the data will not be redistributed.
- **Checksum:** SHA-256 `3f6b8ff0b8e8f5975aa0de9f1f54967e6c8cb867a9bec6cbb64f4c8990f9a645`.
- **Size:** 51,047 customers with labels; 28.8% churn. The class balance is that of the distributed file, which is not a population churn rate.
- **Holdout file not used:** `cell2cellholdout.csv` (20,000 customers) has no churn labels.

**Preprocessing**
- `CustomerID` and `ServiceArea` (747 levels, an identifier-like code) are removed. This leaves 55 predictors: 34 numeric and 21 categorical.
- Missing numeric values (156–909 per column, in 10 columns) are imputed with the median of the training data, inside the preprocessor.
- Numeric features are standardised. Categorical features are one-hot encoded (104 model columns).
- All preprocessing is fitted on the training data only.
- No feature selection and no resampling are used.

## 3. Design (identical to IBM except where stated)

**Unchanged from IBM:**
- K = 10 stratified 80/20 outer splits, seeds 101–110 (40,837 training and 10,210 test customers).
- 500 explained test customers per split.
- B = 50 bootstrap refits and 20 seed-only refits per model and split.
- The same bootstrap sample at each refit index for every model class.
- Models: LR, DT, RF and XGBoost, each tuned once per split by stratified 5-fold CV ROC-AUC.
- The F1 threshold is chosen on out-of-fold predictions.
- Explainers: LinearSHAP with the full training background; exact TreeSHAP; LIME on 50 customers × 10 seeds × 5,000 samples.
- Statistics: corrected resampled t-tests (Nadeau & Bengio, 2003), TOST with a ±0.01 ROC-AUC margin, and Holm correction over all tests in Section 4. These form one family, separate from the IBM family.

**Changed for Cell2Cell, with reasons:**
1. **Leaf-size grids scaled by 7.25,** the ratio of the training-set sizes (40,837 / 5,634):
   - RF `min_samples_leaf` ∈ {22, 36, 72} instead of {3, 5, 10};
   - DT `min_samples_leaf` ∈ {72, 145, 362} instead of {10, 20, 50}.

   This keeps each leaf a comparable share of the training data. It also makes the run feasible: with the IBM grid, a single RF refit with TreeSHAP took 3–6 minutes. All other grids are unchanged.
2. **Collinearity ablation:** drops `DroppedBlockedCalls` instead of `TotalCharges`. `DroppedBlockedCalls` is almost the sum of `DroppedCalls` and `BlockedCalls` (r = 0.998), and Cell2Cell has no `TotalCharges` column.
3. **Customer segments:** tenure bands only, using `MonthsInService` with the IBM cut points (0–12, 13–48, 49–72 months). Cell2Cell has no contract-type variable, so the IBM contract contrast cannot be tested.

## 4. Hypotheses (confirmatory, one Holm family)

| ID | Prediction | Tests |
|---|---|---|
| R-H1 | LR, RF and XGBoost ROC-AUC are equivalent within ±0.01 (TOST, the three pairs) | 3 |
| R-H2a | Per-customer top-3 Jaccard under bootstrap is below the seed-only floor (each model) | 4 |
| R-H2b | Per-customer top-3 Jaccard is below global top-3 Jaccard (each model) | 4 |
| R-H3 | Sign agreement exceeds top-3 membership (each model) | 4 |
| R-H4 | Cross-model agreement is below within-model agreement (LR-RF, LR-XGB, RF-XGB) | 3 |
| R-H5a | Stability rises with the distance of predicted risk from the threshold (each model) | 4 |
| R-H5b | Mid-tenure (13–48 months) customers are less stable than new customers (0–12 months) (each model) | 4 |
| R-H6 | Dropping `DroppedBlockedCalls` raises per-customer top-3 Jaccard for LR and DT | 2 |
| R-H7 | In LR, sign flips concentrate in collinear features (definition below) | 1 |

Total: **29 tests**.

**Definition of R-H7.** The collinear set is fixed now, from the predictors alone and without using the outcome. It contains every numeric feature whose variance inflation factor against all other predictors (median-imputed, categorical one-hot, all 51,047 rows) exceeds 10:

| Feature | VIF |
|---|---|
| `DroppedBlockedCalls` | 264.6 |
| `BlockedCalls` | 133.7 |
| `DroppedCalls` | 91.9 |
| `RetentionCalls` | 14.2 |
| `MonthlyRevenue` | 11.1 |

The test is computed as follows:

1. In each split, for each explained customer, take the three features with the largest consensus |SHAP| in the LR bootstrap refits.
2. For each of those features, compute the flip rate: the share of refits whose sign differs from the consensus sign.
3. Let *f* be the share of the total flip rate that falls on the collinear set.
4. Let *a* be the collinear set's share of the (customer, feature) top-3 instances.
5. R-H7 predicts *f* − *a* > 0. This is tested with a one-sided corrected t-test over the 10 splits.

If no LR sign flips occur in a split, that split is dropped from the R-H7 test. If fewer than 3 splits remain, R-H7 is reported as not testable.

**Reported descriptively (no test):**
- the same measures for RQ4 (LIME) and RQ6 (metric sensitivity);
- the 49–72 month band, which has few customers;
- the LR coefficients of the collinear features across refits;
- whether the IBM pattern that different model classes rank different features first recurs.

## 5. Interpretation rules (fixed in advance)

- A hypothesis **replicates** if all its tests are significant after Holm correction. It **partly replicates** if some are, and does **not replicate** if none are.
- Differences between the IBM and Cell2Cell values (for example lower agreement levels) are reported, but not tested against each other.
- Agreement levels are interpreted against Cell2Cell's own chance baseline. With 55 features, this baseline is lower than with IBM's 19.
- All results are reported, including those that contradict the IBM findings.

## Amendments

(none)
