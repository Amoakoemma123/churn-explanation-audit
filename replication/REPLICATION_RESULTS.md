# Cell2Cell Replication: Results

The run finished on 19 October 2026 at 05:13 (767.6 minutes of compute; 170 raw files). The analysis was run with `analyze.py` and `check_lr_signs.py` (`CHURN_DATASET=cell2cell`), and the output tables are in `pipeline/results_cell2cell/tables/`. The plan (`REPLICATION_PLAN.md`) was not amended, and its interpretation rules are applied unchanged.

## Verdicts (29 tests, one Holm family; 22 significant)

| ID | Verdict | Tests significant | Key numbers |
|---|---|---|---|
| R-H1 | Does not replicate | 0/3 | LR is 0.055 below RF and 0.063 below XGB in ROC-AUC. RF−XGB = −0.009, but the 90% CI [−0.011, −0.007] crosses the −0.01 margin. |
| R-H2a | Replicates | 4/4 | Bootstrap minus seed floor: LR −0.36, DT −0.55, RF −0.19, XGB −0.20. |
| R-H2b | Partly replicates | 3/4 | LR −0.14 has raw p = 0.033 but Holm p = 0.20. DT −0.39, RF −0.24 and XGB −0.24 are significant. |
| R-H3 | Replicates | 4/4 | Sign agreement 0.94–1.00, against top-3 membership. |
| R-H4 | Replicates | 3/3 | Cross-model is 0.11–0.32 below within-model. |
| R-H5a | Replicates | 4/4 | Holm p from 0.0004 to 0.025. |
| R-H5b | Partly replicates | 3/4 | DT has Holm p = 0.077. The share of unstable customers (13–48 vs 0–12 months) is LR 24% vs 14%, RF 10% vs 2% and XGB 18% vs 6%. |
| R-H6 | Partly replicates | 1/2 | DT +0.0014 (Holm p = 0.027) is significant but negligible. LR +0.0009 has p = 0.34. Descriptively, XGB went the other way (−0.005). |
| R-H7 | Does not replicate | 0/1 | Mean f − a is positive in 8 of 10 splits, but p = 0.053 (Holm 0.26). LR sign flips are rare overall: sign agreement is 0.999, against 0.958 in IBM. |

## Descriptive results

- **Local top-3 Jaccard under bootstrap, Cell2Cell vs IBM:** LR 0.64 vs 0.65; DT 0.45 vs 0.60; RF 0.70 vs 0.71; XGB 0.65 vs 0.70. The chance baseline for top-3 is 0.034.
- **Top feature:** all four models rank `CurrentEquipmentDays` first in all 10 splits. The IBM pattern that different model classes rank different features first does **not** recur.
- **Global rank agreement:** Spearman between the tree models is 0.87–0.96, and between LR and the tree models 0.65–0.79.
- **LIME:** self-agreement at top-5 is 0.66–0.79. LIME–SHAP top-5 agreement is 0.27–0.40.
- **LR coefficients of the collinear set** (share of refits with a positive sign): DroppedBlockedCalls 80%, BlockedCalls 75%, DroppedCalls 98%, MonthlyRevenue 99%, RetentionCalls 100%. Signs are unstable for the near-duplicate pair, but these features rarely enter an LR top-3: 1.3–3.2% of top-3 instances.
- **Sign flips in features outside the collinear set:** some features with near-zero coefficients also flip, for example CallForwardingCalls (flip rate 0.33) and OwnsMotorcycle (0.05).

## Correction to the plan's data description

The plan (Section 2) says that missing numeric values occur in 10 columns (156–909 per column). In fact 13 numeric columns have missing values: the 10 named, plus `Handsets`, `HandsetModels` and `CurrentEquipmentDays` with 1 missing value each. This does not affect any result, because the preprocessor median-imputes every numeric column. The frozen plan is left unchanged, and the manuscript states the correct figure (1–909 per column, in 13 columns).
