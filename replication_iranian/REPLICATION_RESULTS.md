# Iranian Churn: results of the confirmatory study

Run on 20 October 2026, following `REPLICATION_PLAN.md` (frozen 2026-10-19T23:26:51Z, SHA-256 `eb5232b4…`) without
deviation. Full experiment 28.5 minutes; results in `pipeline/results_iranian/`. All 37 tests with raw and Holm-adjusted
p-values: `tables/family_tests_holm.csv`; verdicts: `tables/family_verdicts.csv`.

## Verdicts (Holm over 37 tests, α = 0.05)

| ID | Prediction | Significant | Verdict |
|---|---|---|---|
| T-H1 | LR, RF, XGB equivalent within ±0.01 ROC-AUC | 1 of 3 (RF–XGB) | Partly supported |
| T-H2a | Bootstrap stability below seed-only floor | 4 of 4 | Supported |
| T-H2b | Per-customer below global stability | 2 of 4 (LR, DT) | Partly supported |
| T-H3 | Signs more stable than top-3 membership | 4 of 4 | Supported |
| T-H4 | Between-model below within-model agreement | 3 of 3 | Supported |
| T-H5a | Stability rises with distance from threshold | 1 of 4 (DT) | Partly supported |
| T-H6 | Dropping `CustomerValue` raises LR and DT stability | 1 of 2 (DT) | Partly supported |
| T-H7 | LR sign flips concentrate in VIF > 10 features | 0 of 1 (p = 0.007, Holm 0.087) | Not supported |
| **T-R1** | **Averaging over 10 refits raises per-customer stability** | **4 of 4** | **Supported** |
| **T-R2** | **Bagged model not less accurate (margin 0.01)** | **4 of 4** | **Supported** |
| **T-R3** | **5-refit flag identifies unstable customers with ROC-AUC > 0.70** | **1 of 4 (XGB)** | **Partly supported** |

## Key numbers

- **Accuracy is much higher than on the other datasets:** ROC-AUC LR 0.931, DT 0.952, RF 0.987, XGB 0.988.
- **Stability is still well below the floor.** Bootstrap per-customer top-3 agreement is LR 0.76, DT 0.55, RF 0.75,
  XGB 0.72, against seed-only floors of 0.86–1.00. Weak models therefore do not explain the instability.
- **Signs are stable:** 0.95–1.00. Global top-5 inclusion is 0.85–0.93.
- **Model classes disagree strongly per customer.** Between-model agreement is 0.12–0.53, against within-model
  agreement of 0.64–0.76. LR disagrees most with the trees: LR ranks `FrequencyOfUse` second and `Status` eighth,
  while all three tree models rank `Status` first.
- **Averaging works.** With m = 10, agreement is LR 0.93, DT 0.77, RF 0.91 and XGB 0.89. With m = 25 it is 0.84–0.95.
- **The bagged model is more accurate,** by +0.001 (LR) to +0.036 (DT) ROC-AUC.
- **Flagging works descriptively but has little to detect.** The 5-refit flag reaches ROC-AUC 0.81–0.93 (mean over
  evaluable splits). But with such accurate models, on average only 2–4% of customers per split (range 1–7%) had
  held-out stability below 0.5 for LR, RF and XGB, against 34% for DT (range 21–47%). The plan's rule (at least 10 such customers per split, at least 3 splits) left LR with
  only 2 evaluable splits (not evaluable, so it counts as not significant) and RF with 3. DT (raw p = 0.010, Holm
  0.11) and RF (raw 0.022, Holm 0.20) were not significant after Holm; XGB was (Holm 0.001). T-R3 is therefore
  underpowered rather than contradicted. It should be reported as partly supported.
- **Collinearity.** Dropping `CustomerValue` raised stability for DT (+0.021), RF (+0.029) and XGB (+0.040), but not
  for LR (−0.008). Only DT was in the hypothesis. The AUC change was ≤ 0.002.
- **LIME** agrees with itself across seeds at 0.78–0.92 and with SHAP at 0.51–0.64.

## Correction to the plan

The plan (Section 2) says that 11 predictors remain after dropping `Age`. In fact 12 remain: `CallFailure`,
`Complains`, `SubscriptionLength`, `ChargeAmount`, `SecondsOfUse`, `FrequencyOfUse`, `FrequencyOfSMS`,
`DistinctCalledNumbers`, `AgeGroup`, `TariffPlan`, `Status` and `CustomerValue`. The design was not affected: all
predictors are numeric and standardised, and the loader checks the exact feature list. The frozen plan was not edited.
