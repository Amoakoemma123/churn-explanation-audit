"""Experimental constants (proposal Section 9). Change here, nowhere else.

Two datasets share the pipeline. IBM Telco is the default; the Cell2Cell replication is selected with the
environment variable CHURN_DATASET=cell2cell and writes to results_cell2cell/ (plan: replication/REPLICATION_PLAN.md).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = os.environ.get("CHURN_DATASET", "ibm").strip().lower()   # cmd's "set X=y && ..." keeps a trailing space
assert DATASET in ("ibm", "cell2cell", "iranian"), DATASET

# Outer splits. The pilot used 42, so it is excluded (pilot-derived hypotheses are tested on new splits).
OUTER_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]  # 106-110 added to resolve borderline tests
TEST_SIZE = 0.20
N_EXPLAIN = 500

B_BOOT = 50          # bootstrap refits per model and split
B_SEED = 20          # seed-only refits (same data): the stability floor
CV_FOLDS = 5

MODELS = ["LR", "DT", "RF", "XGB"]

# LIME (RQ4): customers per split, seeds per customer, perturbation samples
LIME_CUSTOMERS = 50
LIME_SEEDS = 10
LIME_SAMPLES = 5000

# Pre-specified analysis settings
AUC_MARGIN = 0.01            # H1 equivalence margin
TOP_K = (1, 3, 5)
MIN_CHURNERS = 30            # segments with fewer test churners are descriptive only
TENURE_BINS = [-1, 12, 48, 72]
TENURE_LABELS = ["0-12", "13-48", "49-72"]

N_JOBS = int(os.environ.get("CHURN_N_JOBS", "6").strip())   # 3.3 GB free RAM on this machine: keep workers modest

if DATASET == "ibm":
    DATA = ROOT.parent / "data" / "Telco-Customer-Churn.csv"
    DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    RESULTS = ROOT / "results"
    NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
    TENURE = "tenure"
    IMPUTE = False
    ABLATION = "boot_noTC"                       # H6 condition
    DROP = ["TotalCharges"]
    SEGMENTS = [("Contract", "contract"), ("Tenure", "tenure_band")]   # (label, column in the customer table)
    CONTRASTS = [("Contract", "One year", "Month-to-month"), ("Tenure", "13-48", "0-12")]
elif DATASET == "cell2cell":
    DATA = ROOT.parent / "data" / "cell2celltrain.csv"
    DATA_SHA256 = "3f6b8ff0b8e8f5975aa0de9f1f54967e6c8cb867a9bec6cbb64f4c8990f9a645"
    RESULTS = ROOT / "results_cell2cell"
    NUM = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "DirectorAssistedCalls", "OverageMinutes",
           "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues", "DroppedCalls", "BlockedCalls",
           "UnansweredCalls", "CustomerCareCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls",
           "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut", "DroppedBlockedCalls", "CallForwardingCalls",
           "CallWaitingCalls", "MonthsInService", "UniqueSubs", "ActiveSubs", "Handsets", "HandsetModels",
           "CurrentEquipmentDays", "AgeHH1", "AgeHH2", "RetentionCalls", "RetentionOffersAccepted",
           "ReferralsMadeBySubscriber", "IncomeGroup", "AdjustmentsToCreditRating"]
    TENURE = "MonthsInService"
    IMPUTE = True                                # 1-909 missing values in 13 numeric columns
    ABLATION = "boot_noDBC"                      # H6' condition
    DROP = ["DroppedBlockedCalls"]               # ~ DroppedCalls + BlockedCalls (r = 0.998)
    SEGMENTS = [("Tenure", "tenure_band")]       # Cell2Cell has no contract type
    CONTRASTS = [("Tenure", "13-48", "0-12")]
    # R-H7: numeric features with VIF > 10 against all other predictors (fixed before any result; plan Section 4)
    COLLINEAR = ["DroppedBlockedCalls", "BlockedCalls", "DroppedCalls", "RetentionCalls", "MonthlyRevenue"]
else:   # third dataset: confirmatory study of the remedy (plan: replication_iranian/REPLICATION_PLAN.md)
    DATA = ROOT.parent / "data" / "iranian_churn.csv"
    DATA_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
    RESULTS = ROOT / "results_iranian"
    NUM = ["CallFailure", "Complains", "SubscriptionLength", "ChargeAmount", "SecondsOfUse", "FrequencyOfUse",
           "FrequencyOfSMS", "DistinctCalledNumbers", "AgeGroup", "TariffPlan", "Status", "CustomerValue"]
    TENURE = "SubscriptionLength"
    IMPUTE = False                               # no missing values
    ABLATION = "boot_noCV"                       # T-H6 condition
    DROP = ["CustomerValue"]                     # highest VIF (50.0), "the calculated value of the customer"
    SEGMENTS = [("Tenure", "tenure_band")]       # descriptive only: about 4 test churners per split in 0-12 months
    CONTRASTS = []                               # so the tenure contrast (H5b) is not tested (plan Section 4)
    COLLINEAR = ["CustomerValue", "FrequencyOfSMS", "SecondsOfUse", "FrequencyOfUse"]   # T-H7: VIF > 10

# Conditions: name -> (bootstrap?, features dropped)
CONDITIONS = {"seed": (False, []), "boot": (True, []), ABLATION: (True, DROP)}

RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"


def quick_mode():
    """Tiny configuration for smoke tests."""
    global OUTER_SEEDS, B_BOOT, B_SEED, LIME_CUSTOMERS, LIME_SEEDS, N_EXPLAIN, RAW, TABLES, FIGS
    OUTER_SEEDS, B_BOOT, B_SEED, LIME_CUSTOMERS, LIME_SEEDS, N_EXPLAIN = [101], 4, 2, 5, 2, 100
    q = RESULTS.parent / RESULTS.name.replace("results", "results_quick")
    RAW, TABLES, FIGS = (q / d for d in ("raw", "tables", "figures"))
