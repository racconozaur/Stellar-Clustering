import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
from adapters import run_existing_sslpa, apply_existing_ncsf
from common_holdout import evaluate_binary, evaluate_multiclass

FIXED = Path(__file__).parent / "step6_holdout" / "fixed_test"
OUT = Path(__file__).parent / "step7_holdout" / "final_test"
OUT.mkdir(parents=True, exist_ok=True)

# Load final thresholds
with open(OUT / "final_threshold_binary.json") as f:
    final_bin = json.load(f)
with open(OUT / "final_threshold_multiclass.json") as f:
    final_mul = json.load(f)

# Binary final run on full development
dev_bin = FIXED / "fixed_dev_labels_binary.csv"
test_bin = pd.read_csv(FIXED / "fixed_test_labels_binary.csv")
raw_bin_pred = run_existing_sslpa(dev_bin, mode="binary")
raw_bin_pred.to_csv(OUT / "final_sslpa_predictions_raw_binary.csv", index=False)

raw_bin_metrics = evaluate_binary(raw_bin_pred, test_bin)
pd.DataFrame([raw_bin_metrics]).to_csv(OUT / "final_test_metrics_raw_sslpa_binary.csv", index=False)

ncsf_bin_pred = apply_existing_ncsf(
    raw_bin_pred.copy(),
    d_min=final_bin["d_min"],
    r_min=final_bin["r_min"],
    mode="binary"
)
pd.DataFrame([evaluate_binary(ncsf_bin_pred, test_bin)]).to_csv(
    OUT / "final_test_metrics_ncsf_binary.csv", index=False
)

# Multiclass final run on full development
dev_mul = FIXED / "fixed_dev_labels_multiclass.csv"
test_mul = pd.read_csv(FIXED / "fixed_test_labels_multiclass.csv")
raw_mul_pred = run_existing_sslpa(dev_mul, mode="multiclass")
raw_mul_pred.to_csv(OUT / "final_sslpa_predictions_raw_multiclass.csv", index=False)

raw_mul_metrics = evaluate_multiclass(raw_mul_pred, test_mul)
pd.DataFrame([raw_mul_metrics]).to_csv(OUT / "final_test_metrics_raw_sslpa_multiclass.csv", index=False)

ncsf_mul_pred = apply_existing_ncsf(
    raw_mul_pred.copy(),
    d_min=final_mul["d_min"],
    r_min=final_mul["r_min"],
    mode="multiclass"
)
pd.DataFrame([evaluate_multiclass(ncsf_mul_pred, test_mul)]).to_csv(
    OUT / "final_test_metrics_ncsf_multiclass.csv", index=False
)

print("Done: final frozen-test evaluation complete.")