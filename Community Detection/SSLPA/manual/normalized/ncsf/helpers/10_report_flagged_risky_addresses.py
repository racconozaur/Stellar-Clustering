import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adapters import apply_existing_ncsf

FIXED_DIR = Path(__file__).parent / "step6_holdout" / "fixed_test"
FINAL_DIR = Path(__file__).parent / "step7_holdout" / "final_test"

# Load blind test labels
test_df = pd.read_csv(FIXED_DIR / "fixed_test_labels_binary.csv")
# expected columns: node_id, label

# Load final raw SSLPA predictions
raw_pred = pd.read_csv(FINAL_DIR / "final_sslpa_predictions_raw_binary.csv")
# expected columns: node_id, predicted_label

# Load final chosen thresholds
with open(FINAL_DIR / "final_threshold_binary.json", "r") as f:
    cfg = json.load(f)

d_min = int(cfg["d_min"])
r_min = float(cfg["r_min"])

# Re-apply NCRF/NCLF to the raw predictions
ncrf_pred = apply_existing_ncsf(
    raw_pred.copy(),
    d_min=d_min,
    r_min=r_min,
    mode="binary"
)

# Restrict to blind test nodes
merged = test_df.merge(
    ncrf_pred,
    on="node_id",
    how="left"
)

merged["predicted_label"] = merged["predicted_label"].fillna("UNKNOWN")

# Flagged risky addresses = predicted SCAM
flagged = merged[merged["predicted_label"] == "SCAM"].copy()

X = len(flagged)
Y = int((flagged["label"] == "SCAM").sum())
Z = int((merged["label"] == "SCAM").sum())

print(f"Flagged risky blind-test addresses (X): {X}")
print(f"Of these, true SCAM in blind test (Y): {Y}")
print(f"Total true SCAM in blind test (Z): {Z}")

if X > 0:
    print(f"Precision-style overlap Y/X: {Y/X:.4f}")
if Z > 0:
    print(f"Recall-style overlap Y/Z: {Y/Z:.4f}")

# Save outputs for paper writing
flagged.to_csv(FINAL_DIR / "final_test_flagged_risky_addresses.csv", index=False)

summary = pd.DataFrame([{
    "flagged_risky_addresses_X": X,
    "true_scam_among_flagged_Y": Y,
    "total_true_scam_in_blind_test_Z": Z,
    "overlap_fraction_Y_over_X": (Y / X if X > 0 else 0.0),
    "capture_fraction_Y_over_Z": (Y / Z if Z > 0 else 0.0)
}])

summary.to_csv(FINAL_DIR / "final_test_risky_overlap_summary.csv", index=False)

print("Saved:")
print(FINAL_DIR / "final_test_flagged_risky_addresses.csv")
print(FINAL_DIR / "final_test_risky_overlap_summary.csv")