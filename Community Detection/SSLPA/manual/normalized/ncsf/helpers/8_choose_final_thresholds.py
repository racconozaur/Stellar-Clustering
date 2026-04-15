import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import json

BIN_SUM = Path(__file__).parent / "step7_holdout" / "dev_binary" / "threshold_selection_summary_binary.csv"
MUL_SUM = Path(__file__).parent / "step7_holdout" / "dev_multiclass" / "threshold_selection_summary_multiclass.csv"
OUT = Path(__file__).parent / "step7_holdout" / "final_test"
OUT.mkdir(parents=True, exist_ok=True)

bin_df = pd.read_csv(BIN_SUM)
mul_df = pd.read_csv(MUL_SUM)

# choose final threshold pair by mean selection value over repeated chosen configs
bin_grouped = (
    bin_df.groupby(["d_min", "r_min"])["selection_value"]
    .mean()
    .reset_index()
    .sort_values("selection_value", ascending=False)
)
mul_grouped = (
    mul_df.groupby(["d_min", "r_min"])["selection_value"]
    .mean()
    .reset_index()
    .sort_values("selection_value", ascending=False)
)

final_binary = bin_grouped.iloc[0].to_dict()
final_multiclass = mul_grouped.iloc[0].to_dict()

with open(OUT / "final_threshold_binary.json", "w") as f:
    json.dump({
        "branch": "binary",
        "d_min": int(final_binary["d_min"]),
        "r_min": float(final_binary["r_min"]),
        "selection_metric": "mean_validation_f1",
        "selection_value": float(final_binary["selection_value"])
    }, f, indent=2)

with open(OUT / "final_threshold_multiclass.json", "w") as f:
    json.dump({
        "branch": "multiclass",
        "d_min": int(final_multiclass["d_min"]),
        "r_min": float(final_multiclass["r_min"]),
        "selection_metric": "mean_validation_macro_f1",
        "selection_value": float(final_multiclass["selection_value"])
    }, f, indent=2)

# combined summary
combined = pd.concat([
    pd.DataFrame([{
        "split_id": "final",
        "branch": "binary",
        "d_min": int(final_binary["d_min"]),
        "r_min": float(final_binary["r_min"]),
        "selection_metric": "mean_validation_f1",
        "selection_value": float(final_binary["selection_value"]),
    }]),
    pd.DataFrame([{
        "split_id": "final",
        "branch": "multiclass",
        "d_min": int(final_multiclass["d_min"]),
        "r_min": float(final_multiclass["r_min"]),
        "selection_metric": "mean_validation_macro_f1",
        "selection_value": float(final_multiclass["selection_value"]),
    }]),
], ignore_index=True)

combined.to_csv(OUT / "threshold_selection_summary.csv", index=False)
print("Done: final thresholds chosen.")