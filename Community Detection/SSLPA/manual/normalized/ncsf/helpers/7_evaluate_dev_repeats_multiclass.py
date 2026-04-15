import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import json
from adapters import apply_existing_ncsf
from common_holdout import evaluate_multiclass

REPEATS_DIR = Path(__file__).parent / "step6_holdout" / "dev_repeats"
PRED_DIR = Path(__file__).parent / "step6_holdout" / "dev_repeats" / "predictions"
OUT = Path(__file__).parent / "step7_holdout" / "dev_multiclass"
OUT.mkdir(parents=True, exist_ok=True)

D_GRID = [1, 2, 3, 5, 10]
R_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

summary_rows = []

for val_csv in sorted(REPEATS_DIR.glob("repeat_*_val_labels_multiclass.csv")):
    repeat_id = val_csv.name.replace("_val_labels_multiclass.csv", "")
    val_df = pd.read_csv(val_csv)
    raw_pred = pd.read_csv(PRED_DIR / f"{repeat_id}_sslpa_predictions_raw_multiclass.csv")

    raw_val = evaluate_multiclass(raw_pred, val_df)
    pd.DataFrame([raw_val]).to_csv(OUT / f"{repeat_id}_val_metrics_raw_sslpa_multiclass.csv", index=False)

    grid_rows = []
    for d_min in D_GRID:
        for r_min in R_GRID:
            pred_ncsf = apply_existing_ncsf(raw_pred.copy(), d_min, r_min, mode="multiclass")
            metrics = evaluate_multiclass(pred_ncsf, val_df)
            row = {
                "repeat_id": repeat_id,
                "d_min": d_min,
                "r_min": r_min,
                **metrics,
            }
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT / f"{repeat_id}_threshold_grid_multiclass.csv", index=False)

    best = grid_df.sort_values(["macro_f1", "coverage"], ascending=False).iloc[0]
    best_cfg = {
        "repeat_id": repeat_id,
        "branch": "multiclass",
        "d_min": int(best["d_min"]),
        "r_min": float(best["r_min"]),
        "selection_metric": "macro_f1",
        "selection_value": float(best["macro_f1"]),
    }

    with open(OUT / f"{repeat_id}_selected_thresholds_multiclass.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    summary_rows.append(best_cfg)

pd.DataFrame(summary_rows).to_csv(OUT / "threshold_selection_summary_multiclass.csv", index=False)
print("Done: multiclass development threshold tuning complete.")