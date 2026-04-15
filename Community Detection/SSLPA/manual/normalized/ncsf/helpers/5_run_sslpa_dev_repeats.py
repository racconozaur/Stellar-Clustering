import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from adapters import run_existing_sslpa

REPEATS_DIR = Path(__file__).parent / "step6_holdout" / "dev_repeats"
PRED_DIR = Path(__file__).parent / "step6_holdout" / "dev_repeats" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

for train_csv in sorted(REPEATS_DIR.glob("repeat_*_train_labels_binary.csv")):
    repeat_id = train_csv.name.replace("_train_labels_binary.csv", "")
    pred_df = run_existing_sslpa(train_csv, mode="binary")
    pred_df.to_csv(PRED_DIR / f"{repeat_id}_sslpa_predictions_raw_binary.csv", index=False)

for train_csv in sorted(REPEATS_DIR.glob("repeat_*_train_labels_multiclass.csv")):
    repeat_id = train_csv.name.replace("_train_labels_multiclass.csv", "")
    pred_df = run_existing_sslpa(train_csv, mode="multiclass")
    pred_df.to_csv(PRED_DIR / f"{repeat_id}_sslpa_predictions_raw_multiclass.csv", index=False)

print("Done: SSLPA raw predictions for development repeats saved.")