import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from common_holdout import save_json, save_binary_counts, save_multiclass_counts

FIXED_DIR = Path(__file__).parent / "step6_holdout" / "fixed_test"
OUT = Path(__file__).parent / "step6_holdout" / "dev_repeats"
OUT.mkdir(parents=True, exist_ok=True)

N_REPEATS = 5
TRAIN_RATIO_WITHIN_DEV = 0.875   # yields overall 70/10/20 when test is 20%
VAL_RATIO_WITHIN_DEV = 0.125
RANDOM_SEED_BASE = 100

# Binary
dev_bin = pd.read_csv(FIXED_DIR / "fixed_dev_labels_binary.csv")
dev_mul = pd.read_csv(FIXED_DIR / "fixed_dev_labels_multiclass.csv")

for i in range(1, N_REPEATS + 1):
    repeat_id = f"repeat_{i:02d}"
    seed = RANDOM_SEED_BASE + i

    # Binary repeat
    sss_b = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO_WITHIN_DEV, random_state=seed)
    train_idx_b, val_idx_b = next(sss_b.split(dev_bin[["node_id"]], dev_bin["label"]))

    train_b = dev_bin.iloc[train_idx_b].copy()
    val_b = dev_bin.iloc[val_idx_b].copy()

    membership_b = pd.concat([
        train_b.assign(split_role="train"),
        val_b.assign(split_role="validation"),
    ], ignore_index=True)
    membership_b.to_csv(OUT / f"{repeat_id}_membership_binary.csv", index=False)

    train_b.to_csv(OUT / f"{repeat_id}_train_labels_binary.csv", index=False)
    val_b.to_csv(OUT / f"{repeat_id}_val_labels_binary.csv", index=False)

    save_binary_counts(
        train_b.rename(columns={"label": "binary_label"}),
        repeat_id,
        "train",
        OUT / f"{repeat_id}_binary_counts.csv"
    )
    save_binary_counts(
        val_b.rename(columns={"label": "binary_label"}),
        repeat_id,
        "validation",
        OUT / f"{repeat_id}_binary_counts.csv"
    )

    # Multiclass repeat
    sss_m = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO_WITHIN_DEV, random_state=seed)
    train_idx_m, val_idx_m = next(sss_m.split(dev_mul[["node_id"]], dev_mul["label"]))

    train_m = dev_mul.iloc[train_idx_m].copy()
    val_m = dev_mul.iloc[val_idx_m].copy()

    membership_m = pd.concat([
        train_m.assign(split_role="train"),
        val_m.assign(split_role="validation"),
    ], ignore_index=True)
    membership_m.to_csv(OUT / f"{repeat_id}_membership_multiclass.csv", index=False)

    train_m.to_csv(OUT / f"{repeat_id}_train_labels_multiclass.csv", index=False)
    val_m.to_csv(OUT / f"{repeat_id}_val_labels_multiclass.csv", index=False)

    save_multiclass_counts(
        train_m.rename(columns={"label": "multiclass_label"}),
        repeat_id,
        "train",
        OUT / f"{repeat_id}_multiclass_counts.csv"
    )
    save_multiclass_counts(
        val_m.rename(columns={"label": "multiclass_label"}),
        repeat_id,
        "validation",
        OUT / f"{repeat_id}_multiclass_counts.csv"
    )

    save_json(OUT / f"{repeat_id}_split_config.json", {
        "repeat_id": repeat_id,
        "random_seed": seed,
        "train_ratio_within_development": TRAIN_RATIO_WITHIN_DEV,
        "val_ratio_within_development": VAL_RATIO_WITHIN_DEV,
    })

print("Done: development repeats created.")