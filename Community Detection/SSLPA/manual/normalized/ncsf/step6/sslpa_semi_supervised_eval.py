"""
sslpa_semi_supervised_eval.py

Prepare semi-supervised SSLPA experiments by masking a fraction of NCSF/NCLF labels.

What this script does
---------------------
1. Loads the NCSF-cleaned label file (all labels on the SSLPA LCC).
2. For each mask fraction in MASK_FRACTIONS:
   - Randomly splits labeled nodes into:
        * train set  (kept as seeds for SSLPA)
        * test set   (labels hidden from SSLPA)
   - Saves three CSVs under a dedicated folder per mask:

        maskXX/sslpa_train_labels.csv          -> seed labels you KEEP
        maskXX/sslpa_test_labels_true.csv      -> labels you HIDE but later evaluate on
        maskXX/sslpa_all_labels_with_split.csv -> full overview with a 'split' column

This script does *not* run SSLPA itself – you plug the train split into your
existing SSLPA pipeline, treating test nodes as unlabeled.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------

# Base repo directory on your machine.
# If the repo lives somewhere else, just change this one line.
BASE_DIR = Path("/home/user/jfayzullaev/stellar-clustering/publication")

# MODE: Choose which label set to use
# Options: "scam_only" or "all_labels"
MODE = "scam_only"

# NCSF (NCLF) output with per-node labels on the SSLPA LCC.
NCSF_DIR = BASE_DIR / "Community Detection" / "SSLPA" / "manual" / "normalized" / "ncsf"
STEP1_DIR = NCSF_DIR / "step1" / MODE

# Where to put the semi-supervised splits (separate directory per mode)
# Save in the same directory as this script (step6)
SCRIPT_DIR = Path(__file__).parent
SEMI_BASE_DIR = SCRIPT_DIR / "semi_supervised" / MODE

# Fractions of labels to hide (for evaluation only)
MASK_FRACTIONS = [0.1, 0.2, 0.3, 0.5]

# Fixed RNG for reproducibility
RANDOM_STATE = 46

# Name of the node id and label columns in the NCSF output
NODE_COL = "node"
LABEL_COL = "label"


# ----------------------------- LOGIC ----------------------------------

def make_split(labels_df: pd.DataFrame, mask_fraction: float, rng: np.random.Generator):
    """Return (train_df, test_df, split_flag_array) for a given mask fraction."""
    if not (0.0 < mask_fraction < 1.0):
        raise ValueError(f"mask_fraction must be in (0,1). Got {mask_fraction}.")

    n = len(labels_df)
    mask = rng.random(n) < mask_fraction
    test_df = labels_df[mask].copy()
    train_df = labels_df[~mask].copy()

    split_flag = np.where(mask, "test", "train")
    return train_df, test_df, split_flag


def process_file(ncsf_file: Path, file_name: str):
    """Process a single NCSF label file and create splits."""
    print(f"\n{'='*80}")
    print(f"Processing file: {file_name}")
    print(f"{'='*80}")

    print(f"Loading labels from: {ncsf_file}")
    labels = pd.read_csv(ncsf_file)

    # Basic sanity checks
    for col in (NODE_COL, LABEL_COL):
        if col not in labels.columns:
            raise KeyError(
                f"Expected column '{col}' in {ncsf_file.name}, "
                f"but got columns: {list(labels.columns)}.\n"
                "→ Adjust NODE_COL / LABEL_COL at the top of the script."
            )

    # Drop duplicates just in case
    labels = labels.drop_duplicates(subset=[NODE_COL]).reset_index(drop=True)
    print(f"Loaded {len(labels):,} labeled nodes")

    # Create output directory for this file
    # Extract parameter name from filename (e.g., "d3_r0.5" from "sslpa_tx_lcc_ncsf_d3_r0.5_scam_only.csv")
    file_base = file_name.replace(f"_{MODE}.csv", "").replace("sslpa_tx_lcc_ncsf_", "")
    file_output_dir = SEMI_BASE_DIR / file_base

    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Creating splits for mask fractions: {MASK_FRACTIONS}")

    for frac in MASK_FRACTIONS:
        pct = int(frac * 100)
        split_dir = file_output_dir / f"mask{pct}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df, split_flag = make_split(labels, frac, rng)

        # Full overview with split column
        all_with_split = labels.copy()
        all_with_split["split"] = split_flag

        # Save files
        train_path = split_dir / "sslpa_train_labels.csv"
        test_true_path = split_dir / "sslpa_test_labels_true.csv"
        all_split_path = split_dir / "sslpa_all_labels_with_split.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_true_path, index=False)
        all_with_split.to_csv(all_split_path, index=False)

        print(f"\n  [mask={frac:.0%}]")
        print(f"    Train seeds: {len(train_df):,}")
        print(f"    Test (masked) nodes: {len(test_df):,}")
        print(f"    Output directory: {split_dir}")

    print(f"\n  ✓ Completed splits for {file_name}")
    return file_output_dir


def main():
    if MODE not in ("scam_only", "all_labels"):
        raise ValueError(
            f"Invalid MODE: {MODE}\n"
            "MODE must be either 'scam_only' or 'all_labels'."
        )

    if not STEP1_DIR.exists():
        raise FileNotFoundError(
            f"Step1 directory not found: {STEP1_DIR}\n"
            f"→ Make sure you've run the NCSF labeling step with MODE='{MODE}' "
            "and that the output path matches this config. "
            f"Available modes: 'scam_only', 'all_labels'"
        )

    print(f"Running in MODE: {MODE}")
    print(f"Looking for NCSF files in: {STEP1_DIR}")
    print(f"Mask fractions: {MASK_FRACTIONS}")

    # Find all CSV files in the step1 directory for this mode
    ncsf_files = sorted(STEP1_DIR.glob("*.csv"))

    if not ncsf_files:
        raise FileNotFoundError(
            f"No CSV files found in {STEP1_DIR}\n"
            f"→ Make sure NCSF labeling step has been completed for MODE='{MODE}'"
        )

    print(f"\nFound {len(ncsf_files)} NCSF file(s) to process:")
    for f in ncsf_files:
        print(f"  - {f.name}")

    # Process each file
    processed_dirs = []
    for ncsf_file in ncsf_files:
        output_dir = process_file(ncsf_file, ncsf_file.name)
        processed_dirs.append(output_dir)

    print(f"\n{'='*80}")
    print("DONE! All files processed successfully.")
    print(f"{'='*80}")
    print(f"\nMODE: {MODE}")
    print(f"Processed {len(ncsf_files)} file(s)")
    print(f"Mask fractions: {MASK_FRACTIONS}")
    print(f"\nOutput directories:")
    for d in processed_dirs:
        print(f"  - {d}")


if __name__ == '__main__':
    main()

