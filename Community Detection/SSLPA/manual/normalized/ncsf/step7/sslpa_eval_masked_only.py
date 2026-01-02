"""
sslpa_eval_masked_only.py

Evaluate semi-supervised SSLPA runs on the *masked* nodes only.

Assumptions
-----------
1. You have already run `sslpa_semi_supervised_eval.py`, which created
   one folder per mask fraction:

       Community Detection/SSLPA/manual/ncsf/semi_supervised/maskXX/

   Each folder contains:
      - sslpa_train_labels.csv       (used as seeds for SSLPA)
      - sslpa_test_labels_true.csv   (ground-truth labels for the masked nodes)
      - sslpa_all_labels_with_split.csv  (for debugging, not strictly needed)

2. After running SSLPA **using only the train labels as seeds**, you exported
   predictions for all nodes in that split to:

       sslpa_predictions.csv

   with columns:
      - account_id
      - predicted_label

3. The existing evaluation summaries live at:

       eval_filter_A/all_methods_filterA_summary.csv
       eval_filter_B/all_methods_filterB_summary.csv

This script will:
  * For each mask fraction in MASK_FRACTIONS:
      - merge true + predicted labels **on the test nodes only**
      - compute external clustering metrics (NMI, ARI, AMI, FMI, homogeneity,
        completeness, V-measure) → Filter A style
      - compute binary SCAM vs NON_SCAM metrics (precision, recall, F1,
        NMI/ARI/FMI) → Filter B style
      - append one row per mask to each of the two summary CSVs, with
        method names like "SSLPA_NCSF_mask30".
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

# ----------------------------- CONFIG ---------------------------------

BASE_DIR = Path("/home/user/jfayzullaev/stellar-clustering/publication")
NCSF_DIR = BASE_DIR / "Community Detection" / "SSLPA" / "manual" / "normalized" / "ncsf"

# MODE: Choose which label set to use
# Options: "scam_only" or "all_labels"
# Must match what you used in sslpa_semi_supervised_eval.py
MODE = "scam_only"

# Semi-supervised splits directory
SCRIPT_DIR = Path(__file__).parent
STEP6_DIR = SCRIPT_DIR.parent / "step6"
SEMI_DIR = STEP6_DIR / "semi_supervised" / MODE

# Output evaluation results in the same directory as this script (step7)
EVAL_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / MODE
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_A_PATH = EVAL_OUTPUT_DIR / "all_methods_filterA_summary.csv"
EVAL_B_PATH = EVAL_OUTPUT_DIR / "all_methods_filterB_summary.csv"

# Must match the fractions you used in sslpa_semi_supervised_eval.py
MASK_FRACTIONS = [0.3]  # e.g. [0.1, 0.3, 0.5]

# Which original label values are treated as "SCAM" for the binary evaluation
# Extend this set if you have other scam-specific labels (e.g., hacks).
# NOTE: UltraCapital is *not* a scam label and is therefore NOT included here.
SCAM_LABELS = {"SCAM"}

NODE_COL = "node"
TRUE_LABEL_COL = "label"
PRED_LABEL_COL = "predicted_label"


# ----------------------------- HELPERS --------------------------------

def _load_summary(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def _to_binary(label: str) -> int:
    """Map raw labels to binary 1=SCAM, 0=NON_SCAM."""
    return 1 if label in SCAM_LABELS else 0


def eval_mask_fraction(mask_fraction: float):
    """Compute Filter A & B style metrics for a single mask fraction."""
    pct = int(mask_fraction * 100)
    split_dir = SEMI_DIR / f"mask{pct}"

    true_path = split_dir / "sslpa_test_labels_true.csv"
    pred_path = split_dir / "sslpa_predictions.csv"

    if not true_path.exists():
        print(f"[mask={mask_fraction:.2f}] Missing file: {true_path}")
        return None, None

    if not pred_path.exists():
        print(f"[mask={mask_fraction:.2f}] Missing predictions file: {pred_path}")
        print("  → Run your SSLPA implementation for this mask fraction,")
        print("    then export predictions as 'sslpa_predictions.csv' in that folder.")
        return None, None

    true_df = pd.read_csv(true_path)
    pred_df = pd.read_csv(pred_path)

    # Normalise column names in case something changed
    if NODE_COL not in true_df.columns:
        if "node" in true_df.columns:
            true_df = true_df.rename(columns={"node": NODE_COL})
        else:
            raise KeyError(
                f"Expected '{NODE_COL}' column in {true_path.name}, "
                f"but got columns: {list(true_df.columns)}"
            )

    if TRUE_LABEL_COL not in true_df.columns:
        if "label_true" in true_df.columns:
            true_df = true_df.rename(columns={"label_true": TRUE_LABEL_COL})
        else:
            raise KeyError(
                f"Expected '{TRUE_LABEL_COL}' column in {true_path.name}, "
                f"but got columns: {list(true_df.columns)}"
            )

    # Predictions: allow either 'predicted_label' or 'label'
    if NODE_COL not in pred_df.columns:
        if "node" in pred_df.columns:
            pred_df = pred_df.rename(columns={"node": NODE_COL})
        else:
            raise KeyError(
                f"Expected '{NODE_COL}' column in {pred_path.name}, "
                f"but got columns: {list(pred_df.columns)}"
            )

    if PRED_LABEL_COL not in pred_df.columns:
        if "label" in pred_df.columns:
            pred_df = pred_df.rename(columns={"label": PRED_LABEL_COL})
        else:
            raise KeyError(
                f"Expected '{PRED_LABEL_COL}' column in {pred_path.name}, "
                f"but got columns: {list(pred_df.columns)}"
            )

    # Merge *only* on the test nodes
    merged = true_df.merge(pred_df, on=NODE_COL, how="inner")

    if merged.empty:
        raise ValueError(
            f"[mask={mask_fraction:.2f}] No overlapping nodes between "
            f"true and predicted labels."
        )

    coverage = len(merged) / len(true_df)

    y_true = merged[TRUE_LABEL_COL].astype(str).to_numpy()
    y_pred = merged[PRED_LABEL_COL].astype(str).to_numpy()

    # -------- Filter A style metrics (multi-class entities) --------
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    hom, comp, v = homogeneity_completeness_v_measure(y_true, y_pred)

    method_name = f"SSLPA_NCSF_mask{pct}"

    row_A = {
        "method": method_name,
        "coverage": coverage,
        "Avg_NMI_train": np.nan,
        "Avg_NMI_test": nmi,
        "Std_NMI_test": 0.0,
        "Avg_ARI_train": np.nan,
        "Avg_ARI_test": ari,
        "Std_ARI_test": 0.0,
        "Avg_AMI_train": np.nan,
        "Avg_AMI_test": ami,
        "Std_AMI_test": 0.0,
        "Avg_FMI_train": np.nan,
        "Avg_FMI_test": fmi,
        "Std_FMI_test": 0.0,
        "Avg_Homogeneity_train": np.nan,
        "Avg_Homogeneity_test": hom,
        "Std_Homogeneity_test": 0.0,
        "Avg_Completeness_train": np.nan,
        "Avg_Completeness_test": comp,
        "Std_Completeness_test": 0.0,
        "Avg_V-measure_train": np.nan,
        "Avg_V-measure_test": v,
        "Std_V-measure_test": 0.0,
    }

    # -------- Filter B style metrics (binary SCAM vs NON_SCAM) --------
    y_true_bin = np.array([_to_binary(lbl) for lbl in y_true])
    y_pred_bin = np.array([_to_binary(lbl) for lbl in y_pred])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    nmi_b = normalized_mutual_info_score(y_true_bin, y_pred_bin)
    ari_b = adjusted_rand_score(y_true_bin, y_pred_bin)
    fmi_b = fowlkes_mallows_score(y_true_bin, y_pred_bin)

    row_B = {
        "method": method_name,
        "precision_SCAM_mean": prec,
        "precision_SCAM_std": 0.0,
        "recall_SCAM_mean": rec,
        "recall_SCAM_std": 0.0,
        "f1_SCAM_mean": f1,
        "f1_SCAM_std": 0.0,
        "NMI_mean": nmi_b,
        "NMI_std": 0.0,
        "ARI_mean": ari_b,
        "ARI_std": 0.0,
        "FMI_mean": fmi_b,
        "FMI_std": 0.0,
    }

    print(
        f"[mask={mask_fraction:.2f}] coverage={coverage:.3f}, "
        f"NMI={nmi:.3f}, ARI={ari:.3f}, AMI={ami:.3f}, FMI={fmi:.3f}"
    )
    print(
        f"[mask={mask_fraction:.2f}] SCAM precision={prec:.3f}, "
        f"recall={rec:.3f}, F1={f1:.3f}"
    )

    return row_A, row_B


def main():
    if MODE not in ("scam_only", "all_labels"):
        raise ValueError(
            f"Invalid MODE: {MODE}\n"
            "MODE must be either 'scam_only' or 'all_labels'."
        )

    print(f"Running in MODE: {MODE}")
    print(f"Looking for semi-supervised splits in: {SEMI_DIR}")

    rows_A = []
    rows_B = []

    for frac in MASK_FRACTIONS:
        row_A, row_B = eval_mask_fraction(frac)
        if row_A is not None:
            rows_A.append(row_A)
        if row_B is not None:
            rows_B.append(row_B)

    if rows_A:
        df_new_A = pd.DataFrame(rows_A)
        df_old_A = _load_summary(EVAL_A_PATH)
        if df_old_A is not None:
            df_out_A = pd.concat([df_old_A, df_new_A], ignore_index=True)
        else:
            df_out_A = df_new_A
        df_out_A.to_csv(EVAL_A_PATH, index=False)
        print(f"Appended {len(rows_A)} rows to {EVAL_A_PATH}")

    if rows_B:
        df_new_B = pd.DataFrame(rows_B)
        df_old_B = _load_summary(EVAL_B_PATH)
        if df_old_B is not None:
            df_out_B = pd.concat([df_old_B, df_new_B], ignore_index=True)
        else:
            df_out_B = df_new_B
        df_out_B.to_csv(EVAL_B_PATH, index=False)
        print(f"Appended {len(rows_B)} rows to {EVAL_B_PATH}")

    if not rows_A and not rows_B:
        print("No new rows written – check that your semi-supervised prediction files exist.")


if __name__ == "__main__":
    main()

