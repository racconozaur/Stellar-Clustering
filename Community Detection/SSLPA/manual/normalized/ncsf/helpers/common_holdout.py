from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
)

NODE_COL = "node_id"
# Use name_normalized (not entity): name_normalized carries "SCAM" for malicious nodes,
# which is the label that SSLPA propagates. The entity column uses "MALICIOUS" for the
# same rows — keeping name_normalized ensures consistency with the SSLPA seed format.
LABEL_COL = "name_normalized"
SCAM_LABEL = "SCAM"


def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df[[NODE_COL, LABEL_COL]].dropna().drop_duplicates()
    return df


def make_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["binary_label"] = out[LABEL_COL].apply(
        lambda x: "SCAM" if x == SCAM_LABEL else "NON_SCAM"
    )
    return out


def apply_multiclass_policy(df: pd.DataFrame, min_class_size: int = 10) -> tuple[pd.DataFrame, list, list]:
    counts = df[LABEL_COL].value_counts()
    keep_classes = counts[counts >= min_class_size].index.tolist()
    merged = sorted([c for c in counts.index if c not in keep_classes])

    out = df.copy()
    out["multiclass_label"] = out[LABEL_COL].apply(
        lambda x: x if x in keep_classes else "OTHER"
    )
    return out, keep_classes, merged


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_binary_counts(df_split: pd.DataFrame, split_id: str, role: str, outpath: Path):
    n_scam = int((df_split["binary_label"] == "SCAM").sum())
    n_non = int((df_split["binary_label"] == "NON_SCAM").sum())
    row = pd.DataFrame([{
        "split_id": split_id,
        "role": role,
        "n_scam": n_scam,
        "n_non_scam": n_non,
        "total": len(df_split)
    }])
    if outpath.exists():
        old = pd.read_csv(outpath)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(outpath, index=False)


def save_multiclass_counts(df_split: pd.DataFrame, split_id: str, role: str, outpath: Path):
    rows = (
        df_split.groupby("multiclass_label")
        .size()
        .reset_index(name="count")
        .rename(columns={"multiclass_label": "label"})
    )
    rows.insert(0, "role", role)
    rows.insert(0, "split_id", split_id)
    if outpath.exists():
        old = pd.read_csv(outpath)
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(outpath, index=False)


def evaluate_binary(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    pred_df = pred_df.copy()
    truth_df = truth_df.copy()
    pred_df[NODE_COL] = pred_df[NODE_COL].astype(str)
    truth_df[NODE_COL] = truth_df[NODE_COL].astype(str)
    merged = truth_df.merge(pred_df, on=NODE_COL, how="left")
    merged["predicted_label"] = merged["predicted_label"].fillna("UNKNOWN")

    covered = merged[merged["predicted_label"] != "UNKNOWN"].copy()
    coverage = len(covered) / len(merged) if len(merged) else 0.0

    if len(covered) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "coverage": coverage}

    tp = ((covered["label"] == "SCAM") & (covered["predicted_label"] == "SCAM")).sum()
    fp = ((covered["label"] != "SCAM") & (covered["predicted_label"] == "SCAM")).sum()
    fn = ((covered["label"] == "SCAM") & (covered["predicted_label"] != "SCAM")).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
    }


def evaluate_multiclass(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    pred_df = pred_df.copy()
    truth_df = truth_df.copy()
    pred_df[NODE_COL] = pred_df[NODE_COL].astype(str)
    truth_df[NODE_COL] = truth_df[NODE_COL].astype(str)
    merged = truth_df.merge(pred_df, on=NODE_COL, how="left")
    merged["predicted_label"] = merged["predicted_label"].fillna("UNKNOWN")

    covered = merged[merged["predicted_label"] != "UNKNOWN"].copy()
    coverage = len(covered) / len(merged) if len(merged) else 0.0

    if len(covered) == 0:
        return {
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "nmi": 0.0,
            "ari": 0.0,
            "ami": 0.0,
            "fmi": 0.0,
            "coverage": coverage,
        }

    y_true = covered["label"].astype(str).values
    y_pred = covered["predicted_label"].astype(str).values

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "ami": adjusted_mutual_info_score(y_true, y_pred),
        "fmi": fowlkes_mallows_score(y_true, y_pred),
        "coverage": coverage,
    }