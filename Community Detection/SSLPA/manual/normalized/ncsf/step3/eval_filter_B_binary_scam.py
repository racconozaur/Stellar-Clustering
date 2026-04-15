import os
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score as NMI,
    adjusted_rand_score as ARI,
    fowlkes_mallows_score as FMI,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# =========================================================
# CONFIG
# =========================================================

MODE = "all_labels"  # "all_labels", "scam_only"

NORM_LABELS_PATH = os.path.expanduser(
    "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv"
)

OUT_DIR = os.path.expanduser(
    f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step3/eval_filter_B_{MODE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

EXCLUDE_LABELS = {
    "UNLABELED",
    "Burn Account",
    "Spam Issuer",
    "UltraCapital",
}

SCAM_LABELS = {"SCAM"}

# Non-empty set -> scam_only uses only these as the negative class.
# Empty set      -> scam_only keeps all non-excluded non-SCAM labels (same as all_labels).
TRUSTED_NON_SCAM_LABELS: set = set()

# Methods whose cluster column already contains direct binary labels
# ("SCAM" / "NON_SCAM") and should NOT go through majority-vote mapping.
DIRECT_LABEL_METHODS = {"SSLPA_NCSF"}

METHOD_CONFIG = {
    "Louvain": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Community Detection/Louvain/resolutions/louvain_result_res0.5.csv"
        ),
        "account_col": "account_id",
        "cluster_col": "community",
    },
    "LPA": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Community Detection/LPA/lpa_tx_lcc_lpa_communities.csv"
        ),
        "account_col": "account_id",
        "cluster_col": "community",
    },
    "SSLPA_NCSF": {
        "file": os.path.expanduser(
            f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step1/{MODE}/sslpa_tx_lcc_ncsf_d3_r0.5_{MODE}.csv"
        ),
        "account_col": "node",
        "cluster_col": "label",  # direct binary label column
    },
    "KMeans_Line_k10": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/LINE_res/transactions_line_kmeans_results.csv"
        ),
        "account_col": "node_id",
        "cluster_col": "kmeans_10",
    },
    "KMeans_Node2Vec_k10": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/Node2Vec_res/transactions_node2vec_kmeans_results.csv"
        ),
        "account_col": "account_id",
        "cluster_col": "kmeans_10",
    },
    "HDBSCAN_LINE_mcs20": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/DBSCAN/LINE_res/tx_line_hdbscan_cosine_pca64.csv"
        ),
        "account_col": "node_id",
        "cluster_col": "hdbscan_mcs20",
    },
    "HDBSCAN_Node2Vec_mcs50": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/DBSCAN/Node2Vec_res/tx_node2vec_hdbscan_cosine_pca64.csv"
        ),
        "account_col": "account_id",
        "cluster_col": "hdbscan_mcs50",
    },
}


# =========================================================
# HELPERS
# =========================================================

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def load_binary_labels() -> pd.DataFrame:
    """
    Build binary SCAM vs NON_SCAM labels from external normalized labels.

    Modes
    -----
    all_labels : keep every non-excluded label; map non-SCAM -> NON_SCAM.
    scam_only  : if TRUSTED_NON_SCAM_LABELS is non-empty, restrict negatives to
                 that set; otherwise behaves identically to all_labels (with a
                 warning so the caller is aware).
    """
    print(f"[{ts()}] Loading external labels from {NORM_LABELS_PATH}")
    df = pd.read_csv(NORM_LABELS_PATH)

    df = df.rename(columns={"node_id": "account_id", "name_normalized": "name"})

    required = {"account_id", "name"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required} in {NORM_LABELS_PATH}, got: {list(df.columns)}"
        )

    df = df.dropna(subset=["account_id", "name"]).copy()
    df["account_id"] = df["account_id"].astype(str)
    df["name"] = df["name"].astype(str)

    # One label per account
    df = df.drop_duplicates(subset=["account_id"])

    # Remove junk labels
    df = df[~df["name"].isin(EXCLUDE_LABELS)].copy()

    if MODE == "scam_only":
        if TRUSTED_NON_SCAM_LABELS:
            keep = df["name"].isin(SCAM_LABELS | TRUSTED_NON_SCAM_LABELS)
            df = df[keep].copy()
            print(
                f"[{ts()}] scam_only mode: restricting negatives to "
                f"{TRUSTED_NON_SCAM_LABELS}"
            )
        else:
            # No trusted subset provided — scam_only degrades to all_labels.
            # Warn explicitly rather than silently producing the same output.
            print(
                f"[{ts()}] WARNING: MODE='scam_only' but TRUSTED_NON_SCAM_LABELS is "
                "empty. Falling back to all non-excluded labels (same as all_labels). "
                "Populate TRUSTED_NON_SCAM_LABELS for a stricter benchmark."
            )

    def to_binary(lbl: str) -> str:
        return "SCAM" if lbl in SCAM_LABELS else "NON_SCAM"

    df["binary_label"] = df["name"].map(to_binary)

    vc = df["binary_label"].value_counts().to_dict()
    print(f"[{ts()}] Binary label distribution: {vc}")
    print(f"[{ts()}] Unique raw labels retained: {df['name'].nunique():,}")
    print(f"[{ts()}] Total benchmark accounts:   {len(df):,}")

    if df["binary_label"].nunique() < 2:
        raise ValueError(
            "Binary benchmark has fewer than 2 classes after filtering. "
            "Adjust EXCLUDE_LABELS / TRUSTED_NON_SCAM_LABELS / MODE."
        )

    return df[["account_id", "binary_label", "name"]]


def load_clusters(method_name: str, cfg: dict) -> pd.DataFrame:
    path = cfg["file"]
    acc_col = cfg["account_col"]
    clus_col = cfg["cluster_col"]

    print(f"[{ts()}] [{method_name}] Loading clusters from {path}")
    df = pd.read_csv(path)

    for col in (acc_col, clus_col):
        if col not in df.columns:
            raise ValueError(
                f"{method_name}: column '{col}' not found in {path}. "
                f"Available: {list(df.columns)}"
            )

    out = df[[acc_col, clus_col]].copy()
    out = out.dropna()
    out = out.rename(columns={acc_col: "account_id", clus_col: "community"})

    out["account_id"] = out["account_id"].astype(str)
    out["community"] = out["community"].astype(str)

    # One community per account — keep first occurrence
    out = out.drop_duplicates(subset=["account_id"])

    print(
        f"[{ts()}] [{method_name}] rows={len(out):,}, "
        f"unique_accounts={out['account_id'].nunique():,}, "
        f"unique_clusters={out['community'].nunique():,}"
    )

    return out


def _map_via_majority_vote(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn cluster -> binary label mapping on df_tr; apply to df_tr and df_te.
    Unseen clusters in df_te default to NON_SCAM.
    Returns (y_pred_train, y_pred_test).
    """
    ctab = pd.crosstab(df_tr["community"], df_tr["binary_label"])
    majority_map: dict[str, str] = ctab.idxmax(axis=1).to_dict()

    def apply_map(series: pd.Series) -> np.ndarray:
        return series.map(majority_map).fillna("NON_SCAM").values

    return apply_map(df_tr["community"]), apply_map(df_te["community"])


def _map_direct_labels(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    valid_labels: set = frozenset({"SCAM", "NON_SCAM"}),
) -> tuple[np.ndarray, np.ndarray]:
    """
    For methods that already output binary labels directly.
    Validates that the label column contains only expected values and falls
    back to NON_SCAM for any unexpected entries.
    """
    def apply(series: pd.Series) -> np.ndarray:
        cleaned = series.where(series.isin(valid_labels), other="NON_SCAM")
        unexpected = series[~series.isin(valid_labels)].unique()
        if len(unexpected) > 0:
            print(
                f"  WARNING: unexpected label values {unexpected.tolist()} "
                "treated as NON_SCAM."
            )
        return cleaned.values

    return apply(df_tr["community"]), apply(df_te["community"])


def _compute_binary_metrics(
    yt: np.ndarray,
    yp: np.ndarray,
) -> dict:
    """Compute precision, recall, F1, NMI, ARI, FMI for one split."""
    prec = precision_score(yt, yp, pos_label="SCAM", zero_division=0)
    rec  = recall_score(yt,  yp, pos_label="SCAM", zero_division=0)
    f1   = f1_score(yt,     yp, pos_label="SCAM", zero_division=0)

    le = LabelEncoder().fit(np.concatenate([yt, yp]))
    yt_enc = le.transform(yt)
    yp_enc = le.transform(yp)

    return {
        "precision_SCAM": prec,
        "recall_SCAM":    rec,
        "f1_SCAM":        f1,
        "NMI":            NMI(yt_enc, yp_enc),
        "ARI":            ARI(yt_enc, yp_enc),
        "FMI":            FMI(yt_enc, yp_enc),
    }


def run_for_method(
    method_name: str,
    cfg: dict,
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Evaluate one method against external binary labels via stratified K-fold CV.

    - Cluster -> binary label mapping is learned on the TRAIN fold only
      (not applicable to direct-label methods).
    - Coverage is computed globally (fraction of benchmark accounts covered).
    """
    is_direct = method_name in DIRECT_LABEL_METHODS

    comm_df = load_clusters(method_name, cfg)

    # For direct-label methods (e.g. SSLPA_NCSF), the community column contains
    # raw entity names ("Binance", "UNLABELED", …) in addition to "SCAM".
    # Binarise here so _map_direct_labels never sees unexpected values.
    if is_direct:
        comm_df["community"] = comm_df["community"].apply(
            lambda x: "SCAM" if x in SCAM_LABELS else "NON_SCAM"
        )

    joined = labels_df.merge(comm_df, on="account_id", how="inner")
    if joined.empty:
        raise ValueError(
            f"{method_name}: no overlap between benchmark labels and method output."
        )

    coverage = len(joined) / len(labels_df)
    print(
        f"[{ts()}] [{method_name}] joined={len(joined):,}/{len(labels_df):,} "
        f"(coverage={coverage:.2%})"
    )

    # Stratify on binary label
    y_for_split = LabelEncoder().fit_transform(joined["binary_label"].values)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_rows = []
    cm_rows  = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(joined.index, y_for_split), start=1):
        df_tr = joined.iloc[tr_idx].copy()
        df_te = joined.iloc[te_idx].copy()

        if is_direct:
            y_pred_tr, y_pred_te = _map_direct_labels(df_tr, df_te)
        else:
            y_pred_tr, y_pred_te = _map_via_majority_vote(df_tr, df_te)

        y_true_tr = df_tr["binary_label"].values
        y_true_te = df_te["binary_label"].values

        for split_name, yt, yp in [
            ("train", y_true_tr, y_pred_tr),
            ("test",  y_true_te, y_pred_te),
        ]:
            metrics = _compute_binary_metrics(yt, yp)
            all_rows.append({
                "method":    method_name,
                "fold":      fold,
                "split":     split_name,
                "coverage":  coverage,
                "n_samples": len(yt),
                **metrics,
            })

        # Confusion matrix on test split only
        cm_test = confusion_matrix(
            y_true_te,
            y_pred_te,
            labels=["SCAM", "NON_SCAM"],
        )
        cm_rows.append({
            "method":   method_name,
            "fold":     fold,
            "coverage": coverage,
            "TP_SCAM":  int(cm_test[0, 0]),
            "FN_SCAM":  int(cm_test[0, 1]),
            "FP_SCAM":  int(cm_test[1, 0]),
            "TN_SCAM":  int(cm_test[1, 1]),
        })

    return pd.DataFrame(all_rows), pd.DataFrame(cm_rows), coverage


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    labels_df = load_binary_labels()

    all_summaries = []

    for method, cfg in METHOD_CONFIG.items():
        print(f"\n{'=' * 80}")
        print(f"METHOD: {method}")
        print(f"{'=' * 80}")

        try:
            per_fold, cms, coverage = run_for_method(method, cfg, labels_df)
        except Exception as e:
            print(f"[{ts()}] Skipping {method} – error: {e}")
            continue

        per_fold_out = os.path.join(OUT_DIR, f"{method}_filterB_{MODE}_per_fold.csv")
        cm_out       = os.path.join(OUT_DIR, f"{method}_filterB_{MODE}_confusion.csv")

        per_fold.to_csv(per_fold_out, index=False)
        cms.to_csv(cm_out,            index=False)

        print(f"[{ts()}] Saved per-fold metrics    → {per_fold_out}")
        print(f"[{ts()}] Saved confusion matrices  → {cm_out}")

        test_rows = per_fold[per_fold["split"] == "test"]

        def mean_std(col: str) -> tuple[float, float]:
            return test_rows[col].mean(), test_rows[col].std()

        p_mu, p_sd   = mean_std("precision_SCAM")
        r_mu, r_sd   = mean_std("recall_SCAM")
        f_mu, f_sd   = mean_std("f1_SCAM")
        nmi_mu, nmi_sd = mean_std("NMI")
        ari_mu, ari_sd = mean_std("ARI")
        fmi_mu, fmi_sd = mean_std("FMI")

        summary = {
            "method":               method,
            "coverage":             coverage,
            "n_folds":              len(test_rows),
            "precision_SCAM_mean":  p_mu,
            "precision_SCAM_std":   p_sd,
            "recall_SCAM_mean":     r_mu,
            "recall_SCAM_std":      r_sd,
            "f1_SCAM_mean":         f_mu,
            "f1_SCAM_std":          f_sd,
            "NMI_mean":             nmi_mu,
            "NMI_std":              nmi_sd,
            "ARI_mean":             ari_mu,
            "ARI_std":              ari_sd,
            "FMI_mean":             fmi_mu,
            "FMI_std":              fmi_sd,
        }
        all_summaries.append(summary)

        print(
            f"[{ts()}] [{method}] "
            f"coverage={coverage:.2%}  "
            f"precision={p_mu:.3f}±{p_sd:.3f}  "
            f"recall={r_mu:.3f}±{r_sd:.3f}  "
            f"f1={f_mu:.3f}±{f_sd:.3f}"
        )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_out = os.path.join(OUT_DIR, "all_methods_filterB_summary.csv")
        summary_df.to_csv(summary_out, index=False)
        print(f"\n[{ts()}] Wrote combined binary SCAM summary → {summary_out}")
    else:
        print(f"\n[{ts()}] No method summaries were produced.")


if __name__ == "__main__":
    main()