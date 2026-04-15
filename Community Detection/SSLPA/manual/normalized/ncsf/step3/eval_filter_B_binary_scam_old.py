import os
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

# ---------- CONFIG ----------

MODE = 'all_labels'

# NCSF labels (after scam-only filtering)
NCSF_LABELS = os.path.expanduser(
    f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step1/sslpa_tx_lcc_ncsf_d3_r0.5_{MODE}.csv"
)

OUT_DIR = os.path.expanduser(
    f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step3/eval_filter_B_{MODE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

# Map: method_name → { "file", "account_col", "cluster_col" }
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
            f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step1/sslpa_tx_lcc_ncsf_d3_r0.5_{MODE}.csv"
        ),
        "account_col": "node",
        "cluster_col": "label",
    },
    # ==== Embedding-based (examples – adjust paths/column names!) ====
    "KMeans_Line_k10": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/LINE_res/transactions_line_kmeans_results.csv"
        ),
        "account_col": "node_id",
        "cluster_col": "kmeans_10",  # for example; pick the chosen k
    },
    "KMeans_Node2Vec_k10": {
        "file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/Node2Vec_res/transactions_node2vec_kmeans_results.csv"
        ),
        "account_col": "account_id",
        "cluster_col": "kmeans_10",  # for example; pick the chosen k
    },
    # Add HDBSCAN+LINE, HDBSCAN+Node2Vec, KMeans+LINE similarly...
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
    }

}


def ts():
    return datetime.now().strftime("%H:%M:%S")


def load_binary_labels():
    """
    Build SCAM vs NON_SCAM labels from NCSF output.
    """
    df = pd.read_csv(NCSF_LABELS)
    if "node" in df.columns:
        df = df.rename(columns={"node": "account_id", "label": "raw_label"})
    else:
        df = df.rename(columns={"label": "raw_label"})

    df["account_id"] = df["account_id"].astype(str)

    def to_binary(lbl):
        return "SCAM" if lbl == "SCAM" else "NON_SCAM"

    df["binary_label"] = df["raw_label"].astype(str).map(to_binary)
    print(f"[{ts()}] Binary labels: {df['binary_label'].value_counts().to_dict()}")
    return df[["account_id", "binary_label"]]


def load_clusters(method_name, cfg):
    path = cfg["file"]
    acc_col = cfg["account_col"]
    clus_col = cfg["cluster_col"]

    print(f"[{ts()}] [{method_name}] Loading clusters from {path}")
    df = pd.read_csv(path)

    if acc_col not in df.columns:
        raise ValueError(f"{method_name}: account column '{acc_col}' not in {path}")
    if clus_col not in df.columns:
        raise ValueError(f"{method_name}: cluster column '{clus_col}' not in {path}")

    out = df[[acc_col, clus_col]].copy()
    out = out.dropna().drop_duplicates()
    out = out.rename(columns={acc_col: "account_id", clus_col: "community"})

    out["account_id"] = out["account_id"].astype(str)
    out["community"] = out["community"].astype(str)
    return out


def evaluate_binary_cv(labels_df, comm_df, n_splits=5, random_state=42):
    """
    Evaluate binary SCAM vs NON_SCAM via 5-fold CV on *labeled nodes only*.
    Treat each cluster as a predicted label class (assign majority class per cluster).
    """
    joined = labels_df.merge(comm_df, on="account_id", how="inner")
    print(f"[{ts()}] Joined rows: {len(joined):,}")

    y_true_all = joined["binary_label"].values
    clusters_all = joined["community"].values

    # We'll treat cluster → majority label mapping *inside each fold* to avoid leakage.
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true_all)
    X_ids = joined["account_id"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_ids, y_true_enc), start=1):
        df_tr = joined.iloc[tr_idx].copy()
        df_te = joined.iloc[te_idx].copy()

        # build mapping cluster -> majority class on TRAIN
        cm = pd.crosstab(df_tr["community"], df_tr["binary_label"])
        majority_map = cm.idxmax(axis=1).to_dict()

        def map_clusters(s):
            return s.map(majority_map).fillna("NON_SCAM")

        y_true_tr = df_tr["binary_label"].values
        y_pred_tr = map_clusters(df_tr["community"])

        y_true_te = df_te["binary_label"].values
        y_pred_te = map_clusters(df_te["community"])

        # metrics: treat SCAM as positive
        for split_name, yt, yp in [
            ("train", y_true_tr, y_pred_tr),
            ("test", y_true_te, y_pred_te),
        ]:
            prec = precision_score(yt, yp, pos_label="SCAM", zero_division=0)
            rec = recall_score(yt, yp, pos_label="SCAM", zero_division=0)
            f1 = f1_score(yt, yp, pos_label="SCAM", zero_division=0)

            # encode for NMI/ARI/FMI
            le2 = LabelEncoder()
            all_labels = np.concatenate([yt, yp])
            le2.fit(all_labels)
            yt_enc = le2.transform(yt)
            yp_enc = le2.transform(yp)

            nmi = NMI(yt_enc, yp_enc)
            ari = ARI(yt_enc, yp_enc)
            fmi = FMI(yt_enc, yp_enc)

            rows.append({
                "fold": fold,
                "split": split_name,
                "precision_SCAM": prec,
                "recall_SCAM": rec,
                "f1_SCAM": f1,
                "NMI": nmi,
                "ARI": ari,
                "FMI": fmi,
            })

        # also store confusion matrix on TEST for inspection
        cm_test = confusion_matrix(y_true_te, y_pred_te,
                                   labels=["SCAM", "NON_SCAM"])
        cm_flat = {
            "fold": fold,
            "TP_SCAM": int(cm_test[0, 0]),
            "FN_SCAM": int(cm_test[0, 1]),
            "FP_SCAM": int(cm_test[1, 0]),
            "TN_SCAM": int(cm_test[1, 1]),
        }

        yield rows, cm_flat

    # (Note: we aggregate outside in main(); generator used only once.)


def run_for_method(method_name, cfg, labels_df):
    comm_df = load_clusters(method_name, cfg)

    all_rows = []
    cm_rows = []

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # We reimplement the split loop because evaluate_binary_cv already hides it;
    # easier: do NOT call evaluate_binary_cv as generator. Just copy pattern above.

    joined = labels_df.merge(comm_df, on="account_id", how="inner")
    y_true_all = joined["binary_label"].values
    clusters_all = joined["community"].values

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true_all)
    X_ids = joined["account_id"].values

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_ids, y_true_enc), start=1):
        df_tr = joined.iloc[tr_idx].copy()
        df_te = joined.iloc[te_idx].copy()

        cm = pd.crosstab(df_tr["community"], df_tr["binary_label"])
        majority_map = cm.idxmax(axis=1).to_dict()

        def map_clusters(s):
            return s.map(majority_map).fillna("NON_SCAM")

        y_true_tr = df_tr["binary_label"].values
        y_pred_tr = map_clusters(df_tr["community"])

        y_true_te = df_te["binary_label"].values
        y_pred_te = map_clusters(df_te["community"])

        for split_name, yt, yp in [
            ("train", y_true_tr, y_pred_tr),
            ("test", y_true_te, y_pred_te),
        ]:
            prec = precision_score(yt, yp, pos_label="SCAM", zero_division=0)
            rec = recall_score(yt, yp, pos_label="SCAM", zero_division=0)
            f1 = f1_score(yt, yp, pos_label="SCAM", zero_division=0)

            le2 = LabelEncoder()
            all_labels = np.concatenate([yt, yp])
            le2.fit(all_labels)
            yt_enc = le2.transform(yt)
            yp_enc = le2.transform(yp)

            nmi = NMI(yt_enc, yp_enc)
            ari = ARI(yt_enc, yp_enc)
            fmi = FMI(yt_enc, yp_enc)

            all_rows.append({
                "method": method_name,
                "fold": fold,
                "split": split_name,
                "precision_SCAM": prec,
                "recall_SCAM": rec,
                "f1_SCAM": f1,
                "NMI": nmi,
                "ARI": ari,
                "FMI": fmi,
            })

        cm_test = confusion_matrix(y_true_te, y_pred_te,
                                   labels=["SCAM", "NON_SCAM"])
        cm_rows.append({
            "method": method_name,
            "fold": fold,
            "TP_SCAM": int(cm_test[0, 0]),
            "FN_SCAM": int(cm_test[0, 1]),
            "FP_SCAM": int(cm_test[1, 0]),
            "TN_SCAM": int(cm_test[1, 1]),
        })

    per_fold = pd.DataFrame(all_rows)
    cms = pd.DataFrame(cm_rows)

    return per_fold, cms


def main():
    labels_df = load_binary_labels()

    all_per_fold = []
    all_summaries = []
    all_cms = []

    for method, cfg in METHOD_CONFIG.items():
        print(f"\n======== {method} ========")
        try:
            per_fold, cms = run_for_method(method, cfg, labels_df)
        except Exception as e:
            print(f"[{ts()}] Skipping {method} – error: {e}")
            continue

        # save per-fold and confusion matrices
        per_fold_out = os.path.join(OUT_DIR, f"{method}_filterB_{MODE}_per_fold.csv")
        cm_out = os.path.join(OUT_DIR, f"{method}_filterB_{MODE}_confusion.csv")

        per_fold.to_csv(per_fold_out, index=False)
        cms.to_csv(cm_out, index=False)

        print(f"[{ts()}] Saved per-fold metrics → {per_fold_out}")
        print(f"[{ts()}] Saved confusion matrices → {cm_out}")

        # build summary
        test_rows = per_fold[per_fold["split"] == "test"]
        summary = {
            "method": method,
            "precision_SCAM_mean": test_rows["precision_SCAM"].mean(),
            "precision_SCAM_std": test_rows["precision_SCAM"].std(),
            "recall_SCAM_mean": test_rows["recall_SCAM"].mean(),
            "recall_SCAM_std": test_rows["recall_SCAM"].std(),
            "f1_SCAM_mean": test_rows["f1_SCAM"].mean(),
            "f1_SCAM_std": test_rows["f1_SCAM"].std(),
            "NMI_mean": test_rows["NMI"].mean(),
            "NMI_std": test_rows["NMI"].std(),
            "ARI_mean": test_rows["ARI"].mean(),
            "ARI_std": test_rows["ARI"].std(),
            "FMI_mean": test_rows["FMI"].mean(),
            "FMI_std": test_rows["FMI"].std(),
        }
        all_summaries.append(summary)
        all_per_fold.append(per_fold)
        all_cms.append(cms)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(os.path.join(OUT_DIR, "all_methods_filterB_summary.csv"),
                          index=False)
        print(f"[{ts()}] Wrote combined binary SCAM summary.")


if __name__ == "__main__":
    main()


