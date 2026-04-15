import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score as NMI,
    adjusted_rand_score as ARI,
    adjusted_mutual_info_score as AMI,
    fowlkes_mallows_score as FMI,
    homogeneity_completeness_v_measure,
)

# ---------- CONFIG ----------

MODE = 'scam_only'  # scam_only    all_labels

NORM_LABELS_PATH = os.path.expanduser(
    "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv"
)

# OUTPUT DIR
OUT_DIR = os.path.expanduser(
    f"~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step2/eval_filter_A_{MODE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

# Labels to exclude from evaluation (generic / junk buckets).
# "Spam Issuer" and "UNLABELED" are not present in the new label file but kept
# here for safety in case they appear in future label refreshes.
EXCLUDE_LABELS = {
    "SCAM",
    "UltraCapital",
    "UNLABELED",
    "Burn Account",
    "Spam Issuer",
}

# StratifiedKFold requires at least N_SPLITS samples per class.
# Labels with fewer nodes than this are merged into "OTHER" rather than
# dropped, so their nodes still contribute to coverage but don't cause a crash.
MIN_CLASS_SIZE = N_SPLITS  # = 5

# If ALLOWED_LABELS is empty → we use "all labels except EXCLUDE_LABELS".
ALLOWED_LABELS = set()  # you can later hard-code only exchanges/services here.

# Map: method_name → { "file": path, "account_col": "..", "cluster_col": ".." }
# >>> Jasur MUST fix the paths below to match his actual output files <<<
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


# ---------- HELPERS ----------

def ts():
    return datetime.now().strftime("%H:%M:%S")


def load_filtered_labels():
    """
    Load normalized labels and keep only "entity labels" for Filter A.
    Labels with fewer than MIN_CLASS_SIZE nodes are merged into "OTHER" so
    StratifiedKFold (which needs >= N_SPLITS samples per class) does not crash.
    """
    df = pd.read_csv(NORM_LABELS_PATH)
    df = df.dropna(subset=["node_id", "name_normalized"]).drop_duplicates(subset=["node_id"])
    df = df.rename(columns={"node_id": "account_id", "name_normalized": "name"})

    df["name"] = df["name"].astype(str)
    df["account_id"] = df["account_id"].astype(str)

    if ALLOWED_LABELS:
        df = df[df["name"].isin(ALLOWED_LABELS)]
    else:
        df = df[~df["name"].isin(EXCLUDE_LABELS)]

    # Merge rare classes into "OTHER" to satisfy StratifiedKFold minimum.
    counts = df["name"].value_counts()
    rare = set(counts[counts < MIN_CLASS_SIZE].index)
    if rare:
        print(f"[{ts()}] Merging {len(rare)} rare labels (< {MIN_CLASS_SIZE} nodes) → 'OTHER'")
        df["name"] = df["name"].apply(lambda x: "OTHER" if x in rare else x)

    print(f"[{ts()}] Filter A – labeled entity accounts: {len(df):,}")
    print(f"[{ts()}] Unique labels after filtering: {df['name'].nunique():,}")

    return df[["account_id", "name"]]


def load_clusters(method_name, cfg):
    """
    Returns df with columns: account_id, community.
    """
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

    print(f"[{ts()}] [{method_name}] rows: {len(out):,}, "
          f"unique accounts: {out['account_id'].nunique():,}, "
          f"clusters: {out['community'].nunique():,}")
    return out


def evaluate_cv(labels_df, comm_df, n_splits=5, random_state=42):
    """
    Generic 5-fold CV evaluation on external labels.
    """
    labels = labels_df.copy()
    comm = comm_df.copy()

    joined = labels.merge(comm, on="account_id", how="inner")

    n_labeled = len(labels)
    n_joined = len(joined)
    coverage = (n_joined / n_labeled) if n_labeled else 0.0

    print(f"[{ts()}] Joined labeled + clusters: {n_joined:,}/{n_labeled:,} "
          f"({coverage:.2%})")

    le = LabelEncoder()
    y_all = le.fit_transform(joined["name"].values)
    X_ids = joined["account_id"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_ids, y_all), start=1):
        df_tr = joined.iloc[tr_idx].copy()
        df_te = joined.iloc[te_idx].copy()

        y_true_tr = le.transform(df_tr["name"])
        y_pred_tr = df_tr["community"].values

        nmi_tr = NMI(y_true_tr, y_pred_tr)
        ari_tr = ARI(y_true_tr, y_pred_tr)
        ami_tr = AMI(y_true_tr, y_pred_tr)
        fmi_tr = FMI(y_true_tr, y_pred_tr)
        homo_tr, comp_tr, v_tr = homogeneity_completeness_v_measure(
            y_true_tr, y_pred_tr
        )

        y_true_te = le.transform(df_te["name"])
        y_pred_te = df_te["community"].values

        nmi_te = NMI(y_true_te, y_pred_te)
        ari_te = ARI(y_true_te, y_pred_te)
        ami_te = AMI(y_true_te, y_pred_te)
        fmi_te = FMI(y_true_te, y_pred_te)
        homo_te, comp_te, v_te = homogeneity_completeness_v_measure(
            y_true_te, y_pred_te
        )

        rows.append({
            "fold": fold,
            "n_train": len(df_tr),
            "n_test": len(df_te),
            "NMI_train": nmi_tr,
            "ARI_train": ari_tr,
            "AMI_train": ami_tr,
            "FMI_train": fmi_tr,
            "Homogeneity_train": homo_tr,
            "Completeness_train": comp_tr,
            "V-measure_train": v_tr,
            "NMI_test": nmi_te,
            "ARI_test": ari_te,
            "AMI_test": ami_te,
            "FMI_test": fmi_te,
            "Homogeneity_test": homo_te,
            "Completeness_test": comp_te,
            "V-measure_test": v_te,
        })

    per_fold = pd.DataFrame(rows)

    metrics = ["NMI", "ARI", "AMI", "FMI", "Homogeneity", "Completeness", "V-measure"]
    summary = {"coverage": coverage}
    for m in metrics:
        summary[f"Avg_{m}_train"] = per_fold[f"{m}_train"].mean()
        summary[f"Avg_{m}_test"] = per_fold[f"{m}_test"].mean()
        summary[f"Std_{m}_test"] = per_fold[f"{m}_test"].std()

    return per_fold, summary


# ---------- MAIN ----------

def main():
    labels_df = load_filtered_labels()

    all_summaries = []
    for method, cfg in METHOD_CONFIG.items():
        try:
            comm_df = load_clusters(method, cfg)
        except Exception as e:
            print(f"[{ts()}] Skipping {method} – error: {e}")
            continue

        per_fold, summary = evaluate_cv(labels_df, comm_df,
                                        n_splits=N_SPLITS,
                                        random_state=RANDOM_STATE)

        per_fold_out = os.path.join(OUT_DIR, f"{method}_filterA_{MODE}_per_fold.csv")
        summary_out = os.path.join(OUT_DIR, f"{method}_filterA_{MODE}_summary.csv")

        per_fold.to_csv(per_fold_out, index=False)
        pd.DataFrame([summary]).to_csv(summary_out, index=False)

        print(f"[{ts()}] Saved {method} per-fold → {per_fold_out}")
        print(f"[{ts()}] Saved {method} summary → {summary_out}")

        row = {"method": method}
        row.update(summary)
        all_summaries.append(row)

    if all_summaries:
        all_df = pd.DataFrame(all_summaries)
        all_df.to_csv(os.path.join(OUT_DIR, "all_methods_filterA_summary.csv"),
                      index=False)
        print(f"[{ts()}] Wrote combined summary for Filter A methods.")


if __name__ == "__main__":
    main()
