import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# ---------- CONFIG ----------

OUT_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step4/internal_metrics"
)
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    # ==== KMeans + Node2Vec ====
    {
        "method": "KMeans_Node2Vec_k10",
        "emb_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv"
        ),
        "cluster_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/Node2Vec_res/transactions_node2vec_kmeans_results.csv"
        ),
        "cluster_col": "kmeans_10",
    },
    # ==== KMeans + LINE ====
    {
        "method": "KMeans_LINE_k10",
        "emb_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/Embeddings Generation/LINE/stellar_line_embeddings_order2.csv"
        ),
        "cluster_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/K-Means/LINE_res/transactions_line_kmeans_results.csv"
        ),
        "cluster_col": "kmeans_10",
    },

    # ==== HDBSCAN + Node2Vec ====
    {
        "method": "HDBSCAN_Node2Vec_mcs50",
        "emb_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv"
        ),
        "cluster_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/DBSCAN/Node2Vec_res/tx_node2vec_hdbscan_cosine_pca64.csv"
        ),
        "cluster_col": "hdbscan_mcs50",
    },
    # ==== HDBSCAN + LINE ====
    {
        "method": "HDBSCAN_LINE_mcs20",
        "emb_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/Embeddings Generation/LINE/stellar_line_embeddings_order2.csv"
        ),
        "cluster_file": os.path.expanduser(
            "~/stellar-clustering/publication/Clustering/DBSCAN/LINE_res/tx_line_hdbscan_cosine_pca64.csv"
        ),
        "cluster_col": "hdbscan_mcs20",
    },
]


def ts():
    return datetime.now().strftime("%H:%M:%S")


def load_embeddings(path):
    df = pd.read_csv(path)
    # Handle both 'account_id' and 'node_id' columns
    if "node_id" in df.columns:
        df = df.rename(columns={"node_id": "account_id"})
    elif "account_id" not in df.columns:
        raise ValueError(f"Embeddings file {path} must have 'account_id' or 'node_id' column.")

    feat_cols = [c for c in df.columns if c not in ["account_id", "node_id"]]
    X = df[feat_cols].values
    return df[["account_id"]].copy(), X


def run_internal_metrics(method, emb_file, cluster_file, cluster_col):
    print(f"\n[{ts()}] Running internal metrics for {method}")
    emb_ids_df, X = load_embeddings(emb_file)
    clusters_df = pd.read_csv(cluster_file)

    # Handle both 'account_id' and 'node_id' columns in cluster file
    if "node_id" in clusters_df.columns:
        clusters_df = clusters_df.rename(columns={"node_id": "account_id"})
    elif "account_id" not in clusters_df.columns:
        raise ValueError(f"{cluster_file} has no 'account_id' or 'node_id' column")

    if cluster_col not in clusters_df.columns:
        raise ValueError(f"{cluster_file} has no '{cluster_col}' column")

    clusters_df["account_id"] = clusters_df["account_id"].astype(str)
    emb_ids_df["account_id"] = emb_ids_df["account_id"].astype(str)

    merged = emb_ids_df.merge(
        clusters_df[["account_id", cluster_col]],
        on="account_id",
        how="inner"
    )
    print(f"[{ts()}] Merged {len(merged):,} rows for internal metrics")

    y = merged[cluster_col].values
    # Align X to merged order
    X_aligned = X[merged.index.values, :]

    # Some internal metrics require at least 2 clusters & >1 sample per cluster
    n_clusters = len(np.unique(y))
    if n_clusters < 2:
        print(f"[{ts()}] {method}: only {n_clusters} cluster(s) – skipping.")
        return None

    # sil = silhouette_score(X_aligned, y)
    ch = calinski_harabasz_score(X_aligned, y)
    db = davies_bouldin_score(X_aligned, y)

    print(f"[{ts()}] {method}: CH={ch:.2f}, DB={db:.4f}")

    # "silhouette": sil,


    row = {
        "method": method,
        "n_samples": len(merged),
        "n_clusters": n_clusters,
        
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }
    return row


def main():
    rows = []
    for cfg in CONFIGS:
        res = run_internal_metrics(
            method=cfg["method"],
            emb_file=cfg["emb_file"],
            cluster_file=cfg["cluster_file"],
            cluster_col=cfg["cluster_col"],
        )
        if res is not None:
            rows.append(res)

    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(OUT_DIR, "embedding_internal_metrics.csv")
        df.to_csv(out_csv, index=False)
        print(f"[{ts()}] Saved internal metrics → {out_csv}")


if __name__ == "__main__":
    main()
