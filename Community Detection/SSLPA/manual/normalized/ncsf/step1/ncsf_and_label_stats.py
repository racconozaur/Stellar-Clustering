import os
import pickle
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import Counter

# ---------- CONFIG ----------
GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)

SEED_LABELS = os.path.expanduser(
    "~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv"
)

SSLPA_LABELS = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/sslpa_tx_lcc_manual_labels.csv"
)
# ^ adjust path if different

OUT_NCSF_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf"
)
OUT_STATS_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/manual/normalized/ncsf/stats"
)

os.makedirs(OUT_NCSF_DIR, exist_ok=True)
os.makedirs(OUT_STATS_DIR, exist_ok=True)

D_MIN = 3
R_MIN = 0.5
MODE = "all_labels"      # 'scam_only' or 'all_labels'


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ---------- CORE HELPERS ----------

def load_graph(path):
    print(f"[{ts()}] Loading graph: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"[{ts()}] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_sslpa_labels(csv_path):
    """
    sslpa_tx_lcc_manual_labels.csv:
        columns: node, label
    or: account_id, label
    """
    print(f"[{ts()}] Loading SSLPA labels: {csv_path}")
    df = pd.read_csv(csv_path)
    if "node" not in df.columns and "account_id" in df.columns:
        df = df.rename(columns={"account_id": "node"})
    return df[["node", "label"]]


def ncsf_filter(df_labels, G, d_min=3, r_min=0.5, mode="scam_only"):
    """
    Neighborhood-Constrained Scam Filter.

    mode:
      - 'scam_only'  -> only constrain SCAM
      - 'all_labels' -> constrain every label (Binance, Lobstr, SCAM, ...)
    """
    print(f"[{ts()}] Applying NCSF (d_min={d_min}, r_min={r_min}, mode={mode})")

    lab = dict(zip(df_labels["node"], df_labels["label"]))
    before_counts = Counter(lab.values())
    total_nodes = len(lab)

    new_lab = lab.copy()
    changed = 0

    for node, label in lab.items():
        if label is None or label == "UNLABELED":
            continue

        if mode == "scam_only" and label != "SCAM":
            continue

        if node not in G:
            new_lab[node] = "UNLABELED"
            changed += 1
            continue

        neighbors = list(G.neighbors(node))
        deg = len(neighbors)
        if deg == 0:
            new_lab[node] = "UNLABELED"
            changed += 1
            continue

        same = 0
        for nb in neighbors:
            nb_label = lab.get(nb, None)
            if nb_label == label:
                same += 1

        ratio = same / deg

        if deg < d_min or ratio < r_min:
            new_lab[node] = "UNLABELED"
            changed += 1

    after_counts = Counter(new_lab.values())

    print(f"[{ts()}] NCSF changed {changed:,} / {total_nodes:,} nodes "
          f"({changed/total_nodes:.2%})")
    print(f"[{ts()}] Top labels BEFORE:", before_counts.most_common(10))
    print(f"[{ts()}] Top labels AFTER:", after_counts.most_common(10))

    out_df = pd.DataFrame({"node": list(new_lab.keys()),
                           "label": list(new_lab.values())})
    return out_df, before_counts, after_counts


def label_summary(df, label_col, out_csv):
    """
    Basic label distribution: top 20, #>100k, #=1
    """
    counts = df[label_col].value_counts()
    summary_df = counts.reset_index()
    summary_df.columns = ["label", "count"]

    gt_100k = (summary_df["count"] > 100_000).sum()
    eq_1 = (summary_df["count"] == 1).sum()

    summary_df["gt_100k_flag"] = summary_df["count"] > 100_000
    summary_df["eq_1_flag"] = summary_df["count"] == 1

    summary_df.to_csv(out_csv, index=False)

    print(f"[{ts()}] Saved label summary → {out_csv}")
    print(f"[{ts()}] Top 10 labels:")
    print(summary_df.head(10).to_string(index=False))
    print(f"[{ts()}] Labels with >100k accounts: {gt_100k}")
    print(f"[{ts()}] Labels with exactly 1 account: {eq_1}")


# ---------- MAIN PIPELINE ----------

def main():
    G = load_graph(GRAPH_PKL)

    # 1) Seeds (normalized labels_mapped_normalized.csv)
    seed_df = pd.read_csv(SEED_LABELS).dropna(subset=["account_id", "name"])
    seed_summary_csv = os.path.join(OUT_STATS_DIR, "labels_seed_summary.csv")
    label_summary(seed_df.rename(columns={"name": "label"}),
                  "label", seed_summary_csv)

    # 2) SSLPA output
    sslpa_df = load_sslpa_labels(SSLPA_LABELS)
    sslpa_summary_csv = os.path.join(OUT_STATS_DIR, "labels_sslpa_summary.csv")
    label_summary(sslpa_df.rename(columns={"label": "label"}),
                  "label", sslpa_summary_csv)

    # 3) NCSF (SCAM-focused)
    ncsf_df, before_counts, after_counts = ncsf_filter(
        sslpa_df, G, d_min=D_MIN, r_min=R_MIN, mode=MODE
    )
    out_ncsf_csv = os.path.join(
        OUT_NCSF_DIR, f"sslpa_tx_lcc_ncsf_d{D_MIN}_r{R_MIN}_{MODE}.csv"
    )
    ncsf_df.to_csv(out_ncsf_csv, index=False)
    print(f"[{ts()}] Saved NCSF labels to {out_ncsf_csv}")

    ncsf_summary_csv = os.path.join(OUT_STATS_DIR, "labels_ncsf_summary.csv")
    label_summary(ncsf_df.rename(columns={"label": "label"}),
                  "label", ncsf_summary_csv)


if __name__ == "__main__":
    main()
