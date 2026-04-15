import os
import pickle
from collections import defaultdict, Counter

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as skmetrics

# Config
LABELS = os.path.expanduser(
    "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv"
)
GRAPH_PATH = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)
OUTPUT_DIR = os.path.expanduser(
    "sslpa-igr-out/evaluation_80_20"
)
N_FOLDS = 5
RANDOM_STATE = 42



# Purity
def purity_score(y_true, y_pred):
    contingency = skmetrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


# Run SSLPA with a given set of train seeds return predicted labels for all
def run_sslpa(G: ig.Graph, node_id_map: dict, train_seeds: dict,
              all_seeds: dict) -> dict:
  
    # Encode seed labels  integers
    le = LabelEncoder()
    unique_labels = sorted(set(train_seeds.values()))
    le.fit(unique_labels)
    num_seed_classes = len(unique_labels)
    seed_label_to_int = {lbl: int(le.transform([lbl])[0]) for lbl in unique_labels}

    # Build initial-label and fixed-mask arrays
    initial_labels = []
    fixed_mask = []
    next_unlabeled_id = num_seed_classes

    for i in range(G.vcount()):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            initial_labels.append(seed_label_to_int[train_seeds[node_id]])
            fixed_mask.append(True)
        else:
            # Each non-train node starts as its own singleton
            initial_labels.append(next_unlabeled_id)
            next_unlabeled_id += 1
            fixed_mask.append(False)

    # Run LPA
    communities = G.community_label_propagation(
        weights="weight" if G.is_weighted() else None,
        initial=initial_labels,
        fixed=fixed_mask,
    )
    membership = communities.membership

    # Majority-vote mapping: community id  seed label
    # UTRAIN seeds only for the mapping
    comm_label_votes = defaultdict(Counter)
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            comm_label_votes[comm_id][train_seeds[node_id]] += 1

    comm_to_label = {}
    for comm_id, votes in comm_label_votes.items():
        comm_to_label[comm_id] = votes.most_common(1)[0][0]

    # Assign final labels
    labels = {}
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            labels[node_id] = train_seeds[node_id]
        elif comm_id in comm_to_label:
            labels[node_id] = comm_to_label[comm_id]
        else:
            labels[node_id] = f"CLUSTER_{comm_id}"

    return labels


# Compute metrics
def compute_metrics(y_true, y_pred):

    # Encode labels to integers (union of all labels in both arrays)
    all_labels = sorted(set(y_true) | set(y_pred))
    label_map = {lbl: idx for idx, lbl in enumerate(all_labels)}
    yt = np.array([label_map[l] for l in y_true])
    yp = np.array([label_map[l] for l in y_pred])

    homo, comp, vm = skmetrics.homogeneity_completeness_v_measure(yt, yp)

    return {
        "NMI": skmetrics.normalized_mutual_info_score(yt, yp),
        "ARI": skmetrics.adjusted_rand_score(yt, yp),
        "AMI": skmetrics.adjusted_mutual_info_score(yt, yp),
        "FMI": skmetrics.fowlkes_mallows_score(yt, yp),
        "Homogeneity": homo,
        "Completeness": comp,
        "V-measure": vm,
        "Purity": purity_score(yt, yp),
    }


# Main
def main():
    print("=" * 70)
    print("SSLPA Held-Out Evaluation")
    print("=" * 70)

    # 1 Load graph 
    print(f"\nLoading graph: {GRAPH_PATH}")
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    with open(GRAPH_PATH, "rb") as f:
        G_nx = pickle.load(f)
    if isinstance(G_nx, nx.DiGraph):
        G_nx = nx.Graph(G_nx)

    G = ig.Graph.from_networkx(G_nx)
    print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")

    # Build node_id_map once
    node_id_map = {i: G.vs[i]["_nx_name"] for i in range(G.vcount())}

    # 2. Load seeds
    if not os.path.exists(LABELS):
        raise FileNotFoundError(f"Labels file not found: {LABELS}")

    seeds_df = (
        pd.read_csv(LABELS, usecols=["node_id", "name_normalized"])
        .dropna()
        .drop_duplicates()
    )
    try:
        seeds_df["node_id"] = seeds_df["node_id"].astype("int64")
    except Exception:
        pass

    seeds_all = dict(zip(seeds_df["node_id"].tolist(),
                         seeds_df["name_normalized"].tolist()))
    nx_nodes = set(G_nx.nodes())
    seeds = {n: lbl for n, lbl in seeds_all.items() if n in nx_nodes}
    print(f"Seeds in graph: {len(seeds)}")

    # 3 Prepare arrays for stratified splitting
    seed_nodes = np.array(list(seeds.keys()))
    seed_labels = np.array([seeds[n] for n in seed_nodes])

    # Filter out classes with < N_FOLDS members
    label_counts = Counter(seed_labels)
    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= N_FOLDS}
    mask = np.array([seeds[n] in valid_labels for n in seed_nodes])

    dropped = len(seed_nodes) - mask.sum()
    if dropped > 0:
        print(f"Dropping {dropped} seeds from {len(label_counts) - len(valid_labels)} "
              f"classes with < {N_FOLDS} members (can't stratify)")

    seed_nodes = seed_nodes[mask]
    seed_labels = seed_labels[mask]
    print(f"Seeds used for CV: {len(seed_nodes)} "
          f"({len(valid_labels)} classes)")

    # 4 Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(seed_nodes, seed_labels)):
        print(f"\n{'─' * 50}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─' * 50}")

        train_nodes = seed_nodes[train_idx]
        test_nodes = seed_nodes[test_idx]

        train_seeds = {n: seeds[n] for n in train_nodes}
        test_seeds = {n: seeds[n] for n in test_nodes}

        print(f"  Train seeds: {len(train_seeds)}")
        print(f"  Held-out seeds: {len(test_seeds)}")

        # Run SSLPA with only train seeds
        print("  Running SSLPA ...")
        predicted_labels = run_sslpa(G, node_id_map, train_seeds, seeds)

        # Collect predictions for held-out nodes
        y_true = []
        y_pred = []
        missed = 0

        for node_id in test_nodes:
            true_label = seeds[node_id]
            pred_label = predicted_labels.get(node_id, "UNKNOWN")

            # Skip held-out nodes that ended up in a community with no
            # train seed (CLUSTER_X) — they're genuinely unreachable
            # We still track them for reporting but exclude from metrics
            # to avoid penalizing the algorithm for coverage gaps
            y_true.append(true_label)
            y_pred.append(pred_label)

            if pred_label.startswith("CLUSTER_"):
                missed += 1

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Metrics on ALL held-out including CLUSTER_ predictions
        fold_metrics_all = compute_metrics(y_true, y_pred)

        # Metrics on only REACHED held-out pred != CLUSTER_X
        reached_mask = np.array([not p.startswith("CLUSTER_") for p in y_pred])
        reached_count = reached_mask.sum()

        if reached_count > 0:
            fold_metrics_reached = compute_metrics(y_true[reached_mask], y_pred[reached_mask])
        else:
            fold_metrics_reached = {k: float("nan") for k in fold_metrics_all}

        # Accuracy on reached nodes (simple match rate)
        if reached_count > 0:
            accuracy_reached = np.mean( y_true[reached_mask] == y_pred[reached_mask])
        else:
            accuracy_reached = float("nan")

        coverage = reached_count / len(test_nodes) if len(test_nodes) > 0 else 0.0

        fold_result = {
            "fold": fold_idx + 1,
            "train_seeds": len(train_seeds),
            "test_seeds": len(test_seeds),
            "reached": int(reached_count),
            "missed_cluster_x": missed,
            "coverage": coverage,
            "accuracy_reached": accuracy_reached,
        }
        # Add "all" metrics
        for k, v in fold_metrics_all.items():
            fold_result[f"{k}_all"] = v
        # Add "reached-only" metrics
        for k, v in fold_metrics_reached.items():
            fold_result[f"{k}_reached"] = v

        all_fold_metrics.append(fold_result)

        print(f"  Reached: {reached_count}/{len(test_nodes)} "
              f"({coverage:.1%}), Missed (CLUSTER_X): {missed}")
        print(f"  Accuracy (reached): {accuracy_reached:.4f}")
        print(f"  [All held-out]     NMI={fold_metrics_all['NMI']:.4f}  "
              f"ARI={fold_metrics_all['ARI']:.4f}  "
              f"Purity={fold_metrics_all['Purity']:.4f}")
        print(f"  [Reached only]     NMI={fold_metrics_reached['NMI']:.4f}  "
              f"ARI={fold_metrics_reached['ARI']:.4f}  "
              f"Purity={fold_metrics_reached['Purity']:.4f}")

        # Per-node detail for this fold
        for i, node_id in enumerate(test_nodes):
            all_fold_details.append({
                "fold": fold_idx + 1,
                "node": node_id,
                "true_label": y_true[i],
                "pred_label": y_pred[i],
                "correct": y_true[i] == y_pred[i],
                "reached": not y_pred[i].startswith("CLUSTER_"),
            })

    # 5 Aggregate results across folds 
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS (mean +- std across folds)")
    print(f"{'=' * 70}")

    df_folds = pd.DataFrame(all_fold_metrics)

    metric_names = ["NMI", "ARI", "AMI", "FMI", "Homogeneity",
                    "Completeness", "V-measure", "Purity"]

    summary_rows = []
    for suffix, label in [("_all", "All held-out"), ("_reached", "Reached only")]:
        print(f"\n  [{label}]")
        for m in metric_names:
            col = f"{m}{suffix}"
            vals = df_folds[col].dropna()
            mean_v = vals.mean()
            std_v = vals.std()
            print(f"    {m:15s}  {mean_v:.4f} +- {std_v:.4f}")
            summary_rows.append({
                "scope": label, "metric": m,
                "mean": mean_v, "std": std_v,
            })

    # Coverage and accuracy
    print(f"\n  Coverage: {df_folds['coverage'].mean():.4f} "
          f"+- {df_folds['coverage'].std():.4f}")
    print(f"  Accuracy (reached):{df_folds['accuracy_reached'].mean():.4f} "
          f"+- {df_folds['accuracy_reached'].std():.4f}")

    # Save outputs 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_folds = os.path.join(OUTPUT_DIR, "sslpa_eval_per_fold.csv")
    df_folds.to_csv(out_folds, index=False)
    print(f"\nSaved per-fold metrics  {out_folds}")

    out_summary = os.path.join(OUTPUT_DIR, "sslpa_eval_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"Saved summary  {out_summary}")

    out_details = os.path.join(OUTPUT_DIR, "sslpa_eval_node_details.csv")
    pd.DataFrame(all_fold_details).to_csv(out_details, index=False)
    print(f"Saved per-node details  {out_details}")

    #7 Confusion matrix for most common classes
    df_details = pd.DataFrame(all_fold_details)
    reached_df = df_details[df_details["reached"]]

    if len(reached_df) > 0:
        top_labels = (reached_df["true_label"]
                      .value_counts()
                      .head(15)
                      .index.tolist())
        top_df = reached_df[reached_df["true_label"].isin(top_labels)]
        confusion = pd.crosstab(
            top_df["true_label"], top_df["pred_label"],
            margins=True
        )
        out_conf = os.path.join(OUTPUT_DIR, "sslpa_eval_confusion_top15.csv")
        confusion.to_csv(out_conf)
        print(f"Saved confusion matrix (top 15 classes)  {out_conf}")

    print("\nDone.")


if __name__ == "__main__":
    main()