import os
import pickle
from collections import defaultdict, Counter

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as skmetrics

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
LABELS = os.path.expanduser(
    "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv"
)
GRAPH_PATH = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)
OUTPUT_DIR = os.path.expanduser(
    "sslpa-igr-out/eval_novel_discovery"
)

# -- Evaluation parameters --
MIN_ENTITY_SIZE = 2       # include entities with >=2 nodes (1-node entities
                          # can't fragment, so concentration is trivially 1.0)
N_REMOVE = 10             # remove this many entities per round (fixed count)
N_ROUNDS = 5
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Purity
# --------------------------------------------------------------------------
def purity_score(y_true, y_pred):
    contingency = skmetrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


# --------------------------------------------------------------------------
# Run SSLPA
# --------------------------------------------------------------------------
def run_sslpa(G: ig.Graph, node_id_map: dict, train_seeds: dict) -> tuple:
    le = LabelEncoder()
    unique_labels = sorted(set(train_seeds.values()))
    le.fit(unique_labels)
    num_seed_entities = len(unique_labels)
    seed_label_to_int = {lbl: int(le.transform([lbl])[0])
                         for lbl in unique_labels}

    initial_labels = []
    fixed_mask = []
    next_unlabeled_id = num_seed_entities

    for i in range(G.vcount()):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            initial_labels.append(seed_label_to_int[train_seeds[node_id]])
            fixed_mask.append(True)
        else:
            initial_labels.append(next_unlabeled_id)
            next_unlabeled_id += 1
            fixed_mask.append(False)

    communities = G.community_label_propagation(
        weights="weight" if G.is_weighted() else None,
        initial=initial_labels,
        fixed=fixed_mask,
    )
    membership = communities.membership

    # Majority-vote mapping using train seeds only
    comm_label_votes = defaultdict(Counter)
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            comm_label_votes[comm_id][train_seeds[node_id]] += 1

    comm_to_label = {}
    for comm_id, votes in comm_label_votes.items():
        comm_to_label[comm_id] = votes.most_common(1)[0][0]

    labels = {}
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in train_seeds:
            labels[node_id] = train_seeds[node_id]
        elif comm_id in comm_to_label:
            labels[node_id] = comm_to_label[comm_id]
        else:
            labels[node_id] = f"CLUSTER_{comm_id}"

    return labels, membership


# --------------------------------------------------------------------------
# Compute clustering metrics (requires >=2 unique true entities)
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    all_labels = sorted(set(y_true) | set(y_pred))
    label_map = {lbl: idx for idx, lbl in enumerate(all_labels)}
    yt = np.array([label_map[l] for l in y_true])
    yp = np.array([label_map[l] for l in y_pred])

    n_true_entities = len(set(yt))

    # Guard: most metrics are undefined or trivially 0/1 with <2 entities
    if n_true_entities < 2:
        return {m: float("nan") for m in [
            "NMI", "ARI", "AMI", "FMI",
            "Homogeneity", "Completeness", "V-measure", "Purity"
        ]}

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


# --------------------------------------------------------------------------
# Per-entity discovery analysis
# --------------------------------------------------------------------------
def analyze_discovery(removed_entity_nodes: dict, predicted_labels: dict,
                      membership: list, node_id_map: dict):
    node_to_idx = {v: k for k, v in node_id_map.items()}

    per_entity = []
    for true_label, node_ids in removed_entity_nodes.items():
        comm_assignments = []
        pred_labels = []
        for nid in node_ids:
            if nid in node_to_idx:
                idx = node_to_idx[nid]
                comm_assignments.append(membership[idx])
            pred_labels.append(predicted_labels.get(nid, "UNKNOWN"))

        comm_counter = Counter(comm_assignments)
        pred_counter = Counter(pred_labels)

        dominant_comm, dominant_count = comm_counter.most_common(1)[0]
        concentration = dominant_count / len(node_ids) if node_ids else 0

        named_count = sum(1 for p in pred_labels
                          if not p.startswith("CLUSTER_"))
        cluster_x_count = sum(1 for p in pred_labels
                              if p.startswith("CLUSTER_"))

        per_entity.append({
            "true_label": true_label,
            "num_nodes": len(node_ids),
            "num_communities": len(comm_counter),
            "dominant_comm_id": dominant_comm,
            "dominant_comm_size": dominant_count,
            "concentration": concentration,
            "named_predictions": named_count,
            "cluster_x_predictions": cluster_x_count,
            "top_predicted_labels": dict(pred_counter.most_common(5)),
        })

    return per_entity


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SSLPA Novel Entity Discovery Evaluation (v2)")
    print("=" * 70)

    # -- 1. Load graph --
    print(f"\nLoading graph: {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        G_nx = pickle.load(f)
    if isinstance(G_nx, nx.DiGraph):
        G_nx = nx.Graph(G_nx)

    G = ig.Graph.from_networkx(G_nx)
    print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")

    node_id_map = {i: G.vs[i]["_nx_name"] for i in range(G.vcount())}

    # -- 2. Load seeds --
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

    # -- 3. Group by entity label --
    entity_to_nodes = defaultdict(list)
    for node_id, label in seeds.items():
        entity_to_nodes[label].append(node_id)

    # Show full entity size distribution
    print(f"\nTotal unique entities: {len(entity_to_nodes)}")
    entity_sizes = {lbl: len(nodes) for lbl, nodes in entity_to_nodes.items()}
    size_dist = Counter()
    for s in entity_sizes.values():
        if s == 1:
            size_dist["1 node"] += 1
        elif s <= 4:
            size_dist["2-4 nodes"] += 1
        elif s <= 10:
            size_dist["5-10 nodes"] += 1
        elif s <= 50:
            size_dist["11-50 nodes"] += 1
        elif s <= 100:
            size_dist["51-100 nodes"] += 1
        else:
            size_dist[">100 nodes"] += 1
    print("Entity size distribution:")
    for bucket in ["1 node", "2-4 nodes", "5-10 nodes",
                   "11-50 nodes", "51-100 nodes", ">100 nodes"]:
        if bucket in size_dist:
            print(f"  {bucket:15s}: {size_dist[bucket]} entities")

    # Eligible = entities with >= MIN_ENTITY_SIZE nodes
    eligible_entities = sorted(
        [lbl for lbl, nodes in entity_to_nodes.items()
         if len(nodes) >= MIN_ENTITY_SIZE],
        key=lambda x: -entity_sizes[x]
    )
    print(f"\nEligible for removal (>= {MIN_ENTITY_SIZE} nodes): "
          f"{len(eligible_entities)} entities")
    print("Eligible entities:")
    for lbl in eligible_entities:
        print(f"  {lbl:35s}  {entity_sizes[lbl]:>5} nodes")

    # Adjust N_REMOVE if needed
    n_remove = min(N_REMOVE, len(eligible_entities) - 1)
    # Keep at least 2 entities so metrics work, and leave some for training
    if n_remove < 2:
        print(f"\nERROR: Need at least 2 removable entities but only "
              f"{len(eligible_entities)} eligible. Lower MIN_ENTITY_SIZE or "
              f"add more seeds.")
        return
    print(f"\nRemoving {n_remove} entities per round")

    # -- 4. Run multiple rounds --
    rng = np.random.RandomState(RANDOM_STATE)
    all_round_metrics = []
    all_per_entity_details = []

    for round_idx in range(N_ROUNDS):
        print(f"\n{'-' * 60}")
        print(f"Round {round_idx + 1}/{N_ROUNDS}")
        print(f"{'-' * 60}")

        # Randomly select entities to remove
        removed_entities = list(rng.choice(eligible_entities, size=n_remove,
                                           replace=False))
        removed_set = set(removed_entities)

        removed_entity_nodes = {lbl: entity_to_nodes[lbl]
                                for lbl in removed_entities}
        removed_node_set = set()
        for nodes in removed_entity_nodes.values():
            removed_node_set.update(nodes)

        train_seeds = {n: lbl for n, lbl in seeds.items()
                       if lbl not in removed_set}

        print(f"  Removed {n_remove} entities:")
        for lbl in removed_entities:
            print(f"    - {lbl} ({entity_sizes[lbl]} nodes)")
        print(f"  Total removed nodes: {len(removed_node_set)}")
        print(f"  Remaining train seeds: {len(train_seeds)} "
              f"({len(set(train_seeds.values()))} entities)")

        # Run SSLPA
        print("  Running SSLPA ...")
        predicted_labels, membership = run_sslpa(G, node_id_map, train_seeds)

        # -- Evaluate --
        y_true = []
        y_pred = []
        y_pred_comm = []

        node_to_idx = {v: k for k, v in node_id_map.items()}

        for lbl in removed_entities:
            for nid in entity_to_nodes[lbl]:
                y_true.append(lbl)
                y_pred.append(predicted_labels.get(nid, "UNKNOWN"))
                if nid in node_to_idx:
                    y_pred_comm.append(membership[node_to_idx[nid]])
                else:
                    y_pred_comm.append(-1)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_comm = np.array(y_pred_comm)

        # Community-based metrics
        metrics_comm = compute_metrics(
            y_true.tolist(),
            [str(c) for c in y_pred_comm.tolist()]
        )

        # Mapped-label metrics
        metrics_mapped = compute_metrics(y_true.tolist(), y_pred.tolist())

        # Per-entity discovery stats
        per_entity_stats = analyze_discovery(
            removed_entity_nodes, predicted_labels, membership, node_id_map
        )

        concentrations = [pe["concentration"] for pe in per_entity_stats]
        mean_concentration = np.mean(concentrations)

        discovered_50 = sum(1 for c in concentrations if c >= 0.50)
        discovered_80 = sum(1 for c in concentrations if c >= 0.80)

        named_wrong = sum(
            1 for t, p in zip(y_true, y_pred)
            if not p.startswith("CLUSTER_") and p != t
        )
        cluster_x = sum(1 for p in y_pred if p.startswith("CLUSTER_"))

        round_result = {
            "round": round_idx + 1,
            "n_removed_entities": n_remove,
            "n_removed_nodes": len(removed_node_set),
            "n_train_seeds": len(train_seeds),
            "mean_concentration": mean_concentration,
            "discovery_rate_50pct": discovered_50 / n_remove,
            "discovery_rate_80pct": discovered_80 / n_remove,
            "cluster_x_count": cluster_x,
            "named_wrong_count": named_wrong,
            "cluster_x_frac": cluster_x / len(y_true) if len(y_true) > 0 else 0,
            "named_wrong_frac": named_wrong / len(y_true) if len(y_true) > 0 else 0,
        }

        for k, v in metrics_comm.items():
            round_result[f"{k}_community"] = v
        for k, v in metrics_mapped.items():
            round_result[f"{k}_mapped"] = v

        all_round_metrics.append(round_result)

        for pe in per_entity_stats:
            pe["round"] = round_idx + 1
            pe["top_predicted_labels"] = str(pe["top_predicted_labels"])
            all_per_entity_details.append(pe)

        print(f"\n  Results:")
        print(f"    Mean concentration: {mean_concentration:.4f}")
        print(f"    Discovery (>50%): {discovered_50}/{n_remove} "
              f"({discovered_50/n_remove:.1%})")
        print(f"    Discovery (>80%): {discovered_80}/{n_remove} "
              f"({discovered_80/n_remove:.1%})")
        print(f"    CLUSTER_X: {cluster_x}/{len(y_true)} "
              f"({cluster_x/len(y_true):.1%})")
        print(f"    Wrong named: {named_wrong}/{len(y_true)} "
              f"({named_wrong/len(y_true):.1%})")

        if not np.isnan(metrics_comm.get("NMI", float("nan"))):
            print(f"    [Community] NMI={metrics_comm['NMI']:.4f}  "
                  f"ARI={metrics_comm['ARI']:.4f}  "
                  f"V-measure={metrics_comm['V-measure']:.4f}  "
                  f"Purity={metrics_comm['Purity']:.4f}")
        else:
            print(f"    [Community] Metrics=NaN (only 1 true entity in "
                  f"this round -- should not happen with n_remove>=2)")

        # Per-entity breakdown
        print(f"\n    Per-entity:")
        for pe in per_entity_stats:
            print(f"      {pe['true_label']:30s}  nodes={pe['num_nodes']:>5}  "
                  f"conc={pe['concentration']:.2f}  "
                  f"comms={pe['num_communities']}  "
                  f"named={pe['named_predictions']}  "
                  f"cluster_x={pe['cluster_x_predictions']}")

    # -- 5. Aggregate --
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS (mean +/- std across rounds)")
    print(f"{'=' * 70}")

    df_rounds = pd.DataFrame(all_round_metrics)

    metric_names = ["NMI", "ARI", "AMI", "FMI", "Homogeneity",
                    "Completeness", "V-measure", "Purity"]

    summary_rows = []

    for suffix, label in [("_community", "Community-based"),
                          ("_mapped", "Mapped-label")]:
        print(f"\n  [{label}]")
        for m in metric_names:
            col = f"{m}{suffix}"
            vals = df_rounds[col].dropna()
            if len(vals) > 0:
                mean_v = vals.mean()
                std_v = vals.std()
                print(f"    {m:15s}  {mean_v:.4f} +/- {std_v:.4f}")
            else:
                mean_v = std_v = float("nan")
                print(f"    {m:15s}  N/A (all rounds had <2 true entities)")
            summary_rows.append({"scope": label, "metric": m,
                                 "mean": mean_v, "std": std_v})

    print(f"\n  Concentration:         "
          f"{df_rounds['mean_concentration'].mean():.4f} "
          f"+/- {df_rounds['mean_concentration'].std():.4f}")
    print(f"  Discovery (>50%):      "
          f"{df_rounds['discovery_rate_50pct'].mean():.1%} "
          f"+/- {df_rounds['discovery_rate_50pct'].std():.1%}")
    print(f"  Discovery (>80%):      "
          f"{df_rounds['discovery_rate_80pct'].mean():.1%} "
          f"+/- {df_rounds['discovery_rate_80pct'].std():.1%}")
    print(f"  CLUSTER_X fraction:    "
          f"{df_rounds['cluster_x_frac'].mean():.1%} "
          f"+/- {df_rounds['cluster_x_frac'].std():.1%}")
    print(f"  Wrong-named fraction:  "
          f"{df_rounds['named_wrong_frac'].mean():.1%} "
          f"+/- {df_rounds['named_wrong_frac'].std():.1%}")

    # -- 6. Save --
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_rounds = os.path.join(OUTPUT_DIR, "novel_discovery_per_round.csv")
    df_rounds.to_csv(out_rounds, index=False)
    print(f"\nSaved per-round: {out_rounds}")

    out_summary = os.path.join(OUTPUT_DIR, "novel_discovery_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")

    out_entities = os.path.join(OUTPUT_DIR, "novel_discovery_per_entity.csv")
    pd.DataFrame(all_per_entity_details).to_csv(out_entities, index=False)
    print(f"Saved per-entity: {out_entities}")

    print("\nDone.")


if __name__ == "__main__":
    main()