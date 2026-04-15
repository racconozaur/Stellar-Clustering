import os
import pickle
from collections import defaultdict, Counter
from statistics import median

import igraph as ig
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABELS = os.path.expanduser(
    "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv"
)


def run_sslpa_on_graph(graph_path, output_prefix):
    print(f"\nRunning SSLPA on {graph_path}")
    graph_path = os.path.expanduser(graph_path)
    output_prefix = os.path.expanduser(output_prefix)

    # 1 Load graph
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path, "rb") as f:
        G_nx = pickle.load(f)

    if isinstance(G_nx, nx.DiGraph):
        G_nx = nx.Graph(G_nx)

    print(f"Graph loaded: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")

    # 2 Convert to igraph
    G = ig.Graph.from_networkx(G_nx)
    print(f"Converted to igraph: {G.vcount()} vertices, {G.ecount()} edges")

    # 3 Load seeds and filter to nodes present in graph
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

    seeds_all: dict = dict(zip(seeds_df["node_id"].tolist(), seeds_df["name_normalized"].tolist()))

    nx_nodes = set(G_nx.nodes())
    seeds: dict = {n: lbl for n, lbl in seeds_all.items() if n in nx_nodes}
    print(f"Seeds: {len(seeds_all)} total, {len(seeds)} present in graph")

 
    #4 Encode string labels to integers for igraph

    le = LabelEncoder()
    unique_labels = sorted(set(seeds.values()))
    le.fit(unique_labels)
    num_seed_classes = len(unique_labels)

    # Seed string label to integer in [0, K-1]
    seed_label_to_int: dict = {
        lbl: int(le.transform([lbl])[0]) for lbl in unique_labels
    }
    int_to_seed_label: dict = {v: k for k, v in seed_label_to_int.items()}

   

    # 5 Build initial-label and fixed-mask arrays
  
    initial_labels: list[int] = []
    fixed_mask: list[bool] = []
    node_id_map: dict[int, object] = {} 

    next_unlabeled_id = num_seed_classes

    for i, v in enumerate(G.vs):
        node_id = v["_nx_name"]
        node_id_map[i] = node_id

        if node_id in seeds:
            initial_labels.append(seed_label_to_int[seeds[node_id]])
            fixed_mask.append(True)
        else:
            # Each unlabeled node starts as its own singleton community
            initial_labels.append(next_unlabeled_id)
            next_unlabeled_id += 1
            fixed_mask.append(False)

    print(f"Fixed seed nodes: {sum(fixed_mask)}")
    print(f"Seed classes: {num_seed_classes}, "
          f"total initial communities (seeds + singletons): {next_unlabeled_id}")

 
    # 6 Run igraph semi-supervised label propagation
    print("Running igraph community_label_propagation …")
    communities = G.community_label_propagation(
        weights="weight" if G.is_weighted() else None,
        initial=initial_labels,
        fixed=fixed_mask,
    )
    membership: list[int] = communities.membership


    #7 majority vote mapping  igraph community id to seed label

    comm_label_votes: dict[int, Counter] = defaultdict(Counter)
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in seeds:
            comm_label_votes[comm_id][seeds[node_id]] += 1

    comm_to_label: dict[int, str] = {}
    for comm_id, votes in comm_label_votes.items():
        comm_to_label[comm_id] = votes.most_common(1)[0][0]

    # 8 Assign final labels to every node 
    labels: dict = {}
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        if node_id in seeds:
            labels[node_id] = seeds[node_id]
        elif comm_id in comm_to_label:
            labels[node_id] = comm_to_label[comm_id]
        else:
            labels[node_id] = f"CLUSTER_{comm_id}"

    # 9 Compute community stats on the igraph partition
    comm_sizes = Counter(membership)
    sizes = list(comm_sizes.values())
    num_communities = len(comm_sizes)

    mean_size = sum(sizes) / len(sizes)
    median_size = median(sizes)  # correct median for even-length lists

    try:
        mod = G.modularity(membership, weights="weight" if G.is_weighted() else None)
    except Exception as e:
        print(f"Warning: Could not compute modularity: {e}")
        mod = float("nan")

    print(f"Communities (igraph partition): {num_communities}")
    print(f"Community size — min={min(sizes)}, max={max(sizes)}, "
          f"mean={mean_size:.2f}, median={median_size}")
    print(f"Modularity: {mod:.4f}")

    groups: dict[str, set] = defaultdict(set)
    for node, lbl in labels.items():
        groups[lbl].add(node)
    print(f"Distinct label groups (after majority-vote mapping): {len(groups)}")



    # 10 Persist outputs
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    df_labels = pd.DataFrame(list(labels.items()), columns=["node", "label"])
    out_labels = f"{output_prefix}_sslpa_labels.csv"
    df_labels.to_csv(out_labels, index=False)
    print(f"Saved node label mapping → {out_labels}")

    df_comm = pd.DataFrame(
        [(node_id_map[i], membership[i]) for i in range(len(membership))],
        columns=["node", "community"],
    )
    out_comm = f"{output_prefix}_lpa_communities.csv"
    df_comm.to_csv(out_comm, index=False)
    print(f"Saved community partition → {out_comm}")

    stats = {
        "graph": graph_path,
        "nodes": G.vcount(),
        "edges": G.ecount(),
        "num_igraph_communities": num_communities,
        "num_label_groups": len(groups),
        "modularity": mod,
        "min_comm_size": min(sizes),
        "max_comm_size": max(sizes),
        "mean_comm_size": mean_size,
        "median_comm_size": median_size,
        "num_frozen_seeds": sum(fixed_mask),
        "seeds_csv": LABELS,
        "method": "igraph_sslpa_majority_vote",
    }
    out_stats = f"{output_prefix}_sslpa_stats.csv"
    pd.DataFrame([stats]).to_csv(out_stats, index=False)
    print(f"Saved summary stats → {out_stats}")

    return communities, labels, stats


# Entry point
if __name__ == "__main__":
    run_sslpa_on_graph(
        "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl",
        "sslpa-igr-out/full/sslpa_tx_lcc",
    )