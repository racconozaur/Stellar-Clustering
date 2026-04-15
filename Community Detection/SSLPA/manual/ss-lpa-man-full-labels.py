import os
import pickle
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


def timestamp():
    return datetime.now().strftime('%H:%M:%S')


def sslpa_manual(G_nx, seeds, max_iter=100, convergence_threshold=0.001):

    print(f"[{timestamp()}] Initializing SSLPA with {len(seeds):,} seeds")
    
    labels = {node: seeds.get(node, None) for node in G_nx.nodes()}
    unlabeled_nodes = [n for n in G_nx.nodes() if n not in seeds]
    
    print(f"[{timestamp()}] Unlabeled nodes: {len(unlabeled_nodes):,}")
    
    for iteration in range(max_iter):
        changed = 0
        
        np.random.shuffle(unlabeled_nodes)
        
        for node in unlabeled_nodes:

            neighbor_votes = defaultdict(float)
            
            for neighbor in G_nx.neighbors(node):
                if labels[neighbor] is not None:
                    weight = G_nx[node][neighbor].get('weight', 1.0)
                    neighbor_votes[labels[neighbor]] += weight
            
            if neighbor_votes:
                new_label = max(neighbor_votes, key=neighbor_votes.get)
                
                if new_label != labels[node]:
                    labels[node] = new_label
                    changed += 1
        
        change_pct = changed / len(unlabeled_nodes) if unlabeled_nodes else 0
        print(f"[{timestamp()}] Iteration {iteration+1}: {changed:,} nodes changed ({change_pct:.4%})")
        
        if change_pct < convergence_threshold:
            print(f"[{timestamp()}] Converged after {iteration+1} iterations")
            break
    else:
        print(f"[{timestamp()}] Reached max iterations ({max_iter})")
    
    num_unlabeled = sum(1 for v in labels.values() if v is None)
    for node in labels:
        if labels[node] is None:
            labels[node] = "UNLABELED"
    
    return labels, iteration + 1, num_unlabeled


def run_sslpa_manual(graph_path, LABELS, output_prefix):
    

    start_time = datetime.now()
    
    graph_path = os.path.expanduser(graph_path)
    LABELS = os.path.expanduser(LABELS)
    output_prefix = os.path.expanduser(output_prefix)

    print(f"[{timestamp()}] Loading graph")
    with open(graph_path, "rb") as f:
        G_nx = pickle.load(f)
    print(f"[{timestamp()}] Graph loaded: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

    print(f"[{timestamp()}] Loading seed labels")
    seeds_df = pd.read_csv(LABELS, usecols=["node_id", "name_normalized"]).dropna().drop_duplicates()
    seeds_all = dict(zip(seeds_df["node_id"].tolist(), seeds_df["name_normalized"].tolist()))
    
    # Filter seeds present in graph
    nx_nodes = set(G_nx.nodes())
    seeds = {n: lbl for n, lbl in seeds_all.items() if n in nx_nodes}
    print(f"[{timestamp()}] Seeds: {len(seeds):,} in graph out of {len(seeds_all):,} total")

    # Run SSLPA
    print(f"[{timestamp()}] Starting label propagation")
    lpa_start = datetime.now()
    
    labels, num_iterations, num_unlabeled = sslpa_manual(G_nx, seeds, max_iter=100)
    
    lpa_time = (datetime.now() - lpa_start).total_seconds()
    print(f"[{timestamp()}] SSLPA completed in {lpa_time:.2f}s")
    
    # Calculate statistics
    print(f"[{timestamp()}] Calculating statistics")
    label_counts = Counter(labels.values())
    sizes = list(label_counts.values())
    
    num_labeled = len(G_nx) - num_unlabeled
    
    print("\nLABEL PROPAGATION RESULTS")
    print(f"Total nodes: {len(G_nx):,}")
    print(f"Seeds: {len(seeds):,}")
    print(f"Successfully labeled: {num_labeled:,} ({100*num_labeled/len(G_nx):.2f}%)")
    print(f"Unlabeled: {num_unlabeled:,} ({100*num_unlabeled/len(G_nx):.2f}%)")
    print(f"Unique labels: {len(label_counts):,}")
    print(f"Iterations: {num_iterations}")
    
    print(f"\nLabel size stats:")
    print(f"Min: {min(sizes):,}")
    print(f"Max: {max(sizes):,}")
    print(f"Mean: {sum(sizes)/len(sizes):.2f}")
    print(f"Median: {sorted(sizes)[len(sizes)//2]:,}")
    
    print(f"\nTop 15 labels:")
    for label, count in label_counts.most_common(15):
        print(f"  {label}: {count:,} ({100*count/len(G_nx):.2f}%)")
    

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    

    print(f"\n[{timestamp()}] Saving results")
    df_labels = pd.DataFrame(list(labels.items()), columns=["node", "label"])
    out_labels = f"{output_prefix}_manual_labels.csv"
    df_labels.to_csv(out_labels, index=False)
    print(f"[{timestamp()}] Saved: {out_labels}")
    

    stats = {
        "graph": graph_path,
        "nodes": len(G_nx),
        "edges": G_nx.number_of_edges(),
        "num_labels": len(label_counts),
        "num_seeds": len(seeds),
        "num_labeled_final": num_labeled,
        "num_unlabeled": num_unlabeled,
        "pct_labeled": 100 * num_labeled / len(G_nx),
        "iterations": num_iterations,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes) / len(sizes),
        "median_size": sorted(sizes)[len(sizes) // 2],
        "seeds_csv": LABELS,
        "method": "manual_sslpa"
    }
    out_stats = f"{output_prefix}_manual_stats.csv"
    pd.DataFrame([stats]).to_csv(out_stats, index=False)
    print(f"[{timestamp()}] Saved: {out_stats}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"[{timestamp()}] Total time: {total_time:.2f}s")
    
    return labels, stats


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{timestamp()}] Starting SSLPA")
    

    labels, stats = run_sslpa_manual(
        "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl",
        "~/stellar-clustering/publication/new-labled-data/normalization/labeled_nodes_in_graph.csv",
        "normalized/sslpa_tx_lcc",
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[{timestamp()}] Total execution time: {elapsed:.2f}s")