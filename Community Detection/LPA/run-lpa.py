import os
import pickle
import networkx as nx
import pandas as pd
from networkx.algorithms.community import asyn_lpa_communities, modularity
from datetime import datetime

def timestamp():
    return datetime.now().strftime('%H:%M:%S')

def run_lpa_on_graph(graph_path, output_prefix, seed=42):
    
    print(f"\n[{timestamp()}] Running LPA on {graph_path}")
    step_start = datetime.now()
    

    print(f"[{timestamp()}] Loading graph")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"[{timestamp()}] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Run LPA
    print(f"[{timestamp()}] Running LPA")
    lpa_start = datetime.now()
    comms = list(asyn_lpa_communities(G, weight="weight", seed=seed))
    lpa_time = (datetime.now() - lpa_start).total_seconds()
    print(f"[{timestamp()}] LPA done in {lpa_time:.2f}s")
    
    # node to community mapping
    node2cid = {node: i for i, comm in enumerate(comms) for node in comm}
    print(f"[{timestamp()}] LPA found {len(comms):,} communities")
    
    # statistics
    sizes = [len(c) for c in comms]
    mod = modularity(G, comms, weight="weight")
    
    print(f"Community size stats:")
    print(f"min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.2f}, median={sorted(sizes)[len(sizes)//2]}")
    print(f"Modularity: {mod:.4f}")
    

    print(f"[{timestamp()}] Saving")
    df = pd.DataFrame(list(node2cid.items()), columns=["node", "community"])
    csv_path = f"{output_prefix}_lpa_communities.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{timestamp()}] Saved node-community mapping: {csv_path} ({len(df):,} rows)")
    

    stats = {
        "graph": graph_path,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "num_communities": len(comms),
        "modularity": mod,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes)/len(sizes),
        "median_size": sorted(sizes)[len(sizes)//2],
    }
    stats_df = pd.DataFrame([stats])
    stats_path = f"{output_prefix}_lpa_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[{timestamp()}] Saved stats: {stats_path}")
    
    total_time = (datetime.now() - step_start).total_seconds()
    print(f"[{timestamp()}] Total time: {total_time:.2f}s")
    
    return comms, node2cid, stats


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{timestamp()}] Starting LPA")

    graph_path = os.path.expanduser("~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl")
    
    comms, node2cid, stats = run_lpa_on_graph(
        graph_path=graph_path,
        output_prefix="lpa_tx_lcc",
        seed=42
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[{timestamp()}] Total execution time: {elapsed:.2f}s")
