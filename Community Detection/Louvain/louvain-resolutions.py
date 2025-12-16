import os
import pickle
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
from datetime import datetime

def timestamp():
    return datetime.now().strftime('%H:%M:%S')


# [0.5, 0.8, 1.0, 1.2]

RESOLUTIONS = [0.8, 1.0, 1.2]
THRESHOLD = 1e-7
SEED = 42
WEIGHT = "weight" 

GRAPH_PATH = os.path.expanduser("~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl")

OUT_DIR = "resolutions"
SUMMARY_CSV = os.path.join(OUT_DIR, "louvain_summary_by_resolution.csv")

start_time = datetime.now()


print(f"[{timestamp()}] Loading graph")
with open(GRAPH_PATH, "rb") as f:
    G_lcc = pickle.load(f)
print(f"[{timestamp()}] LCC nodes: {G_lcc.number_of_nodes():,}")
print(f"[{timestamp()}] LCC edges: {G_lcc.number_of_edges():,}")



def stable_sort_communities(communities):
    def key_fn(c):
        try:
            mn = min(c)
        except ValueError:
            mn = float("inf")
        return (-len(c), mn)
    return sorted(communities, key=key_fn)



#----
summary_rows = []

for RESOLUTION in RESOLUTIONS:
    print(f"\n[{timestamp()}] Running Louvain on resolution={RESOLUTION}, threshold={THRESHOLD}, seed={SEED}, weight='{WEIGHT}'")
    
    step_start = datetime.now()
    communities = louvain_communities(
        G_lcc,
        resolution=RESOLUTION,
        threshold=THRESHOLD,
        seed=SEED,
        weight=WEIGHT
    )
    step_time = (datetime.now() - step_start).total_seconds()
    print(f"[{timestamp()}] Louvain done in {step_time:.2f}s")
    
    # Stable ID assignment
    communities_sorted = stable_sort_communities(communities)
    node_to_community = {}
    for cid, community in enumerate(communities_sorted):
        for node in community:
            node_to_community[node] = cid
    
    # node-community mapping
    df_result = pd.DataFrame({
        "account_id": list(node_to_community.keys()),
        "community": list(node_to_community.values()),
        "resolution": RESOLUTION
    })
    

    out_path = os.path.join(OUT_DIR, f"louvain_result_res{RESOLUTION}.csv")
    df_result.to_csv(out_path, index=False)
    print(f"[{timestamp()}] Saved {out_path} ({len(df_result):,} rows)")
    
    # weighted modularity
    modularity_score = nx.algorithms.community.quality.modularity(
        G_lcc, communities_sorted, weight=WEIGHT
    )
    n_comms = len(communities_sorted)
    print(f"[{timestamp()}] Found {n_comms:,} communities, Modularity = {modularity_score:.6f}")
    
    summary_rows.append({
        "resolution": RESOLUTION,
        "n_communities": n_comms,
        "modularity": modularity_score
    })




#

df_summary = pd.DataFrame(summary_rows).sort_values("resolution")
df_summary.to_csv(SUMMARY_CSV, index=False)
print(f"[{timestamp()}] Saved summary: {SUMMARY_CSV}")
print("SUMMARY")
print(df_summary.to_string(index=False))

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n[{timestamp()}] Total time: {elapsed:.2f}s")
