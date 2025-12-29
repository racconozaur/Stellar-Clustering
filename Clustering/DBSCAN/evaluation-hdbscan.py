import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score



CLUSTERING_RESULTS = "LINE_res/tx_line_hdbscan_cosine_pca64.csv" 
NODE_ID_COLUMN = "node_id"  # "account_id"  "node_id"

HDBSCAN_COLUMN = "hdbscan_mcs20"  
METHOD_NAME = "HDBSCAN_LINE_mcs50"  
OUTPUT_FILE = "LINE_res/evaluation/hdbscan_node2vec_mcs50_evaluation.csv"


def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(np.array(y_pred), np.array(y_true))
    return np.sum(np.amax(contingency_matrix.values, axis=1)) / np.sum(contingency_matrix.values)

start_time = time.time()

LABELS_CSV = "~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv"
FULL_GRAPH = os.path.expanduser("~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl")

print("Loading graph...")
load_start = time.time()
with open(FULL_GRAPH, 'rb') as f:
    G = pickle.load(f)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time()-load_start:.2f}s)")

print("\nLoading ground truth labels...")
load_start = time.time()
labels_df = pd.read_csv(os.path.expanduser(LABELS_CSV))
print(f"Loaded {len(labels_df)} labeled nodes ({time.time()-load_start:.2f}s)")

print(f"\nLoading HDBSCAN results from {HDBSCAN_COLUMN}...")
load_start = time.time()
hdbscan_df = pd.read_csv(CLUSTERING_RESULTS)
print(f"Loaded {len(hdbscan_df)} node assignments ({time.time()-load_start:.2f}s)")

print("\nExtracting cluster assignments...")

cluster_data = hdbscan_df[[NODE_ID_COLUMN, HDBSCAN_COLUMN]].copy()
cluster_data.columns = ['node_id', 'cluster_id']

# Count noise points
noise_count = len(cluster_data[cluster_data['cluster_id'] == -1])
print(f"Noise points (label -1): {noise_count} ({noise_count/len(cluster_data)*100:.2f}%)")

print("\nConverting to communities (excluding noise)...")
convert_start = time.time()

cluster_data_no_noise = cluster_data[cluster_data['cluster_id'] != -1].copy()
communities_dict = cluster_data_no_noise.groupby('cluster_id')['node_id'].apply(set).to_dict()
hdbscan_communities = list(communities_dict.values())
print(f"Found {len(hdbscan_communities)} communities (excluding noise) ({time.time()-convert_start:.2f}s)")

sizes = [len(comm) for comm in hdbscan_communities]
results = {
    'method': METHOD_NAME,
    'n_clusters': len(hdbscan_communities),
    'n_noise': noise_count,
    'noise_ratio': noise_count / len(cluster_data),
    'mean_size': np.mean(sizes) if sizes else 0,
    'max_size': np.max(sizes) if sizes else 0,
    'min_size': np.min(sizes) if sizes else 0,
    'median_size': np.median(sizes) if sizes else 0,
    'std_size': np.std(sizes) if sizes else 0
}

print("\nComputing intrinsic metrics (excluding noise points)...")
if len(hdbscan_communities) > 0:
    clustered_nodes_set = set()
    for comm in hdbscan_communities:
        clustered_nodes_set.update(comm)
    
    print(f"  Creating subgraph with {len(clustered_nodes_set)} clustered nodes...")
    metric_start = time.time()
    G_clustered = G.subgraph(clustered_nodes_set).copy()
    print(f"  Subgraph created: {G_clustered.number_of_nodes()} nodes, {G_clustered.number_of_edges()} edges ({time.time()-metric_start:.2f}s)")
    
    metric_start = time.time()
    results['modularity'] = nx.community.modularity(G_clustered, hdbscan_communities)
    print(f"  Modularity: {results['modularity']:.4f} ({time.time()-metric_start:.2f}s)")

    metric_start = time.time()
    results['coverage'] = nx.community.coverage(G_clustered, hdbscan_communities)
    print(f"  Coverage: {results['coverage']:.4f} ({time.time()-metric_start:.2f}s)")

    print(f"  Computing conductance for {len(hdbscan_communities)} communities...")
    metric_start = time.time()
    conductances = []
    for i, comm in enumerate(hdbscan_communities):
        if len(comm) > 1:
            try:
                conductances.append(nx.cuts.conductance(G_clustered, comm))
            except:
                continue
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(hdbscan_communities)} communities...")
            
    results['conductance'] = np.mean(conductances) if conductances else np.nan
    print(f"  Conductance: {results['conductance']:.4f} ({time.time()-metric_start:.2f}s)")
else:
    print("  No clusters found (all noise)!")
    results['modularity'] = np.nan
    results['coverage'] = np.nan
    results['conductance'] = np.nan




print("\nComputing extrinsic metrics (including noise as cluster -1)...")
metric_start = time.time()

node_to_cluster = dict(zip(cluster_data['node_id'], cluster_data['cluster_id']))

labeled_nodes = set(labels_df.iloc[:, 0].values)
clustered_nodes = set(node_to_cluster.keys())
common_nodes = labeled_nodes & clustered_nodes

print(f"  Labeled nodes: {len(labeled_nodes)}")
print(f"  Common nodes for evaluation: {len(common_nodes)}")
print(f"  Coverage: {len(common_nodes)/len(labeled_nodes)*100:.2f}%")

if len(common_nodes) > 0:
    y_true = []
    y_pred = []
    for node in common_nodes:
        true_label = labels_df[labels_df.iloc[:, 0] == node].iloc[0, 1]
        y_true.append(true_label)
        y_pred.append(node_to_cluster[node])
    
    results['nmi'] = normalized_mutual_info_score(y_true, y_pred)
    results['ari'] = adjusted_rand_score(y_true, y_pred)
    results['purity'] = purity_score(y_true, y_pred)
    results['labeled_coverage'] = len(common_nodes) / len(labeled_nodes)
    
    print(f"  NMI:     {results['nmi']:.4f}")
    print(f"  ARI:     {results['ari']:.4f}")
    print(f"  Purity:  {results['purity']:.4f}")
else:
    results['nmi'] = np.nan
    results['ari'] = np.nan
    results['purity'] = np.nan
    results['labeled_coverage'] = 0.0
    print(f"  No common nodes found!")

print(f"  Extrinsic metrics computed ({time.time()-metric_start:.2f}s)")

print("\n" + "="*60)
print(f"{METHOD_NAME} EVALUATION RESULTS")
print("="*60)
print(f"Number of clusters: {results['n_clusters']}")
print(f"Noise points:       {results['n_noise']} ({results['noise_ratio']*100:.2f}%)")
print(f"\nIntrinsic Metrics (excluding noise):")
print(f"  Modularity:         {results['modularity']:.4f}")
print(f"  Conductance:        {results['conductance']:.4f}")
print(f"  Coverage:           {results['coverage']:.4f}")
print(f"\nExtrinsic Metrics (noise as separate cluster):")
print(f"  NMI:                {results['nmi']:.4f}")
print(f"  ARI:                {results['ari']:.4f}")
print(f"  Purity:             {results['purity']:.4f}")
print(f"  Labeled Coverage:   {results['labeled_coverage']:.4f}")
print(f"\nCluster statistics (excluding noise):")
print(f"  Mean size:   {results['mean_size']:.1f}")
print(f"  Median size: {results['median_size']:.1f}")
print(f"  Max size:    {results['max_size']}")
print(f"  Min size:    {results['min_size']}")
print(f"  Std size:    {results['std_size']:.1f}")

pd.DataFrame([results]).to_csv(OUTPUT_FILE, index=False)
print(f"\nResults saved to '{OUTPUT_FILE}'")
print(f"\nTotal time: {time.time()-start_time:.2f}s")