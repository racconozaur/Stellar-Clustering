import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def purity_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    contingency_matrix = pd.crosstab(y_pred, y_true)
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

print("\nLoading Louvain results...")
load_start = time.time()
louvain_df = pd.read_csv('../louvain_result_res0.5.csv')
print(f"Loaded {len(louvain_df)} node assignments ({time.time()-load_start:.2f}s)")

print("\nConverting to communities...")
convert_start = time.time()
communities_dict = louvain_df.groupby(louvain_df.columns[1])[louvain_df.columns[0]].apply(set).to_dict()
louvain_communities = list(communities_dict.values())
print(f"Found {len(louvain_communities)} communities ({time.time()-convert_start:.2f}s)")

sizes = [len(comm) for comm in louvain_communities]
results = {
    'method': 'Louvain_res0.5',
    'n_clusters': len(louvain_communities),
    'mean_size': np.mean(sizes),
    'max_size': np.max(sizes),
    'min_size': np.min(sizes),
    'median_size': np.median(sizes),
    'std_size': np.std(sizes)
}

print("\nComputing intrinsic metrics...")
metric_start = time.time()
results['modularity'] = nx.community.modularity(G, louvain_communities)
print(f"  Modularity: {results['modularity']:.4f} ({time.time()-metric_start:.2f}s)")

metric_start = time.time()
results['coverage'] = nx.community.coverage(G, louvain_communities)
print(f"  Coverage: {results['coverage']:.4f} ({time.time()-metric_start:.2f}s)")

print(f"  Computing conductance for {len(louvain_communities)} communities...")
metric_start = time.time()
conductances = []
for i, comm in enumerate(louvain_communities):
    if len(comm) > 1:
        try:
            conductances.append(nx.cuts.conductance(G, comm))
        except:
            continue
    if (i + 1) % 10 == 0:
        print(f"    Processed {i+1}/{len(louvain_communities)} communities...")
        
results['conductance'] = np.mean(conductances) if conductances else np.nan
print(f"  Conductance: {results['conductance']:.4f} ({time.time()-metric_start:.2f}s)")

print("\nComputing extrinsic metrics...")
metric_start = time.time()

node_to_cluster = {}
for cluster_id, comm in enumerate(louvain_communities):
    for node in comm:
        node_to_cluster[node] = cluster_id

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

print("LOUVAIN EVALUATION RESULTS")


print(f"Number of clusters: {results['n_clusters']}")
print(f"\nIntrinsic Metrics:")
print(f"  Modularity:         {results['modularity']:.4f}")
print(f"  Conductance:        {results['conductance']:.4f}")
print(f"  Coverage:           {results['coverage']:.4f}")
print(f"\nExtrinsic Metrics:")
print(f"  NMI:                {results['nmi']:.4f}")
print(f"  ARI:                {results['ari']:.4f}")
print(f"  Purity:             {results['purity']:.4f}")
print(f"  Labeled Coverage:   {results['labeled_coverage']:.4f}")
print(f"\nCluster statistics:")
print(f"  Mean size:   {results['mean_size']:.1f}")
print(f"  Median size: {results['median_size']:.1f}")
print(f"  Max size:    {results['max_size']}")
print(f"  Min size:    {results['min_size']}")
print(f"  Std size:    {results['std_size']:.1f}")

pd.DataFrame([results]).to_csv('louvain_evaluation_res0.5.csv', index=False)
print(f"\nResults saved to 'louvain_evaluation_res0.5.csv'")
print(f"\nTotal time: {time.time()-start_time:.2f}s")