import os
import pickle
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def timestamp():
    return datetime.now().strftime('%H:%M:%S')


def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_pred, y_true)
    return np.sum(np.amax(contingency_matrix.values, axis=1)) / np.sum(contingency_matrix.values)


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
    
    for node in labels:
        if labels[node] is None:
            labels[node] = "UNLABELED"
    
    return labels, iteration + 1


def compute_intrinsic_metrics(G, labels_dict):


    print(f"[{timestamp()}] Computing intrinsic metrics...")
    
    communities_dict = defaultdict(set)
    labeled_nodes = set()  
    for node, label in labels_dict.items():
        if label != "UNLABELED":
            communities_dict[label].add(node)
            labeled_nodes.add(node)
    
    communities = list(communities_dict.values())
    
    print(f"[{timestamp()}]   Total communities: {len(communities)}")
    print(f"[{timestamp()}]   Labeled nodes: {len(labeled_nodes):,}")
    

    G_labeled = G.subgraph(labeled_nodes).copy()
    print(f"[{timestamp()}]   Labeled subgraph: {G_labeled.number_of_nodes():,} nodes, {G_labeled.number_of_edges():,} edges")
    
    metric_start = datetime.now()
    modularity = nx.algorithms.community.quality.modularity(G_labeled, communities)
    print(f"[{timestamp()}]   Modularity: {modularity:.4f} ({(datetime.now()-metric_start).total_seconds():.2f}s)")
    
    metric_start = datetime.now()
    coverage = nx.community.coverage(G_labeled, communities)
    print(f"[{timestamp()}]   Coverage: {coverage:.4f} ({(datetime.now()-metric_start).total_seconds():.2f}s)")
    

    metric_start = datetime.now()
    multi_node_communities = [c for c in communities if len(c) > 1]
    print(f"[{timestamp()}]   Computing conductance for {len(multi_node_communities)} multi-node communities...")
    
    conductances = []
    for i, comm in enumerate(multi_node_communities):
        try:
            conductances.append(nx.algorithms.cuts.conductance(G, comm))
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"[{timestamp()}]     Processed {i+1}/{len(multi_node_communities)} communities...")
    
    conductance = np.mean(conductances) if conductances else np.nan
    print(f"[{timestamp()}]   Conductance: {conductance:.4f} ({(datetime.now()-metric_start).total_seconds():.2f}s)")
    
    return {
        'modularity': modularity,
        'coverage': coverage,
        'conductance': conductance,
        'n_communities': len(communities),
        'n_singletons': len(communities) - len(multi_node_communities),
        'n_labeled_nodes': len(labeled_nodes),  
        'n_labeled_edges': G_labeled.number_of_edges()
    }


def compute_extrinsic_metrics(y_true, y_pred):

    mask = y_pred != "UNLABELED"
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        return {
            'nmi': np.nan,
            'ari': np.nan,
            'purity': np.nan,
            'test_labeled': 0,
            'test_total': len(y_true)
        }
    
    nmi = normalized_mutual_info_score(y_true_filtered, y_pred_filtered)
    ari = adjusted_rand_score(y_true_filtered, y_pred_filtered)
    purity = purity_score(y_true_filtered, y_pred_filtered)
    
    return {
        'nmi': nmi,
        'ari': ari,
        'purity': purity,
        'test_labeled': len(y_true_filtered),
        'test_total': len(y_true)
    }


def run_sslpa_5fold_cv(graph_path, labels_path, output_prefix, n_splits=5, max_iter=100):

    
    start_time = datetime.now()
    
    graph_path = os.path.expanduser(graph_path)
    labels_path = os.path.expanduser(labels_path)
    output_prefix = os.path.expanduser(output_prefix)
    
    print(f"[{timestamp()}] Loading graph from {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"[{timestamp()}] Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    print(f"[{timestamp()}] Loading labeled data from {labels_path}")
    labels_df = pd.read_csv(labels_path, usecols=["account_id", "name"]).dropna().drop_duplicates()
    
    graph_nodes = set(G.nodes())
    labels_df = labels_df[labels_df["account_id"].isin(graph_nodes)]
    
    print(f"[{timestamp()}] Labeled nodes in graph: {len(labels_df):,}")
    
    labeled_nodes = labels_df["account_id"].values
    labeled_labels = labels_df["name"].values
    

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_fold_results = []
    
    print(f"\n{'='*80}")
    print(f"STARTING STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}\n")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(labeled_nodes, labeled_labels), 1):
        fold_start = datetime.now()
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'='*80}")
        
        train_nodes = labeled_nodes[train_idx]
        train_labels = labeled_labels[train_idx]
        test_nodes = labeled_nodes[test_idx]
        test_labels = labeled_labels[test_idx]
        
        print(f"[{timestamp()}] Train (seeds): {len(train_nodes):,} nodes")
        print(f"[{timestamp()}] Test (held-out): {len(test_nodes):,} nodes")
        
  
        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        print(f"[{timestamp()}] Train classes: {len(train_dist)}, Test classes: {len(test_dist)}")
        
        seeds = dict(zip(train_nodes, train_labels))
        
        # Run SSLPA
        print(f"[{timestamp()}] Running SSLPA...")
        sslpa_start = datetime.now()
        labels_dict, num_iterations = sslpa_manual(G, seeds, max_iter=max_iter)
        sslpa_time = (datetime.now() - sslpa_start).total_seconds()
        print(f"[{timestamp()}] SSLPA completed in {sslpa_time:.2f}s ({num_iterations} iterations)")
        
        intrinsic_metrics = compute_intrinsic_metrics(G, labels_dict)
        
        print(f"[{timestamp()}] Computing extrinsic metrics on test set...")
        test_predictions = np.array([labels_dict[node] for node in test_nodes])
        extrinsic_metrics = compute_extrinsic_metrics(test_labels, test_predictions)
        
        label_counts = Counter(labels_dict.values())
        sizes = [count for label, count in label_counts.items() if label != "UNLABELED"]
        num_unlabeled = label_counts.get("UNLABELED", 0)
        
        fold_results = {
            'fold': fold_idx,
            'train_size': len(train_nodes),
            'test_size': len(test_nodes),
            'train_classes': len(train_dist),
            'test_classes': len(test_dist),
            'iterations': num_iterations,
            'sslpa_time': sslpa_time,
            
            'modularity': intrinsic_metrics['modularity'],
            'coverage': intrinsic_metrics['coverage'],
            'conductance': intrinsic_metrics['conductance'],
            'n_communities': intrinsic_metrics['n_communities'],
            'n_singletons': intrinsic_metrics['n_singletons'],
            
            'nmi': extrinsic_metrics['nmi'],
            'ari': extrinsic_metrics['ari'],
            'purity': extrinsic_metrics['purity'],
            'test_labeled': extrinsic_metrics['test_labeled'],
            'test_labeled_pct': 100 * extrinsic_metrics['test_labeled'] / extrinsic_metrics['test_total'],
            
            'n_labels': len(label_counts),
            'n_unlabeled': num_unlabeled,
            'pct_unlabeled': 100 * num_unlabeled / len(G),
            'mean_size': np.mean(sizes) if sizes else np.nan,
            'median_size': np.median(sizes) if sizes else np.nan,
            'max_size': np.max(sizes) if sizes else np.nan,
            'min_size': np.min(sizes) if sizes else np.nan,
        }
        
        all_fold_results.append(fold_results)
        
        fold_time = (datetime.now() - fold_start).total_seconds()
        
        print(f"FOLD {fold_idx} RESULTS (completed in {fold_time:.2f}s)")


        print(f"Intrinsic metrics:")
        print(f"  Modularity:   {fold_results['modularity']:.4f}")
        print(f"  Conductance:  {fold_results['conductance']:.4f}")
        print(f"  Coverage:     {fold_results['coverage']:.4f}")
        print(f"\nExtrinsic metrics (on held-out test set):")
        print(f"  NMI:          {fold_results['nmi']:.4f}")
        print(f"  ARI:          {fold_results['ari']:.4f}")
        print(f"  Purity:       {fold_results['purity']:.4f}")
        print(f"  Test labeled: {fold_results['test_labeled']}/{fold_results['test_size']} ({fold_results['test_labeled_pct']:.2f}%)")
        print(f"\nClustering info:")
        print(f"  Communities:  {fold_results['n_communities']} (singletons: {fold_results['n_singletons']})")
        print(f"  Unlabeled:    {fold_results['n_unlabeled']:,} ({fold_results['pct_unlabeled']:.2f}%)")
    


    print(f"CROSS-VALIDATION SUMMARY")

    
    df_results = pd.DataFrame(all_fold_results)
    
    avg_results = {
        'method': 'SSLPA_5fold_StratifiedCV',
        'n_folds': n_splits,
        'total_labeled_nodes': len(labeled_nodes),
        
        # Intrinsic metrics
        'modularity_mean': df_results['modularity'].mean(),
        'modularity_std': df_results['modularity'].std(),
        'coverage_mean': df_results['coverage'].mean(),
        'coverage_std': df_results['coverage'].std(),
        'conductance_mean': df_results['conductance'].mean(),
        'conductance_std': df_results['conductance'].std(),
        
        # Extrinsic metrics 
        'nmi_mean': df_results['nmi'].mean(),
        'nmi_std': df_results['nmi'].std(),
        'ari_mean': df_results['ari'].mean(),
        'ari_std': df_results['ari'].std(),
        'purity_mean': df_results['purity'].mean(),
        'purity_std': df_results['purity'].std(),
        

        'avg_iterations': df_results['iterations'].mean(),
        'avg_communities': df_results['n_communities'].mean(),
        'avg_singletons': df_results['n_singletons'].mean(),
        'avg_unlabeled_pct': df_results['pct_unlabeled'].mean(),
        'avg_test_labeled_pct': df_results['test_labeled_pct'].mean(),
    }
    
    print(f"\nIntrinsic metrics (mean ± std):")
    print(f"  Modularity:   {avg_results['modularity_mean']:.4f} ± {avg_results['modularity_std']:.4f}")
    print(f"  Conductance:  {avg_results['conductance_mean']:.4f} ± {avg_results['conductance_std']:.4f}")
    print(f"  Coverage:     {avg_results['coverage_mean']:.4f} ± {avg_results['coverage_std']:.4f}")
    
    print(f"\nExtrinsic metrics (mean ± std):")
    print(f"  NMI:          {avg_results['nmi_mean']:.4f} ± {avg_results['nmi_std']:.4f}")
    print(f"  ARI:          {avg_results['ari_mean']:.4f} ± {avg_results['ari_std']:.4f}")
    print(f"  Purity:       {avg_results['purity_mean']:.4f} ± {avg_results['purity_std']:.4f}")
    
    print(f"\nOther statistics:")
    print(f"  Avg iterations:        {avg_results['avg_iterations']:.1f}")
    print(f"  Avg communities:       {avg_results['avg_communities']:.1f} (singletons: {avg_results['avg_singletons']:.1f})")
    print(f"  Avg unlabeled:         {avg_results['avg_unlabeled_pct']:.2f}%")
    print(f"  Avg test labeled:      {avg_results['avg_test_labeled_pct']:.2f}%")
    

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    

    fold_results_file = f"{output_prefix}_fold_results.csv"
    df_results.to_csv(fold_results_file, index=False)
    print(f"\n[{timestamp()}] Saved fold results to: {fold_results_file}")
    

    summary_file = f"{output_prefix}_cv_summary.csv"
    pd.DataFrame([avg_results]).to_csv(summary_file, index=False)
    print(f"[{timestamp()}] Saved CV summary to: {summary_file}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n[{timestamp()}] Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    return df_results, avg_results


if __name__ == "__main__":
    print(f"[{timestamp()}] Starting SSLPA Stratified 5-Fold Cross-Validation\n")
    
    

    print("\n" + "="*80)
    print("EVALUATING WITH NORMALIZED LABELS")
    print("="*80 + "\n")
    
    df_folds_norm, summary_norm = run_sslpa_5fold_cv(
        graph_path="~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl",
        labels_path="~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv",
        output_prefix="sslpa",
        n_splits=5,
        max_iter=100
    )
    
    print(f"\n[{timestamp()}] All evaluations complete!")