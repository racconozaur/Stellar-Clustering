import os
import pickle
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score as NMI,
    adjusted_rand_score as ARI,
    adjusted_mutual_info_score as AMI,
    fowlkes_mallows_score as FMI,
    homogeneity_completeness_v_measure,
)


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def overall_purity(y_true_enc, y_pred_enc):
    df = pd.DataFrame({"true": y_true_enc, "pred": y_pred_enc})
    counts = df.groupby(["pred", "true"]).size().reset_index(name="cnt")
    totals = counts.groupby("pred")["cnt"].sum()
    max_per_pred = counts.groupby("pred")["cnt"].max()
    return float(max_per_pred.sum() / totals.sum())


def sslpa_manual(
    G_nx,
    seeds,
    max_iter=100,
    min_iter=10,
    convergence_threshold=0.001,
    rng=None,
    weight_key="weight",
):

    if rng is None:
        rng = np.random.default_rng(42)

    print(f"[{timestamp()}] Initializing SSLPA with {len(seeds):,} seeds")

    # seeds get their labels, everyone else gets None
    labels = {}
    for node in G_nx.nodes():
        if node in seeds:
            labels[node] = seeds[node]
        else:
            labels[node] = None

    unlabeled_nodes = [n for n in G_nx.nodes() if n not in seeds]
    print(f"[{timestamp()}] Unlabeled nodes: {len(unlabeled_nodes):,}")

    # Find nodes that can receive labels 
    active = set()
    for node in unlabeled_nodes:
        for neighbor in G_nx.neighbors(node):
            if labels[neighbor] is not None:
                active.add(node)
                break

    print(f"[{timestamp()}] Initially active nodes: {len(active):,}")

  
    iterations_run = 0

    # propagation loop
    for iteration in range(max_iter):
        if len(active) == 0:
            print(f"[{timestamp()}] No active nodes left")
            break

        iterations_run = iteration + 1

        active_list = list(active)
        rng.shuffle(active_list)

        changed = 0
        newly_activated = set()
        still_active = set()

        for node in active_list:
            # count votes from labeled neighbors
            neighbor_votes = defaultdict(float)
            for neighbor in G_nx.neighbors(node):
                neighbor_label = labels[neighbor]
                if neighbor_label is not None:
                    edge_weight = G_nx[node][neighbor].get(weight_key, 1.0)
                    neighbor_votes[neighbor_label] += float(edge_weight)

            if len(neighbor_votes) == 0:
                continue 

            # Pick the label with most votes
            new_label = max(neighbor_votes, key=neighbor_votes.get)

            if new_label != labels[node]:
                labels[node] = new_label
                changed += 1

                # When this node gets labeled, its unlabeled neighbors become active
                for nb in G_nx.neighbors(node):
                    if labels[nb] is None and nb not in seeds:
                        newly_activated.add(nb)

            still_active.add(node)


        active = still_active | newly_activated

        change_pct = changed / len(active_list) if len(active_list) > 0 else 0

        # Print progress
        if iteration < 3 or iteration % 20 == 0:
            print(
                f"[{timestamp()}] Iter {iteration+1}: changed={changed:,} "
                f"({change_pct:.4%} of active={len(active_list):,}), "
                f"next_active={len(active):,}"
            )

        # convergence
        if iteration + 1 >= min_iter:
            if changed == 0:
                print(f"[{timestamp()}] Converged (no changes) after {iteration+1} iterations")
                break
            if change_pct < convergence_threshold:
                print(f"[{timestamp()}] Converged (change < threshold) after {iteration+1} iterations")
                break


    num_unlabeled = sum(1 for label in labels.values() if label is None)

    for node in labels:
        if labels[node] is None:
            labels[node] = "UNLABELED"


    return labels, iterations_run, num_unlabeled


def sslpa_cross_validation(
    graph_path,
    labels_path,
    output_dir="cross-validation",
    n_splits=5,
    random_state=42,
    max_iter=100,
    min_iter=10,
    convergence_threshold=0.001,
    weight_key="weight",
):


    print("SSLPA 5-FOLD CROSS-VALIDATION")


    cv_start_time = datetime.now()


    graph_path = os.path.expanduser(graph_path)
    labels_path = os.path.expanduser(labels_path)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(graph_path, "rb") as f:
        G_nx = pickle.load(f)
    print(f"[{timestamp()}] Graph loaded: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

    print(f"[{timestamp()}] Loading labels from {labels_path}")
    labels_df = pd.read_csv(labels_path, usecols=["account_id", "name"])
    labels_df = labels_df.dropna()
    labels_df = labels_df.drop_duplicates(subset=["account_id"])


    sample_node = list(G_nx.nodes())[0]
    if isinstance(sample_node, str):
        labels_df["account_id"] = labels_df["account_id"].astype(str).str.strip()
    else:

        labels_df["account_id"] = pd.to_numeric(labels_df["account_id"], errors="coerce")
        labels_df = labels_df.dropna(subset=["account_id"])
        labels_df["account_id"] = labels_df["account_id"].astype(type(sample_node))

    labels_df["name"] = labels_df["name"].astype(str).str.strip()




    graph_node_set = set(G_nx.nodes())
    labels_df = labels_df[labels_df["account_id"].isin(graph_node_set)].copy()

    print(f"[{timestamp()}] Labeled nodes in graph: {len(labels_df):,}")
    print(f"[{timestamp()}] Unique labels: {labels_df['name'].nunique():,}")



    account_ids = labels_df["account_id"].values
    label_names = labels_df["name"].values

    le_stratify = LabelEncoder()
    y_encoded = le_stratify.fit_transform(label_names)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []

    # Run cv
    for fold, (train_idx, test_idx) in enumerate(skf.split(account_ids, y_encoded), start=1):

        print(f"FOLD {fold}/{n_splits}")


        fold_start = datetime.now()


        # train test
        train_accounts = account_ids[train_idx]
        train_labels = label_names[train_idx]
        test_accounts = account_ids[test_idx]
        test_labels = label_names[test_idx]

        print(f"[{timestamp()}] Train seeds: {len(train_accounts):,}")
        print(f"[{timestamp()}] Test nodes (hidden): {len(test_accounts):,}")


        seeds = {}
        for acc, lab in zip(train_accounts, train_labels):
            seeds[acc] = lab


        fold_rng = np.random.default_rng(random_state + fold)

        
        print(f"[{timestamp()}] Running SSLPA")
        propagated_labels, num_iterations, num_unlabeled = sslpa_manual(
            G_nx,
            seeds,
            max_iter=max_iter,
            min_iter=min_iter,
            convergence_threshold=convergence_threshold,
            rng=fold_rng,
            weight_key=weight_key,
        )

        print(f"[{timestamp()}] Propagation complete: {num_iterations} iterations")
        print(f"[{timestamp()}] Nodes still unlabeled: {num_unlabeled:,}")


        print(f"\n[{timestamp()}] Evaluating on TEST set")

        y_true_test = test_labels
        y_pred_test = []
        for acc in test_accounts:
            pred = propagated_labels.get(acc, "UNLABELED")
            y_pred_test.append(pred)
        y_pred_test = np.array(y_pred_test)

  
        test_mask = y_pred_test != "UNLABELED"
        n_test_labeled = test_mask.sum()
        test_coverage = n_test_labeled / len(test_mask) if len(test_mask) > 0 else 0

        print(f"[{timestamp()}] Test coverage: {test_coverage:.2%} ({n_test_labeled}/{len(test_mask)})")

        if n_test_labeled == 0:
            print(f"[{timestamp()}] No test nodes were labeled, skipping this fold")
            continue



   
        y_true_test_valid = y_true_test[test_mask]
        y_pred_test_valid = y_pred_test[test_mask]

   

        le_test = LabelEncoder()
        all_labels = np.concatenate([y_true_test_valid, y_pred_test_valid])
        le_test.fit(all_labels)

        y_true_enc = le_test.transform(y_true_test_valid)
        y_pred_enc = le_test.transform(y_pred_test_valid)

        # metrics
        nmi_test = NMI(y_true_enc, y_pred_enc)
        ari_test = ARI(y_true_enc, y_pred_enc)
        ami_test = AMI(y_true_enc, y_pred_enc)
        fmi_test = FMI(y_true_enc, y_pred_enc)
        homo_test, comp_test, v_test = homogeneity_completeness_v_measure(y_true_enc, y_pred_enc)
        purity_test = overall_purity(y_true_enc, y_pred_enc)


        print(f"\n[{timestamp()}] Evaluating on TRAIN set")

        y_true_train = train_labels
        y_pred_train = []
        for acc in train_accounts:
            pred = propagated_labels.get(acc, "UNLABELED")
            y_pred_train.append(pred)
        y_pred_train = np.array(y_pred_train)

        train_mask = y_pred_train != "UNLABELED"
        y_true_train_valid = y_true_train[train_mask]
        y_pred_train_valid = y_pred_train[train_mask]

        if len(y_true_train_valid) > 0:
            le_train = LabelEncoder()
            all_train = np.concatenate([y_true_train_valid, y_pred_train_valid])
            le_train.fit(all_train)

            y_true_train_enc = le_train.transform(y_true_train_valid)
            y_pred_train_enc = le_train.transform(y_pred_train_valid)

            nmi_train = NMI(y_true_train_enc, y_pred_train_enc)
            ari_train = ARI(y_true_train_enc, y_pred_train_enc)
            ami_train = AMI(y_true_train_enc, y_pred_train_enc)
            fmi_train = FMI(y_true_train_enc, y_pred_train_enc)
            homo_train, comp_train, v_train = homogeneity_completeness_v_measure(
                y_true_train_enc, y_pred_train_enc
            )
            purity_train = overall_purity(y_true_train_enc, y_pred_train_enc)
        else:
            nmi_train = ari_train = ami_train = fmi_train = np.nan
            homo_train = comp_train = v_train = purity_train = np.nan

        fold_time = (datetime.now() - fold_start).total_seconds()

        print(f"\n[{timestamp()}] Fold {fold} Summary:")
        print(f"Time: {fold_time:.2f}s")
        print(f"Test  - NMI: {nmi_test:.4f}, ARI: {ari_test:.4f}, Purity: {purity_test:.4f}")
        print(f"Train - NMI: {nmi_train:.4f}, ARI: {ari_train:.4f}, Purity: {purity_train:.4f}")


        fold_result = {
            "fold": fold,
            "n_train": len(train_accounts),
            "n_test": len(test_accounts),
            "n_test_labeled": int(n_test_labeled),
            "test_coverage": float(test_coverage),
            "iterations": int(num_iterations),
            "num_unlabeled": int(num_unlabeled),
        }


        fold_result["NMI_train"] = float(nmi_train)
        fold_result["ARI_train"] = float(ari_train)
        fold_result["AMI_train"] = float(ami_train)
        fold_result["FMI_train"] = float(fmi_train)
        fold_result["Homogeneity_train"] = float(homo_train)
        fold_result["Completeness_train"] = float(comp_train)
        fold_result["V-measure_train"] = float(v_train)
        fold_result["Purity_train"] = float(purity_train)


        fold_result["NMI_test"] = float(nmi_test)
        fold_result["ARI_test"] = float(ari_test)
        fold_result["AMI_test"] = float(ami_test)
        fold_result["FMI_test"] = float(fmi_test)
        fold_result["Homogeneity_test"] = float(homo_test)
        fold_result["Completeness_test"] = float(comp_test)
        fold_result["V-measure_test"] = float(v_test)
        fold_result["Purity_test"] = float(purity_test)

        results.append(fold_result)


    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\nNo folds produced valid results")
        return results_df, {}

    # summary stats
    metric_names = ["NMI", "ARI", "AMI", "FMI", "Homogeneity", "Completeness", "V-measure", "Purity"]

    summary = {}
    for metric in metric_names:
        summary[f"Avg_{metric}_train"] = results_df[f"{metric}_train"].mean()
        summary[f"Avg_{metric}_test"] = results_df[f"{metric}_test"].mean()
        summary[f"Std_{metric}_test"] = results_df[f"{metric}_test"].std()

    summary["Avg_test_coverage"] = results_df["test_coverage"].mean()
    summary["Avg_iterations"] = results_df["iterations"].mean()

    total_time = (datetime.now() - cv_start_time).total_seconds()



    print("CROSS-VALIDATION SUMMARY")

    summary_data = []
    for metric in metric_names:
        summary_data.append(
            {"Metric": metric, "Mean": summary[f"Avg_{metric}_test"], "Std": summary[f"Std_{metric}_test"]}
        )

    summary_df = pd.DataFrame(summary_data)
    print("\nTest Set Performance:")
    print(summary_df.to_string(index=False))
    print(f"\nAverage test coverage: {summary['Avg_test_coverage']:.2%}")
    print(f"Average iterations: {summary['Avg_iterations']:.1f}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")





    results_df.to_csv(f"{output_dir}/sslpa_cv_results_per_fold.csv", index=False)
    pd.DataFrame([summary]).to_csv(f"{output_dir}/sslpa_cv_summary.csv", index=False)
    summary_df.to_csv(f"{output_dir}/sslpa_cv_summary_table.csv", index=False)
    print(f"[{timestamp()}] Results saved")

    return results_df, summary


if __name__ == "__main__":
    start_time = datetime.now()


    results_df, summary = sslpa_cross_validation(
        graph_path="~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl",
        labels_path="~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv",
        output_dir="cross-validation-res",
        n_splits=5,
        random_state=42,
        max_iter=100,
        min_iter=10,
        convergence_threshold=0.001,
        weight_key="weight",
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[{timestamp()}] Total execution time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
