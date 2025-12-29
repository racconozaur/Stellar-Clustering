import os
import pickle
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


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def overall_purity(y_true_enc: np.ndarray, y_pred_enc: np.ndarray) -> float:
    if len(y_true_enc) == 0:
        return np.nan
    df = pd.DataFrame({"true": y_true_enc, "pred": y_pred_enc})
    counts = df.groupby(["pred", "true"]).size().reset_index(name="cnt")
    totals = counts.groupby("pred")["cnt"].sum()
    max_per_pred = counts.groupby("pred")["cnt"].max()
    return float(max_per_pred.sum() / totals.sum())


def sslpa_manual(
    G_nx,
    seeds: dict,
    max_iter: int = 100,
    min_iter: int = 10,
    convergence_threshold: float = 0.001,
    rng=None,
    weight_key: str = "weight",
):

    if rng is None:
        rng = np.random.default_rng(42)

    print(f"[{timestamp()}] Initializing SSLPA with {len(seeds):,} seeds")

    labels = {node: seeds.get(node, None) for node in G_nx.nodes()}

    unlabeled_nodes = [n for n in G_nx.nodes() if n not in seeds]
    print(f"[{timestamp()}] Unlabeled nodes: {len(unlabeled_nodes):,}")

    active = set()
    for node in unlabeled_nodes:
        for neighbor in G_nx.neighbors(node):
            if labels[neighbor] is not None:
                active.add(node)
                break
    print(f"[{timestamp()}] Initially active nodes: {len(active):,}")

    iterations_run = 0

    for iteration in range(max_iter):
        if not active:
            print(f"[{timestamp()}] No active nodes left")
            break

        iterations_run = iteration + 1
        active_list = list(active)
        rng.shuffle(active_list)

        changed = 0
        newly_activated = set()
        still_active = set()

        for node in active_list:
            neighbor_votes = defaultdict(float)

            for neighbor in G_nx.neighbors(node):
                neighbor_label = labels[neighbor]
                if neighbor_label is not None:
                    w = G_nx[node][neighbor].get(weight_key, 1.0)
                    neighbor_votes[neighbor_label] += float(w)

            if not neighbor_votes:
                continue

            new_label = max(neighbor_votes, key=neighbor_votes.get)

            if new_label != labels[node]:
                labels[node] = new_label
                changed += 1

                for nb in G_nx.neighbors(node):
                    if labels[nb] is None and nb not in seeds:
                        newly_activated.add(nb)

            still_active.add(node)

        active = still_active | newly_activated
        change_pct = changed / len(active_list) if active_list else 0.0

        if iteration < 3 or iteration % 20 == 0:
            print(
                f"[{timestamp()}] Iter {iteration+1}: changed={changed:,} "
                f"({change_pct:.4%} of active={len(active_list):,}), "
                f"next_active={len(active):,}"
            )

        if iteration + 1 >= min_iter:
            if changed == 0:
                print(f"[{timestamp()}] Converged (no changes) after {iteration+1} iterations")
                break
            if change_pct < convergence_threshold:
                print(f"[{timestamp()}] Converged (change < threshold) after {iteration+1} iterations")
                break

    num_unlabeled = sum(1 for v in labels.values() if v is None)

    for node in labels:
        if labels[node] is None:
            labels[node] = "UNLABELED"

    return labels, iterations_run, num_unlabeled


def sslpa_cross_validation(
    graph_path: str,
    labels_path: str,
    output_dir: str = "cross-validation",
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 100,
    min_iter: int = 10,
    convergence_threshold: float = 0.001,
    weight_key: str = "weight",
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
    labels_df = pd.read_csv(labels_path, usecols=["account_id", "name"]).dropna()
    labels_df = labels_df.drop_duplicates(subset=["account_id"])
    labels_df["name"] = labels_df["name"].astype(str).str.strip()

    sample_node = next(iter(G_nx.nodes()))
    if isinstance(sample_node, str):
        labels_df["account_id"] = labels_df["account_id"].astype(str).str.strip()
    else:
        labels_df["account_id"] = pd.to_numeric(labels_df["account_id"], errors="coerce")
        labels_df = labels_df.dropna(subset=["account_id"])
        labels_df["account_id"] = labels_df["account_id"].astype(type(sample_node))

    graph_nodes = set(G_nx.nodes())
    labels_df = labels_df[labels_df["account_id"].isin(graph_nodes)].copy()

    print(f"[{timestamp()}] Labeled nodes in graph: {len(labels_df):,}")
    print(f"[{timestamp()}] Unique labels: {labels_df['name'].nunique():,}")

    account_ids = labels_df["account_id"].values
    label_names = labels_df["name"].values

    le_stratify = LabelEncoder()
    y_encoded = le_stratify.fit_transform(label_names)
    min_count = pd.Series(y_encoded).value_counts().min()
    if min_count < n_splits:
        raise ValueError(
            f"Some labels appear only {min_count} times, cannot run StratifiedKFold(n_splits={n_splits}). "
            f"Reduce n_splits or filter rare labels."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(account_ids, y_encoded), start=1):
        print(f"\nFOLD {fold}/{n_splits}")
        fold_start = datetime.now()

        train_accounts = account_ids[train_idx]
        train_labels = label_names[train_idx]
        test_accounts = account_ids[test_idx]
        test_labels = label_names[test_idx]

        print(f"[{timestamp()}] Train seeds: {len(train_accounts):,}")
        print(f"[{timestamp()}] Test nodes (hidden): {len(test_accounts):,}")

        seeds = {acc: lab for acc, lab in zip(train_accounts, train_labels)}
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
        print(f"[{timestamp()}] Nodes still UNLABELED in full graph: {num_unlabeled:,}")

        #-------------------------

        print(f"[{timestamp()}] Evaluating on TEST set")

        y_true_test = test_labels.astype(str)
        y_pred_test = np.array([propagated_labels.get(acc, "UNLABELED") for acc in test_accounts], dtype=object)

        test_mask = (y_pred_test != "UNLABELED")
        n_test_labeled = int(test_mask.sum())
        n_test_total = int(len(test_accounts))
        n_test_unlabeled = int(n_test_total - n_test_labeled)
        test_coverage = (n_test_labeled / n_test_total) if n_test_total else 0.0

        print(f"[{timestamp()}] Test coverage: {test_coverage:.2%} ({n_test_labeled}/{n_test_total})")

        if n_test_labeled == 0:
            nmi_test = ari_test = ami_test = fmi_test = np.nan
            homo_test = comp_test = v_test = purity_test = np.nan
        else:
            y_true_test_valid = y_true_test[test_mask]
            y_pred_test_valid = y_pred_test[test_mask]

            le_test = LabelEncoder()
            le_test.fit(np.concatenate([y_true_test_valid, y_pred_test_valid]))

            y_true_enc = le_test.transform(y_true_test_valid)
            y_pred_enc = le_test.transform(y_pred_test_valid)

            nmi_test = NMI(y_true_enc, y_pred_enc)
            ari_test = ARI(y_true_enc, y_pred_enc)
            ami_test = AMI(y_true_enc, y_pred_enc)
            fmi_test = FMI(y_true_enc, y_pred_enc)
            homo_test, comp_test, v_test = homogeneity_completeness_v_measure(y_true_enc, y_pred_enc)
            purity_test = overall_purity(y_true_enc, y_pred_enc)




        print(f"[{timestamp()}] Evaluating on TRAIN set (sanity check)")

        y_true_train = train_labels.astype(str)
        y_pred_train = np.array([propagated_labels.get(acc, "UNLABELED") for acc in train_accounts], dtype=object)

        train_mask = (y_pred_train != "UNLABELED")
        if int(train_mask.sum()) == 0:
            nmi_train = ari_train = ami_train = fmi_train = np.nan
            homo_train = comp_train = v_train = purity_train = np.nan
        else:
            y_true_train_valid = y_true_train[train_mask]
            y_pred_train_valid = y_pred_train[train_mask]

            le_train = LabelEncoder()
            le_train.fit(np.concatenate([y_true_train_valid, y_pred_train_valid]))

            y_true_train_enc = le_train.transform(y_true_train_valid)
            y_pred_train_enc = le_train.transform(y_pred_train_valid)

            nmi_train = NMI(y_true_train_enc, y_pred_train_enc)
            ari_train = ARI(y_true_train_enc, y_pred_train_enc)
            ami_train = AMI(y_true_train_enc, y_pred_train_enc)
            fmi_train = FMI(y_true_train_enc, y_pred_train_enc)
            homo_train, comp_train, v_train = homogeneity_completeness_v_measure(y_true_train_enc, y_pred_train_enc)
            purity_train = overall_purity(y_true_train_enc, y_pred_train_enc)

        fold_time = (datetime.now() - fold_start).total_seconds()
        print(f"[{timestamp()}] Fold {fold} time: {fold_time:.2f}s")
        print(f"[{timestamp()}] TEST  NMI={nmi_test:.4f} ARI={ari_test:.4f} AMI={ami_test:.4f} FMI={fmi_test:.4f} Coverage={test_coverage:.2%}")

        fold_result = {
            "fold": fold,
            "n_train": int(len(train_accounts)),
            "n_test": int(len(test_accounts)),
            "n_test_total": n_test_total,
            "n_test_labeled": n_test_labeled,
            "n_test_unlabeled": n_test_unlabeled,
            "test_coverage": float(test_coverage),
            "iterations": int(num_iterations),
            "num_unlabeled_fullgraph": int(num_unlabeled),

            "NMI_train": float(nmi_train),
            "ARI_train": float(ari_train),
            "AMI_train": float(ami_train),
            "FMI_train": float(fmi_train),
            "Homogeneity_train": float(homo_train),
            "Completeness_train": float(comp_train),
            "V-measure_train": float(v_train),
            "Purity_train": float(purity_train),

            "NMI_test": float(nmi_test),
            "ARI_test": float(ari_test),
            "AMI_test": float(ami_test),
            "FMI_test": float(fmi_test),
            "Homogeneity_test": float(homo_test),
            "Completeness_test": float(comp_test),
            "V-measure_test": float(v_test),
            "Purity_test": float(purity_test),
        }

        results.append(fold_result)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("\nNo folds produced valid results.")
        return results_df, {}

    metric_names = ["NMI", "ARI", "AMI", "FMI", "Homogeneity", "Completeness", "V-measure", "Purity"]
    summary = {}

    for metric in metric_names:
        summary[f"Avg_{metric}_train"] = float(np.nanmean(results_df[f"{metric}_train"].values))
        summary[f"Avg_{metric}_test"] = float(np.nanmean(results_df[f"{metric}_test"].values))
        summary[f"Std_{metric}_test"] = float(np.nanstd(results_df[f"{metric}_test"].values))

    summary["Avg_test_coverage"] = float(np.nanmean(results_df["test_coverage"].values))
    summary["Std_test_coverage"] = float(np.nanstd(results_df["test_coverage"].values))
    summary["Avg_iterations"] = float(np.nanmean(results_df["iterations"].values))

    total_time = (datetime.now() - cv_start_time).total_seconds()

    print("\nCROSS-VALIDATION SUMMARY (TEST)")
    summary_rows = [{"Metric": m, "Mean": summary[f"Avg_{m}_test"], "Std": summary[f"Std_{m}_test"]} for m in metric_names]
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print(f"\nAvg test coverage: {summary['Avg_test_coverage']:.2%} ± {summary['Std_test_coverage']:.2%}")
    print(f"Avg iterations: {summary['Avg_iterations']:.1f}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")

    # Save
    results_df.to_csv(f"{output_dir}/sslpa_cv_results_per_fold.csv", index=False)
    pd.DataFrame([summary]).to_csv(f"{output_dir}/sslpa_cv_summary.csv", index=False)
    summary_df.to_csv(f"{output_dir}/sslpa_cv_summary_table.csv", index=False)

    print(f"[{timestamp()}] Results saved to: {output_dir}")
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
