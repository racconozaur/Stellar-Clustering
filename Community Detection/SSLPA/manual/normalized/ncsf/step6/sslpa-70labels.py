"""
sslpa-70labels.py

Run SSLPA with semi-supervised training labels (70% train, 30% test).

This script:
1. Loads the graph from pickle
2. Loads only the train labels (70%) from step6 splits
3. Runs SSLPA treating test nodes as unlabeled
4. Saves predictions for ALL nodes to CSV for evaluation

This bridges step6 (creating splits) and step7 (evaluation).
"""

import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import networkx as nx


# ----------------------------- CONFIG ---------------------------------

BASE_DIR = Path("/home/user/jfayzullaev/stellar-clustering/publication")

# MODE: Must match what you used in step6
MODE = "scam_only"  # Options: "scam_only" or "all_labels"

# Graph file
GRAPH_PATH = BASE_DIR / "data" / "LCC" / "LCC_G_tx_undirected_weighted.pkl"

# Semi-supervised splits directory (from step6)
NCSF_DIR = BASE_DIR / "Community Detection" / "SSLPA" / "manual" / "normalized" / "ncsf"
STEP6_DIR = NCSF_DIR / "step6"
SEMI_BASE_DIR = STEP6_DIR / "semi_supervised" / MODE
STEP1_DIR = NCSF_DIR / "step1" / MODE

# Mask fractions to process
MASK_FRACTIONS = [0.1, 0.2, 0.3, 0.5]  # Must match step6 settings

# SSLPA parameters
MAX_ITER = 100
CONVERGENCE_THRESHOLD = 0.001


# ----------------------------- FUNCTIONS ------------------------------

def timestamp():
    return datetime.now().strftime('%H:%M:%S')


def sslpa_manual(G_nx, seeds, max_iter=100, convergence_threshold=0.001):
    """
    Run SSLPA label propagation algorithm.

    Parameters:
    -----------
    G_nx : networkx.Graph
        The graph to propagate labels on
    seeds : dict
        Dictionary mapping node -> label for seed nodes
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Stop when change percentage falls below this

    Returns:
    --------
    labels : dict
        Dictionary mapping all nodes to predicted labels
    num_iterations : int
        Number of iterations completed
    """
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

    # Mark any remaining unlabeled nodes
    for node in labels:
        if labels[node] is None:
            labels[node] = "UNLABELED"

    return labels, iteration + 1


def run_sslpa_for_mask(file_base: str, mask_fraction: float, G):
    """Run SSLPA for a single file and mask fraction and save predictions."""
    pct = int(mask_fraction * 100)
    split_dir = SEMI_BASE_DIR / file_base / f"mask{pct}"

    train_path = split_dir / "sslpa_train_labels.csv"
    output_path = split_dir / "sslpa_predictions.csv"

    if not train_path.exists():
        print(f"[ERROR] Train labels file not found: {train_path}")
        print(f"  → Run step6/sslpa_semi_supervised_eval.py first to create splits")
        return False

    print(f"\n{'='*80}")
    print(f"FILE: {file_base} | MASK FRACTION: {mask_fraction:.0%}")
    print(f"{'='*80}\n")

    # Load train labels (seeds only)
    print(f"[{timestamp()}] Loading train labels from {train_path}")
    train_df = pd.read_csv(train_path)

    # Validate columns
    if "node" not in train_df.columns or "label" not in train_df.columns:
        raise KeyError(
            f"Expected columns 'node' and 'label' in {train_path.name}, "
            f"but got: {list(train_df.columns)}"
        )

    # Create seeds dictionary
    seeds = dict(zip(train_df["node"].values, train_df["label"].values))
    print(f"[{timestamp()}] Loaded {len(seeds):,} seed labels")

    # Run SSLPA
    print(f"[{timestamp()}] Running SSLPA...")
    sslpa_start = datetime.now()
    labels_dict, num_iterations = sslpa_manual(
        G,
        seeds,
        max_iter=MAX_ITER,
        convergence_threshold=CONVERGENCE_THRESHOLD
    )
    sslpa_time = (datetime.now() - sslpa_start).total_seconds()
    print(f"[{timestamp()}] SSLPA completed in {sslpa_time:.2f}s ({num_iterations} iterations)")

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame([
        {"node": node, "label": label}
        for node, label in labels_dict.items()
    ])

    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    print(f"[{timestamp()}] Saved predictions to: {output_path}")

    # Summary statistics
    label_counts = predictions_df["label"].value_counts()
    n_unlabeled = label_counts.get("UNLABELED", 0)
    n_labeled = len(predictions_df) - n_unlabeled

    print(f"\nPrediction Summary:")
    print(f"  Total nodes:     {len(predictions_df):,}")
    print(f"  Labeled:         {n_labeled:,} ({100*n_labeled/len(predictions_df):.2f}%)")
    print(f"  Unlabeled:       {n_unlabeled:,} ({100*n_unlabeled/len(predictions_df):.2f}%)")
    print(f"  Unique labels:   {len(label_counts)}")

    return True


def main():
    if MODE not in ("scam_only", "all_labels"):
        raise ValueError(
            f"Invalid MODE: {MODE}\n"
            "MODE must be either 'scam_only' or 'all_labels'."
        )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_PATH}\n"
            "→ Check that the graph pickle file exists at this path."
        )

    if not SEMI_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Semi-supervised splits directory not found: {SEMI_BASE_DIR}\n"
            f"→ Run step6/sslpa_semi_supervised_eval.py first with MODE='{MODE}'"
        )

    # Find all parameter directories (e.g., d2_r0.4, d3_r0.5, d5_r0.6)
    param_dirs = sorted([d for d in SEMI_BASE_DIR.iterdir() if d.is_dir()])

    if not param_dirs:
        raise FileNotFoundError(
            f"No parameter directories found in {SEMI_BASE_DIR}\n"
            f"→ Run step6/sslpa_semi_supervised_eval.py first with MODE='{MODE}'"
        )

    print(f"Running in MODE: {MODE}")
    print(f"Semi-supervised splits directory: {SEMI_BASE_DIR}")
    print(f"\nFound {len(param_dirs)} parameter configuration(s) to process:")
    for d in param_dirs:
        print(f"  - {d.name}")

    # Load graph once (shared across all runs)
    print(f"\n[{timestamp()}] Loading graph from {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"[{timestamp()}] Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    start_time = datetime.now()
    total_runs = 0
    successful_runs = 0

    # Process each parameter configuration
    for param_dir in param_dirs:
        file_base = param_dir.name
        print(f"\n{'#'*80}")
        print(f"# Processing parameter configuration: {file_base}")
        print(f"{'#'*80}")

        # Process each mask fraction
        for frac in MASK_FRACTIONS:
            total_runs += 1
            success = run_sslpa_for_mask(file_base, frac, G)
            if success:
                successful_runs += 1
            else:
                print(f"\n[WARNING] Failed to process {file_base} with mask fraction {frac:.0%}")

    total_time = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"MODE: {MODE}")
    print(f"Parameter configurations processed: {len(param_dirs)}")
    print(f"Mask fractions: {MASK_FRACTIONS}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_runs - successful_runs}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"{'='*80}")
    print("DONE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
