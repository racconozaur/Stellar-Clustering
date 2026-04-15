"""
2_adapters.py
=============
Connects the fixed-test pipeline (helpers 3–9) to the existing
SSLPA runner and NCSF filter implementations.

run_existing_sslpa   — seeds SSLPA from a train-labels CSV, returns predictions
                       for every graph node as a DataFrame (node_id, predicted_label).

apply_existing_ncsf  — applies the Neighborhood-Constrained Scam Filter to a
                       predictions DataFrame and returns a filtered copy.

Both functions are self-contained (load the graph internally on first call via a
module-level cache) so the calling scripts don't need to manage the graph object.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---- shared config ----------------------------------------------------------------

BASE_DIR = Path("/home/user/jfayzullaev/stellar-clustering/publication")

GRAPH_PATH = BASE_DIR / "data" / "LCC" / "LCC_G_tx_undirected_weighted.pkl"

# Label used when SSLPA has no prediction for a node.
# Must match the sentinel checked in 1_common_holdout.py evaluate_binary /
# evaluate_multiclass (both filter with != "UNKNOWN"). Using a different string
# here would cause all unpredicted nodes to be counted as covered, breaking
# coverage and all metrics.
UNKNOWN_LABEL = "UNKNOWN"

# SSLPA convergence settings (mirror ss-lpa-man-full-labels.py).
SSLPA_MAX_ITER = 100
SSLPA_CONVERGENCE_THRESHOLD = 0.001

# ---- graph cache ------------------------------------------------------------------
# Loaded once the first time either adapter function is called.
_G_CACHE = None


def _get_graph():
    global _G_CACHE
    if _G_CACHE is None:
        print(f"[{_ts()}] Loading graph from {GRAPH_PATH} …")
        with open(GRAPH_PATH, "rb") as f:
            _G_CACHE = pickle.load(f)
        print(
            f"[{_ts()}] Graph loaded: "
            f"{_G_CACHE.number_of_nodes():,} nodes, "
            f"{_G_CACHE.number_of_edges():,} edges"
        )
    return _G_CACHE


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---- SSLPA ------------------------------------------------------------------------

def _run_sslpa_on_graph(G, seeds: dict) -> dict:
    """
    Weighted semi-supervised label propagation (from ss-lpa-man-full-labels.py).

    Parameters
    ----------
    G     : networkx Graph
    seeds : {node_id (int or str) -> label (str)}

    Returns
    -------
    labels : {node_id_str -> label_str}
             UNKNOWN_LABEL for nodes that received no label.
    """
    # Normalise seed keys to int (graph nodes are ints).
    seeds_int = {int(k): v for k, v in seeds.items()}

    labels: dict = {node: seeds_int.get(node, None) for node in G.nodes()}
    unlabeled = [n for n in G.nodes() if n not in seeds_int]

    print(f"[{_ts()}]   seeds={len(seeds_int):,}  unlabeled={len(unlabeled):,}")

    for iteration in range(SSLPA_MAX_ITER):
        changed = 0
        np.random.shuffle(unlabeled)

        for node in unlabeled:
            votes: dict[str, float] = defaultdict(float)
            for nb in G.neighbors(node):
                lbl = labels[nb]
                if lbl is not None:
                    weight = G[node][nb].get("weight", 1.0)
                    votes[lbl] += weight

            if votes:
                best = max(votes, key=votes.__getitem__)
                if best != labels[node]:
                    labels[node] = best
                    changed += 1

        pct = changed / len(unlabeled) if unlabeled else 0.0
        print(f"[{_ts()}]   iter {iteration + 1:3d}: {changed:,} changed ({pct:.4%})")

        if pct < SSLPA_CONVERGENCE_THRESHOLD:
            print(f"[{_ts()}]   Converged at iteration {iteration + 1}")
            break
    else:
        print(f"[{_ts()}]   Reached max iterations ({SSLPA_MAX_ITER})")

    # Fill None -> UNKNOWN
    for node in labels:
        if labels[node] is None:
            labels[node] = UNKNOWN_LABEL

    return {str(k): v for k, v in labels.items()}


def run_existing_sslpa(train_labels_csv: Path, mode: str) -> pd.DataFrame:
    """
    Run SSLPA seeded from train_labels_csv.

    The CSV must have columns: node_id, label
      - node_id : integer graph node ID
      - label   : name_normalized value (e.g. "SCAM", "Binance", …)
        These are the values that SSLPA propagates. Binary and multiclass
        modes use the same propagation; 'mode' is accepted for interface
        consistency but does not change the SSLPA run itself.

    Returns
    -------
    DataFrame with columns: node_id (str), predicted_label (str)
    One row per graph node; UNKNOWN_LABEL for nodes with no prediction.
    """
    print(f"[{_ts()}] run_existing_sslpa: loading seeds from {train_labels_csv}")
    seeds_df = pd.read_csv(train_labels_csv)

    # The split files written by 4_prepare_dev_repeats.py have a 'label' column.
    if "label" not in seeds_df.columns:
        raise KeyError(
            f"Expected column 'label' in {train_labels_csv}, "
            f"got: {list(seeds_df.columns)}"
        )
    if "node_id" not in seeds_df.columns:
        raise KeyError(
            f"Expected column 'node_id' in {train_labels_csv}, "
            f"got: {list(seeds_df.columns)}"
        )

    seeds_df = seeds_df.dropna(subset=["node_id", "label"]).drop_duplicates(subset=["node_id"])
    seeds_all = dict(zip(seeds_df["node_id"].astype(int), seeds_df["label"].astype(str)))

    # Filter to only seeds whose node_id is actually in the graph
    # (mirrors ss-lpa-man-full-labels.py line 81).
    G = _get_graph()
    nx_nodes = set(G.nodes())
    seeds = {n: lbl for n, lbl in seeds_all.items() if n in nx_nodes}
    print(f"[{_ts()}] Seeds: {len(seeds):,} in graph out of {len(seeds_all):,} total")
    print(f"[{_ts()}] Running SSLPA (mode={mode}) …")
    label_map = _run_sslpa_on_graph(G, seeds)

    result = pd.DataFrame(
        [{"node_id": nid, "predicted_label": lbl} for nid, lbl in label_map.items()]
    )
    print(
        f"[{_ts()}] SSLPA done: "
        f"{(result['predicted_label'] != UNKNOWN_LABEL).sum():,} nodes labeled, "
        f"{(result['predicted_label'] == UNKNOWN_LABEL).sum():,} unknown"
    )
    return result


# ---- NCSF -------------------------------------------------------------------------

def apply_existing_ncsf(
    pred_df: pd.DataFrame,
    d_min: int,
    r_min: float,
    mode: str,
) -> pd.DataFrame:
    """
    Apply the Neighborhood-Constrained Scam Filter (from step1/ncsf_and_label_stats.py).

    For every labeled node (predicted_label != UNKNOWN_LABEL):
      - if degree in the graph < d_min, OR
      - fraction of neighbors with the same label < r_min
    → set predicted_label = UNKNOWN_LABEL.

    The filter is applied to ALL labels (not just SCAM), matching the
    'all_labels' mode of the original ncsf_filter() function.  The
    'mode' parameter is accepted for interface consistency.

    Parameters
    ----------
    pred_df : DataFrame with columns node_id (str), predicted_label (str)
    d_min   : minimum degree threshold
    r_min   : minimum same-label neighbor ratio threshold
    mode    : "binary" | "multiclass" (unused internally, kept for API consistency)

    Returns
    -------
    Filtered copy of pred_df with the same columns.
    """
    G = _get_graph()

    # Build fast lookup: node_id_str -> label
    label_map: dict[str, str] = dict(
        zip(pred_df["node_id"].astype(str), pred_df["predicted_label"].astype(str))
    )

    out_map = label_map.copy()
    changed = 0

    for node_str, lbl in label_map.items():
        if lbl == UNKNOWN_LABEL:
            continue

        # Graph nodes are stored as ints; convert for lookup.
        try:
            node_int = int(node_str)
        except ValueError:
            continue

        if node_int not in G:
            out_map[node_str] = UNKNOWN_LABEL
            changed += 1
            continue

        neighbors = list(G.neighbors(node_int))
        deg = len(neighbors)

        if deg < d_min:
            out_map[node_str] = UNKNOWN_LABEL
            changed += 1
            continue

        same = sum(
            1 for nb in neighbors
            if label_map.get(str(nb), UNKNOWN_LABEL) == lbl
        )

        if same / deg < r_min:
            out_map[node_str] = UNKNOWN_LABEL
            changed += 1

    print(
        f"[{_ts()}] NCSF (d_min={d_min}, r_min={r_min}): "
        f"removed {changed:,} labels"
    )

    result = pred_df.copy()
    result["predicted_label"] = result["node_id"].astype(str).map(out_map)
    return result
