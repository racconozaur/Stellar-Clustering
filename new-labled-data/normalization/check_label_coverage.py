"""
Label Coverage Analysis
=======================
Checks how many nodes in the LCC graph have labels in stellar_directory_normalized.csv.

The graph nodes are integer IDs. Labels use Stellar addresses (G...).
The mapping is built from transaction_edges_metadata.csv (sender_id/receiver_id <-> address).
"""

import csv
import pickle
import os
from collections import Counter

# ---- paths ----
BASE = os.path.expanduser("~/stellar-clustering/publication")
GRAPH_PKL   = os.path.join(BASE, "data/LCC/LCC_G_tx_undirected_weighted.pkl")
METADATA    = os.path.join(BASE, "data/transaction_edges_metadata.csv")
LABELS_CSV  = os.path.join(BASE, "new-labled-data/normalization/stellar_directory_normalized.csv")

# ---- 1. load graph ----
print("Loading graph...")
with open(GRAPH_PKL, "rb") as f:
    G = pickle.load(f)

graph_nodes = set(G.nodes())
print(f"  LCC nodes:  {len(graph_nodes):,}")
print(f"  LCC edges:  {G.number_of_edges():,}")

# ---- 2. build id -> address map from metadata ----
print("\nBuilding id <-> address map from metadata CSV...")
id_to_address = {}
with open(METADATA, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for id_col, addr_col in [("sender_id", "sender_address"), ("receiver_id", "receiver_address")]:
            nid = int(row[id_col])
            addr = row[addr_col].strip()
            if nid not in id_to_address and addr:
                id_to_address[nid] = addr

print(f"  Unique node IDs mapped to addresses: {len(id_to_address):,}")

# ---- 3. load labels ----
print("\nLoading labels from stellar_directory_normalized.csv...")
labeled_addresses = {}   # address -> {entity, category, name_normalized}
with open(LABELS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        addr = row["address"].strip()
        labeled_addresses[addr] = {
            "entity": row["entity"],
            "category": row["category"],
            "name_normalized": row["name_normalized"],
        }

print(f"  Total labeled addresses: {len(labeled_addresses):,}")

# ---- 4. coverage analysis ----
print("\nAnalyzing coverage...")

in_graph_with_label = 0
in_graph_no_label   = 0
labeled_not_in_graph = 0

node_labels = {}   # node_id -> label info, for nodes that have a label

for node_id in graph_nodes:
    addr = id_to_address.get(node_id)
    if addr and addr in labeled_addresses:
        in_graph_with_label += 1
        node_labels[node_id] = labeled_addresses[addr]
    else:
        in_graph_no_label += 1

for addr in labeled_addresses:
    # check if this address appears as any graph node
    pass

# labeled addresses that don't appear in graph at all
graph_addresses = {id_to_address[nid] for nid in graph_nodes if nid in id_to_address}
labeled_not_in_graph = sum(1 for a in labeled_addresses if a not in graph_addresses)

total_nodes = len(graph_nodes)
coverage_pct = 100.0 * in_graph_with_label / total_nodes

print("\n" + "=" * 60)
print("COVERAGE SUMMARY")
print("=" * 60)
print(f"Total LCC graph nodes:            {total_nodes:>10,}")
print(f"Nodes WITH a label:               {in_graph_with_label:>10,}  ({coverage_pct:.4f}%)")
print(f"Nodes WITHOUT a label:            {in_graph_no_label:>10,}  ({100-coverage_pct:.4f}%)")
print(f"Labeled addresses NOT in graph:   {labeled_not_in_graph:>10,}")
print(f"Total labeled addresses:          {len(labeled_addresses):>10,}")
print()

# ---- 5. breakdown by category ----
category_counter = Counter()
entity_counter   = Counter()

for info in node_labels.values():
    category_counter[info["category"]] += 1
    entity_counter[info["entity"]] += 1

print("-" * 60)
print("LABELED NODES BY CATEGORY")
print("-" * 60)
for cat, count in category_counter.most_common():
    pct_of_labeled = 100.0 * count / in_graph_with_label if in_graph_with_label else 0
    pct_of_total   = 100.0 * count / total_nodes
    print(f"  {cat:<30s} {count:>6,}  ({pct_of_labeled:5.1f}% of labeled, {pct_of_total:.4f}% of graph)")

print()
print("-" * 60)
print("TOP 30 ENTITIES IN GRAPH")
print("-" * 60)
for ent, count in entity_counter.most_common(30):
    print(f"  {ent:<40s} {count:>6,}")

print()
print("=" * 60)
print(f"Unique entities in graph: {len(entity_counter)}")
print(f"Unique categories in graph: {len(category_counter)}")

# ---- 6. save labeled nodes to CSV ----
OUTPUT_CSV = "labeled_nodes_in_graph.csv"
print(f"\nSaving labeled nodes to {OUTPUT_CSV}...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["node_id", "address", "name_normalized", "entity", "category"])
    writer.writeheader()
    for node_id, info in sorted(node_labels.items()):
        writer.writerow({
            "node_id": node_id,
            "address": id_to_address[node_id],
            "name_normalized": info["name_normalized"],
            "entity": info["entity"],
            "category": info["category"],
        })

print(f"  Saved {len(node_labels):,} rows to {OUTPUT_CSV}")
