import os
import pickle
import numpy as np
import pandas as pd

GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)
with open(GRAPH_PKL, "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

def load_line_embeddings(filepath):
    filepath = os.path.expanduser(filepath)
    embeddings = {}
    with open(filepath, "r") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise ValueError(f"Bad header line in {filepath}: {header}")
        num_nodes, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.rstrip().split()
            if not parts:
                continue
            node_id = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.size != dim:
                continue
            embeddings[node_id] = vec
    if len(embeddings) != num_nodes:
        print(f"NOTE: header says {num_nodes:,} nodes, but loaded {len(embeddings):,} embeddings")
    return embeddings, dim

embeddings, dim = load_line_embeddings(
    "~/stellar-clustering/publication/Clustering/Embeddings Generation/LINE/stellar_line_order2.txt"
)

node_list = list(G.nodes())
missing = [n for n in node_list if str(n) not in embeddings]
if missing:
    print(f"WARNING: {len(missing):,} nodes missing embeddings. Example: {missing[:5]}")
    node_list = [n for n in node_list if str(n) in embeddings]

if not node_list:
    raise RuntimeError("No nodes left after filtering missing embeddings. Check ID mapping.")

X = np.vstack([embeddings[str(n)] for n in node_list]).astype(np.float32)
print(f"Embedding dim (from header): {dim}")
print(f"Nodes used: {len(node_list):,}")
print(f"Embedding matrix shape: {X.shape}")





print("\nSaving embeddings to CSV...")
OUTPUT_CSV = "stellar_line_embeddings_order2.csv"




df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(dim)])
df.insert(0, "node_id", node_list)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df):,} embeddings to: {OUTPUT_CSV}")
print(f"File size: {os.path.getsize(OUTPUT_CSV) / (1024**3):.2f} GB")