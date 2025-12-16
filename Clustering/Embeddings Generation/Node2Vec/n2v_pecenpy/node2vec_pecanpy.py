import os
import pickle
import pandas as pd
import networkx as nx
from pecanpy import pecanpy as node2vec
from gensim.models import Word2Vec
from datetime import datetime

start_time = datetime.now()
ts = lambda: datetime.now().strftime('%H:%M:%S')

GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)
OUT_CSV = "txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv"

EMBED_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 10
WINDOW = 10
P = 1.0
Q = 2.0
WORKERS = 10

print(f"[{ts()}] Loading graph")
with open(GRAPH_PKL, "rb") as f:
    G_full = pickle.load(f)

print(f"[{ts()}] Graph: {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges")

edgelist_file = "temp_edgelist.txt"
print(f"[{ts()}] Writing edgelist")
with open(edgelist_file, 'w') as f:
    for u, v, data in G_full.edges(data=True):
        weight = data.get('weight', 1.0)
        f.write(f"{u}\t{v}\t{weight}\n")

del G_full 



print(f"[{ts()}] Init pecanpy SparseOTF")
g = node2vec.SparseOTF(
    p=P, 
    q=Q, 
    workers=WORKERS,
    verbose=True
    )

print(f"[{ts()}] Reading graph")
g.read_edg(edgelist_file, weighted=True, directed=False, delimiter='\t')

print(f"[{ts()}] Generating walks")
walks = g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)




print(f"[{ts()}] Training Word2Vec")
model = Word2Vec(
    walks,
    vector_size=EMBED_DIM,
    window=WINDOW,
    min_count=1,
    sg=1,
    workers=WORKERS,
    epochs=1,
    seed=42
)

print(f"[{ts()}] Saving embeddings")
nodes = list(model.wv.index_to_key)
vectors = model.wv.vectors

df = pd.DataFrame(vectors, columns=[f"z{i+1}" for i in range(EMBED_DIM)])
df.insert(0, "account_id", nodes)

try:
    df["account_id"] = df["account_id"].astype("int64")
except Exception:
    pass

df.to_csv(OUT_CSV, index=False)

print(f"[{ts()}] Done {len(df):,} embeddings")
elapsed = (datetime.now() - start_time).total_seconds()
print(f"[{ts()}] Total time: {elapsed/3600:.2f} hours")

os.remove(edgelist_file)
