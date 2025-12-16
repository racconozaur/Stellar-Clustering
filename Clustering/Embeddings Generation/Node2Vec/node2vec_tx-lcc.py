import os
import pickle
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

import time
from datetime import datetime



start_time = datetime.now()


GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)

OUT_CSV = "txlcc_node2vec_d128_p1_q2_wl30_nw10_w10.csv"

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()
TEMP_FOLDER = os.path.join(BASE_DIR, ".n2v_cache")
os.makedirs(TEMP_FOLDER, exist_ok=True)

EMBED_DIM = 128
WALK_LENGTH = 20
NUM_WALKS = 4
WINDOW = 10
P = 1.0
Q = 2.0
SEED = 42
EDGE_WEIGHT_KEY = "weight"
WORKERS = 10

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading graph")
start = time.time()
with open(GRAPH_PKL, "rb") as f:
    G_full = pickle.load(f)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Graph loaded: {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges")



print(f"[{datetime.now().strftime('%H:%M:%S')}] keep only weight")
H = nx.Graph()
H.add_nodes_from(G_full.nodes())
for u, v, d in G_full.edges(data=True):
    H.add_edge(u, v, weight=float(d.get(EDGE_WEIGHT_KEY, 1.0)))
del G_full


print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing Node2Vec walks")
n2v = Node2Vec(
    H,
    dimensions=EMBED_DIM,
    walk_length=WALK_LENGTH,
    num_walks=NUM_WALKS,
    workers=WORKERS,
    p=P, q=Q,
    weight_key=EDGE_WEIGHT_KEY,
    seed=SEED,
    quiet=False,
    temp_folder=TEMP_FOLDER,
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Word2Vec")
model = n2v.fit(
    window=WINDOW,
    min_count=1,
    sg=1,
    negative=5,
    seed=SEED,
    workers=WORKERS
)


print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving embeddings")
nodes = list(model.wv.index_to_key)
vectors = model.wv.vectors
df = pd.DataFrame(vectors, columns=[f"z{i+1}" for i in range(EMBED_DIM)])
df.insert(0, "account_id", nodes)
try:
    df["account_id"] = df["account_id"].astype("int64")
except Exception:
    pass
df.to_csv(OUT_CSV, index=False)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Done. Saved {len(df):,} embeddings to: {OUT_CSV}")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Total time: {elapsed:.2f}s")