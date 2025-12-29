import os
import pickle
import networkx as nx


GRAPH_PKL = os.path.expanduser("~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl")
OUT_TXT   = os.path.expanduser("tx_line_edgelist.txt")


with open(GRAPH_PKL, "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

self_loops = list(nx.selfloop_edges(G))
if self_loops:
    print(f"Removing {len(self_loops):,} self-loops")
    G.remove_edges_from(self_loops)



edges_written = 0
with open(OUT_TXT, "w") as f:
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)

        if w is None or w <= 0:
            continue
        f.write(f"{u} {v} {w}\n")
        f.write(f"{v} {u} {w}\n")
        edges_written += 2


print(f"Wrote {edges_written:,} directed edges ({G.number_of_edges():,} undirected) to: {OUT_TXT}")
print(f"File size: {os.path.getsize(OUT_TXT) / (1024**2):.1f} MB")



# ./line -train ../../stellar_edgelist.txt \
#       -output ../../stellar_line_embeddings.txt \
#       -size 128 \
#       -order 2 \
#       -negative 5 \
#       -samples 100 \
#       -threads 8