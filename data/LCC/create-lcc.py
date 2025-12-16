import pickle
import networkx as nx
import os


GRAPH = os.path.expanduser("../pkl/tx_undirected_weighted_graph.pkl")

# load the full graph
print("Reading the pkl data")
with open(GRAPH, "rb") as f:
    G_full = pickle.load(f)

print(f"Number of full nodes: {G_full.number_of_nodes()}")
print(f"Number of full edges: {G_full.number_of_edges()}")


print("Extracting llc")
largest_cc_nodes = max(nx.connected_components(G_full), key=len)
G_lcc = G_full.subgraph(largest_cc_nodes).copy()

print(f"LCC has {G_lcc.number_of_nodes()} nodes and {G_lcc.number_of_edges()} edges")

sample_node = list(G_lcc.nodes())[0]
print("Sample node:", sample_node)
print("Attributes:", G_lcc.nodes[sample_node])

# Save LCC
output_path = "LCC_G_tx_undirected_weighted.pkl"
with open(output_path, "wb") as f:
    pickle.dump(G_lcc, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved lcc as .pkl")
