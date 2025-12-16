import pandas as pd
import networkx as nx
import pickle
from datetime import datetime


MINIMAL_CSV = "../transaction_edges_minimal.csv"
GRAPH_PICKLE = "tx_undirected_weighted_graph.pkl"

start_time = datetime.now()


print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading CSV...")
transaction_edges = pd.read_csv(MINIMAL_CSV)



transaction_edges = transaction_edges.dropna(subset=["sender_id", "receiver_id"])

transaction_edges["amount"] = pd.to_numeric(transaction_edges["amount"], errors="coerce").fillna(0)


# Remove self-loops
transaction_edges = transaction_edges[transaction_edges["sender_id"] != transaction_edges["receiver_id"]]


# undirected pairs (u, v) with u < v
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating undirected pairs")
transaction_edges["u"] = transaction_edges[["sender_id", "receiver_id"]].min(axis=1)
transaction_edges["v"] = transaction_edges[["sender_id", "receiver_id"]].max(axis=1)

# Aggregate multiple operations between the same account pair
print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregating edges")
edges_agg = (
    transaction_edges
    .groupby(["u", "v"])
    .agg(
        tx_count=("operation_id", "count"),
        amount_sum=("amount", "sum"),
    )
    .reset_index()
)


edges_agg["weight"] = edges_agg["tx_count"]


print(f"[{datetime.now().strftime('%H:%M:%S')}] Building graph")
G = nx.from_pandas_edgelist(
    edges_agg,
    source="u",
    target="v",
    edge_attr=["weight", "tx_count", "amount_sum"],
    create_using=nx.Graph,
)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")




with open(GRAPH_PICKLE, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Graph saved to {GRAPH_PICKLE}")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Total time: {elapsed:.2f}s")