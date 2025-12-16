import os
import time
import pickle
import pandas as pd
import networkx as nx
from datetime import datetime
from karateclub import LINE

def timestamp():
    return datetime.now().strftime("%H:%M:%S")



GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl"
)
OUT_CSV = "tx_lcc_line2nd_d128.csv"

def run_line_embeddings(graph_pkl, out_csv, dim=128, epochs=5):


    print(f"{timestamp()} Loading graph")
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    
    print(f"{timestamp()} Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    

    # Train LINE
    print(f"{timestamp()} Training LINE: dim={dim}, epochs={epochs}")

    
    model = LINE(
        dimensions=dim,
        order=2,
        epochs=epochs,
        learning_rate=0.025
    )
    
    start = time.time()
    model.fit(G)
    end = time.time()
    
    print(f"{timestamp()} Training finished in {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
    

    print(f"{timestamp()} Extracting embeddings")
    embeddings = model.get_embedding()  
    nodes = list(G.nodes())
    



    df = pd.DataFrame(embeddings, index=nodes)
    df.index.name = "account_id"
    df.reset_index(inplace=True)
    
    # Rename cols
    df.columns = ["account_id"] + [f"line2_{i+1}" for i in range(dim)]
    




    out_csv = os.path.expanduser(out_csv)
    df.to_csv(out_csv, index=False)
    
    print(f"{timestamp()} Saved embeddings to: {out_csv}")
    print(f"{timestamp()} Shape: {df.shape}")
    
    return out_csv

def main():
    run_line_embeddings(GRAPH_PKL, OUT_CSV, dim=128, epochs=5)
    print(f"[{timestamp()}] Done!")

if __name__ == "__main__":
    main()