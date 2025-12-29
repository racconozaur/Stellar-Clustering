import os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from datetime import datetime

EMB_PATH = os.path.expanduser('~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv')
OUT = "Node2Vec_res/tx_node2vec_dbscan_cosine_pca64_kgrid_test.csv"



MIN_SAMPLES_LIST = [5, 10, 15]
PERCENTILES = [70, 80, 85, 90, 95]   

start_time = datetime.now()

def timestamp():
    return datetime.now().strftime('%H:%M:%S')

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"[{timestamp()}] Reading embeddings")
emb = pd.read_csv(EMB_PATH)
X = emb.drop(columns=["account_id"]).to_numpy(dtype=float)

# PCA 
print(f"[{timestamp()}] PCA")
X = PCA(n_components=64, random_state=42).fit_transform(X)

print(f"[{timestamp()}] Running DBSCAN in gridsearch")
for k in MIN_SAMPLES_LIST:

    print(f"[{timestamp()}] Starting k={k}")


    print(f"[{timestamp()}]   Computing k-NN for k={k}")
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dists, _ = nn.kneighbors(X)

    print(f"[{timestamp()}]   Computing percentiles for k={k}")
    kth = np.sort(dists[:, -1])
    seeds = {p: float(np.percentile(kth, p)) for p in PERCENTILES}
    print(f"[{timestamp()}] k={k} seeds:", {p: round(e, 4) for p, e in seeds.items()})
    
    for p, eps in seeds.items():
        print(f"[{timestamp()}] Running DBSCAN ms={k} p={p} eps={eps:.6f}")

        labels = DBSCAN(eps=eps, min_samples=k, metric="cosine", algorithm="brute", n_jobs=1).fit_predict(X)
        col = f"dbscan_ms{k}_p{p}_eps_{eps:.6f}"
        emb[col] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"[{timestamp()}][ms={k} p={p} eps={eps:.4f}] clusters={n_clusters} , noise={n_noise}")

emb.to_csv(OUT, index=False)
print("Saved:", OUT)
elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n[{timestamp()}] Total execution time: {elapsed:.2f}s")