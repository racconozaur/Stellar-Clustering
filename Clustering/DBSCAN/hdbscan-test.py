import os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import hdbscan
from datetime import datetime



#EMB_PATH = os.path.expanduser('~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv')


EMB_PATH = os.path.expanduser('~/stellar-clustering/publication/Clustering/Embeddings Generation/LINE/stellar_line_embeddings_order2.csv')


OUT = "LINE_res/tx_line_hdbscan_cosine_pca64.csv"

MIN_CLUSTER_SIZES = [20, 50, 100]



start_time = datetime.now()
def timestamp():
    return datetime.now().strftime('%H:%M:%S')

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"[{timestamp()}] Reading embeddings")
emb = pd.read_csv(EMB_PATH)
print(f"[{timestamp()}] Dataset size: {len(emb):,} nodes")





X = emb.drop(columns=["node_id"]).to_numpy(dtype=np.float32, copy=False)






print(f"[{timestamp()}] PCA to 64 dimensions")
pca = PCA(n_components=64, random_state=42, svd_solver="randomized")
X = pca.fit_transform(X).astype(np.float32, copy=False)
print(f"[{timestamp()}] Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

print(f"[{timestamp()}] Normalizing L2")
X = normalize(X, norm="l2", axis=1).astype(np.float32, copy=False)
X = np.ascontiguousarray(X)

print(f"[{timestamp()}] Running HDBSCAN")
for i, mcs in enumerate(MIN_CLUSTER_SIZES, 1):
    print(f"\n[{timestamp()}] === {i}/{len(MIN_CLUSTER_SIZES)}: mcs={mcs} ===")

    min_samples = mcs

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="boruvka_kdtree",
        approx_min_span_tree=True,
        gen_min_span_tree=False,
        prediction_data=False,
        core_dist_n_jobs= 10,
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(X)
    col = f"hdbscan_mcs{mcs}"
    emb[col] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[{timestamp()}] clusters={n_clusters} | noise={n_noise} ({100*n_noise/len(labels):.1f}%)")

    emb.to_csv(OUT, index=False)
    print(f"[{timestamp()}] Checkpoint saved {OUT}")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n[{timestamp()}] Total time: {elapsed/60:.1f} minutes")
print(f"[{timestamp()}] Final results saved: {OUT}")



















# import os, numpy as np, pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
# import hdbscan
# from datetime import datetime



# EMB_PATH = os.path.expanduser('~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv')
# OUT = "Node2Vec_res/tx_node2vec_hdbscan_cosine_pca64.csv"



# MIN_CLUSTER_SIZES = [5, 10, 15, 20, 50] 

# start_time = datetime.now()

# def timestamp():
#     return datetime.now().strftime('%H:%M:%S')



# os.makedirs(os.path.dirname(OUT), exist_ok=True)


# print(f"[{timestamp()}] Reading embeddings")
# emb = pd.read_csv(EMB_PATH)
# X = emb.drop(columns=["account_id"]).to_numpy(dtype=float)

# print(f"[{timestamp()}] PCA to 64 dimensions")
# X = PCA(n_components=64, random_state=42).fit_transform(X)

# print(f"[{timestamp()}] Normalizing for cosine distance")
# X_norm = normalize(X, norm='l2', axis=1)

# print(f"[{timestamp()}] Running HDBSCAN with different min_cluster_sizes")
# for mcs in MIN_CLUSTER_SIZES:
#     print(f"[{timestamp()}] Starting HDBSCAN with min_cluster_size={mcs}")
    
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=mcs,
#         metric='euclidean',
#         core_dist_n_jobs=-1,
#         algorithm='best'
#     )
    
#     labels = clusterer.fit_predict(X_norm)
    
#     col = f"hdbscan_mcs{mcs}"
#     emb[col] = labels
    
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = int((labels == -1).sum())
    
#     print(f"[{timestamp()}] [mcs={mcs}] clusters={n_clusters} | noise={n_noise} | noise%={100*n_noise/len(labels):.1f}%")

# print(f"[{timestamp()}] Saving results")
# emb.to_csv(OUT, index=False)
# print(f"[{timestamp()}] Saved:", OUT)

# elapsed = (datetime.now() - start_time).total_seconds()
# print(f"\n[{timestamp()}] Total time: {elapsed/60:.1f} minutes")