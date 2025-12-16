import pandas as pd
from sklearn.cluster import KMeans

PATH = '~/stellar-clustering/publication/Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/txlcc_node2vec_d128_p1_q2_wl30_nw10_w10_pecanpy.csv'

embeddings = pd.read_csv(PATH)
X = embeddings.drop(columns=["account_id"]).values


K_VALUES = [10, 15, 20, 30, 40, 50, 65, 70, 75, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400]

for k in K_VALUES:
    print(f"Running K-means with k={k}")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    embeddings[f"kmeans_{k}"] = kmeans.fit_predict(X)



embeddings.to_csv("Node2Vec_res/transactions_node2vec_kmeans_results.csv", index=False)
print("Saved results")
