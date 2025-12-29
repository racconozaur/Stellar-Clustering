import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter
import os




LBL_NORM = os.path.expanduser("~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv")


FN = "Node2Vec_res/tx_node2vec_hdbscan_cosine_pca64.csv" 
#FN = "LINE_res/tx_line_hdbscan_cosine_pca64.csv" 


COLS = 'z'


OUT = 'Node2Vec_res/evaluation'
#OUT = 'LINE_res/evaluation'


os.makedirs(OUT, exist_ok=True)






print('reading file ', FN)

df = pd.read_csv(FN)

if df.columns[0] != "account_id":
    print(f"Renaming first column '{df.columns[0]}' to 'account_id'")
    df.rename(columns={df.columns[0]: "account_id"}, inplace=True)

X = df[[c for c in df.columns if c.startswith(COLS)]].values
hdbscan_cols = sorted([c for c in df.columns if c.startswith("hdbscan_")],
                      key=lambda s: int(s.split("mcs")[1]))

# intr
print("Computing intrinsic metrics")
rows = []
for col in hdbscan_cols:
    y = df[col].values
    mcs = int(col.split("mcs")[1])  
    

    mask = y != -1
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    n_clusters = len(set(y_filtered))
    n_noise = (y == -1).sum()
    
    if n_clusters < 2:
        print(f"Skipping mcs={mcs}: only {n_clusters} cluster(s)")
        continue
    
    print(f"Scoring mcs={mcs} (n={len(y_filtered):,}, clusters={n_clusters}, noise={n_noise:,})")
    db = davies_bouldin_score(X_filtered, y_filtered)
    ch = calinski_harabasz_score(X_filtered, y_filtered)
    rows.append({"min_cluster_size": mcs, "n_clusters": n_clusters, "n_noise": n_noise, 
                 "davies_bouldin": db, "calinski_harabasz": ch})

out = pd.DataFrame(rows).sort_values("min_cluster_size")
out.to_csv(f"{OUT}/transaction_hdbscan_intrinsic_scores.csv", index=False)
print(out)
print("\nIntrinsic scores saved")

# extr 
print("\nComputing extrinsic metrics...")
lblnorm = pd.read_csv(LBL_NORM)[["account_id", "name"]]
# Ensure compatible dtypes for merging
for col in ["account_id"]:
    try:
        df[col] = df[col].astype("Int64")
        lblnorm[col] = lblnorm[col].astype("Int64")
    except Exception:
        df[col] = df[col].astype(str)
        lblnorm[col] = lblnorm[col].astype(str)

# Merge with ground truth labels
df_labeled = df.merge(lblnorm, on="account_id", how="inner")
print(f"Matched accounts for extrinsic eval: {len(df_labeled):,} / {len(df):,}")

# Purity function
def purity(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")
    total = len(y_true)
    score = 0
    for c in set(y_pred):
        idx = (y_pred == c)
        if idx.any():
            score += Counter(y_true[idx]).most_common(1)[0][1]
    return score / total

rows = []
y_ref = df_labeled["name"].to_numpy()
for col in hdbscan_cols:
    y = df_labeled[col].to_numpy()
    mcs = int(col.split("mcs")[1]) 
    rows.append({
        "min_cluster_size": mcs,
        "NMI_vs_LBL_NORM": normalized_mutual_info_score(y_ref, y),
        "ARI_vs_LBL_NORM": adjusted_rand_score(y_ref, y),
        "Purity_vs_LBL_NORM": purity(y_ref, y),
    })

ext = pd.DataFrame(rows).sort_values("min_cluster_size").reset_index(drop=True)
ext.to_csv(f"{OUT}/transaction_hdbscan_extrinsic_scores.csv", index=False)
print(ext)
print("\nExtrinsic scores saved")