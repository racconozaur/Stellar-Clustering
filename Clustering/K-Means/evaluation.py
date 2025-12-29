import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter
import os

LBL_NORM = os.path.expanduser("~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv")


#FN = "Node2Vec_res/transactions_node2vec_kmeans_results.csv" 
FN = "LINE_res/transactions_line_kmeans_results.csv"






COLS = 'dim_'

#OUT = 'Node2Vec_res/evaluation'
OUT = 'LINE_res/evaluation'

os.makedirs(OUT, exist_ok=True)


print('reading file ', FN)

df = pd.read_csv(FN)


if df.columns[0] != "account_id":
    print(f"Renaming first column '{df.columns[0]}' to 'account_id'")
    df.rename(columns={df.columns[0]: "account_id"}, inplace=True)


X = df[[c for c in df.columns if c.startswith(COLS)]].values
k_cols = sorted([c for c in df.columns if c.startswith("kmeans_")],
                key=lambda s: int(s.split("_")[1]))

# intr
print("Computing intrinsic metrics")
rows = []
for col in k_cols:
    y = df[col].values
    k = int(col.split("_")[1])
    print(f"Scoring k={k}")
    # sil = silhouette_score(X, y, metric="euclidean")
    db  = davies_bouldin_score(X, y)
    ch  = calinski_harabasz_score(X, y)
    # rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

    rows.append({"k": k, "davies_bouldin": db, "calinski_harabasz": ch})

out = pd.DataFrame(rows).sort_values("k")
out.to_csv(f"{OUT}/transaction_kmeans_intrinsic_scores.csv", index=False)
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

# Compute extrinsic metrics
rows = []
y_ref = df_labeled["name"].to_numpy()
for kcol in k_cols:
    y = df_labeled[kcol].to_numpy()
    k = int(kcol.split("_")[1])
    rows.append({
        "k": k,
        "NMI_vs_LBL_NORM": normalized_mutual_info_score(y_ref, y),
        "ARI_vs_LBL_NORM": adjusted_rand_score(y_ref, y),
        "Purity_vs_LBL_NORM": purity(y_ref, y),
    })

ext = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
ext.to_csv(f"{OUT}/transaction_kmeans_extrinsic_scores.csv", index=False)
print(ext)
print("\nExtrinsic scores saved")