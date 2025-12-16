import pandas as pd

TYPE = "trust_lcc"

RAW_EMB = f"role2vec_io/{TYPE}/role2vec_raw.csv"     
MAPPING = f"role2vec_io/{TYPE}/node_mapping.csv"   
 
OUT_CSV = f"embeddings/{TYPE}_role2vec_d128.csv"    


E = pd.read_csv(RAW_EMB)
M = pd.read_csv(MAPPING)  

assert "id" in E.columns, "Expected id col in output"
emb_cols = [c for c in E.columns if c != "id"]

E["id"] = E["id"].astype(int)
M["node_idx"] = M["node_idx"].astype(int)
df = M.merge(E, left_on="node_idx", right_on="id", how="inner").drop(columns=["node_idx", "id"])


rename_map = {c: f"role{i+1}" for i, c in enumerate(sorted(emb_cols, key=lambda s:int(s.split('_')[1])))}
df = df.rename(columns=rename_map)

df = df.sort_values("account_id")
df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with shape {df.shape}")
