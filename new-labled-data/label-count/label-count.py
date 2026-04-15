
import pandas as pd
from collections import Counter
from pathlib import Path

CSV = "../normalization/labeled_nodes_in_graph.csv"
OUTDIR = Path("labels_summary_out")
OUTDIR.mkdir(exist_ok=True)

cols = pd.read_csv(CSV, nrows=0).columns.tolist()

account_col = "node_id"
label_col   = "entity"

total_rows = 0
label_counter = Counter()
null_account = 0
null_label = 0

for chunk in pd.read_csv(CSV, usecols=[account_col, label_col], chunksize=200000):
    total_rows += len(chunk)
    null_account += chunk[account_col].isna().sum()
    null_label += chunk[label_col].isna().sum()
    vc = chunk[label_col].fillna("<<NULL>>").value_counts()
    label_counter.update(vc.to_dict())

counts_df = pd.DataFrame(
    [{"label": k, "count": v} for k, v in label_counter.items()]
).sort_values("count", ascending=False)

counts_df.to_csv(OUTDIR / "label_counts.csv", index=False)
counts_df[counts_df["count"] < 5].to_csv(OUTDIR / "label_counts_lt5.csv", index=False)
counts_df[counts_df["count"] < 10].to_csv(OUTDIR / "label_counts_lt10.csv", index=False)
counts_df[counts_df["count"] < 20].to_csv(OUTDIR / "label_counts_lt20.csv", index=False)

with open(OUTDIR / "basic_info.txt", "w") as f:
    f.write(f"CSV: {CSV}\\n")
    f.write(f"Columns: {cols}\\n")
    f.write(f"account_col: {account_col}\\n")
    f.write(f"label_col: {label_col}\\n")
    f.write(f"total_rows: {total_rows}\\n")
    f.write(f"null_account: {null_account}\\n")
    f.write(f"null_label: {null_label}\\n")
    f.write(f"n_unique_labels: {len(label_counter)}\\n")

print("Wrote:")
print(OUTDIR / "basic_info.txt")
print(OUTDIR / "label_counts.csv")
print(OUTDIR / "label_counts_lt5.csv")
print(OUTDIR / "label_counts_lt10.csv")
print(OUTDIR / "label_counts_lt20.csv")
