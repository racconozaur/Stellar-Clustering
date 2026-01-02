import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
EVAL_DIR_A = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step2/eval_filter_A"
)
EVAL_DIR_B = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step3/eval_filter_B"
)
INTERNAL_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step4/internal_metrics"
)
OUT_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step5/plots"
)

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")

def load_summary(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: Could not find {path}")
    return None

# 1. Plot Filter A (Entity Evaluation)
print("Plotting Filter A results...")
df_a = load_summary(os.path.join(EVAL_DIR_A, "all_methods_filterA_summary.csv"))
if df_a is not None:
    metrics = ['Avg_NMI_test', 'Avg_ARI_test', 'Avg_AMI_test', 'Avg_FMI_test']
    df_melt = df_a.melt(id_vars=['method'], value_vars=metrics,
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='method', y='Score', hue='Metric')
    plt.title("Filter A: External Metrics by Method (Entity Subset)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "FilterA_External_Metrics.png"))
    plt.close()

# 2. Plot Filter B (Binary SCAM detection)
print("Plotting Filter B results...")
df_b = load_summary(os.path.join(EVAL_DIR_B, "all_methods_filterB_summary.csv"))
if df_b is not None:
    metrics = ['precision_SCAM_mean', 'recall_SCAM_mean', 'f1_SCAM_mean']
    df_melt = df_b.melt(id_vars=['method'], value_vars=metrics,
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='method', y='Score', hue='Metric')
    plt.title("Filter B: Binary SCAM Detection Performance")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "FilterB_Scam_Detection.png"))
    plt.close()

# 3. Plot Internal Metrics (Embedding Quality)
print("Plotting Internal Metrics...")
df_int = load_summary(os.path.join(INTERNAL_DIR, "embedding_internal_metrics.csv"))
if df_int is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # sns.barplot(data=df_int, x='method', y='silhouette', ax=axes[0])
    # axes[0].set_title("Silhouette Score (higher is better)")
    # axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df_int, x='method', y='calinski_harabasz', ax=axes[1])
    axes[1].set_title("Calinski–Harabasz (higher is better)")
    axes[1].tick_params(axis='x', rotation=45)

    sns.barplot(data=df_int, x='method', y='davies_bouldin', ax=axes[2])
    axes[2].set_title("Davies–Bouldin (lower is better)")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Internal_Embedding_Metrics.png"))
    plt.close()

print(f"Done! Plots saved to {OUT_DIR}/")
