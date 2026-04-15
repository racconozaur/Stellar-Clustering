from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================

MODE = "all_labels"  # "scam_only" | "all_labels"

BASE_DIR = os.path.expanduser(
    "~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf"
)

EVAL_DIR_A   = os.path.join(BASE_DIR, "step2", f"eval_filter_A_{MODE}")
EVAL_DIR_B   = os.path.join(BASE_DIR, "step3", f"eval_filter_B_{MODE}")
INTERNAL_DIR = os.path.join(BASE_DIR, "step4", "internal_metrics")
OUT_DIR      = os.path.join(BASE_DIR, "step5", "plots", MODE)

os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# Canonical display names – note both case variants of KMeans+LINE
METHOD_NAME_MAP: dict[str, str] = {
    "Louvain":               "Louvain",
    "LPA":                   "LPA",
    "SSLPA_NCSF":            "SSLPA+NCSF",
    "KMeans_Line_k10":       "KMeans+LINE",   # handles lowercase 'i'
    "KMeans_LINE_k10":       "KMeans+LINE",   # handles uppercase 'I'
    "KMeans_Node2Vec_k10":   "KMeans+Node2Vec",
    "HDBSCAN_LINE_mcs20":    "HDBSCAN+LINE",
    "HDBSCAN_Node2Vec_mcs50":"HDBSCAN+Node2Vec",
}

# =========================================================
# HELPERS
# =========================================================

def load_summary(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded: {path}")
        return df
    print(f"Warning: Could not find {path}")
    return None


def prettify_method_names(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    df = df.copy()
    if method_col in df.columns:
        df[method_col] = df[method_col].map(lambda x: METHOD_NAME_MAP.get(x, x))
    return df


def sort_methods(
    df: pd.DataFrame,
    method_col: str = "method",
    score_col: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return df with method column as an ordered Categorical."""
    df = df.copy()
    if score_col is not None and score_col in df.columns:
        order = (
            df.groupby(method_col)[score_col]
            .mean()
            .sort_values(ascending=ascending)
            .index.tolist()
        )
        df[method_col] = pd.Categorical(df[method_col], categories=order, ordered=True)
        df = df.sort_values(method_col)
    return df


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: int = 10) -> None:
    for container in ax.containers:
        labels = [fmt.format(bar.get_height()) if pd.notna(bar.get_height()) else "" for bar in container]
        ax.bar_label(container, labels=labels, fontsize=fontsize, padding=2, rotation=90)


def add_error_bars(ax: plt.Axes, df_melt: pd.DataFrame, metric_col: str = "Metric") -> None:
    """
    Correctly align error bars with seaborn grouped bar patches.

    Seaborn lays patches out as: all x-positions for hue-group-0, then all
    x-positions for hue-group-1, … (one flat list, NOT interleaved by x).
    We therefore build a lookup (method, metric) → Std and iterate patches
    in the same order seaborn drew them.
    """
    if "Std" not in df_melt.columns:
        return

    hue_order = df_melt[metric_col].cat.categories.tolist() if hasattr(df_melt[metric_col], "cat") else df_melt[metric_col].unique().tolist()
    method_order = df_melt["method"].cat.categories.tolist() if hasattr(df_melt["method"], "cat") else df_melt["method"].unique().tolist()

    # Build lookup: (method, metric) → Std
    std_lookup: dict[tuple, float] = {}
    for _, row in df_melt.iterrows():
        std_lookup[(row["method"], row[metric_col])] = row["Std"] if pd.notna(row["Std"]) else 0.0

    # Seaborn patch order: for each hue group, iterate over x positions
    patch_keys: list[tuple[str, str]] = []
    for metric in hue_order:
        for method in method_order:
            patch_keys.append((method, metric))

    for patch, key in zip(ax.patches, patch_keys):
        std = std_lookup.get(key, 0.0)
        if std == 0.0:
            continue
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(x=x, y=y, yerr=std, fmt="none", ecolor="black", elinewidth=1, capsize=3)


# =========================================================
# 1. FILTER A PLOT
# =========================================================

def plot_filter_a() -> None:
    print("\nPlotting Filter A results...")

    summary_path = os.path.join(EVAL_DIR_A, "all_methods_filterA_summary.csv")
    df_a = load_summary(summary_path)

    if df_a is None or df_a.empty:
        print("Skipping Filter A plot.")
        return

    df_a = prettify_method_names(df_a)

    metric_map = {
        "Avg_NMI_test": ("NMI", "Std_NMI_test"),
        "Avg_ARI_test": ("ARI", "Std_ARI_test"),
        "Avg_AMI_test": ("AMI", "Std_AMI_test"),
        "Avg_FMI_test": ("FMI", "Std_FMI_test"),
    }

    frames = []
    for metric_col, (metric_name, std_col) in metric_map.items():
        if metric_col not in df_a.columns:
            print(f"Warning: Missing {metric_col} in Filter A summary.")
            continue
        temp = df_a[["method", metric_col]].rename(columns={metric_col: "Score"}).copy()
        temp["Metric"] = metric_name
        temp["Std"] = df_a[std_col].values if std_col in df_a.columns else 0.0
        frames.append(temp)

    if not frames:
        print("No valid Filter A metrics found.")
        return

    df_melt = pd.concat(frames, ignore_index=True)

    # Sort methods by NMI (primary metric); fall back to Score mean
    primary = "NMI" if "NMI" in df_melt["Metric"].values else None
    if primary:
        order = (
            df_melt[df_melt["Metric"] == primary]
            .groupby("method")["Score"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = df_melt.groupby("method")["Score"].mean().sort_values(ascending=False).index.tolist()

    metric_order = [m for m in ["NMI", "ARI", "AMI", "FMI"] if m in df_melt["Metric"].unique()]

    df_melt["method"] = pd.Categorical(df_melt["method"], categories=order, ordered=True)
    df_melt["Metric"] = pd.Categorical(df_melt["Metric"], categories=metric_order, ordered=True)
    df_melt = df_melt.sort_values(["method", "Metric"])

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=df_melt,
        x="method",
        y="Score",
        hue="Metric",
        hue_order=metric_order,
        order=order,
        errorbar=None,
    )
    add_error_bars(ax, df_melt)

    ax.set_title(f"Filter A: External Multi-class Metrics ({MODE})", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"FilterA_External_Metrics_{MODE}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================
# 2. FILTER B PLOT
# =========================================================

def plot_filter_b() -> None:
    print("\nPlotting Filter B results...")

    summary_path = os.path.join(EVAL_DIR_B, "all_methods_filterB_summary.csv")
    df_b = load_summary(summary_path)

    if df_b is None or df_b.empty:
        print("Skipping Filter B plot.")
        return

    df_b = prettify_method_names(df_b)

    metric_map = {
        "precision_SCAM_mean": ("Precision", "precision_SCAM_std"),
        "recall_SCAM_mean":    ("Recall",    "recall_SCAM_std"),
        "f1_SCAM_mean":        ("F1",        "f1_SCAM_std"),
    }

    frames = []
    for metric_col, (metric_name, std_col) in metric_map.items():
        if metric_col not in df_b.columns:
            print(f"Warning: Missing {metric_col} in Filter B summary.")
            continue
        temp = df_b[["method", metric_col]].rename(columns={metric_col: "Score"}).copy()
        temp["Metric"] = metric_name
        temp["Std"] = df_b[std_col].values if std_col in df_b.columns else 0.0
        frames.append(temp)

    if not frames:
        print("No valid Filter B metrics found.")
        return

    df_melt = pd.concat(frames, ignore_index=True)

    # Sort by F1 as the primary metric
    primary = "F1" if "F1" in df_melt["Metric"].values else None
    if primary:
        order = (
            df_melt[df_melt["Metric"] == primary]
            .groupby("method")["Score"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = df_melt.groupby("method")["Score"].mean().sort_values(ascending=False).index.tolist()

    metric_order = [m for m in ["Precision", "Recall", "F1"] if m in df_melt["Metric"].unique()]

    df_melt["method"] = pd.Categorical(df_melt["method"], categories=order, ordered=True)
    df_melt["Metric"] = pd.Categorical(df_melt["Metric"], categories=metric_order, ordered=True)
    df_melt = df_melt.sort_values(["method", "Metric"])

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=df_melt,
        x="method",
        y="Score",
        hue="Metric",
        hue_order=metric_order,
        order=order,
        errorbar=None,
    )
    add_error_bars(ax, df_melt)

    ax.set_title(f"Filter B: Binary SCAM Detection ({MODE})", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"FilterB_Scam_Detection_{MODE}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================
# 3. INTERNAL METRICS PLOTS
# =========================================================

def plot_internal_metrics() -> None:
    print("\nPlotting internal embedding metrics...")

    summary_path = os.path.join(INTERNAL_DIR, "embedding_internal_metrics.csv")
    df_int = load_summary(summary_path)

    if df_int is None or df_int.empty:
        print("Skipping internal metrics plot.")
        return

    df_int = prettify_method_names(df_int)

    # Each subplot has a different "better" direction – sort independently
    ch_order = (
        df_int.groupby("method")["calinski_harabasz"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    db_order = (
        df_int.groupby("method")["davies_bouldin"]
        .mean()
        .sort_values(ascending=True)   # lower is better → best on left
        .index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(
        data=df_int,
        x="method",
        y="calinski_harabasz",
        order=ch_order,
        ax=axes[0],
        errorbar=None,
    )
    axes[0].set_title("Calinski–Harabasz (higher is better)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CH Score")
    axes[0].tick_params(axis="x", rotation=30)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")

    sns.barplot(
        data=df_int,
        x="method",
        y="davies_bouldin",
        order=db_order,
        ax=axes[1],
        errorbar=None,
    )
    axes[1].set_title("Davies–Bouldin (lower is better)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("DB Score")
    axes[1].tick_params(axis="x", rotation=30)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "Internal_Embedding_Metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================
# 4. COVERAGE PLOT FOR FILTER A
# =========================================================

def plot_filter_a_coverage() -> None:
    print("\nPlotting Filter A coverage...")

    summary_path = os.path.join(EVAL_DIR_A, "all_methods_filterA_summary.csv")
    df_a = load_summary(summary_path)

    if df_a is None or df_a.empty or "coverage" not in df_a.columns:
        print("Skipping Filter A coverage plot.")
        return

    df_a = prettify_method_names(df_a)
    df_a = sort_methods(df_a, score_col="coverage", ascending=False)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_a, x="method", y="coverage", errorbar=None)
    ax.set_title(f"Filter A Coverage by Method ({MODE})", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    annotate_bars(ax, fmt="{:.2f}", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"FilterA_Coverage_{MODE}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print(f"MODE      = {MODE}")
    print(f"Output    = {OUT_DIR}")

    plot_filter_a()
    plot_filter_b()
    plot_internal_metrics()
    plot_filter_a_coverage()

    print(f"\nAll plots saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()