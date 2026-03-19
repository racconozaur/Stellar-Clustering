# NCSF Pipeline README

## Overview

This folder contains the NCSF-based evaluation pipeline built around the normalized SSLPA results on the Stellar transaction graph.

This part of the project is structured as a multi-step workflow rather than a single script. The main idea is:

1. Start from the normalized SSLPA node labels.
2. Apply an NCSF filtering rule to make labels more locally consistent in the graph.
3. Evaluate the filtered labels from two external perspectives:
   - multi-class entity matching
   - binary `SCAM` vs `NON_SCAM`
4. Compare embedding-based clustering methods with internal metrics.
5. Visualize the comparison results.
6. Run a semi-supervised masking experiment using the NCSF labels.
7. Evaluate SSLPA predictions only on the masked nodes.

This is publication-oriented part of the SSLPA workflow. It tries to move from raw propagation output toward a more controlled evaluation setup.

## What NCSF means here

In the code, `NCSF` is implemented as a neighborhood-based label consistency filter.

The core idea in [`step1/ncsf_and_label_stats.py`](step1/ncsf_and_label_stats.py) is:

- take the SSLPA label assigned to each node
- look at the node's neighborhood in the graph
- keep the label only if the node satisfies two local conditions:
  - degree is at least `d_min`
  - enough neighbors share the same label (`ratio >= r_min`)
- otherwise relabel that node as `UNLABELED`

The script supports two modes:

- `scam_only`
  Only nodes labeled `SCAM` are filtered by the neighborhood rule.
- `all_labels`
  The same neighborhood constraint is applied to every label.

This means NCSF is used here as a post-processing step on top of SSLPA, not as a separate clustering algorithm.

## Folder structure

The structure inside this folder is:

```text
ncsf/
├── step1/
│   └── ncsf_and_label_stats.py
├── step2/
│   └── eval_filter_A_entities.py
├── step3/
│   └── eval_filter_B_binary_scam.py
├── step4/
│   └── eval_internal_metrics_embeddings.py
├── step5/
│   ├── make_plots.py
│   └── plots/
│       ├── FilterA_External_Metrics.png
│       ├── FilterB_Scam_Detection.png
│       └── Internal_Embedding_Metrics.png
├── step6/
│   ├── README.md
│   ├── sslpa_semi_supervised_eval.py
│   └── sslpa-70labels.py
└── step7/
    ├── sslpa_eval_masked_only.py
    └── __pycache__/...
```

Most generated CSV outputs are not stored in git.

## Step-by-step description

### Step 1: Build NCSF-filtered labels

Main file:

- [`step1/ncsf_and_label_stats.py`](step1/ncsf_and_label_stats.py)

Purpose:

- load the transaction LCC graph
- load the normalized seed/entity labels
- load the normalized SSLPA output
- apply the neighborhood-constrained filter
- save filtered labels and basic label-frequency summaries

Important inputs from the script:

- graph pickle:
  `~/stellar-clustering/publication/data/LCC/LCC_G_tx_undirected_weighted.pkl`
- normalized labels:
  `~/stellar-clustering/publication/labeled-data/normalization/labels_mapped_normalized.csv`
- SSLPA labels:
  `~/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/sslpa_tx_lcc_manual_labels.csv`

Important parameters:

- `D_MIN`
- `R_MIN`
- `MODE`

Main outputs:

- filtered NCSF labels in:
  `step1/{MODE}/sslpa_tx_lcc_ncsf_d{D_MIN}_r{R_MIN}_{MODE}.csv`
- summary CSVs for seed labels, SSLPA labels, and NCSF labels

Observed behavior from the code:

- nodes with degree `0` are forced to `UNLABELED`
- nodes missing from the graph are also forced to `UNLABELED`
- in `scam_only` mode, only the `SCAM` label is filtered; all other labels are left unchanged

### Step 2: Filter A evaluation on entity labels

Main file:

- [`step2/eval_filter_A_entities.py`](step2/eval_filter_A_entities.py)

Purpose:

- evaluate multiple methods on a filtered subset of labeled accounts
- focus on entity-style labels rather than generic or noisy buckets

What the script does:

- loads normalized labels from `labels_mapped_normalized.csv`
- removes excluded classes such as:
  - `SCAM`
  - `UNLABELED`
  - `Burn Account`
  - `Spam Issuer`
  - `UltraCapital`
- loads clustering/community assignments from several methods
- intersects labeled accounts with method outputs
- runs 5-fold stratified evaluation
- writes per-fold and summary CSV files

Methods referenced in the tracked config:

- `Louvain`
- `LPA`
- `SSLPA_NCSF`
- `KMeans_Line_k10`
- `KMeans_Node2Vec_k10`
- `HDBSCAN_LINE_mcs20`
- `HDBSCAN_Node2Vec_mcs50`

Metrics used:

- `NMI`
- `ARI`
- `AMI`
- `FMI`
- `Homogeneity`
- `Completeness`
- `V-measure`
- coverage of labeled nodes matched to a method output

Main output directory:

- `step2/eval_filter_A_{MODE}/`

This step is useful for comparing whether the NCSF-cleaned SSLPA labels behave more like meaningful entity groups when compared with other methods.

### Step 3: Filter B evaluation for binary SCAM detection

Main file:

- [`step3/eval_filter_B_binary_scam.py`](step3/eval_filter_B_binary_scam.py)

Purpose:

- convert the NCSF label space into a binary task:
  `SCAM` vs `NON_SCAM`
- compare the same set of methods from a scam-detection viewpoint

What the script does:

- reads one chosen NCSF output from step 1
- maps each label to:
  - `SCAM` if label equals `SCAM`
  - `NON_SCAM` otherwise
- loads method outputs
- performs 5-fold stratified evaluation on labeled nodes
- builds a cluster-to-majority-class mapping inside each training fold
- evaluates the held-out fold using that mapping

Metrics used:

- `precision_SCAM`
- `recall_SCAM`
- `f1_SCAM`
- `NMI`
- `ARI`
- `FMI`
- confusion matrix counts:
  - `TP_SCAM`
  - `FN_SCAM`
  - `FP_SCAM`
  - `TN_SCAM`

Main output directory:

- `step3/eval_filter_B_{MODE}/`

This step makes the NCSF results easier to interpret from a security perspective, especially when the main question is whether a method can separate scam-related behavior from the rest.

### Step 4: Internal metrics for embedding-based methods

Main file:

- [`step4/eval_internal_metrics_embeddings.py`](step4/eval_internal_metrics_embeddings.py)

Purpose:

- compare embedding-based clustering methods with internal clustering quality scores

Methods configured in the script:

- `KMeans_Node2Vec_k10`
- `KMeans_LINE_k10`
- `HDBSCAN_Node2Vec_mcs50`
- `HDBSCAN_LINE_mcs20`

What the script loads:

- embedding files
- cluster assignment files
- account ids from both sides

Metrics used:

- `Calinski-Harabasz`
- `Davies-Bouldin`

The silhouette score appears in comments but is not currently active in the tracked version of the script.

Main output directory:

- `step4/internal_metrics/`

This step is not specifically about SSLPA, but it gives a baseline for comparing the geometry of embedding-based solutions against graph-native approaches.

### Step 5: Plotting and visual summaries

Main file:

- [`step5/make_plots.py`](step5/make_plots.py)

Tracked figures:

- [`step5/plots/FilterA_External_Metrics.png`](step5/plots/FilterA_External_Metrics.png)
- [`step5/plots/FilterB_Scam_Detection.png`](step5/plots/FilterB_Scam_Detection.png)
- [`step5/plots/Internal_Embedding_Metrics.png`](step5/plots/Internal_Embedding_Metrics.png)


Purpose:

- collect summary CSVs from steps 2, 3, and 4
- produce bar plots for publication/reporting

The three figure groups correspond to:

- Filter A external comparison
- Filter B binary scam detection comparison
- internal metrics for embedding methods

### Step 6: Semi-supervised masking experiment

Files:

- [`step6/README.md`](step6/README.md)
- [`step6/sslpa_semi_supervised_eval.py`](step6/sslpa_semi_supervised_eval.py)
- [`step6/sslpa-70labels.py`](step6/sslpa-70labels.py)

Purpose:

- test how SSLPA performs when only a fraction of NCSF labels is visible
- treat part of the labels as seeds and part as hidden ground truth

There are two parts here:

1. `sslpa_semi_supervised_eval.py`
   Creates train/test splits for each NCSF output and each mask fraction.
2. `sslpa-70labels.py`
   Runs SSLPA using only the training seeds, then writes predictions for all nodes.

Mask fractions in the tracked code:

- `0.1`
- `0.2`
- `0.3`
- `0.5`

Expected split structure:

```text
step6/semi_supervised/{MODE}/{param_config}/maskXX/
├── sslpa_train_labels.csv
├── sslpa_test_labels_true.csv
├── sslpa_all_labels_with_split.csv
└── sslpa_predictions.csv
```

This step is important because it turns the pipeline into a more realistic semi-supervised experiment instead of evaluating only on labels already used during propagation.

### Step 7: Evaluate only the masked nodes

Main file:

- [`step7/sslpa_eval_masked_only.py`](step7/sslpa_eval_masked_only.py)

Purpose:

- evaluate the semi-supervised SSLPA predictions from step 6
- restrict evaluation to the hidden test nodes only

What the script does:

- reads `sslpa_test_labels_true.csv`
- reads `sslpa_predictions.csv`
- merges them on the masked nodes
- computes:
  - Filter A style multi-class metrics
  - Filter B style binary scam metrics
- appends the results to evaluation summary CSVs under:
  - `step7/eval_results/{MODE}/all_methods_filterA_summary.csv`
  - `step7/eval_results/{MODE}/all_methods_filterB_summary.csv`

The method names written by this step look like:

- `SSLPA_NCSF_d2_r0.4_mask10`
- `SSLPA_NCSF_d3_r0.5_mask30`

This allows the semi-supervised SSLPA runs to be compared directly with the earlier method summaries.

## Suggested workflow order

Based on the tracked code, the intended execution order looks like this:

1. Produce the normalized SSLPA output in the parent SSLPA pipeline.
2. Run step 1 to create NCSF-filtered labels.
3. Run step 2 for entity-level external evaluation.
4. Run step 3 for binary scam evaluation.
5. Run step 4 for internal metrics on embedding-based baselines.
6. Run step 5 to generate publication plots.
7. Run step 6 to create semi-supervised splits and predictions.
8. Run step 7 to evaluate predictions on masked nodes only.



So this folder is not standalone. It is an evaluation layer on top of earlier graph construction, labeling, embedding, and clustering stages.



