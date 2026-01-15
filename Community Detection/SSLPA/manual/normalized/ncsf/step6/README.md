# Step 6: Semi-Supervised SSLPA Evaluation Pipeline

This directory contains scripts for running semi-supervised learning experiments with the SSLPA (Speaker-Listener Label Propagation Algorithm) on the Stellar network graph.

## Overview

The pipeline consists of two main scripts that work together:

1. **`sslpa_semi_supervised_eval.py`** - Data preparation (creates train/test splits)
2. **`sslpa-70labels.py`** - SSLPA execution (runs label propagation)

---

## Script 1: `sslpa_semi_supervised_eval.py`

### Purpose
Prepares semi-supervised learning experiments by randomly splitting labeled data into training and test sets at multiple mask fractions.

### What It Does
1. Loads all NCSF-labeled CSV files from `step1/{MODE}/` directory
2. For each file and each mask fraction:
   - Randomly splits nodes into **train** (visible labels) and **test** (hidden labels)
   - Creates reproducible splits using a fixed random seed
3. Saves three CSV files per split for evaluation

### Inputs

| Input | Location | Description |
|-------|----------|-------------|
| **NCSF label files** | `step1/{MODE}/*.csv` | Per-node community labels from NCSF clustering |
| **MODE** | Config variable | Either `"scam_only"` or `"all_labels"` |
| **MASK_FRACTIONS** | Config variable | List of test set fractions, e.g., `[0.1, 0.2, 0.3, 0.5]` |
| **RANDOM_STATE** | Config variable | Random seed for reproducibility (default: `46`) |

**Expected NCSF files:**
- `sslpa_tx_lcc_ncsf_d2_r0.4_{MODE}.csv`
- `sslpa_tx_lcc_ncsf_d3_r0.5_{MODE}.csv`
- `sslpa_tx_lcc_ncsf_d5_r0.6_{MODE}.csv`

Each file must contain columns: `node`, `label`

### Outputs

For each NCSF file and mask fraction, creates a directory structure:

```
semi_supervised/{MODE}/{param_config}/mask{XX}/
├── sslpa_train_labels.csv              # Training labels (seed nodes)
├── sslpa_test_labels_true.csv          # Test labels (ground truth, hidden from SSLPA)
└── sslpa_all_labels_with_split.csv     # All labels with 'split' column (train/test)
```

**Example directory structure:**
```
semi_supervised/scam_only/
├── d2_r0.4/
│   ├── mask10/
│   │   ├── sslpa_train_labels.csv         # 90% of labels
│   │   ├── sslpa_test_labels_true.csv     # 10% of labels
│   │   └── sslpa_all_labels_with_split.csv
│   ├── mask20/
│   ├── mask30/
│   └── mask50/
├── d3_r0.5/
│   └── ... (same structure)
└── d5_r0.6/
    └── ... (same structure)
```

### Output File Formats

**sslpa_train_labels.csv:**
```csv
node,label
GABC123...,community_1
GXYZ789...,community_2
```

**sslpa_test_labels_true.csv:**
```csv
node,label
GDEF456...,community_1
GHIJ012...,community_3
```

**sslpa_all_labels_with_split.csv:**
```csv
node,label,split
GABC123...,community_1,train
GDEF456...,community_1,test
GXYZ789...,community_2,train
GHIJ012...,community_3,test
```

### Configuration

Edit these variables at the top of the script:

```python
MODE = "scam_only"                      # "scam_only" or "all_labels"
MASK_FRACTIONS = [0.1, 0.2, 0.3, 0.5]  # Test set fractions
RANDOM_STATE = 46                       # Random seed
```

### Usage

```bash
cd "/home/user/jfayzullaev/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step6"
python sslpa_semi_supervised_eval.py
```

### Expected Output

```
Running in MODE: scam_only
Looking for NCSF files in: .../step1/scam_only
Mask fractions: [0.1, 0.2, 0.3, 0.5]

Found 3 NCSF file(s) to process:
  - sslpa_tx_lcc_ncsf_d2_r0.4_scam_only.csv
  - sslpa_tx_lcc_ncsf_d3_r0.5_scam_only.csv
  - sslpa_tx_lcc_ncsf_d5_r0.6_scam_only.csv

================================================================================
Processing file: sslpa_tx_lcc_ncsf_d2_r0.4_scam_only.csv
================================================================================
Loaded 1,234,567 labeled nodes
Creating splits for mask fractions: [0.1, 0.2, 0.3, 0.5]

  [mask=10%]
    Train seeds: 1,111,110
    Test (masked) nodes: 123,457
    Output directory: .../d2_r0.4/mask10

  ... (continues for all mask fractions and files)
```

---

## Script 2: `sslpa-70labels.py`

### Purpose
Runs the SSLPA label propagation algorithm using training labels as seeds to predict labels for all nodes (including hidden test nodes).

### What It Does
1. Loads the graph once (for efficiency)
2. For each parameter configuration and mask fraction:
   - Loads training labels as seeds
   - Runs weighted SSLPA label propagation
   - Predicts labels for ALL nodes (including test nodes)
   - Saves predictions for evaluation

### Algorithm: SSLPA (Semi-Supervised Label Propagation)

The algorithm propagates labels through the graph using weighted voting:

1. **Initialize:** Seed nodes get their training labels; test nodes start unlabeled
2. **Iterate:** For each unlabeled node:
   - Collect weighted votes from labeled neighbors
   - Assign the label with the highest total weight
3. **Converge:** Stop when change rate < 0.1% or max iterations reached
4. **Finalize:** Mark any unreached nodes as "UNLABELED"

**Key parameters:**
- `MAX_ITER = 100` - Maximum iterations
- `CONVERGENCE_THRESHOLD = 0.001` - Stop when < 0.1% nodes change

### Inputs

| Input | Location | Description |
|-------|----------|-------------|
| **Graph** | `data/LCC/LCC_G_tx_undirected_weighted.pkl` | NetworkX graph with edge weights |
| **Training labels** | `semi_supervised/{MODE}/{param}/mask{XX}/sslpa_train_labels.csv` | Seed labels from Script 1 |
| **MODE** | Config variable | Either `"scam_only"` or `"all_labels"` (must match Script 1) |
| **MASK_FRACTIONS** | Config variable | List of mask fractions (must match Script 1) |

**Graph requirements:**
- Format: Pickled NetworkX Graph object
- Node IDs: Stellar account addresses
- Edge attribute: `weight` (transaction weights)

**Training labels format:**
```csv
node,label
GABC123...,community_1
GXYZ789...,community_2
```

### Outputs

For each parameter configuration and mask fraction:

```
semi_supervised/{MODE}/{param_config}/mask{XX}/
└── sslpa_predictions.csv    # Predictions for ALL nodes
```

**sslpa_predictions.csv format:**
```csv
node,label
GABC123...,community_1
GDEF456...,community_1       # Predicted (was in test set)
GXYZ789...,community_2
GHIJ012...,community_3       # Predicted (was in test set)
GUNKNOWN...,UNLABELED        # Unreached by propagation
```

### Configuration

Edit these variables at the top of the script:

```python
MODE = "scam_only"                      # "scam_only" or "all_labels"
MASK_FRACTIONS = [0.1, 0.2, 0.3, 0.5]  # Must match Script 1
MAX_ITER = 100                          # Maximum iterations
CONVERGENCE_THRESHOLD = 0.001           # Convergence threshold (0.1%)
```

### Usage

```bash
cd "/home/user/jfayzullaev/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step6"
python sslpa-70labels.py
```

### Expected Output

```
Running in MODE: scam_only
Semi-supervised splits directory: .../semi_supervised/scam_only

Found 3 parameter configuration(s) to process:
  - d2_r0.4
  - d3_r0.5
  - d5_r0.6

[04:21:12] Loading graph from .../LCC_G_tx_undirected_weighted.pkl
[04:21:24] Graph loaded: 4,315,652 nodes, 7,100,549 edges

################################################################################
# Processing parameter configuration: d2_r0.4
################################################################################

================================================================================
FILE: d2_r0.4 | MASK FRACTION: 10%
================================================================================

[04:21:25] Loading train labels from .../d2_r0.4/mask10/sslpa_train_labels.csv
[04:21:26] Loaded 1,111,110 seed labels
[04:21:26] Running SSLPA...
[04:21:26] Initializing SSLPA with 1,111,110 seeds
[04:21:26] Unlabeled nodes: 3,204,542
[04:21:35] Iteration 1: 1,234,567 nodes changed (38.5432%)
[04:21:44] Iteration 2: 234,567 nodes changed (7.3211%)
...
[04:23:15] Converged after 12 iterations
[04:23:15] SSLPA completed in 109.23s (12 iterations)
[04:23:20] Saved predictions to: .../d2_r0.4/mask10/sslpa_predictions.csv

Prediction Summary:
  Total nodes:     4,315,652
  Labeled:         4,298,123 (99.59%)
  Unlabeled:       17,529 (0.41%)
  Unique labels:   42

... (continues for all configurations and mask fractions)

================================================================================
SUMMARY
================================================================================
MODE: scam_only
Parameter configurations processed: 3
Mask fractions: [0.1, 0.2, 0.3, 0.5]
Total runs: 12
Successful runs: 12
Failed runs: 0
Total time: 1543.67s (25.73m)
================================================================================
DONE!
================================================================================
```

---

## Complete Workflow

### Step-by-Step Execution

```bash
# 1. Navigate to step6 directory
cd "/home/user/jfayzullaev/stellar-clustering/publication/Community Detection/SSLPA/manual/normalized/ncsf/step6"

# 2. Create train/test splits
python sslpa_semi_supervised_eval.py

# 3. Run SSLPA label propagation
python sslpa-70labels.py

# 4. Evaluate results (step 7 - separate script)
# Compare predictions against test labels
```

### Data Flow Diagram

```
step1/{MODE}/*.csv
    │ (NCSF labeled data)
    │
    ▼
┌─────────────────────────────────┐
│ sslpa_semi_supervised_eval.py   │
│ (Create train/test splits)      │
└─────────────────────────────────┘
    │
    ├── sslpa_train_labels.csv ──────┐
    ├── sslpa_test_labels_true.csv   │
    └── sslpa_all_labels_with_split  │
                                     │
    ┌────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ sslpa-70labels.py               │
│ (Run SSLPA propagation)         │  ◄── Graph (LCC_G_tx_undirected_weighted.pkl)
└─────────────────────────────────┘
    │
    ├── sslpa_predictions.csv
    │
    ▼
┌─────────────────────────────────┐
│ Step 7: Evaluation              │
│ (Compare predictions vs test)   │
└─────────────────────────────────┘
    │
    └── Metrics (accuracy, F1, etc.)
```

---

## Experiments Matrix

With current settings, you will run:

| Parameter Config | Mask Fraction | Train % | Test % | Output Directory |
|-----------------|---------------|---------|--------|------------------|
| d2_r0.4 | 0.1 | 90% | 10% | `d2_r0.4/mask10/` |
| d2_r0.4 | 0.2 | 80% | 20% | `d2_r0.4/mask20/` |
| d2_r0.4 | 0.3 | 70% | 30% | `d2_r0.4/mask30/` |
| d2_r0.4 | 0.5 | 50% | 50% | `d2_r0.4/mask50/` |
| d3_r0.5 | 0.1 | 90% | 10% | `d3_r0.5/mask10/` |
| d3_r0.5 | 0.2 | 80% | 20% | `d3_r0.5/mask20/` |
| d3_r0.5 | 0.3 | 70% | 30% | `d3_r0.5/mask30/` |
| d3_r0.5 | 0.5 | 50% | 50% | `d3_r0.5/mask50/` |
| d5_r0.6 | 0.1 | 90% | 10% | `d5_r0.6/mask10/` |
| d5_r0.6 | 0.2 | 80% | 20% | `d5_r0.6/mask20/` |
| d5_r0.6 | 0.3 | 70% | 30% | `d5_r0.6/mask30/` |
| d5_r0.6 | 0.5 | 50% | 50% | `d5_r0.6/mask50/` |

**Total: 12 experiments** (3 parameter configs × 4 mask fractions)

---

## Parameter Configurations

The three parameter configurations come from different NCSF clustering settings:

- **d2_r0.4**: Depth=2, Radius=0.4
- **d3_r0.5**: Depth=3, Radius=0.5 (default)
- **d5_r0.6**: Depth=5, Radius=0.6

These represent different community detection granularities from step 1.

---

## Troubleshooting

### Error: "Train labels file not found"
**Cause:** Script 2 ran before Script 1, or MODE mismatch
**Solution:** Run `sslpa_semi_supervised_eval.py` first with matching MODE

### Error: "No CSV files found in step1"
**Cause:** NCSF labeling step hasn't been completed
**Solution:** Run step 1 NCSF clustering first to generate label files

### Error: "Graph file not found"
**Cause:** Graph pickle file missing or incorrect path
**Solution:** Verify `GRAPH_PATH` points to correct LCC graph file

### MODE mismatch between scripts
**Cause:** Different MODE settings in the two scripts
**Solution:** Ensure both scripts have the same MODE value

### Mask fractions mismatch
**Cause:** Different MASK_FRACTIONS in the two scripts
**Solution:** Ensure MASK_FRACTIONS lists are identical

---

## Performance Notes

- **Script 1** (data prep): Fast, ~seconds for 3 files × 4 masks
- **Script 2** (SSLPA): Slow, ~20-30 minutes per experiment
  - Graph loading: ~12 seconds (done once)
  - Each SSLPA run: ~1-3 minutes depending on convergence
  - Total time: ~25-40 minutes for 12 experiments

**Optimization:**
- Graph is loaded once and reused for all experiments
- Runs are sequential (could be parallelized for faster execution)

---

## Output Summary

After running both scripts, you will have:

```
semi_supervised/scam_only/
├── d2_r0.4/
│   ├── mask10/
│   │   ├── sslpa_train_labels.csv           # Input for SSLPA
│   │   ├── sslpa_test_labels_true.csv       # Ground truth for evaluation
│   │   ├── sslpa_all_labels_with_split.csv  # Full overview
│   │   └── sslpa_predictions.csv            # SSLPA output
│   ├── mask20/ (same files)
│   ├── mask30/ (same files)
│   └── mask50/ (same files)
├── d3_r0.5/ (same structure)
└── d5_r0.6/ (same structure)
```

**Total files created:** 3 configs × 4 masks × 4 files = **48 CSV files**

---

## Next Steps (Step 7)

After running both scripts, proceed to step 7 for evaluation:

1. Load `sslpa_predictions.csv` and `sslpa_test_labels_true.csv`
2. Calculate metrics:
   - Accuracy on test set
   - Precision, Recall, F1-score per community
   - Confusion matrix
   - Coverage (% of nodes labeled)
3. Compare performance across:
   - Different mask fractions (how does train size affect performance?)
   - Different parameter configurations (which NCSF setting works best?)

---

## Contact & Support

For questions or issues:
- Check that MODE is consistent across scripts
- Verify all input files exist before running
- Review console output for detailed error messages
- Ensure sufficient disk space for output files (~500MB per configuration)
