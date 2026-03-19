# Behavior-Based Address Clustering on Stellar Blockchain

## Project Overview

This repository contains the implementation code for a masters thesis analyzing community detection and clustering algorithms on the Stellar blockchain network. The project evaluates various graph based and embedding based methods for identifying communities within Stellars transaction and trustlines networks.

Detailed NCSF documentation:

- [`Community Detection/SSLPA/manual/normalized/ncsf/README.md`](Community%20Detection/SSLPA/manual/normalized/ncsf/README.md)

## Contents

- **Python scripts**: Data preprocessing, algorithm implementation, and evaluation modules
- **Jupyter notebooks**: For analysis, results, visualization

## Purpose

This codebase implements and evaluates multiple community detection approaches on Stellar network data, comparing unsupervised clustering methods (K-Means, DBSCAN), graph-based community detection algorithms (Louvain, LPA, SSLPA, Spectral Clustering). The research aims to identify optimal methods for detecting meaningful communities in blockchain transaction networks.




## What this repo is about

The general idea of the project is:

1. Build a transaction graph from Stellar transaction data.
2. Extract the largest connected component (LCC).
3. Map known labels/entities to graph accounts.
4. Normalize those labels into cleaner entity names.
5. Generate graph embeddings.
6. Run clustering and community detection methods.
7. Evaluate the results with internal and external metrics.

The main methods present in this repo are:

- `K-Means`
- `DBSCAN / HDBSCAN`
- `Louvain`
- `LPA`
- `SSLPA` (manual / semi-supervised style)

## Important note before running anything

A lot of data/results are **not stored in git**.

The `.gitignore` excludes:

- `*.csv`
- `*.pkl`
- `*.log`
- `*.txt`

The repo mostly contains code and notebooks.

Also, many scripts use hard-coded paths like:

- `~/stellar-clustering/publication/...`
- `~/stellar-clustering/network/...`
- `/home/user/jfayzullaev/...`

So if someone else runs this repo, paths will probably need to be changed first.

## Repository structure

### `data/`

This part is for building and checking the graph.

- [`data/pkl/create-network.py`](data/pkl/create-network.py) reads `transaction_edges_minimal.csv`, removes missing/self-loop rows, aggregates transactions between account pairs, and creates an undirected weighted NetworkX graph saved as `tx_undirected_weighted_graph.pkl`.
- [`data/LCC/create-lcc.py`](data/LCC/create-lcc.py) loads the full graph pickle and saves the largest connected component as `LCC_G_tx_undirected_weighted.pkl`.
- [`data/check-data.ipynb`](data/check-data.ipynb) is a notebook for checking the raw CSV data, the full graph, and the LCC.

Expected missing inputs here:

- `transaction_edges_minimal.csv`
- `transaction_edges_metadata.csv`
- generated `.pkl` graph files

### `labeled-data/`

This folder is for attaching labels/entities to accounts.

- [`labeled-data/full_stellar_directory.json`](labeled-data/full_stellar_directory.json) is the tracked label source in JSON format.
- [`labeled-data/map-labels/map_labels.py`](labeled-data/map-labels/map_labels.py) reads the Stellar directory JSON and transaction metadata, then maps known addresses to account ids and saves `labels_mapped.csv`.
- [`labeled-data/normalization/label-normalization.py`](labeled-data/normalization/label-normalization.py) applies regex-based normalization rules and writes `labels_mapped_normalized.csv`.
- [`labeled-data/normalization/rules.json`](labeled-data/normalization/rules.json) stores the normalization rules used to merge messy names into canonical entities such as exchanges, wallets, bots, scam-related labels, etc.


### `Clustering/`

This folder contains the embedding generation and clustering experiments.

#### `Clustering/Embeddings Generation/`

This is where graph embeddings are prepared.

`LINE`

- [`Clustering/Embeddings Generation/LINE/convert-format.py`](Clustering/Embeddings%20Generation/LINE/convert-format.py) converts the LCC pickle graph into a text edge list format.
- [`Clustering/Embeddings Generation/LINE/line-emb.py`](Clustering/Embeddings%20Generation/LINE/line-emb.py) runs LINE embeddings with `karateclub`.
- [`Clustering/Embeddings Generation/LINE/convert-emb-back.py`](Clustering/Embeddings%20Generation/LINE/convert-emb-back.py) maps raw LINE embeddings back to graph node ids and saves a CSV.
- [`Clustering/Embeddings Generation/LINE/check.ipynb`](Clustering/Embeddings%20Generation/LINE/check.ipynb) checks the embedding file.

`Node2Vec`

- [`Clustering/Embeddings Generation/Node2Vec/node2vec_tx-lcc.py`](Clustering/Embeddings%20Generation/Node2Vec/node2vec_tx-lcc.py) runs Node2Vec on the LCC.
- [`Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/node2vec_pecanpy.py`](Clustering/Embeddings%20Generation/Node2Vec/n2v_pecenpy/node2vec_pecanpy.py) is another Node2Vec implementation using `pecanpy`.
- [`Clustering/Embeddings Generation/Node2Vec/n2v_pecenpy/check-emb.ipynb`](Clustering/Embeddings%20Generation/Node2Vec/n2v_pecenpy/check-emb.ipynb) checks the generated embeddings.

`Role2Vec`

- [`Clustering/Embeddings Generation/Role2Vec/prepare-data.py`](Clustering/Embeddings%20Generation/Role2Vec/prepare-data.py) prepares graph data and node mappings for Role2Vec-style processing.
- [`Clustering/Embeddings Generation/Role2Vec/map-embeddings-back.py`](Clustering/Embeddings%20Generation/Role2Vec/map-embeddings-back.py) maps raw embedding ids back to account ids.

#### `Clustering/K-Means/`

This part contains K-Means experiments on embeddings.

- [`Clustering/K-Means/kmeans-emb.py`](Clustering/K-Means/kmeans-emb.py) runs K-Means for many values of `k` and saves cluster assignments.
- [`Clustering/K-Means/evaluation.py`](Clustering/K-Means/evaluation.py) computes intrinsic and external clustering metrics.
- [`Clustering/K-Means/evaluation-kmeans.py`](Clustering/K-Means/evaluation-kmeans.py) evaluates one selected K-Means result against normalized labels.
- [`Clustering/K-Means/k-means-cross-validation.ipynb`](Clustering/K-Means/k-means-cross-validation.ipynb) does cross-validation style evaluation.
- [`Clustering/K-Means/check.ipynb`](Clustering/K-Means/check.ipynb) checks saved result files.
- `LINE_res/` and `Node2Vec_res/` contain evaluation notebooks for the corresponding embedding source.

#### `Clustering/DBSCAN/`

This part contains density-based clustering experiments.

- [`Clustering/DBSCAN/hdbscan-test.py`](Clustering/DBSCAN/hdbscan-test.py) runs HDBSCAN on embeddings (currently configured for LINE in the tracked file).
- [`Clustering/DBSCAN/dbscan-final.py`](Clustering/DBSCAN/dbscan-final.py) runs DBSCAN with a parameter grid using PCA-reduced embeddings.
- [`Clustering/DBSCAN/evaluation.py`](Clustering/DBSCAN/evaluation.py) computes intrinsic/external scores.
- [`Clustering/DBSCAN/evaluation-hdbscan.py`](Clustering/DBSCAN/evaluation-hdbscan.py) evaluates a selected HDBSCAN result.
- [`Clustering/DBSCAN/check.ipynb`](Clustering/DBSCAN/check.ipynb) checks stored outputs.
- `LINE_res/` and `Node2Vec_res/` again contain evaluation notebooks and a couple of saved log files.

### `Community Detection/`

This folder contains graph-native community detection experiments.

#### `Community Detection/Louvain/`

- [`Community Detection/Louvain/louvain-resolutions.py`](Community%20Detection/Louvain/louvain-resolutions.py) runs Louvain on the LCC for several resolution values and saves one CSV per resolution plus a summary table.
- [`Community Detection/Louvain/resolutions/evaluation/evaluation.py`](Community%20Detection/Louvain/resolutions/evaluation/evaluation.py) evaluates Louvain communities against known labels.
- [`Community Detection/Louvain/resolutions/evaluation-bset.ipynb`](Community%20Detection/Louvain/resolutions/evaluation-bset.ipynb) compares saved resolution results.
- [`Community Detection/Louvain/resolutions/analysis/lcc-louvian0.5-analysis.ipynb`](Community%20Detection/Louvain/resolutions/analysis/lcc-louvian0.5-analysis.ipynb) explores structural properties of selected communities.
- [`Community Detection/Louvain/resolutions/5-fold-cv/llc-louvian-res0.5-cross-validation.ipynb`](Community%20Detection/Louvain/resolutions/5-fold-cv/llc-louvian-res0.5-cross-validation.ipynb) does cross-validation style checking for one result file.

#### `Community Detection/LPA/`

- [`Community Detection/LPA/run-lpa.py`](Community%20Detection/LPA/run-lpa.py) runs asynchronous label propagation (`asyn_lpa_communities`) on the transaction LCC and saves community assignments plus stats.
- [`Community Detection/LPA/evaluation/evaluation-lpa.py`](Community%20Detection/LPA/evaluation/evaluation-lpa.py) evaluates LPA results against normalized labels.
- [`Community Detection/LPA/evaluation.ipynb`](Community%20Detection/LPA/evaluation.ipynb) is a notebook version of result checking.
- [`Community Detection/LPA/unsupervized-lpa-cross-validation.ipynb`](Community%20Detection/LPA/unsupervized-lpa-cross-validation.ipynb) contains unsupervised / cross-validation style analysis.

#### `Community Detection/SSLPA/`

This is the biggest and most custom part of the repo.

- [`Community Detection/SSLPA/manual/ss-lpa-man-full-labels.py`](Community%20Detection/SSLPA/manual/ss-lpa-man-full-labels.py) implements a manual weighted semi-supervised label propagation process using known labels as seeds.
- [`Community Detection/SSLPA/manual/check.ipynb`](Community%20Detection/SSLPA/manual/check.ipynb) checks raw vs normalized SSLPA outputs.
- [`Community Detection/SSLPA/manual/normalized/evaluation.ipynb`](Community%20Detection/SSLPA/manual/normalized/evaluation.ipynb) evaluates normalized SSLPA results.
- [`Community Detection/SSLPA/manual/normalized/cross-validation/sslpa-cv.py`](Community%20Detection/SSLPA/manual/normalized/cross-validation/sslpa-cv.py) runs 5-fold stratified cross-validation for SSLPA and saves per-fold and summary tables.
- [`Community Detection/SSLPA/manual/normalized/evaluation-cv/final-evaluation-cv.py`](Community%20Detection/SSLPA/manual/normalized/evaluation-cv/final-evaluation-cv.py) looks like another evaluation/cross-validation variant for the manual SSLPA pipeline.

There is also a deeper `ncsf/` workflow inside the normalized SSLPA folder:

The `ncsf/` folder contains the detailed SSLPA post-processing and evaluation pipeline. Briefly, the NCSF filter works by checking whether a node's propagated label is supported by its local neighborhood in the graph. This part of the project also includes entity-level evaluation, binary scam-detection evaluation, publication plots, and semi-supervised masking experiments. The detailed explanation is in the dedicated NCSF README linked at the top of this document.



#### `Community Detection/cv-comparison/`

- [`Community Detection/cv-comparison/comp.ipynb`](Community%20Detection/cv-comparison/comp.ipynb) compares cross-validation results from Louvain, LPA, and SSLPA.

### `Evaluation/`

This looks like the final comparison area.

- [`Evaluation/evaluation-all.ipynb`](Evaluation/evaluation-all.ipynb) combines multiple evaluation CSVs.
- [`Evaluation/clustering_comparison.png`](Evaluation/clustering_comparison.png) is a saved comparison figure.

## Methods and libraries used

From the tracked Python files, the main libraries used are:

- `pandas`
- `numpy`
- `networkx`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `node2vec`
- `pecanpy`
- `gensim`
- `karateclub`
- `hdbscan`

There is no `requirements.txt`, `pyproject.toml`, or environment file in the tracked repo, so dependencies have to be installed manually.

## Workflow

1. Prepare raw transaction data.
2. Run [`create-network.py`](data/pkl/create-network.py).
3. Run [`create-lcc.py`](data/LCC/create-lcc.py).
4. Run label mapping with [`map_labels.py`](labeled-data/map-labels/map_labels.py).
5. Normalize entity names with [`label-normalization.py`](labeled-data/normalization/label-normalization.py).
6. Generate embeddings with LINE / Node2Vec / Role2Vec helpers.
7. Run clustering methods like K-Means and DBSCAN/HDBSCAN.
8. Run community detection methods like Louvain, LPA, and SSLPA.
9. Evaluate and compare results using the notebooks/scripts in each result folder.
