"""
Step 00: Setup
Creates all required directories and saves the 60-gene panel,
beta_k weights, and config file to the shared outputs folder.
"""

import os
import json
import numpy as np

BASE = "."
SHARED = os.path.join(BASE, "outputs", "shared")

GENE_PANEL = [
    "LDHA", "LDHB", "SLC2A1", "SLC2A3", "HK2", "PKM", "ENO1", "PFKL", "ALDOA", "GAPDH",
    "HIF1A", "CA9", "VEGFA", "EPO", "PDK1", "BNIP3",
    "MMP2", "MMP9", "MMP14", "ITGB1", "ITGAV", "FN1", "COL1A1", "COL4A1",
    "PIK3CA", "AKT1", "PTEN", "MTOR", "RPS6KB1", "EIF4EBP1",
    "CCND1", "CDK4", "CDKN1A", "CDKN2A", "RB1", "TP53", "BCL2", "BAX", "MKI67", "PCNA",
    "CD44", "CD24", "ALDH1A1", "EPCAM", "VIM", "CDH1", "CDH2", "ZEB1", "SNAI1", "TWIST1",
    "BRCA1", "BRCA2", "RAD51", "CHEK2", "ATM", "H2AFX", "ESR1", "PGR", "ERBB2", "ERBB3"
]

BETA_K = [
    0.89, 0.71, 0.84, 0.65, 0.76, 0.71, 0.60, 0.55, 0.53, 0.50,
    0.87, 0.83, 0.79, 0.63, 0.68, 0.52,
    0.74, 0.77, 0.73, 0.62, 0.61, 0.64, 0.59, 0.55,
    0.81, 0.78, -0.65, 0.76, 0.60, 0.54,
    0.78, 0.55, -0.68, -0.72, -0.72, -0.80, -0.45, -0.75, 0.82, 0.56,
    0.54, -0.55, 0.57, 0.61, 0.60, -0.58, 0.59, -0.60, -0.55, -0.57,
    -0.73, -0.65, -0.60, -0.55, -0.52, -0.57, -0.75, -0.65, 0.80, 0.50
]

DATASETS = ["gse26304", "gse21422", "gse3893", "gse72205", "combined"]

CONFIG = {
    "gene_panel": GENE_PANEL,
    "random_seed": 42,
    "test_size": 0.2,
    "fm_hidden_dim": 128,
    "fm_latent_dim": 8,
    "fm_epochs": 400,
    "fm_lr": 0.001,
    "kan_hidden_layers": [32, 16],
    "kan_grid": 5,
    "kan_k": 3,
    "kan_lr": 0.001,
    "kan_epochs": 300,
    "kan_lasso_C": 0.1,
    "dag_alpha": 0.05,
    "dag_max_cond_set": 3,
    "n_permutations_dag": 1000,
    "n_permutations_null": 10000,
    "n_bootstrap": 1000
}


def create_directories():
    dirs = [SHARED, "data", os.path.join("data", "geo"), "logs"]
    for ds in DATASETS:
        dirs.append(os.path.join(BASE, "outputs", ds, "data"))
        dirs.append(os.path.join(BASE, "outputs", ds, "figures"))
    dirs.append(os.path.join(BASE, "figures"))
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("All directories created.")


def save_shared_files():
    assert len(GENE_PANEL) == 60
    assert len(BETA_K) == 60

    with open(os.path.join(SHARED, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    np.save(os.path.join(SHARED, "beta_k.npy"), np.array(BETA_K, dtype=np.float32))

    with open(os.path.join(SHARED, "gene_panel.txt"), "w") as f:
        for gene in GENE_PANEL:
            f.write(gene + "\n")

    adjacency = np.zeros((60, 60), dtype=np.float32)
    pathway_bounds = [(0,10),(10,16),(16,24),(24,30),(30,40),(40,50),(50,60)]
    for start, end in pathway_bounds:
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    adjacency[i, j] = 1.0
    np.save(os.path.join(SHARED, "adjacency_matrix.npy"), adjacency)

    print("Saved config.json, beta_k.npy, gene_panel.txt, adjacency_matrix.npy")


if __name__ == "__main__":
    create_directories()
    save_shared_files()
    print("Step 00 complete.")
