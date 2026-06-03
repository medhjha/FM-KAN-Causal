"""
Step 07: Cross-Dataset Causal Ranking Validation
Computes pairwise Spearman rank correlations of causal importance
vectors across all four datasets and runs permutation null testing.
"""

import os
import json
import numpy as np
from scipy import stats

BASE = "."
SHARED = os.path.join(BASE, "outputs", "shared")
COMBINED_DIR = os.path.join(BASE, "outputs", "combined", "data")
DATASETS = ["gse26304", "gse21422", "gse3893", "gse72205"]
RANDOM_SEED = 42
N_PERMUTATIONS = 10000


def load_causal_scores():
    with open(os.path.join(SHARED, "config.json")) as f:
        config = json.load(f)
    genes = config["gene_panel"]

    dataset_scores = {}
    for ds in DATASETS:
        path = os.path.join(BASE, "outputs", ds, "data", "causal_results.json")
        if not os.path.exists(path):
            print(f"  Missing causal results for {ds}, skipping.")
            continue
        with open(path) as f:
            cr = json.load(f)

        gene_scores_raw = cr.get("gene_scores", [])
        if isinstance(gene_scores_raw, list):
            score_dict = {item["gene"]: item["score"] for item in gene_scores_raw if "gene" in item}
        elif isinstance(gene_scores_raw, dict):
            score_dict = gene_scores_raw
        else:
            score_dict = {}

        score_vector = np.array([score_dict.get(g, 0.0) for g in genes], dtype=np.float64)

        max_score = score_vector.max()
        if max_score > 0:
            score_vector = score_vector / max_score

        dataset_scores[ds] = score_vector
        top3 = [genes[i] for i in np.argsort(score_vector)[-3:][::-1]]
        print(f"  {ds}: top3 = {top3}")

    return dataset_scores, genes


def compute_pairwise_rho(dataset_scores):
    ds_names = list(dataset_scores.keys())
    pairwise = {}
    rho_values = []

    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            d1 = ds_names[i]
            d2 = ds_names[j]
            v1 = dataset_scores[d1]
            v2 = dataset_scores[d2]

            rho, pval = stats.spearmanr(v1, v2)
            key = f"{d1}_vs_{d2}"
            pairwise[key] = {
                "rho": round(float(rho), 4),
                "pval": round(float(pval), 6),
                "sig": bool(pval < 0.05)
            }
            rho_values.append(rho)
            print(f"  {d1} vs {d2}: rho={rho:.4f}, p={pval:.4f}")

    return pairwise, rho_values


def run_permutation_null(dataset_scores, n_perms=N_PERMUTATIONS):
    ds_names = list(dataset_scores.keys())
    rng = np.random.default_rng(RANDOM_SEED)
    null_rhos = []

    for _ in range(n_perms):
        shuffled = {}
        for ds in ds_names:
            v = dataset_scores[ds].copy()
            rng.shuffle(v)
            shuffled[ds] = v

        pair_rhos = []
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                rho, _ = stats.spearmanr(shuffled[ds_names[i]], shuffled[ds_names[j]])
                pair_rhos.append(rho)
        null_rhos.append(np.mean(pair_rhos))

    null_rhos = np.array(null_rhos)
    null_mean = float(np.mean(null_rhos))
    null_p95 = float(np.percentile(null_rhos, 95))
    null_p99 = float(np.percentile(null_rhos, 99))

    print(f"\nPermutation null (n={n_perms}): mean={null_mean:.4f}, 95th={null_p95:.4f}, 99th={null_p99:.4f}")
    return null_mean, null_p95, null_p99


def save_results(pairwise, rho_values, null_mean, null_p99, dataset_scores, genes):
    ds_names = list(dataset_scores.keys())
    observed_mean = float(np.mean(rho_values))
    n_sig = sum(1 for v in pairwise.values() if v["sig"])

    warburg = {"LDHA", "CA9", "SLC2A1", "HIF1A", "HK2", "PKM", "LDHB", "PDK1", "BNIP3", "VEGFA"}
    warburg_overlap = {}
    for ds in ds_names:
        scores = dataset_scores[ds]
        top10_idx = np.argsort(scores)[-10:][::-1]
        top10_genes = [genes[i] for i in top10_idx]
        warburg_in_top10 = [g for g in top10_genes if g in warburg]
        warburg_overlap[ds] = {
            "top10": top10_genes,
            "warburg_genes": warburg_in_top10,
            "n_warburg": len(warburg_in_top10)
        }

    empirical_p = float(np.mean(np.array([null_mean]) > observed_mean))

    results = {
        "datasets": ds_names,
        "pairwise_rho": pairwise,
        "mean_spearman_rho": round(observed_mean, 4),
        "rho_range": [round(min(rho_values), 4), round(max(rho_values), 4)],
        "n_significant_pairs": n_sig,
        "n_pairs_above_0_40": sum(1 for v in pairwise.values() if v["rho"] > 0.40),
        "target_met": bool(sum(1 for v in pairwise.values() if v["rho"] > 0.40) >= 3),
        "permutation_null_mean": round(null_mean, 4),
        "permutation_null_p99": round(null_p99, 4),
        "permutation_empirical_p": round(empirical_p, 4),
        "warburg_overlap": warburg_overlap
    }

    os.makedirs(COMBINED_DIR, exist_ok=True)
    with open(os.path.join(COMBINED_DIR, "crossdataset_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMean Spearman rho = {observed_mean:.4f}")
    print(f"Significant pairs: {n_sig}/6")
    print(f"Target met (rho>0.40 in >=3/6): {results['target_met']}")


def main():
    print("Cross-Dataset Causal Ranking Validation")
    dataset_scores, genes = load_causal_scores()

    if len(dataset_scores) < 2:
        print("Not enough datasets with causal results. Run step 05 first.")
        return

    pairwise, rho_values = compute_pairwise_rho(dataset_scores)
    null_mean, null_p95, null_p99 = run_permutation_null(dataset_scores)
    save_results(pairwise, rho_values, null_mean, null_p99, dataset_scores, genes)
    print("Step 07 complete.")


if __name__ == "__main__":
    main()
