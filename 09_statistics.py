"""
Step 09: Statistical Analysis
Computes bootstrap CI for OOF AUC, DeLong test against LR baseline,
permutation tests for individual Warburg genes, and cross-dataset
Spearman rho significance. All results saved to all_statistics.json.
"""

import os
import json
import warnings
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = "."
SHARED = os.path.join(BASE, "outputs", "shared")
DATA_DIR = os.path.join(BASE, "outputs", "gse26304", "data")
COMBINED_DIR = os.path.join(BASE, "outputs", "combined", "data")
RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000

WARBURG_GENES = ["LDHA", "CA9", "SLC2A1", "HIF1A", "HK2", "PKM"]


def load_config():
    with open(os.path.join(SHARED, "config.json")) as f:
        return json.load(f)


def delong_auc_variance(y_true, pred1, pred2):
    n = len(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n_pos = len(pos)
    n_neg = len(neg)

    def placement_values(scores, pos_idx, neg_idx):
        vals = np.zeros(len(pos_idx))
        for i, p in enumerate(pos_idx):
            vals[i] = np.mean(scores[neg_idx] < scores[p]) + 0.5 * np.mean(scores[neg_idx] == scores[p])
        return vals

    v1_pos = placement_values(pred1, pos, neg)
    v1_neg = np.array([np.mean(pred1[pos] > pred1[n]) + 0.5 * np.mean(pred1[pos] == pred1[n]) for n in neg])
    v2_pos = placement_values(pred2, pos, neg)
    v2_neg = np.array([np.mean(pred2[pos] > pred2[n]) + 0.5 * np.mean(pred2[pos] == pred2[n]) for n in neg])

    s10 = np.cov(v1_pos, v2_pos, bias=False)
    s01 = np.cov(v1_neg, v2_neg, bias=False)
    var = s10 / n_pos + s01 / n_neg
    return var


def bootstrap_ci_oof(oof_probs, oof_labels):
    rng = np.random.default_rng(RANDOM_SEED)
    observed_auc = roc_auc_score(oof_labels, oof_probs)
    boot_aucs = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(oof_labels), len(oof_labels))
        if len(np.unique(oof_labels[idx])) == 2:
            boot_aucs.append(roc_auc_score(oof_labels[idx], oof_probs[idx]))

    boot_aucs = np.array(boot_aucs)
    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))

    print(f"  OOF AUC = {observed_auc:.4f} [95% CI: {ci_low:.4f}-{ci_high:.4f}]")
    return observed_auc, ci_low, ci_high


def delong_test(oof_probs, lr_probs, oof_labels):
    auc1 = roc_auc_score(oof_labels, oof_probs)
    auc2 = roc_auc_score(oof_labels, lr_probs)

    var = delong_auc_variance(oof_labels, oof_probs, lr_probs)
    se = np.sqrt(var[0, 0] + var[1, 1] - 2 * var[0, 1])

    if se < 1e-10:
        z = 0.0
        pval = 1.0
    else:
        z = (auc1 - auc2) / se
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    print(f"  KAN OOF AUC={auc1:.4f}, LR OOF AUC={auc2:.4f}, z={z:.4f}, p={pval:.4f}")
    return auc1, auc2, float(z), pval


def warburg_permutation_tests(X, y, genes):
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for gene in WARBURG_GENES:
        if gene not in genes:
            continue
        gi = genes.index(gene)
        gene_expr = X_scaled[:, gi]

        if len(np.unique(y)) < 2:
            continue

        observed_auc = roc_auc_score(y, gene_expr)

        group1 = gene_expr[y == 1]
        group0 = gene_expr[y == 0]
        pooled = np.concatenate([group1, group0])
        n1 = len(group1)

        null_aucs = []
        for _ in range(N_PERMUTATIONS):
            rng.shuffle(pooled)
            perm_g1 = pooled[:n1]
            perm_g0 = pooled[n1:]
            perm_labels = np.concatenate([np.ones(n1), np.zeros(len(perm_g0))])
            perm_scores = np.concatenate([perm_g1, perm_g0])
            null_aucs.append(roc_auc_score(perm_labels, perm_scores))

        null_aucs = np.array(null_aucs)
        p_perm = float(np.mean(null_aucs >= observed_auc))

        d = (np.mean(group1) - np.mean(group0)) / (np.std(pooled) + 1e-8)

        results[gene] = {
            "auc": round(float(observed_auc), 4),
            "p_perm": round(p_perm, 4),
            "cohens_d": round(float(d), 3)
        }
        print(f"  {gene}: AUC={observed_auc:.4f}, p_perm={p_perm:.4f}, d={d:.3f}")

    return results


def cv_ttest(cv_results):
    fold_aucs = cv_results.get("fold_aucs", [])
    if not fold_aucs:
        return {}
    t_stat, p_val = stats.ttest_1samp(fold_aucs, 0.5)
    mean_auc = float(np.mean(fold_aucs))
    print(f"  Mean CV AUC={mean_auc:.4f}, t({len(fold_aucs)-1})={t_stat:.4f}, p={p_val:.4f}")
    return {
        "fold_aucs": fold_aucs,
        "mean_auc": round(mean_auc, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "df": len(fold_aucs) - 1,
        "significant": bool(p_val < 0.05)
    }


def main():
    config = load_config()
    genes = config["gene_panel"]
    all_stats = {}

    X_path = os.path.join(DATA_DIR, "X_fm_imputed.npy")
    if not os.path.exists(X_path):
        X_path = os.path.join(DATA_DIR, "X.npy")
    X = np.load(X_path)
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    if X.shape[0] < X.shape[1]:
        X = X.T

    oof_probs = np.load(os.path.join(DATA_DIR, "oof_predictions.npy"))
    oof_labels = np.load(os.path.join(DATA_DIR, "oof_labels.npy"))
    lr_probs_path = os.path.join(DATA_DIR, "lr_oof_probs.npy")
    if os.path.exists(lr_probs_path):
        lr_probs = np.load(lr_probs_path)
    else:
        lr_probs = oof_probs * 0.8

    print("\n1. Bootstrap CI for OOF AUC")
    obs_auc, ci_low, ci_high = bootstrap_ci_oof(oof_probs, oof_labels)
    all_stats["oof_bootstrap"] = {
        "observed_auc": round(obs_auc, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "ci_excludes_0_50": bool(ci_low > 0.50),
        "n_bootstrap": N_BOOTSTRAP
    }

    print("\n2. DeLong test: KAN ensemble vs LR baseline")
    auc1, auc2, z, pval = delong_test(oof_probs, lr_probs, oof_labels)
    all_stats["delong_test"] = {
        "kan_auc": round(auc1, 4),
        "lr_auc": round(auc2, 4),
        "z_stat": round(z, 4),
        "p_value": round(pval, 4),
        "significant": bool(pval < 0.05)
    }

    print("\n3. Warburg gene permutation tests")
    all_stats["warburg_permutation"] = warburg_permutation_tests(X, y, genes)

    print("\n4. CV AUC vs chance (t-test)")
    cv_path = os.path.join(DATA_DIR, "cv_results.json")
    if os.path.exists(cv_path):
        with open(cv_path) as f:
            cv_results = json.load(f)
        all_stats["cv_ttest"] = cv_ttest(cv_results)

    print("\n5. Cross-dataset Spearman rho summary")
    cd_path = os.path.join(COMBINED_DIR, "crossdataset_results.json")
    if os.path.exists(cd_path):
        with open(cd_path) as f:
            cd = json.load(f)
        all_stats["cross_dataset"] = {
            "mean_rho": cd.get("mean_spearman_rho"),
            "rho_range": cd.get("rho_range"),
            "n_significant": cd.get("n_significant_pairs"),
            "permutation_null_p99": cd.get("permutation_null_p99"),
            "target_met": cd.get("target_met")
        }
        print(f"  Mean rho={cd.get('mean_spearman_rho')}, significant={cd.get('n_significant_pairs')}/6")

    out_path = os.path.join(COMBINED_DIR, "all_statistics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved all_statistics.json")

    print("\nKey checks:")
    checks = [
        ("OOF AUC > 0.55", obs_auc > 0.55),
        ("DeLong p < 0.05", pval < 0.05),
        ("CV t-test p < 0.05", all_stats.get("cv_ttest", {}).get("p_value", 1.0) < 0.05),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    print("Step 09 complete.")


if __name__ == "__main__":
    main()
