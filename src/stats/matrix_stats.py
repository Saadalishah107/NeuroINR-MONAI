"""
src/stats/matrix_stats.py

(1) Matrix-computation ML: PCA (via SVD) on the CNN's learned
    penultimate-layer feature embeddings, both as a dimensionality
    reduction step feeding the quantum-feature-map extension, and as a
    standalone diagnostic (explained variance, 2D projection colored by
    tumour label).

(2) Statistical analysis: paired t-test comparing the traveling-model vs
    FedAvg final AUPRC across repeated runs (src/federated/traveling_model.py
    `compare_strategies` output) -- answers "is one strategy reliably
    better, or is the difference within noise across repeats?".
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def pca_via_svd(X: np.ndarray, n_components: int = 2):
    """
    X: (N, D) feature matrix (rows = samples).
    Returns dict with: components (D, k), explained_variance_ratio (k,),
    projected (N, k), mean (D,).

    Implemented via SVD (not sklearn.decomposition.PCA) to make the
    "matrix computation ML" claim concrete: X_centered = U S V^T, and the
    principal components are exactly the right singular vectors V.
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum()
    explained_variance_ratio = (S ** 2) / total_var
    components = Vt[:n_components]                      # (k, D)
    projected = Xc @ components.T                        # (N, k)
    return {
        "mean": mean.squeeze(0),
        "components": components,
        "singular_values": S[:n_components],
        "explained_variance_ratio": explained_variance_ratio[:n_components],
        "projected": projected,
    }


def paired_ttest(sample_a, sample_b, alternative="two-sided"):
    """
    Paired t-test between two equal-length sequences of matched
    measurements (e.g. traveling-model AUPRC vs FedAvg AUPRC, same seed
    per pair). Returns statistic, p-value, and the mean difference with
    its 95% CI (Student's t).
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    assert a.shape == b.shape, "paired samples must be the same length"

    result = sp_stats.ttest_rel(a, b, alternative=alternative)
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t_crit = sp_stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se) if n > 1 else (float("nan"),) * 2

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_difference": float(mean_diff),
        "ci_95": (float(ci[0]), float(ci[1])),
        "n_pairs": n,
    }
