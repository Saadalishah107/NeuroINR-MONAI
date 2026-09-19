"""Classification and image-quality metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(thresholds) == 0:
        return 0.5
    candidates = np.concatenate(([0.0], thresholds, [1.0]))
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_report_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = float("nan")
        auprc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "threshold": float(threshold),
        "n_positive": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def psnr(img_true: np.ndarray, img_pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((img_true.astype(np.float64) - img_pred.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(img_true: np.ndarray, img_pred: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(img_true, img_pred, data_range=data_range))
