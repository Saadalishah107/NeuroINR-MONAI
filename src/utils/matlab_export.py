"""
src/utils/matlab_export.py

Exports pipeline artefacts (INR reconstructions at multiple resolutions,
Grad-CAM heatmaps, evaluation metrics) as .mat files for clinical /
collaborator review in MATLAB or Octave (free, file-compatible), via
scipy.io.savemat -- no MATLAB Engine / license required to WRITE the
files, only to open them.
"""
from __future__ import annotations

import numpy as np
from scipy.io import savemat


def export_resolution_experiment(path: str, results: dict, original_image: np.ndarray):
    """`results` is the dict returned by src.training.train_inr.resolution_experiment:
    {scale_name: {"reconstruction":..., "target":..., "psnr":..., "ssim":...}}"""
    mat_dict = {"original_image": original_image.astype(np.float64)}
    for scale_name, r in results.items():
        key = scale_name.replace("x", "_x").replace("px", "").replace("__", "_")
        mat_dict[f"recon_{key}"] = r["reconstruction"].astype(np.float64)
        mat_dict[f"target_{key}"] = r["target"].astype(np.float64)
        mat_dict[f"psnr_{key}"] = float(r["psnr"])
        mat_dict[f"ssim_{key}"] = float(r["ssim"])
    savemat(path, mat_dict)


def export_gradcam(path: str, image: np.ndarray, heatmap: np.ndarray, predicted_prob: float, true_label: int):
    savemat(path, {
        "image": image.astype(np.float64),
        "gradcam_heatmap": heatmap.astype(np.float64),
        "predicted_prob_tumour": float(predicted_prob),
        "true_label": int(true_label),
    })


def _flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            _flatten(key, v, out)
    elif isinstance(value, (int, float, str, bool)):
        out[prefix] = value
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            out[prefix] = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            # e.g. a list of mixed types / strings -- store as a MATLAB cell array
            out[prefix] = np.array(value, dtype=object)
    else:
        out[prefix] = str(value)


def export_metrics_table(path: str, metrics_dict: dict):
    """metrics_dict: any (arbitrarily nested) dict of numbers/strings/arrays/
    sub-dicts, e.g. run_pipeline.py's full `report`. Nested keys are joined
    with underscores (e.g. report['cnn_test']['auroc'] -> MATLAB variable
    `cnn_test_auroc`) so every leaf value becomes its own .mat variable."""
    clean: dict = {}
    _flatten("", metrics_dict, clean)
    clean = {(k if k else "value"): v for k, v in clean.items()}
    savemat(path, clean)
