#!/usr/bin/env python3
"""End-to-end NeuroINR-MONAI experimental pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _save_figures(out_dir, cnn_history, heatmap, example_image, res_results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["epoch"] for h in cnn_history], [h["train_loss"] for h in cnn_history], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("CNN training loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "cnn_training_loss.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(example_image, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")
    ax.set_title("Grad-CAM tumour-class attribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "gradcam.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(res_results.keys())
    psnr_vals = [res_results[k]["psnr"] for k in names]
    ax.plot(names, psnr_vals, marker="o")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("Query resolution")
    ax.set_title("SIREN resolution experiment")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "siren_resolution_psnr.png"), dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic_fallback", action="store_true")
    ap.add_argument("--real_quick", action="store_true")
    ap.add_argument("--modality_mode", choices=["flair", "multimodal"], default="flair")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--limit_patients", type=int, default=40)
    ap.add_argument("--slices_per_patient", type=int, default=8)
    ap.add_argument("--cnn_epochs", type=int, default=None)
    ap.add_argument("--inr_steps", type=int, default=None)
    ap.add_argument("--fed_rounds", type=int, default=None)
    ap.add_argument("--fed_local_steps", type=int, default=None)
    ap.add_argument("--fed_repeats", type=int, default=None)
    ap.add_argument("--n_sites", type=int, default=4)
    ap.add_argument("--run_vit", action="store_true")
    ap.add_argument("--vit_epochs", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    args = ap.parse_args()

    if args.quick:
        args.cnn_epochs = args.cnn_epochs or 5
        args.inr_steps = args.inr_steps or 150
        args.fed_rounds = args.fed_rounds or 2
        args.fed_local_steps = args.fed_local_steps or 10
        args.fed_repeats = args.fed_repeats or 2
        args.limit_patients = min(args.limit_patients, 12)
        args.slices_per_patient = min(args.slices_per_patient, 4)
    elif args.real_quick:
        args.cnn_epochs = args.cnn_epochs or 3
        args.inr_steps = args.inr_steps or 100
        args.fed_rounds = args.fed_rounds or 1
        args.fed_local_steps = args.fed_local_steps or 5
        args.fed_repeats = args.fed_repeats or 1
        args.limit_patients = min(args.limit_patients, 12)
        args.slices_per_patient = min(args.slices_per_patient, 4)
    else:
        args.cnn_epochs = args.cnn_epochs or 20
        args.inr_steps = args.inr_steps or 500
        args.fed_rounds = args.fed_rounds or 6
        args.fed_local_steps = args.fed_local_steps or 30
        args.fed_repeats = args.fed_repeats or 5

    for sub in ["checkpoints", "figures", "matlab_exports"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")

    report = {
        "device": str(device),
        "modality_mode": args.modality_mode,
        "in_channels": 4 if args.modality_mode == "multimodal" and not args.synthetic_fallback else 1,
    }

    print("\n=== [1/8] Loading data ===")
    if args.synthetic_fallback:
        from src.data.synthetic_fallback import make_phantom_dataset
        from src.data.dataset import SliceRecord
        imgs, labels = make_phantom_dataset(n_per_class=60, size=args.img_size)
        all_records = [SliceRecord(
            f"phantom_{i}", 0, imgs[i],
            (imgs[i] > 0.9).astype(np.uint8) * int(labels[i]), int(labels[i])
        ) for i in range(len(labels))]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(all_records))
        all_records = [all_records[i] for i in perm]
        n = len(all_records)
        splits = {
            "train": all_records[:int(n * 0.70)],
            "val": all_records[int(n * 0.70):int(n * 0.85)],
            "test": all_records[int(n * 0.85):],
        }
        train_patients_for_sites = None
        report["in_channels"] = 1
    else:
        from src.data.dataset import download_task01_braintumour, split_patients, build_slice_dataset
        data_root = os.path.join(os.path.dirname(args.out_dir), "data")
        patient_list = download_task01_braintumour(data_root, limit_patients=args.limit_patients)
        patient_splits = split_patients(patient_list)
        modality = "multimodal" if args.modality_mode == "multimodal" else "FLAIR"
        splits = {
            k: build_slice_dataset(
                v, modality=modality, img_size=args.img_size,
                slices_per_patient=args.slices_per_patient,
            ) for k, v in patient_splits.items()
        }
        train_patients_for_sites = patient_splits["train"]

    print({k: len(v) for k, v in splits.items()})
    if min(len(v) for v in splits.values()) == 0:
        raise RuntimeError("One or more splits contain zero usable slices.")

    from src.data.dataset import make_torch_dataset
    train_loader = DataLoader(make_torch_dataset(splits["train"]), batch_size=16, shuffle=True)
    val_loader = DataLoader(make_torch_dataset(splits["val"]), batch_size=16, shuffle=False)
    test_loader = DataLoader(make_torch_dataset(splits["test"]), batch_size=16, shuffle=False)

    print("\n=== [2/8] Training CNN baseline ===")
    from src.training.train_cnn import train as train_cnn_fn, evaluate as eval_cnn_fn, calibrate_threshold
    from src.models.cnn import BrainMRICNN
    from src.utils.metrics import find_best_f1_threshold

    train_labels = np.array([r.tumour_present for r in splits["train"]])
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    class_weights = [1.0, float(n_neg) / max(1, n_pos)] if n_pos > 0 else None

    in_channels = report["in_channels"]
    cnn = BrainMRICNN(in_channels=in_channels)
    cnn, cnn_history = train_cnn_fn(
        cnn, train_loader, val_loader, device,
        epochs=args.cnn_epochs, class_weights=class_weights,
    )
    threshold, val_calibrated = calibrate_threshold(cnn, val_loader, device)
    cnn_test_report, cnn_test_probs, cnn_test_labels = eval_cnn_fn(
        cnn, test_loader, device, threshold=threshold,
    )
    report["cnn_validation_calibrated"] = val_calibrated
    report["cnn_test"] = cnn_test_report
    report["cnn_test"]["n_train"] = len(train_loader.dataset)
    torch.save({
        "state_dict": cnn.state_dict(),
        "in_channels": in_channels,
        "threshold": threshold,
        "test_report": cnn_test_report,
    }, os.path.join(args.out_dir, "checkpoints", "cnn_brain_mri.pt"))
    print("CNN test report:", cnn_test_report)

    if args.run_vit:
        print("\n=== [3/8] Vision Transformer comparison ===")
        from src.training.train_cnn import build_model, train as train_model
        vit = build_model("vit", args.img_size, in_channels=in_channels)
        vit, vit_history = train_model(
            vit, train_loader, val_loader, device,
            epochs=args.vit_epochs, class_weights=class_weights,
        )
        vit_threshold, _ = calibrate_threshold(vit, val_loader, device)
        vit_report, _, _ = eval_cnn_fn(vit, test_loader, device, threshold=vit_threshold)
        report["vit_test"] = vit_report
    else:
        vit_history = []

    print("\n=== [4/8] Training SIREN INR + resolution experiment ===")
    from src.training.train_inr import fit_siren, resolution_experiment
    example = splits["test"][0]
    example_image = example.image[0] if np.asarray(example.image).ndim == 3 else example.image
    siren_model, siren_losses = fit_siren(
        example_image, device=device, steps=args.inr_steps,
        verbose=True,
    )
    res_results = resolution_experiment(
        siren_model, example_image, args.img_size, device=device,
    )
    report["siren_resolution_experiment"] = {
        k: {"psnr": v["psnr"], "ssim": v["ssim"]} for k, v in res_results.items()
    }
    print(report["siren_resolution_experiment"])
    torch.save({"state_dict": siren_model.state_dict()}, os.path.join(args.out_dir, "checkpoints", "siren_mri.pt"))

    print("\n=== [5/8] Explainability ===")
    from src.explainability.gradcam import GradCAM
    from src.explainability.coordinate_shapley import make_torch_model_callable, shapley_xy_decomposition
    cnn.to(device)
    ex_tensor = torch.from_numpy(np.asarray(example.image)).float()
    if ex_tensor.ndim == 2:
        ex_tensor = ex_tensor.unsqueeze(0)
    ex_tensor = ex_tensor.unsqueeze(0).to(device)
    heatmap = GradCAM(cnn, cnn.block4.conv)(ex_tensor, class_idx=1)
    f_callable = make_torch_model_callable(siren_model, device=device)
    phi_x, phi_y, f_base, f_full = shapley_xy_decomposition(f_callable, x=0.1, y=-0.1)
    report["coordinate_shapley_example"] = {
        "phi_x": phi_x,
        "phi_y": phi_y,
        "efficiency_check": phi_x + phi_y - (f_full - f_base),
    }
    print(report["coordinate_shapley_example"])

    print("\n=== [6/8] Traveling-model vs FedAvg ===")
    from src.federated.traveling_model import compare_strategies
    from src.stats.matrix_stats import paired_ttest
    if train_patients_for_sites is not None:
        from src.data.dataset import split_into_sites, build_slice_dataset
        modality = "multimodal" if args.modality_mode == "multimodal" else "FLAIR"
        site_patient_groups = split_into_sites(train_patients_for_sites, n_sites=args.n_sites)
        site_slice_groups = [build_slice_dataset(
            g, modality=modality, img_size=args.img_size,
            slices_per_patient=args.slices_per_patient,
        ) for g in site_patient_groups]
    else:
        chunks = np.array_split(np.arange(len(splits["train"])), args.n_sites)
        site_slice_groups = [[splits["train"][i] for i in idxs] for idxs in chunks]

    site_loaders = [
        DataLoader(make_torch_dataset(g), batch_size=8, shuffle=True)
        for g in site_slice_groups if len(g) > 0
    ]
    fed_results = compare_strategies(
        site_loaders, test_loader, device,
        rounds=args.fed_rounds, local_steps=args.fed_local_steps,
        n_repeats=args.fed_repeats, in_channels=in_channels,
    )
    ttest_result = paired_ttest(
        fed_results["travel_final_auprc"], fed_results["fedavg_final_auprc"]
    ) if args.fed_repeats > 1 else {
        "statistic": float("nan"), "p_value": float("nan"),
        "mean_difference": float(fed_results["travel_final_auprc"][0] - fed_results["fedavg_final_auprc"][0]),
        "ci_95": [float("nan"), float("nan")], "n_pairs": 1,
    }
    report["traveling_vs_fedavg"] = {
        "travel_final_auprc": fed_results["travel_final_auprc"],
        "fedavg_final_auprc": fed_results["fedavg_final_auprc"],
        "paired_ttest": ttest_result,
    }
    print(report["traveling_vs_fedavg"])

    print("\n=== [7/8] PCA/SVD + quantum feature map extension ===")
    from src.stats.matrix_stats import pca_via_svd
    from src.quantum.quantum_feature_map import compare_classical_vs_quantum_features
    cnn.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for img, y, _mask in DataLoader(make_torch_dataset(splits["train"] + splits["test"]), batch_size=32):
            all_feats.append(cnn.features(img.to(device)).cpu().numpy())
            all_labels.append(y.numpy())
    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    pca_result = pca_via_svd(all_feats, n_components=min(4, all_feats.shape[1]))
    report["pca_explained_variance_ratio"] = pca_result["explained_variance_ratio"].tolist()
    proj = pca_result["projected"]
    split_at = max(4, int(len(proj) * 0.7))
    report["quantum_vs_classical"] = compare_classical_vs_quantum_features(
        proj[:split_at, :2], all_labels[:split_at],
        proj[split_at:, :2], all_labels[split_at:],
    )
    print(report["quantum_vs_classical"])

    print("\n=== [8/8] MATLAB export and report ===")
    from src.utils.matlab_export import export_resolution_experiment, export_gradcam, export_metrics_table
    export_resolution_experiment(
        os.path.join(args.out_dir, "matlab_exports", "resolution_experiment.mat"),
        res_results, example_image,
    )
    export_gradcam(
        os.path.join(args.out_dir, "matlab_exports", "gradcam_example.mat"),
        example_image, heatmap,
        predicted_prob=float(cnn_test_probs[0]) if len(cnn_test_probs) else float("nan"),
        true_label=example.tumour_present,
    )
    export_metrics_table(
        os.path.join(args.out_dir, "matlab_exports", "summary_metrics.mat"), report,
    )
    _save_figures(args.out_dir, cnn_history, heatmap, example_image, res_results)

    report["settings"] = vars(args)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print("\nDone. See outputs/report.json, outputs/figures/, outputs/checkpoints/, outputs/matlab_exports/")


if __name__ == "__main__":
    main()
