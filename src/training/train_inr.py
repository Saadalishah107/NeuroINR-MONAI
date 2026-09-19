"""
src/training/train_inr.py

Fits a PyTorch SIREN to a single MRI slice, then queries the SAME trained
weights at 1x, 2x, and 4x the training resolution, measuring PSNR/SSIM
against a matching-resolution ground truth at each scale. This is the
core "resolution-agnostic representation" experiment: no retraining
happens between the three evaluations, only the coordinate grid changes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.siren import SIREN, make_coordinate_grid, query_at_resolution
from src.utils.metrics import psnr, ssim


def image_to_coords_and_values(img: np.ndarray):
    h, w = img.shape
    coords = make_coordinate_grid(h, w)
    values = torch.from_numpy(img.reshape(-1, 1).astype(np.float32))
    return coords, values


def fit_siren(img: np.ndarray, device="cpu", steps=500, hidden_dim=128, depth=4, lr=1e-4, verbose=True):
    """Trains a SIREN to reproduce `img` (H, W) float32 in [0,1]. Returns
    the trained model and a loss curve."""
    coords, values = image_to_coords_and_values(img)
    coords, values = coords.to(device), values.to(device)

    model = SIREN(hidden_dim=hidden_dim, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        opt.zero_grad()
        pred = model(coords)
        loss = torch.mean((pred - values) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and step % max(1, steps // 10) == 0:
            print(f"  step {step:4d}  mse={loss.item():.5f}")
    return model, losses


def resolution_experiment(model: SIREN, ground_truth_full_res: np.ndarray, train_size: int,
                           device="cpu", scales=(1, 2, 4)):
    """
    ground_truth_full_res: the ORIGINAL (pre-downsample) image, at the
    highest resolution we have available, used as the reference when
    scoring the 2x/4x queries (downsampled to match each query size with
    the same anti-aliased resize used elsewhere in the pipeline, so PSNR/
    SSIM at every scale is measured against a fair like-for-like target).
    """
    from skimage.transform import resize as sk_resize

    results = {}
    for s in scales:
        size = train_size * s
        recon = query_at_resolution(model, size, size, device=device)
        target = sk_resize(ground_truth_full_res, (size, size), order=1,
                            preserve_range=True, anti_aliasing=True).astype(np.float32)
        results[f"{s}x_{size}px"] = {
            "reconstruction": recon,
            "target": target,
            "psnr": psnr(target, recon),
            "ssim": ssim(target, recon),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--synthetic_fallback", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    if args.synthetic_fallback:
        from src.data.synthetic_fallback import make_phantom_slice
        full_res = make_phantom_slice(size=args.train_size * 4, lesion=True, seed=1)
    else:
        from src.data.dataset import download_task01_braintumour, extract_slices_for_patient
        patients = download_task01_braintumour(args.out_dir + "/../data", limit_patients=1)
        recs = extract_slices_for_patient(patients[0], img_size=args.train_size * 4, slices_per_patient=1)
        full_res = recs[0].image

    from skimage.transform import resize as sk_resize
    train_img = sk_resize(full_res, (args.train_size, args.train_size), order=1,
                           preserve_range=True, anti_aliasing=True).astype(np.float32)

    print(f"Training SIREN at {args.train_size}x{args.train_size} for {args.steps} steps on {device} ...")
    model, losses = fit_siren(train_img, device=device, steps=args.steps,
                               hidden_dim=args.hidden_dim, depth=args.depth, lr=args.lr)

    results = resolution_experiment(model, full_res, args.train_size, device=device)
    for scale_name, r in results.items():
        print(f"{scale_name}: PSNR={r['psnr']:.2f} dB  SSIM={r['ssim']:.3f}")

    ckpt_path = os.path.join(args.out_dir, "checkpoints", "siren_mri.pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "final_loss": losses[-1]}, ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")
    return model, results, losses


if __name__ == "__main__":
    main()
