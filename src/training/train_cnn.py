"""Training and evaluation utilities for the CNN and ViT classifiers."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.cnn import BrainMRICNN
from src.models.vit import TinyViT
from src.utils.metrics import classification_report_dict, find_best_f1_threshold


def build_model(name: str, img_size: int, in_channels: int = 1):
    if name == "cnn":
        return BrainMRICNN(in_channels=in_channels)
    if name == "vit":
        return TinyViT(img_size=img_size, in_channels=in_channels)
    raise ValueError(f"unknown model {name}")


def predict(model, loader, device):
    model.to(device).eval()
    probs, labels = [], []
    with torch.no_grad():
        for img, y, _mask in loader:
            p = torch.softmax(model(img.to(device)), dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate(model, loader, device, threshold=0.5):
    probs, labels = predict(model, loader, device)
    return classification_report_dict(labels, probs, threshold=threshold), probs, labels


def train(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
          class_weights=None, verbose=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state, best_auprc = None, -np.inf
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img, y, _mask in train_loader:
            img, y = img.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(img), y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * img.size(0)
        sched.step()
        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_report, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_report})
        if verbose:
            print(f"epoch {epoch:02d} loss={train_loss:.4f} val_auroc={val_report['auroc']:.3f} val_auprc={val_report['auprc']:.3f}")
        if val_report["auprc"] > best_auprc:
            best_auprc = val_report["auprc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def calibrate_threshold(model, val_loader, device):
    probs, labels = predict(model, val_loader, device)
    threshold = find_best_f1_threshold(labels, probs)
    return threshold, classification_report_dict(labels, probs, threshold=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "vit"], default="cnn")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_patients", type=int, default=40)
    ap.add_argument("--slices_per_patient", type=int, default=8)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--synthetic_fallback", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    if args.synthetic_fallback:
        from src.data.synthetic_fallback import make_phantom_dataset
        from src.data.dataset import SliceRecord
        imgs, labels = make_phantom_dataset(n_per_class=60, size=args.img_size)
        records = [SliceRecord(f"phantom_{i}", 0, imgs[i],
                               (imgs[i] > 0.9).astype(np.uint8) * int(labels[i]),
                               int(labels[i])) for i in range(len(labels))]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(records))
        records = [records[i] for i in perm]
        n = len(records)
        splits = {"train": records[:int(n * 0.7)], "val": records[int(n * 0.7):int(n * 0.85)], "test": records[int(n * 0.85):]}
        in_channels = 1
    else:
        from src.data.dataset import download_task01_braintumour, split_patients, build_slice_dataset
        patients = download_task01_braintumour(args.data_root, limit_patients=args.limit_patients)
        patient_splits = split_patients(patients)
        splits = {k: build_slice_dataset(v, img_size=args.img_size, slices_per_patient=args.slices_per_patient) for k, v in patient_splits.items()}
        in_channels = 1

    from src.data.dataset import make_torch_dataset
    train_loader = DataLoader(make_torch_dataset(splits["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make_torch_dataset(splits["val"]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(make_torch_dataset(splits["test"]), batch_size=args.batch_size, shuffle=False)

    train_labels = np.array([r.tumour_present for r in splits["train"]])
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    class_weights = [1.0, float(n_neg) / max(1, n_pos)] if n_pos > 0 else None

    model = build_model(args.model, args.img_size, in_channels=in_channels)
    model, history = train(model, train_loader, val_loader, device, epochs=args.epochs,
                           lr=args.lr, class_weights=class_weights)
    threshold, val_report = calibrate_threshold(model, val_loader, device)
    test_report, _, _ = evaluate(model, test_loader, device, threshold=threshold)
    print("VALIDATION:", json.dumps(val_report, indent=2))
    print("TEST:", json.dumps(test_report, indent=2))

    ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{args.model}_brain_mri.pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "threshold": threshold, "test_report": test_report}, ckpt_path)
    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history, threshold, test_report


if __name__ == "__main__":
    main()
