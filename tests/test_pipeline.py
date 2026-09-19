"""
tests/test_pipeline.py

Run with:  pytest -v

Design: every test checks a FALSIFIABLE property, not just "did it run
without crashing". Tests that need torch/MONAI use pytest.importorskip
so this file still gives useful signal in an environment where those
heavy packages aren't installed (they always are on the target Colab
runtime, per requirements.txt / notebook Step 0-1).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------
# Quantum feature map (pure numpy -- always runs)
# ---------------------------------------------------------------------
def test_quantum_embedding_is_valid_probability_vector():
    from src.quantum.quantum_feature_map import quantum_feature_embedding
    x = np.array([0.3, -0.7, 0.1])
    probs = quantum_feature_embedding(x)
    assert probs.shape == (2 ** 3,)
    assert np.all(probs >= -1e-9)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_quantum_embedding_distinguishes_different_inputs():
    from src.quantum.quantum_feature_map import quantum_feature_embedding
    p1 = quantum_feature_embedding(np.array([0.2, 0.5]))
    p2 = quantum_feature_embedding(np.array([-0.4, 0.9]))
    assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------
# Coordinate Shapley (pure numpy -- always runs)
# ---------------------------------------------------------------------
def test_shapley_efficiency_axiom():
    from src.explainability.coordinate_shapley import shapley_xy_decomposition

    # arbitrary smooth synthetic function standing in for a trained model
    def f(coords):
        x, y = coords[:, 0], coords[:, 1]
        return x ** 2 + 3 * x * y - y

    phi_x, phi_y, f_base, f_full = shapley_xy_decomposition(f, x=0.6, y=-0.2)
    assert abs((phi_x + phi_y) - (f_full - f_base)) < 1e-9


# ---------------------------------------------------------------------
# PCA / SVD stats (pure numpy -- always runs)
# ---------------------------------------------------------------------
def test_pca_via_svd_reconstruction_and_variance():
    from src.stats.matrix_stats import pca_via_svd

    rng = np.random.default_rng(0)
    # data that truly lives on a 2D plane embedded in 5D + tiny noise
    latent = rng.normal(size=(200, 2))
    projection = rng.normal(size=(2, 5))
    X = latent @ projection + rng.normal(scale=1e-4, size=(200, 5))

    result = pca_via_svd(X, n_components=2)
    assert result["projected"].shape == (200, 2)
    # 2 components should explain almost all variance since X is ~2D
    assert result["explained_variance_ratio"].sum() > 0.99


def test_paired_ttest_detects_real_difference():
    from src.stats.matrix_stats import paired_ttest
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.80, scale=0.02, size=10)   # consistently higher
    b = rng.normal(loc=0.70, scale=0.02, size=10)
    result = paired_ttest(a, b)
    assert result["p_value"] < 0.01
    assert result["mean_difference"] > 0


# ---------------------------------------------------------------------
# MATLAB export (needs scipy only -- always runs)
# ---------------------------------------------------------------------
def test_matlab_export_roundtrip(tmp_path):
    from scipy.io import loadmat
    from src.utils.matlab_export import export_gradcam

    img = np.random.rand(32, 32).astype(np.float32)
    heat = np.random.rand(32, 32).astype(np.float32)
    out_path = tmp_path / "gradcam_test.mat"
    export_gradcam(str(out_path), img, heat, predicted_prob=0.87, true_label=1)

    loaded = loadmat(str(out_path))
    assert np.allclose(loaded["image"], img, atol=1e-5)
    assert np.allclose(loaded["gradcam_heatmap"], heat, atol=1e-5)
    assert int(loaded["true_label"][0, 0]) == 1


# ---------------------------------------------------------------------
# Synthetic fallback data (pure numpy -- always runs)
# ---------------------------------------------------------------------
def test_synthetic_fallback_is_balanced_and_bounded():
    from src.data.synthetic_fallback import make_phantom_dataset
    imgs, labels = make_phantom_dataset(n_per_class=10, size=64)
    assert imgs.shape == (20, 64, 64)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    assert labels.sum() == 10   # exactly half positive by construction


# ---------------------------------------------------------------------
# Classification metrics (needs sklearn -- always runs)
# ---------------------------------------------------------------------
def test_classification_report_known_values():
    from src.utils.metrics import classification_report_dict
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.8, 0.9])  # perfectly separable
    report = classification_report_dict(y_true, y_prob)
    assert report["auroc"] == 1.0
    assert report["auprc"] == 1.0


def test_psnr_and_ssim_identical_images_are_maximal():
    from src.utils.metrics import psnr, ssim
    img = np.random.rand(32, 32).astype(np.float32)
    assert psnr(img, img) >= 50.0
    assert ssim(img, img) > 0.999


# ---------------------------------------------------------------------
# Torch-dependent tests -- skipped automatically if torch isn't installed
# (always installed on the Colab target runtime; may be absent in a
#  lightweight CI/offline sandbox, which is fine)
# ---------------------------------------------------------------------
def test_siren_forward_shape_and_resolution_query():
    torch = pytest.importorskip("torch")
    from src.models.siren import SIREN, query_at_resolution

    model = SIREN(hidden_dim=32, depth=3)
    coords = torch.rand(10, 2) * 2 - 1
    out = model(coords)
    assert out.shape == (10, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)

    recon_64 = query_at_resolution(model, 64, 64)
    recon_128 = query_at_resolution(model, 128, 128)
    assert recon_64.shape == (64, 64)
    assert recon_128.shape == (128, 128)   # same weights, different query resolution


def test_cnn_forward_shape():
    torch = pytest.importorskip("torch")
    from src.models.cnn import BrainMRICNN

    model = BrainMRICNN()
    x = torch.rand(4, 1, 128, 128)
    logits = model(x)
    assert logits.shape == (4, 2)
    feats = model.features(x)
    assert feats.shape == (4, model.embedding_dim)


def test_gradcam_output_shape():
    torch = pytest.importorskip("torch")
    from src.models.cnn import BrainMRICNN
    from src.explainability.gradcam import GradCAM

    model = BrainMRICNN()
    cam = GradCAM(model, model.block4.conv)
    img = torch.rand(1, 1, 128, 128)
    heatmap = cam(img, class_idx=1)
    assert heatmap.shape == (128, 128)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


def test_vit_forward_shape():
    torch = pytest.importorskip("torch")
    from src.models.vit import TinyViT

    model = TinyViT(img_size=128, patch_size=16)
    x = torch.rand(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_cnn_multimodal_forward_shape():
    torch = pytest.importorskip("torch")
    from src.models.cnn import BrainMRICNN
    model = BrainMRICNN(in_channels=4)
    x = torch.rand(2, 4, 128, 128)
    assert model(x).shape == (2, 2)


def test_fedavg_accepts_multichannel_inputs():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset
    from src.data.dataset import SliceRecord, make_torch_dataset
    from src.federated.traveling_model import run_fedavg

    records_a = [SliceRecord(f"a{i}", 0, np.random.rand(4, 32, 32).astype(np.float32), np.zeros((32, 32), dtype=np.uint8), i % 2) for i in range(4)]
    records_b = [SliceRecord(f"b{i}", 0, np.random.rand(4, 32, 32).astype(np.float32), np.zeros((32, 32), dtype=np.uint8), (i + 1) % 2) for i in range(4)]
    site_loaders = [
        DataLoader(make_torch_dataset(records_a), batch_size=2),
        DataLoader(make_torch_dataset(records_b), batch_size=2),
    ]
    test_loader = DataLoader(make_torch_dataset(records_a + records_b), batch_size=2)
    model, history = run_fedavg(site_loaders, test_loader, torch.device("cpu"), rounds=1, local_steps=1, in_channels=4)
    assert model.in_channels == 4
    assert len(history) == 1
