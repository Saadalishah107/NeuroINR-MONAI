"""
src/models/siren.py

Implicit Neural Representation (INR) of a single 2D image using a SIREN
(Sinusoidal Representation Network, Sitzmann et al. 2020), implemented
in PyTorch with autograd (implemented directly in PyTorch with autograd --
that hand-derived backprop was mathematically correct but did not
exercise PyTorch/CUDA, which is the whole point of this rebuild).

The model learns a continuous function f_theta(x, y) -> intensity.
Because it never sees a fixed pixel grid during training, it can be
queried at ANY resolution after training completes, with no retraining
and no architecture change -- this is what "resolution-agnostic" means
here, and Section 5 of the notebook measures it quantitatively (PSNR /
SSIM at 1x, 2x, 4x the training resolution).
"""
import math

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """A single Linear -> sin(omega_0 * x) layer with SIREN's principled
    weight initialisation (Sitzmann et al. 2020, Sec. 3.2), which keeps
    activations distributed consistently across depth."""

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    coordinates (B, 2) in [-1, 1]^2  ->  intensity (B, 1) in [0, 1]

    Parameters mirror the original paper's defaults, tuned down slightly
    (hidden_dim=128, depth=4) so a single image fits comfortably and fast
    on a Colab T4 (~5-15 seconds for 500 training steps at 128x128).
    """
    def __init__(self, in_features=2, hidden_dim=128, depth=4, out_features=1,
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        layers = [SineLayer(in_features, hidden_dim, is_first=True, omega_0=first_omega_0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

        final_linear = nn.Linear(hidden_dim, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)
        # sigmoid keeps output in [0, 1] to match normalised MRI intensities
        self.final = nn.Sequential(final_linear, nn.Sigmoid())

    def forward(self, coords):
        h = self.net(coords)
        return self.final(h)


def make_coordinate_grid(h: int, w: int) -> torch.Tensor:
    """Returns (h*w, 2) tensor of (x, y) coordinates in [-1, 1], row-major
    (matches numpy/torch reshape order of an (h, w) image)."""
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return coords


@torch.no_grad()
def query_at_resolution(model: SIREN, h: int, w: int, device="cpu", batch_size=65536) -> np.ndarray:
    """Query a TRAINED SIREN at an arbitrary (h, w) -- this is the function
    the whole 'resolution-agnostic' claim rests on: no weights change,
    only the coordinate grid we ask it to evaluate."""
    model.eval()
    coords = make_coordinate_grid(h, w).to(device)
    outs = []
    for i in range(0, coords.shape[0], batch_size):
        outs.append(model(coords[i:i + batch_size]))
    out = torch.cat(outs, dim=0).cpu().numpy().reshape(h, w)
    return out
