"""
src/explainability/coordinate_shapley.py

Exact Shapley-value decomposition of a trained SIREN's two INPUT
FEATURES (the x and y coordinates) for a single query point, using the
closed-form 2-player cooperative game solution (no sampling needed with
only 2 features):

    phi_x = 0.5 * [ (f(x,y) - f(base_x, base_y))
                   + (f(x, base_y) - f(base_x, base_y)) ]
    phi_y = 0.5 * [ (f(x,y) - f(base_x, base_y))
                   + (f(base_x, y) - f(base_x, base_y)) ]

This is verified against the Shapley "efficiency" axiom in the test
suite: phi_x + phi_y must exactly equal f(x,y) - f(base_x, base_y).

IMPORTANT SCOPE NOTE (read before citing this in an application): this
attributes how much of the predicted pixel intensity is explained by the
horizontal vs. vertical coordinate at that point -- a legitimate but
narrow question about the INR's own two inputs. It is NOT clinical SHAP
analysis of "which anatomical features drove a diagnosis" -- that claim
belongs to the CNN + Grad-CAM pipeline (src/explainability/gradcam.py),
which explains a real diagnostic prediction over real image content.
Keep these two forms of explainability clearly separated in any writeup.
"""
from __future__ import annotations

import numpy as np


def shapley_xy_decomposition(f, x: float, y: float, base_x: float = 0.0, base_y: float = 0.0):
    """
    f: callable, f(np.ndarray[[x, y]]) -> float  (a single scalar prediction)
    Returns (phi_x, phi_y, base_value, full_value)
    """
    # Cooperative-game values for the two orderings of a 2-player game:
    #   v({})      = f_base     (neither coordinate "on")
    #   v({x})     = f_x_only   (only x moved from baseline)
    #   v({y})     = f_y_only   (only y moved from baseline)
    #   v({x,y})   = f_full     (both moved from baseline)
    f_full = f(np.array([[x, y]]))[0]
    f_base = f(np.array([[base_x, base_y]]))[0]
    f_x_only = f(np.array([[x, base_y]]))[0]
    f_y_only = f(np.array([[base_x, y]]))[0]

    # Exact 2-player Shapley value = average marginal contribution over the
    # 2 orderings (x-then-y, y-then-x):
    #   phi_x = 1/2 [ (v({x}) - v({})) + (v({x,y}) - v({y})) ]
    #   phi_y = 1/2 [ (v({y}) - v({})) + (v({x,y}) - v({x})) ]
    phi_x = 0.5 * ((f_x_only - f_base) + (f_full - f_y_only))
    phi_y = 0.5 * ((f_y_only - f_base) + (f_full - f_x_only))
    return float(phi_x), float(phi_y), float(f_base), float(f_full)


def saliency_map(f, h: int, w: int, base_x: float = 0.0, base_y: float = 0.0):
    """Computes phi_x and phi_y at every pixel of an (h, w) grid -- two
    saliency maps showing, per output pixel, how much of the deviation
    from the baseline prediction is attributable to the x- vs y-coordinate."""
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    phi_x_map = np.zeros((h, w), dtype=np.float32)
    phi_y_map = np.zeros((h, w), dtype=np.float32)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            phi_x, phi_y, _, _ = shapley_xy_decomposition(f, xv, yv, base_x, base_y)
            phi_x_map[i, j] = phi_x
            phi_y_map[i, j] = phi_y
    return phi_x_map, phi_y_map


def make_torch_model_callable(model, device="cpu"):
    """Wraps a trained SIREN (src/models/siren.py) as the plain-numpy
    callable `f` that shapley_xy_decomposition expects."""
    import torch

    @torch.no_grad()
    def f(coords_np: np.ndarray) -> np.ndarray:
        model.eval()
        coords = torch.from_numpy(coords_np.astype(np.float32)).to(device)
        out = model(coords)
        return out.squeeze(-1).cpu().numpy()

    return f
