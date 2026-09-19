"""
src/explainability/gradcam.py

Grad-CAM (Selvaraju et al. 2017) for BrainMRICNN, implemented by hand
with forward/backward hooks (no external Grad-CAM library dependency
for this core piece, so the mechanism is fully inspectable). Produces a
heatmap over the ORIGINAL MRI slice showing which spatial regions drove
the tumour-present prediction -- i.e. genuine image-level explainability
of the classifier, as distinct from the coordinate-Shapley analysis of
the INR (see src/explainability/coordinate_shapley.py), which is a
different and smaller claim. Do not conflate the two in a report.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, img: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        img: (1, 1, H, W) single image tensor, on the same device as model.
        Returns an (H, W) heatmap in [0, 1], upsampled to the input size.
        """
        self.model.eval()
        img = img.clone().requires_grad_(True)
        logits = self.model(img)
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward()

        # global-average-pool the gradients over spatial dims -> per-channel weight
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam
