"""
Grad-CAM for QualityNet.

Hooks the last convolutional block of the MobileNetV3 backbone
(model.features[-1]) to produce a class/target-activation map showing
which image regions most influenced a given prediction — either the
overall quality_score or a specific issue's logit. Used both for
explainability (section 10) and, combined with the autoencoder's
reconstruction-error map, for the optional quality-heatmap bonus.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, image: torch.Tensor, engineered_features: torch.Tensor,
                 target: str = "score", issue_idx: int | None = None) -> torch.Tensor:
        """
        image: (1, 3, H, W) — single-image batch (batched CAM works too,
            but per-image normalization below assumes this is the common
            inference case).
        Returns: (1, H, W) tensor in [0, 1], same spatial size as `image`.
        """
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        score, issue_logits = self.model(image, engineered_features)
        target_scalar = score.sum() if target == "score" else issue_logits[:, issue_idx].sum()
        target_scalar.backward()

        if was_training:
            self.model.train()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire — check target_layer is on the forward path")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)         # (B, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1))            # (B, h, w)
        cam = F.interpolate(cam.unsqueeze(1), size=image.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze(1)  # (B, H, W)

        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()
