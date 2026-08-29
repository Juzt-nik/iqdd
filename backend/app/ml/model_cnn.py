"""
Learned deep-learning components.

1. QualityNet: MobileNetV3-small backbone (ImageNet-pretrained, so we
   still satisfy "no external AI services" since weights are downloaded
   once at build/train time via torchvision, not called at inference)
   with two heads:
     - quality_score: single scalar regression, 0-100
     - issue_logits:  6-way multi-label classification (sigmoid per issue)

2. Autoencoder: a small convolutional autoencoder trained ONLY on clean
   images. At inference, reconstruction error (both scalar and per-pixel
   map) flags anomalous / defective regions the classifier wasn't
   explicitly trained to name — this is the "potential visual defect"
   signal, and doubles as a quality-heatmap for the optional bonus.

Both are intentionally lightweight (MobileNetV3-small, a 4-layer conv
autoencoder) so training and CPU/edge inference stay fast, matching the
assessment's ask for "a lightweight deep-learning model."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]
N_ISSUES = len(ISSUE_TYPES)


class QualityNet(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 12):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_out = backbone.classifier[0].in_features  # 576 for mobilenet_v3_small

        # Fuse CNN embedding with the engineered feature vector (hybrid).
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.ReLU(inplace=True)
        )
        fused_dim = backbone_out + 32

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.score_head = nn.Linear(256, 1)
        self.issue_head = nn.Linear(256, N_ISSUES)

    def forward(self, image: torch.Tensor, engineered_features: torch.Tensor):
        """
        image: (B, 3, H, W) normalized tensor
        engineered_features: (B, feature_dim) — output of app/ml/features.py,
            should be standardized (zero mean / unit var) by the caller
            using statistics saved at training time.
        """
        x = self.features(image)
        x = self.pool(x).flatten(1)               # (B, backbone_out)
        f = self.feature_proj(engineered_features)  # (B, 32)
        fused = torch.cat([x, f], dim=1)
        h = self.trunk(fused)
        score = torch.sigmoid(self.score_head(h)) * 100.0   # (B, 1) in [0,100]
        issue_logits = self.issue_head(h)                    # (B, N_ISSUES) raw logits
        return score.squeeze(1), issue_logits


class ConvAutoencoder(nn.Module):
    """Trained only on clean images. Reconstruction error = anomaly score."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # H/2
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),            # H/4
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),            # H/8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # H/4
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # H/2
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),   # H
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel L1 reconstruction error, averaged over channels.
        Returns (B, H, W) in [0, 1]-ish range (depends on input scaling)."""
        recon = self.forward(x)
        return (x - recon).abs().mean(dim=1)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar anomaly score per image (mean reconstruction error)."""
        return self.anomaly_map(x).flatten(1).mean(dim=1)
