"""
EfficientNet-B0 binary classifier for deepfake detection.

Built with timm to match the trained checkpoint architecture.
Output: single logit (BCEWithLogitsLoss during training, sigmoid for inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Backbone — features only (num_classes=0 removes timm's default head)
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        # Custom classifier matching the saved checkpoint
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),   # index 0
            nn.ReLU(),              # index 1
            nn.Dropout(p=0.3),     # index 2
            nn.Linear(512, 1),     # index 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 1280) after global avg pool
        return self.classifier(features)

    def get_target_layer(self) -> nn.Module:
        """Return the last conv block for Grad-CAM."""
        return self.backbone.blocks[-1][-1]
