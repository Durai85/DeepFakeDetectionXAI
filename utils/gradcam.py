"""
Grad-CAM utilities for deepfake detection explainability.

GradCAM          : lightweight wrapper that works with both custom and HF models
overlay_heatmap  : blends the heatmap onto the face crop PIL image
get_target_layer_hf : finds the last convolutional layer in a HF ViT/EfficientNet model
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage
    -----
    gcam = GradCAM(model, target_layer)
    heatmap = gcam.generate(input_tensor)   # numpy H×W in [0, 1]
    gcam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run a forward+backward pass and return a normalised H×W heatmap."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        # For binary classifier with single logit, maximise the fake score
        score = output if output.dim() == 0 else output[0, 0]

        self.model.zero_grad()
        score.backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Global average-pool the gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(
    heatmap: np.ndarray,
    image: Image.Image,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Resize *heatmap* to match *image* and blend them together.

    Returns a PIL.Image (RGB).
    """
    import matplotlib.cm as cm

    target_w, target_h = image.size

    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (target_w, target_h), Image.LANCZOS
        )
    ) / 255.0

    # Apply colourmap
    cmap = cm.get_cmap(colormap)
    coloured = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    base = np.array(image.convert("RGB")).astype(np.float32)
    overlay = coloured.astype(np.float32)
    blended = (1 - alpha) * base + alpha * overlay
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def get_target_layer_hf(model) -> torch.nn.Module | None:
    """
    Heuristically find the last convolutional or patch-embedding layer
    in a HuggingFace image model for Grad-CAM.
    """
    target = None
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d,)):
            target = module
    return target
