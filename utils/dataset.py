"""
Dataset utilities for deepfake detection training and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_samples(root: str) -> list[tuple[str, int]]:
    """
    Walk *root* expecting sub-folders named 'real' and 'fake'.

    Returns a list of (image_path, label) where label=0 for real, 1 for fake.
    """
    samples: list[tuple[str, int]] = []
    # Support both capitalised (Dataset/Train/Real) and lowercase (data/train/real)
    label_map = {"real": 0, "fake": 1}

    for cls_name, label in label_map.items():
        cls_dir = Path(root) / cls_name
        if not cls_dir.is_dir():
            cls_dir = Path(root) / cls_name.capitalize()   # try Real / Fake
        if not cls_dir.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for img_path in cls_dir.glob(ext):
                samples.append((str(img_path), label))

    return samples


def get_transforms(split: str = "train") -> transforms.Compose:
    """Return the appropriate torchvision transform pipeline."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """
    Folder-based dataset.

    Expected layout
    ---------------
    root/
        real/  *.jpg
        fake/  *.jpg
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        use_face_crop: bool = False,
        transform: transforms.Compose | None = None,
    ):
        self.samples = _collect_samples(root)
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {root}. "
                "Expected sub-folders named 'real' and 'fake'."
            )
        self.transform = transform or get_transforms(split)
        self.use_face_crop = use_face_crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.use_face_crop:
            from utils.face_detector import detect_and_crop
            image, _ = detect_and_crop(image)

        tensor = self.transform(image)
        return tensor, label
