"""
Face detection and cropping using MTCNN (facenet-pytorch).

detect_and_crop : returns a PIL face crop (falls back to center crop if no face)
draw_box        : draws a bounding box on the original PIL image
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")
except Exception:
    _mtcnn = None


def detect_and_crop(
    image: Image.Image,
    target_size: int = 224,
    margin: float = 0.2,
) -> tuple[Image.Image, list[int] | None]:
    """
    Detect the largest face in *image* and return a square crop.

    Returns
    -------
    face_crop : PIL.Image  (RGB, target_size × target_size)
    box       : [x1, y1, x2, y2] in pixel coords, or None if no face found
    """
    if _mtcnn is None:
        return _center_crop(image, target_size), None

    try:
        boxes, _ = _mtcnn.detect(image)
    except Exception:
        boxes = None

    if boxes is None or len(boxes) == 0:
        return _center_crop(image, target_size), None

    # Pick the largest face by area
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    box = boxes[int(np.argmax(areas))]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Add margin
    w, h = image.size
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    face = image.crop((x1, y1, x2, y2)).resize(
        (target_size, target_size), Image.LANCZOS
    )
    return face, [x1, y1, x2, y2]


def draw_box(
    image: Image.Image,
    box: list[int] | None,
    label: str = "",
    color: str = "#00FF00",
    width: int = 3,
) -> Image.Image:
    """Draw a bounding box on a copy of *image* and return it."""
    img = image.copy()
    if box is None:
        return img
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        draw.text((x1 + 4, y1 + 2), label, fill=color)
    return img


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    return image.crop((left, top, left + short, top + short)).resize(
        (size, size), Image.LANCZOS
    )
