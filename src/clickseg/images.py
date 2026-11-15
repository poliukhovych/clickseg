import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


def validate_point(image: np.ndarray, x: float, y: float) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image of shape H x W x 3.")
    height, width = image.shape[:2]
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError(f"Click must be inside the image: 0 <= x < {width}, 0 <= y < {height}.")


def overlay_mask(image: np.ndarray, mask: np.ndarray, point: tuple[float, float]) -> Image.Image:
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask must have the original image dimensions.")
    output = image.astype(np.float32).copy()
    output[mask] = 0.55 * output[mask] + 0.45 * np.array([35, 180, 100], dtype=np.float32)
    result = Image.fromarray(output.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    x, y = point
    radius = max(3, round(min(image.shape[:2]) / 100))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="white")
    return result


def synthetic_example() -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Diagnostic fixture, not a segmentation benchmark or a COCO training sample."""
    image = Image.new("RGB", (256, 192), (36, 51, 69))
    target = Image.new("L", image.size, 0)
    box = (45, 30, 175, 160)
    ImageDraw.Draw(image).ellipse(box, fill=(220, 165, 40))
    ImageDraw.Draw(target).ellipse(box, fill=255)
    ImageDraw.Draw(image).rectangle((195, 55, 240, 140), fill=(45, 140, 215))
    return np.array(image), np.array(target) > 0, (110.0, 95.0)
