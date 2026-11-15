"""Binary instance metrics; macro averages, never predicted IoU head scores."""

import numpy as np


def binary_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Predicted and target masks must be equally sized 2-D arrays.")
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    total = int(prediction.sum()) + int(target.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / total if total else 1.0,
    }


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize zero instances.")
    result = {
        "instances": len(rows),
        "images": len({r["image_id"] for r in rows}),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
    }
    result["by_size"] = {}
    for size, lo, hi in [
        ("small", 0, 32**2),
        ("medium", 32**2, 96**2),
        ("large", 96**2, float("inf")),
    ]:
        group = [r for r in rows if lo <= r["area_pixels"] < hi]
        result["by_size"][size] = {
            "instances": len(group),
            "mean_iou": float(np.mean([r["iou"] for r in group])) if group else None,
            "mean_dice": float(np.mean([r["dice"] for r in group])) if group else None,
        }
    return result
