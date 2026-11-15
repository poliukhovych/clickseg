import numpy as np
import pytest

from clickseg.metrics import binary_metrics, summarize


def test_known_overlap_and_empty_prediction():
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    scores = binary_metrics(prediction, target)
    assert scores["iou"] == pytest.approx(1 / 3)
    assert scores["dice"] == 0.5
    assert binary_metrics(np.zeros_like(target), target) == {"iou": 0, "dice": 0}
    assert binary_metrics(target, target) == {"iou": 1, "dice": 1}


def test_macro_average_does_not_weight_by_object_area():
    rows = [
        {"image_id": 1, "iou": 1, "dice": 1, "area_pixels": 500},
        {"image_id": 1, "iou": 0, "dice": 0, "area_pixels": 50000},
    ]
    result = summarize(rows)
    assert result["mean_iou"] == result["mean_dice"] == 0.5
    assert result["images"] == 1
    assert result["instances"] == 2
    assert result["by_size"]["medium"]["mean_iou"] is None
