import json
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as mask_api

from clickseg.data import annotation_mask, check_disjoint, positive_click, select_instances


def test_polygon_and_compressed_and_uncompressed_rle_agree():
    expected = np.zeros((6, 8), dtype=bool)
    expected[1:5, 2:7] = True
    compressed = mask_api.encode(np.asfortranarray(expected.astype(np.uint8)))
    # COCO uses column-major run lengths, beginning with a background run.
    counts, previous, run = [], 0, 0
    for value in expected.ravel(order="F").astype(int):
        if value == previous:
            run += 1
        else:
            counts.append(run)
            previous, run = value, 1
    counts.append(run)
    formats = [[[2, 1, 7, 1, 7, 5, 2, 5]], compressed, {"size": [6, 8], "counts": counts}]
    for segmentation in formats:
        actual = annotation_mask({"segmentation": segmentation}, 6, 8)
        np.testing.assert_array_equal(actual, expected)


def test_click_on_concave_mask_with_hole_never_falls_in_background():
    mask = np.ones((13, 21), dtype=bool)
    mask[3:10, 5:16] = False
    points = [positive_click(mask)]
    rng = np.random.default_rng(42)
    points.extend(positive_click(mask, rng) for _ in range(100))
    assert all(mask[y, x] for x, y in points)
    with pytest.raises(ValueError, match="nonempty"):
        positive_click(np.zeros_like(mask))


def test_overlap_rejected_even_if_ids_differ_but_bytes_match():
    first = SimpleNamespace(rows=[{"image_id": 1, "image_sha256": "same"}])
    second = SimpleNamespace(rows=[{"image_id": 2, "image_sha256": "same"}])
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint(first, second)
    second.rows[0]["image_sha256"] = "other"
    check_disjoint(first, second)
    second.rows[0]["image_id"] = 1
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint(first, second)


def test_selection_is_reproducible_and_filters_crowd_and_tiny(tmp_path):
    images = [{"id": i, "height": 40, "width": 50} for i in range(10)]
    annotations = []
    for i in range(10):
        for offset, crowd, polygon in [
            (0, 0, [5, 5, 30, 5, 30, 30, 5, 30]),
            (1, 1, [5, 5, 30, 5, 30, 30, 5, 30]),
            (2, 0, [1, 1, 2, 1, 2, 2, 1, 2]),
        ]:
            annotations.append(
                {
                    "id": 10 * i + offset,
                    "image_id": i,
                    "category_id": 1,
                    "iscrowd": crowd,
                    "segmentation": [polygon],
                }
            )
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "thing"}],
            }
        )
    )
    first, rejected = select_instances(path, 6, 42, 3)
    second, _ = select_instances(path, 6, 42, 3)
    assert [i["id"] for i, _ in first] == [i["id"] for i, _ in second]
    assert len({i["id"] for i, _ in first}) == 6
    assert all(len(objects) == 1 for _, objects in first)
    assert rejected["crowd"] == 6
    assert rejected["small_or_empty"] == 6
