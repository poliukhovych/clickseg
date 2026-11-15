import json
from types import SimpleNamespace

import pytest
import torch

from clickseg.experiment import compare
from clickseg.features import feature_metadata, feature_path, read_feature, save_torch


def test_cache_invalidates_when_image_or_dimensions_change_and_rejects_bad_metadata(tmp_path):
    row = {"image_sha256": "first", "height": 100, "width": 200}
    path = feature_path(tmp_path, row)
    assert path != feature_path(tmp_path, {**row, "image_sha256": "second"})
    assert path != feature_path(tmp_path, {**row, "height": 200})
    payload = {
        "metadata": feature_metadata(row),
        "input_size": [512, 1024],
        "features": torch.zeros((1, 256, 64, 64), dtype=torch.float32),
    }
    save_torch(path, payload)
    read_feature(path, row)
    payload["metadata"]["base_checkpoint_sha256"] = "wrong"
    save_torch(path, payload)
    with pytest.raises(ValueError, match="stale"):
        read_feature(path, row)


def test_comparison_refuses_changed_clicks_or_manifest(tmp_path):
    before = {
        "manifest_sha256": "data",
        "protocol": "fixed",
        "base_checkpoint_sha256": "base",
        "decoder_sha256": None,
        "split": "test",
        "instances": [{"image_id": 1, "annotation_id": 5, "click_xy": [10, 20]}],
        "summary": {"instances": 1, "mean_iou": 0.5, "mean_dice": 0.6},
    }
    after = json.loads(json.dumps(before))
    args = SimpleNamespace(
        before=tmp_path / "before.json",
        after=tmp_path / "after.json",
        output=tmp_path / "comparison.json",
    )
    args.before.write_text(json.dumps(before))
    after["summary"]["mean_iou"] = 0.7
    args.after.write_text(json.dumps(after))
    result = compare(args)
    assert result["metrics"]["mean_iou"]["delta"] == pytest.approx(0.2)
    after["instances"][0]["click_xy"] = [11, 20]
    args.after.write_text(json.dumps(after))
    with pytest.raises(ValueError, match="clicks differ"):
        compare(args)
    after["manifest_sha256"] = "other"
    args.after.write_text(json.dumps(after))
    with pytest.raises(ValueError, match="manifest_sha256"):
        compare(args)
