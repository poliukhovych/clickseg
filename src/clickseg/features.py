"""Disk-backed frozen image embeddings, keyed by image and encoder provenance."""

import io

import torch
from mobile_sam import SamPredictor
from PIL import __version__ as pillow_version

from clickseg.artifacts import atomic_bytes, progress
from clickseg.checkpoint import CHECKPOINT_SHA256, UPSTREAM_REV
from clickseg.data import Manifest, content_key
from clickseg.model import encode_image, load_model


def feature_metadata(row):
    return {
        "version": 1,
        "image_sha256": row["image_sha256"],
        "height": row["height"],
        "width": row["width"],
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_revision": UPSTREAM_REV,
        "torch_version": str(torch.__version__),
        "pillow_version": pillow_version,
        "transform": "SamPredictor-RGB-ResizeLongestSide1024-float32",
    }


def feature_path(cache, row):
    return cache / f"{content_key(feature_metadata(row))}.pt"


def save_torch(path, payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    atomic_bytes(path, stream.getvalue())


def read_feature(path, row):
    if not path.is_file():
        raise ValueError(f"Missing cached embedding: {path}. Run clickseg cache first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    feature = payload["features"]
    from mobile_sam.utils.transforms import ResizeLongestSide

    input_size = ResizeLongestSide.get_preprocess_shape(row["height"], row["width"], 1024)
    if (
        payload["metadata"] != feature_metadata(row)
        or tuple(payload["input_size"]) != tuple(input_size)
        or feature.shape != (1, 256, 64, 64)
        or feature.dtype != torch.float32
        or not torch.isfinite(feature).all().item()
    ):
        raise ValueError(f"Invalid or stale embedding: {path}. Move it aside and rebuild cache.")
    return payload


class CachedPredictor:
    def __init__(self, model, cache):
        self.predictor = SamPredictor(model)
        self.cache = cache
        self.current_key = None

    def use(self, row):
        path = feature_path(self.cache, row)
        if path != self.current_key:
            payload = read_feature(path, row)
            self.predictor.features = payload["features"].to(self.predictor.device)
            self.predictor.original_size = (row["height"], row["width"])
            self.predictor.input_size = tuple(payload["input_size"])
            self.predictor.is_image_set = True
            self.current_key = path
        return self.predictor


def cache_features(args, device):
    datasets = [Manifest(p) for p in args.manifest]
    rows = {}
    for dataset in datasets:
        for row in dataset.rows:
            rows[feature_path(args.cache, row)] = (dataset, row)
    predictor = SamPredictor(load_model(args.checkpoint, device))
    created, reused, seconds = 0, 0, []
    for index, (path, (dataset, row)) in enumerate(rows.items()):
        if path.exists():
            read_feature(path, row)
            reused += 1
        else:
            duration = encode_image(predictor, dataset.image(row))
            save_torch(
                path,
                {
                    "features": predictor.features.detach().cpu().float(),
                    "input_size": list(predictor.input_size),
                    "metadata": feature_metadata(row),
                },
            )
            created += 1
            seconds.append(duration)
        progress(f"cache: {index + 1}/{len(rows)} images; new={created}, reused={reused}")
    return {
        "images": len(rows),
        "created": created,
        "reused": reused,
        "new_encode_seconds_total": sum(seconds),
        "cache_bytes": sum(p.stat().st_size for p in rows),
        "device": str(device),
        "dtype": "float32",
    }
