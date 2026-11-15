"""COCO instance masks, image-disjoint manifests and reproducible positive clicks."""

import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from scipy.ndimage import distance_transform_edt

from clickseg.artifacts import atomic_json, fetch, progress
from clickseg.checkpoint import sha256_file
from clickseg.images import overlay_mask

COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
DATA_VERSION = 1


def annotation_mask(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rle = mask_api.merge(mask_api.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_api.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_api.decode(rle).astype(bool)
    if mask.shape != (height, width):
        raise ValueError("Annotation mask has incorrect dimensions.")
    return mask


def positive_click(mask, rng=None):
    """Deepest interior pixel for eval; uniform inner-half-of-radius pixel for train."""
    if mask.ndim != 2 or not mask.any():
        raise ValueError("A positive click needs a nonempty 2-D mask.")
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    if rng is None:
        y, x = np.unravel_index(distance.argmax(), mask.shape)
    else:
        pixels = np.argwhere(distance >= max(1.0, 0.5 * distance.max()))
        y, x = pixels[int(rng.integers(len(pixels)))]
    return [int(x), int(y)]


def ensure_annotations(data_dir):
    paths = [data_dir / "annotations" / f"instances_{s}2017.json" for s in ["train", "val"]]
    if all(p.is_file() for p in paths):
        return paths
    archive = data_dir / "downloads" / "annotations_trainval2017.zip"
    progress("Downloading COCO annotation archive once (about 241 MiB).")
    fetch(f"{COCO_BASE}/annotations/annotations_trainval2017.zip", archive)
    with zipfile.ZipFile(archive) as bundle:
        for path in paths:
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.part")
            try:
                with bundle.open(f"annotations/{path.name}") as source:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return paths


def select_instances(annotation_path, count, seed, max_objects, min_pixels=256):
    """Uniform shuffled images; up to max_objects qualifying shuffled annotations/image."""
    progress(f"Reading {annotation_path.name}")
    document = json.loads(annotation_path.read_text())
    by_image = defaultdict(list)
    for annotation in document["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    rng = np.random.default_rng(seed)
    images = sorted(document["images"], key=lambda x: x["id"])
    rng.shuffle(images)
    categories = {c["id"]: c["name"] for c in document["categories"]}
    selected, rejected = [], Counter()
    for info in images:
        accepted = []
        annotations = sorted(by_image[info["id"]], key=lambda x: x["id"])
        rng.shuffle(annotations)
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                rejected["crowd"] += 1
                continue
            mask = annotation_mask(annotation, info["height"], info["width"])
            area = int(mask.sum())
            fraction = area / mask.size
            if area < min_pixels or fraction < 0.001:
                rejected["small_or_empty"] += 1
                continue
            if fraction > 0.90:
                rejected["over_90_percent"] += 1
                continue
            accepted.append(
                {
                    "annotation_id": annotation["id"],
                    "category_id": annotation["category_id"],
                    "category": categories[annotation["category_id"]],
                    "mask_array": mask,
                    "area_pixels": area,
                    "area_fraction": fraction,
                    "click_xy": positive_click(mask),
                }
            )
            if len(accepted) == max_objects:
                break
        if accepted:
            selected.append((info, accepted))
            if len(selected) == count:
                return selected, dict(rejected)
        else:
            rejected["image_without_eligible_instance"] += 1
    raise ValueError(f"Requested {count} eligible images; found only {len(selected)}.")


def prepare(args):
    if min(args.train_images, args.val_images, args.test_images, args.max_objects) < 1:
        raise ValueError("All image counts and max-objects must be positive.")
    config = {
        "version": DATA_VERSION,
        "seed": args.seed,
        "train_images": args.train_images,
        "val_images": args.val_images,
        "test_images": args.test_images,
        "max_objects": args.max_objects,
        "min_pixels": 256,
        "min_fraction": 0.001,
        "max_fraction": 0.90,
        "eval_click": "deepest_inside_pixel_row_major_tie_break",
        "train_click": "random_inner_half_radius",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("This output contains a different dataset config. Use a new --output.")
    atomic_json(config_path, config)
    paths = ensure_annotations(args.data_dir)
    summary = {"config": config, "splits": {}, "annotation_sha256": {}}
    for source_split, annotation_path in zip(["train2017", "val2017"], paths, strict=True):
        summary["annotation_sha256"][source_split] = sha256_file(annotation_path)
        total = (
            args.train_images + args.val_images if source_split == "train2017" else args.test_images
        )
        selected, rejected = select_instances(
            annotation_path, total, args.seed + (source_split == "val2017"), args.max_objects
        )
        groups = (
            {"train": selected[: args.train_images], "val": selected[args.train_images :]}
            if source_split == "train2017"
            else {"test": selected}
        )
        summary.setdefault("rejections_among_scanned_annotations", {})[source_split] = rejected
        for split, group in groups.items():
            rows = []
            for index, (info, objects) in enumerate(group):
                filename = info["file_name"]
                if Path(filename).name != filename:
                    raise ValueError("Unexpected COCO filename.")
                image_path = args.data_dir / "images" / source_split / filename
                fetch(f"{COCO_BASE}/{source_split}/{filename}", image_path)
                with Image.open(image_path) as img:
                    img.load()
                    if img.size != (info["width"], info["height"]):
                        raise ValueError(f"Image size differs from COCO annotation: {image_path}")
                    image = np.array(img.convert("RGB"))
                image_sha = sha256_file(image_path)
                for obj in objects:
                    mask = obj.pop("mask_array")
                    mask_path = args.output / "masks" / split / f"{obj['annotation_id']}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                    row = {
                        **obj,
                        "image_id": info["id"],
                        "source_split": source_split,
                        "width": info["width"],
                        "height": info["height"],
                        "image": os.path.relpath(image_path, args.output),
                        "image_sha256": image_sha,
                        "mask": os.path.relpath(mask_path, args.output),
                        "mask_sha256": sha256_file(mask_path),
                    }
                    rows.append(row)
                    if len(rows) <= 6:
                        preview = args.output / "previews" / f"{split}-{len(rows):02d}.png"
                        preview.parent.mkdir(parents=True, exist_ok=True)
                        overlay_mask(image, mask, row["click_xy"]).save(preview)
                progress(f"prepare {split}: {index + 1}/{len(group)} images, {len(rows)} objects")
            manifest = {
                "version": DATA_VERSION,
                "split": split,
                "config": config,
                "source_annotation_sha256": summary["annotation_sha256"][source_split],
                "samples": rows,
            }
            atomic_json(args.output / f"{split}.json", manifest)
            summary["splits"][split] = {
                "images": len(group),
                "instances": len(rows),
                "categories": dict(Counter(r["category"] for r in rows)),
            }
        del selected, groups
    datasets = [Manifest(args.output / f"{s}.json") for s in ["train", "val", "test"]]
    for i, first in enumerate(datasets):
        for second in datasets[i + 1 :]:
            check_disjoint(first, second)
    atomic_json(args.output / "summary.json", summary)
    return summary


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text())
        if self.document.get("version") != DATA_VERSION:
            raise ValueError("Unsupported dataset manifest version.")
        self.rows = self.document["samples"]
        self.split = self.document["split"]
        self.sha256 = sha256_file(self.path)
        if not self.rows:
            raise ValueError("Empty dataset manifest.")
        seen, verified = set(), set()
        for row in self.rows:
            key = (row["source_split"], row["annotation_id"])
            if key in seen:
                raise ValueError("Duplicate instance in manifest.")
            seen.add(key)
            for kind in ["image", "mask"]:
                path = self.resolve(row[kind])
                expected = row[f"{kind}_sha256"]
                if (path, expected) not in verified:
                    if sha256_file(path) != expected:
                        raise ValueError(f"Dataset file changed: {path}")
                    verified.add((path, expected))

    def resolve(self, relative):
        return (self.path.parent / relative).resolve()

    def image(self, row):
        # COCO annotations refer to the stored pixel grid; do not rotate it using EXIF.
        with Image.open(self.resolve(row["image"])) as image:
            result = np.array(image.convert("RGB"))
        if result.shape[:2] != (row["height"], row["width"]):
            raise ValueError("Manifest image dimensions do not match.")
        return result

    def mask(self, row):
        with Image.open(self.resolve(row["mask"])) as image:
            mask = np.array(image.convert("L")) > 0
        if mask.shape != (row["height"], row["width"]) or not mask.any():
            raise ValueError("Invalid target mask.")
        x, y = row["click_xy"]
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]):
            raise ValueError("Evaluation click must be inside the target mask.")
        return mask


def check_disjoint(first, second):
    ids_a = {r["image_id"] for r in first.rows}
    ids_b = {r["image_id"] for r in second.rows}
    hashes_a = {r["image_sha256"] for r in first.rows}
    hashes_b = {r["image_sha256"] for r in second.rows}
    if ids_a & ids_b or hashes_a & hashes_b:
        raise ValueError("Data leakage: splits share image IDs or image bytes.")


def content_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
