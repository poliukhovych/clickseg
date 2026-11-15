"""Single-positive-click evaluation and decoder-only fine-tuning on COCO instances."""

import csv
import io
import json
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from clickseg.artifacts import atomic_bytes, atomic_json, progress
from clickseg.checkpoint import CHECKPOINT_SHA256, UPSTREAM_REV, sha256_file
from clickseg.data import Manifest, check_disjoint, positive_click
from clickseg.features import CachedPredictor, save_torch
from clickseg.images import overlay_mask
from clickseg.metrics import binary_metrics, summarize
from clickseg.model import decode_for_training, load_model
from clickseg.runtime import environment_report, synchronize
from clickseg.smoke import segmentation_loss


def apply_decoder(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "clickseg-decoder-v1"
        or payload.get("base_checkpoint_sha256") != CHECKPOINT_SHA256
        or payload.get("upstream_revision") != UPSTREAM_REV
    ):
        raise ValueError("Decoder checkpoint does not match the pinned MobileSAM base.")
    model.mask_decoder.load_state_dict(payload["decoder"], strict=True)
    return payload


def score_dataset(model, dataset, cache, output=None, examples=6):
    model.eval()
    cached = CachedPredictor(model, cache)
    # Warm up once at the first example's dimensions, outside timed observations.
    first = dataset.rows[0]
    predictor = cached.use(first)
    with torch.no_grad():
        decode_for_training(predictor, first["click_xy"])
    synchronize(predictor.device)
    rows = []
    for index, sample in enumerate(dataset.rows):
        target = dataset.mask(sample)
        predictor = cached.use(sample)
        synchronize(predictor.device)
        started = perf_counter()
        with torch.no_grad():
            logits = decode_for_training(predictor, sample["click_xy"])
        synchronize(predictor.device)
        elapsed = perf_counter() - started
        mask = (logits[0, 0] > model.mask_threshold).cpu().numpy()
        record = {
            k: sample[k]
            for k in [
                "image_id",
                "annotation_id",
                "category_id",
                "category",
                "area_pixels",
                "click_xy",
            ]
        }
        record.update(binary_metrics(mask, target))
        record["decode_seconds"] = elapsed
        rows.append(record)
        if output is not None and index < examples:
            directory = output / "examples"
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{index:03d}-{sample['annotation_id']}"
            image = dataset.image(sample)
            overlay_mask(image, mask, sample["click_xy"]).save(directory / f"{stem}-pred.png")
            overlay_mask(image, target, sample["click_xy"]).save(directory / f"{stem}-target.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{stem}-mask.png")
        if (index + 1) % 10 == 0 or index + 1 == len(dataset.rows):
            progress(f"eval {dataset.split}: {index + 1}/{len(dataset.rows)} objects")
    summary = summarize(rows)
    times = [r["decode_seconds"] for r in rows]
    summary["decode_seconds"] = {
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "mean": float(np.mean(times)),
        "warmup_calls": 1,
        "scope": "prompt encoding + mask decoding + original-size logits; "
        "excludes cache I/O, image encoder, metrics and UI; "
        "new image shapes may still need first-use setup",
    }
    if output is not None:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_bytes(output / "per-instance.csv", stream.getvalue().encode())
    return summary, rows


def evaluate(args, device):
    dataset = Manifest(args.manifest)
    model = load_model(args.checkpoint, device)
    decoder_info = None
    if args.decoder is not None:
        decoder_info = apply_decoder(model, args.decoder)
        if dataset.split == "test":
            train_ids = set(decoder_info["train_image_ids"])
            val_ids = set(decoder_info["val_image_ids"])
            if (train_ids | val_ids) & {r["image_id"] for r in dataset.rows}:
                raise ValueError("Test images overlap the decoder's training/validation images.")
            prior_hashes = set(
                decoder_info["train_image_hashes"] + decoder_info["val_image_hashes"]
            )
            if prior_hashes & {r["image_sha256"] for r in dataset.rows}:
                raise ValueError("Test image bytes overlap training/validation images.")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, rows = score_dataset(model, dataset, args.cache, args.output, args.examples)
    report = {
        "schema": 1,
        "split": dataset.split,
        "manifest_sha256": dataset.sha256,
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "decoder_sha256": sha256_file(args.decoder) if args.decoder else None,
        "decoder_epoch": decoder_info["epoch"] if decoder_info else None,
        "environment": environment_report(str(device)),
        "protocol": "one fixed interior positive click; one mask; macro instance IoU/Dice",
        "summary": summary,
        "instances": rows,
    }
    atomic_json(args.output / "report.json", report)
    return {"report": str(args.output / "report.json"), **summary}


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def frozen_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("image_encoder.", "prompt_encoder."))
    }


def train(args, device):
    if args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs and lr must be positive and finite.")
    train_set, val_set = Manifest(args.train), Manifest(args.val)
    if train_set.split != "train" or val_set.split != "val":
        raise ValueError(
            "Training requires train and val manifests; test cannot select checkpoints."
        )
    check_disjoint(train_set, val_set)
    config = {
        "train_sha256": train_set.sha256,
        "val_sha256": val_set.sha256,
        "seed": args.seed,
        "lr": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "batch_size": 1,
        "grad_clip": 1.0,
        "loss": "BCEWithLogits + soft Dice",
        "train_click": "random_inner_half_radius",
        "cache_dtype": "float32",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and not args.resume:
        raise ValueError("Training output already exists. Use --resume or a new --output.")
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, device)
    model.mask_decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0.01)
    cached = CachedPredictor(model, args.cache)
    # Fail before a long training run if any train/validation features are missing or invalid.
    seen = set()
    for row in train_set.rows + val_set.rows:
        if row["image_sha256"] not in seen:
            cached.use(row)
            seen.add(row["image_sha256"])
    history, best_iou, start_epoch = [], -1.0, 1
    if args.resume:
        state = apply_decoder(model, args.resume)
        if state["config"] != config:
            raise ValueError("Resume config/manifests differ from the saved training run.")
        optimizer.load_state_dict(state["optimizer"])
        history, best_iou = state["history"], state["best_iou"]
        start_epoch = state["epoch"] + 1
        # best.pt is a sibling; otherwise a new output could lose the earlier best model.
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume into the same --output directory as the checkpoint.")
        if not (args.output / "best.pt").is_file():
            raise ValueError("Resume requires best.pt beside the checkpoint.")
    else:
        baseline, _ = score_dataset(model, val_set, args.cache)
        atomic_json(args.output / "baseline-val.json", baseline)
    if start_epoch > args.epochs:
        raise ValueError("--epochs is the total epoch target; it must exceed the resumed epoch.")
    atomic_json(config_path, config)
    atomic_json(args.output / "environment.json", environment_report(str(device)))
    frozen = frozen_state(model)
    image_groups = {}
    for row in train_set.rows:
        image_groups.setdefault(row["image_id"], []).append(row)
    for epoch in range(start_epoch, args.epochs + 1):
        # Per-epoch seeds allow restart from the last complete epoch without serializing NumPy.
        rng = np.random.default_rng(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        groups = list(image_groups.values())
        rng.shuffle(groups)
        ordered = []
        for group in groups:
            group = list(group)
            rng.shuffle(group)
            ordered.extend(group)
        model.eval()
        model.mask_decoder.train()
        started, losses = perf_counter(), []
        before = {k: v.detach().cpu().clone() for k, v in model.mask_decoder.state_dict().items()}
        for step, row in enumerate(ordered, start=1):
            target = train_set.mask(row)
            point = positive_click(target, rng)
            predictor = cached.use(row)
            tensor = torch.as_tensor(target, device=device, dtype=torch.float32)[None, None]
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(decode_for_training(predictor, point), tensor)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.mask_decoder.parameters(), 1.0, error_if_nonfinite=True
            )
            if not torch.isfinite(norm).item():
                raise RuntimeError("Non-finite gradient norm.")
            optimizer.step()
            losses.append(float(loss.item()))
            if step % 10 == 0 or step == len(ordered):
                progress(
                    f"epoch {epoch}/{args.epochs}: {step}/{len(ordered)} objects, "
                    f"loss={np.mean(losses):.5f}"
                )
        synchronize(device)
        elapsed = perf_counter() - started
        state = model.state_dict()
        if any(
            not torch.equal(value, state[name].detach().cpu()) for name, value in frozen.items()
        ):
            raise RuntimeError("Frozen encoder state changed.")
        if any(
            p.grad is not None
            for enc in [model.image_encoder, model.prompt_encoder]
            for p in enc.parameters()
        ):
            raise RuntimeError("Frozen encoder received gradients.")
        if all(
            torch.equal(v, model.mask_decoder.state_dict()[k].detach().cpu())
            for k, v in before.items()
        ):
            raise RuntimeError("Decoder parameters did not change.")
        validation, _ = score_dataset(model, val_set, args.cache)
        improved = validation["mean_iou"] > best_iou
        best_iou = max(best_iou, validation["mean_iou"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_seconds": elapsed,
                "train_instances": len(ordered),
                "val_mean_iou": validation["mean_iou"],
                "val_mean_dice": validation["mean_dice"],
                "encoders_state_unchanged": True,
                "encoder_gradients_absent": True,
                "decoder_parameters_changed": True,
            }
        )
        payload = {
            "format": "clickseg-decoder-v1",
            "base_checkpoint_sha256": CHECKPOINT_SHA256,
            "upstream_revision": UPSTREAM_REV,
            "epoch": epoch,
            "config": config,
            "decoder": to_cpu(model.mask_decoder.state_dict()),
            "optimizer": to_cpu(optimizer.state_dict()),
            "best_iou": best_iou,
            "history": history,
            "train_image_ids": sorted({r["image_id"] for r in train_set.rows}),
            "val_image_ids": sorted({r["image_id"] for r in val_set.rows}),
            "train_image_hashes": sorted({r["image_sha256"] for r in train_set.rows}),
            "val_image_hashes": sorted({r["image_sha256"] for r in val_set.rows}),
        }
        if improved:
            save_torch(args.output / "best.pt", payload)
        save_torch(args.output / "last.pt", payload)
        atomic_json(args.output / "history.json", history)
        progress(f"epoch {epoch}: val IoU={validation['mean_iou']:.4f}, best={best_iou:.4f}")
    return {
        "best_checkpoint": str(args.output / "best.pt"),
        "best_val_iou": best_iou,
        "last_epoch": args.epochs,
        "history": str(args.output / "history.json"),
    }


def compare(args):
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    for field in ["manifest_sha256", "protocol", "base_checkpoint_sha256"]:
        if before[field] != after[field]:
            raise ValueError(f"Cannot compare reports with different {field}.")

    def key(row):
        return row["image_id"], row["annotation_id"], tuple(row["click_xy"])

    if [key(r) for r in before["instances"]] != [key(r) for r in after["instances"]]:
        raise ValueError("Compared instances or clicks differ.")
    result = {
        "manifest_sha256": before["manifest_sha256"],
        "split": before["split"],
        "instances": before["summary"]["instances"],
        "before_decoder_sha256": before["decoder_sha256"],
        "after_decoder_sha256": after["decoder_sha256"],
        "metrics": {
            metric: {
                "before": before["summary"][metric],
                "after": after["summary"][metric],
                "delta": after["summary"][metric] - before["summary"][metric],
            }
            for metric in ["mean_iou", "mean_dice"]
        },
    }
    atomic_json(args.output, result)
    return result
