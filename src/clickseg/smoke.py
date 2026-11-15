"""Verify inference, decoder gradients, parameter updates, and frozen encoder state."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from mobile_sam import SamPredictor
from PIL import Image
from torch.nn import functional as F

from clickseg.images import overlay_mask, synthetic_example
from clickseg.model import decode_for_training, encode_image, load_model, predict_point
from clickseg.runtime import environment_report, synchronize


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("Logits and target must have the same shape.")
    probability = logits.sigmoid()
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss


def run_smoke(checkpoint: Path, device: torch.device, output: Path) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    model = load_model(checkpoint, device)
    predictor = SamPredictor(model)
    image, target, point = synthetic_example()
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output / "input.png")
    Image.fromarray(target.astype(np.uint8) * 255).save(output / "target-diagnostic.png")

    encode_seconds = encode_image(predictor, image)
    mask, score, predict_seconds = predict_point(predictor, image, *point)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask-before.png")
    overlay_mask(image, mask, point).save(output / "overlay-before.png")

    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("image_encoder.", "prompt_encoder."))
    }
    model.mask_decoder.requires_grad_(True)
    model.mask_decoder.train()
    decoder_before = {
        name: value.detach().cpu().clone() for name, value in model.mask_decoder.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=1e-4)
    target_tensor = torch.as_tensor(target, device=device, dtype=torch.float32)[None, None]

    synchronize(device)
    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = decode_for_training(predictor, point)
    loss = segmentation_loss(logits, target_tensor)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss.")
    loss.backward()
    gradients = [p.grad for p in model.mask_decoder.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("Missing or non-finite decoder gradients.")
    gradient_norm = sum(g.detach().float().square().sum().item() for g in gradients) ** 0.5
    if gradient_norm <= 0:
        raise RuntimeError("Decoder gradients are all zero.")
    torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), max_norm=1.0)
    optimizer.step()
    synchronize(device)
    train_seconds = perf_counter() - started

    decoder_changed = any(
        not torch.equal(decoder_before[name], value.detach().cpu())
        for name, value in model.mask_decoder.named_parameters()
    )
    state_after = model.state_dict()
    encoders_unchanged = all(
        torch.equal(value, state_after[name].detach().cpu())
        for name, value in frozen_before.items()
    )
    encoder_gradients_absent = all(
        p.grad is None
        for encoder in [model.image_encoder, model.prompt_encoder]
        for p in encoder.parameters()
    )
    if not (decoder_changed and encoders_unchanged and encoder_gradients_absent):
        raise RuntimeError("Decoder update or frozen encoder invariant failed.")

    model.eval()
    with torch.no_grad():
        loss_after = segmentation_loss(decode_for_training(predictor, point), target_tensor).item()
    report = {
        "status": "passed",
        "purpose": "synthetic diagnostic, not an accuracy benchmark or COCO fine-tuning",
        "environment": environment_report(str(device)),
        "original_image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "embedding_shape": list(predictor.features.shape),
        "image_encode_seconds": encode_seconds,
        "point_predict_seconds": predict_seconds,
        "decoder_train_step_seconds": train_seconds,
        "diagnostic_loss_before": float(loss.item()),
        "diagnostic_loss_after": float(loss_after),
        "decoder_gradient_norm": gradient_norm,
        "decoder_parameters_changed": decoder_changed,
        "encoders_state_unchanged": encoders_unchanged,
        "encoder_gradients_absent": encoder_gradients_absent,
        "model_quality_estimate_not_measured_iou": score,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "decoder_parameters": sum(p.numel() for p in model.mask_decoder.parameters()),
    }
    return report
