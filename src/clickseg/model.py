from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

from clickseg.checkpoint import verify_checkpoint
from clickseg.images import validate_point
from clickseg.runtime import synchronize


def load_model(checkpoint: Path, device: torch.device):
    verify_checkpoint(checkpoint)
    # Load explicitly on CPU and only deserialize tensors. Upstream's builder has no map_location.
    model = sam_model_registry["vit_t"]()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def encode_image(predictor: SamPredictor, image: np.ndarray) -> float:
    synchronize(predictor.device)
    started = perf_counter()
    predictor.set_image(image)
    synchronize(predictor.device)
    return perf_counter() - started


def predict_point(predictor: SamPredictor, image: np.ndarray, x: float, y: float):
    validate_point(image, x, y)
    synchronize(predictor.device)
    started = perf_counter()
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        multimask_output=False,
    )
    synchronize(predictor.device)
    elapsed = perf_counter() - started
    mask = masks[0].astype(bool)
    if mask.shape != image.shape[:2] or not np.isfinite(scores).all():
        raise RuntimeError("Invalid model output.")
    return mask, float(scores[0]), elapsed


def decode_for_training(predictor: SamPredictor, point: tuple[float, float]) -> torch.Tensor:
    """Use the decoder directly: SamPredictor.predict_torch runs under no_grad."""
    coords = predictor.transform.apply_coords(
        np.array([point], dtype=np.float32), predictor.original_size
    )
    coords = torch.as_tensor(coords, dtype=torch.float32, device=predictor.device)[None]
    labels = torch.ones((1, 1), dtype=torch.int64, device=predictor.device)
    with torch.no_grad():
        sparse, dense = predictor.model.prompt_encoder(
            points=(coords, labels), boxes=None, masks=None
        )
        position = predictor.model.prompt_encoder.get_dense_pe()
    low_res, _ = predictor.model.mask_decoder(
        image_embeddings=predictor.features.detach(),
        image_pe=position,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return predictor.model.postprocess_masks(low_res, predictor.input_size, predictor.original_size)
