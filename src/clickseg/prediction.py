import numpy as np
from mobile_sam import SamPredictor
from PIL import Image

from clickseg.cli import write_json
from clickseg.images import overlay_mask, read_rgb, validate_point
from clickseg.model import encode_image, load_model, predict_point


def predict_image(args, device) -> dict:
    image = read_rgb(args.image)
    validate_point(image, args.x, args.y)
    model = load_model(args.checkpoint, device)
    if args.decoder is not None:
        from clickseg.experiment import apply_decoder

        apply_decoder(model, args.decoder)
    predictor = SamPredictor(model)
    encode_seconds = encode_image(predictor, image)
    mask, score, predict_seconds = predict_point(predictor, image, args.x, args.y)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / "mask.png")
    overlay_mask(image, mask, (args.x, args.y)).save(args.output / "overlay.png")
    report = {
        "device": str(device),
        "decoder": str(args.decoder) if args.decoder else None,
        "image": str(args.image),
        "image_shape": list(image.shape),
        "click_xy": [args.x, args.y],
        "foreground_pixels": int(mask.sum()),
        "image_encode_seconds": encode_seconds,
        "point_predict_seconds": predict_seconds,
        "model_quality_estimate_not_measured_iou": score,
        "mask": str(args.output / "mask.png"),
        "overlay": str(args.output / "overlay.png"),
    }
    write_json(args.output / "report.json", report)
    return report
