"""Local-only single-click demo. Session state stores the original, unannotated image."""

import tempfile
from threading import Lock

import numpy as np
from mobile_sam import SamPredictor
from PIL import Image

from clickseg.images import overlay_mask
from clickseg.model import encode_image, load_model, predict_point


def launch(checkpoint, device, port: int = 7860, decoder=None) -> None:
    import gradio as gr

    model = load_model(checkpoint, device)
    if decoder is not None:
        from clickseg.experiment import apply_decoder

        apply_decoder(model, decoder)
    lock = Lock()

    def new_image(image):
        if image is None:
            return None, None, None, "Завантажте зображення."
        # Drop old embeddings for this session; no image data is added to the shared model.
        state = {"image": image.copy(), "predictor": None}
        return state, None, None, "Натисніть усередині потрібного об’єкта."

    def select_image(state, evt: gr.SelectData):
        if state is None:
            raise gr.Error("Спочатку завантажте зображення.")
        x, y = evt.index
        with lock:
            encode_seconds = 0.0
            if state["predictor"] is None:
                state["predictor"] = SamPredictor(model)
                encode_seconds = encode_image(state["predictor"], state["image"])
            mask, _, seconds = predict_point(state["predictor"], state["image"], x, y)
        handle, filename = tempfile.mkstemp(prefix="clickseg-mask-", suffix=".png")
        import os

        os.close(handle)
        Image.fromarray(mask.astype(np.uint8) * 255).save(filename)
        status = f"Клік ({x}, {y}); відповідь {seconds:.3f} с; пристрій {device}."
        if encode_seconds:
            status += f" Підготовка зображення {encode_seconds:.3f} с."
        return overlay_mask(state["image"], mask, (x, y)), filename, status, state

    with gr.Blocks(title="ClickSeg", delete_cache=(3600, 3600), analytics_enabled=False) as demo:
        gr.Markdown("# ClickSeg\nЗавантажте фотографію й натисніть на об’єкт для отримання маски.")
        gr.Markdown(
            "Модель: **донавчена MobileSAM**" if decoder else "Модель: **початкова MobileSAM**"
        )
        state = gr.State(value=None, time_to_live=600)
        with gr.Row():
            source = gr.Image(
                label="Зображення — натисніть на об’єкт", type="numpy", sources=["upload"]
            )
            result = gr.Image(label="Результат сегментації", type="pil", interactive=False)
        download = gr.File(label="Бінарна маска PNG")
        status = gr.Textbox(label="Стан", interactive=False, value="Завантажте зображення.")
        clear = gr.Button("Очистити")
        source.upload(new_image, inputs=source, outputs=[state, result, download, status])
        source.select(select_image, inputs=state, outputs=[result, download, status, state])
        clear.click(
            lambda: (None, None, None, None, "Завантажте зображення."),
            outputs=[source, state, result, download, status],
        )
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=port, share=False
    )
