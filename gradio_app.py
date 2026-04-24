from __future__ import annotations

import gradio as gr

from src.ocr_pipeline import get_model_bundle, run_pipeline


def predict(image):
    if image is None:
        return None, "Please upload an image."

    bundle = get_model_bundle()
    out_img, preds = run_pipeline(image, bundle)

    if not preds:
        return out_img, "No detections found."

    lines = []
    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        lines.append(f"{p['text']}\tconf={p['conf']:.2f}\tbox=({x1},{y1},{x2},{y2})")

    return out_img, "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown("# OCR demo (YOLO + CRNN)")

    inp = gr.Image(type="pil", label="Upload image")
    btn = gr.Button("Run")

    out_img = gr.Image(type="pil", label="Output image")
    out_txt = gr.Textbox(label="Results", lines=10)

    btn.click(predict, inputs=inp, outputs=[out_img, out_txt])


demo.launch()
