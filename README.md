# OCR Extract Data

A two-stage Optical Character Recognition (OCR) pipeline that detects and reads text from natural scene images. A YOLO model first locates every text region in the image; a custom CRNN then transcribes each crop. The whole system is wrapped in an interactive **Gradio** web interface.

---

## Demo

> **Add a screen-recorded demo here.**
> Replace the placeholder below with your own video file or a hosted link (YouTube, Loom, etc.).

```
[Demo video – coming soon]
```

To embed a local video in GitHub README:

```html
<video src="assets/demo.mp4" controls width="720"></video>
```

---

## Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────┐
│  Stage 1 – Text Detection   │
│  YOLOv8  (text_detection.pt)│
│  → bounding boxes + conf    │
└─────────────┬───────────────┘
              │  crop each box
              ▼
┌─────────────────────────────┐
│  Stage 2 – Text Recognition │
│  CRNN  (ocr_crnn.pt)        │
│  ├─ ResNet-34 backbone      │
│  │   (AdaptiveAvgPool → 1×W)│
│  ├─ Linear projection → 512 │
│  ├─ Bidirectional GRU ×3    │
│  └─ CTC greedy decode       │
└─────────────┬───────────────┘
              │
              ▼
   Annotated image + text results
              │
              ▼
     ┌────────────────┐
     │  Gradio UI     │
     └────────────────┘
```

| Component | Detail |
|---|---|
| Detector | YOLOv8 fine-tuned for scene-text regions |
| Recogniser backbone | ResNet-34 (grayscale, 1-channel input) |
| Sequence model | 3-layer bidirectional GRU, hidden size 256 |
| Decoding | CTC greedy decode |
| Vocabulary | `0-9 a-z -` (37 characters) |
| Input resolution | 100 × 420 px (recognition crops) |
| Interface | Gradio Blocks |

---

## Installation & Running Locally

### Prerequisites

- Python ≥ 3.9
- `pip`
- (Optional but recommended) a virtual environment

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/OCR-extract-data.git
cd OCR-extract-data
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install torch torchvision timm ultralytics gradio Pillow
```

> If you have a CUDA GPU, install the matching PyTorch build from https://pytorch.org/get-started/locally/ before running the command above.

### 4 — Verify model weights

Make sure both weight files are present:

```
models/
├── text_detection.pt   # YOLO detector
└── ocr_crnn.pt         # CRNN recogniser
```

### 5 — Launch the Gradio app

```bash
python gradio_app.py
```

Open the URL printed in the terminal (default: `http://127.0.0.1:7860`), upload an image, and click **Run**.

---

## Disadvantages & Future Work

### Current limitations

| Limitation | Impact |
|---|---|
| Vocabulary limited to `0-9 a-z -` | Cannot recognise uppercase letters, punctuation, or non-Latin scripts |
| Fixed recognition crop size (100 × 420) | Very long or very short words may be distorted, hurting accuracy |
| YOLO detector trained on a single dataset (SceneTrialTrain) | May miss text in significantly different visual domains |
| CTC greedy decode only | Suboptimal for ambiguous sequences; beam-search would improve accuracy |
| Single-GPU / CPU only; no batching | Slow on large images with many text regions |
| No post-processing / spell-check | Raw model output may contain isolated character errors |

### Roadmap

- [ ] Expand vocabulary to include uppercase, punctuation, and Unicode scripts
- [ ] Replace greedy CTC decode with beam-search + language model
- [ ] Add a spellcheck / word-correction post-processing step
- [ ] Support batched inference for faster processing
- [ ] Fine-tune on more diverse scene-text datasets (e.g., ICDAR, TextOCR)
- [ ] Export models to ONNX for faster CPU inference
- [ ] Add confidence threshold slider and multi-language support in the UI
