from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms


class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.2,
        unfreeze_layers: int = 3,
    ) -> None:
        super().__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        self.num_features = backbone.num_features

        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # (T,B,C) for CTC-style decoding
        return x


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vocab(
    chars: str = "0123456789abcdefghijklmnopqrstuvwxyz-",
) -> tuple[dict[str, int], dict[int, str]]:
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def decode(
    tokens: torch.Tensor, idx_to_char: dict[int, str], blank_char: str = "-"
) -> str:
    """Greedy CTC-style decode for a single sequence (B=1).

    Expects tokens shape (B,T) with integer class ids.
    """
    seq = tokens[0]
    decoded: list[str] = []
    prev_char: str | None = None

    for token in seq:
        token_int = int(token.item())
        if token_int == 0:
            continue
        char = idx_to_char[token_int]
        if char != blank_char:
            if char != prev_char or prev_char == blank_char:
                decoded.append(char)
        prev_char = char

    return "".join(decoded)


def build_recognition_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((100, 420)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


_HF_REPO_ID = "nhatkhangnguyen/ocr_extract_data"


def _resolve_model(models_path: Path, filename: str) -> Path:
    """Return local path to model file, downloading from HF Hub if not present."""
    local = models_path / filename
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it first: pip install huggingface_hub"
        ) from e
    hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=filename,
        local_dir=str(models_path),
    )
    return local


def _clip_bbox(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1i = int(max(0, min(w - 1, round(x1))))
    y1i = int(max(0, min(h - 1, round(y1))))
    x2i = int(max(0, min(w, round(x2))))
    y2i = int(max(0, min(h, round(y2))))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


@dataclass(frozen=True)
class ModelBundle:
    yolo: Any
    crnn: CRNN
    device: torch.device
    idx_to_char: dict[int, str]
    rec_transform: transforms.Compose


@lru_cache(maxsize=1)
def get_model_bundle(models_dir: str | Path | None = None) -> ModelBundle:
    """Load YOLO + CRNN once and cache in-process."""
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install it first: pip install ultralytics"
        ) from e

    root = Path(__file__).resolve().parents[1]
    models_path = Path(models_dir) if models_dir is not None else (root / "models")

    yolo = YOLO(str(_resolve_model(models_path, "text_detection.pt")))

    chars = "0123456789abcdefghijklmnopqrstuvwxyz-"
    _, idx_to_char = build_vocab(chars)
    vocab_size = len(chars)

    device = get_device()

    crnn = CRNN(
        vocab_size=vocab_size,
        hidden_size=256,
        n_layers=3,
        dropout=0.2,
        unfreeze_layers=3,
    ).to(device)

    state_dict = torch.load(
        _resolve_model(models_path, "ocr_crnn.pt"), map_location=device
    )
    crnn.load_state_dict(state_dict)
    crnn.eval()

    rec_transform = build_recognition_transform()

    return ModelBundle(
        yolo=yolo,
        crnn=crnn,
        device=device,
        idx_to_char=idx_to_char,
        rec_transform=rec_transform,
    )


def run_pipeline(
    image_pil: Image.Image, bundle: ModelBundle, max_boxes: int = 10
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Run detection + recognition on one image.

    Returns annotated image and list of predictions.
    Each prediction: {bbox, conf, class_name, text}
    """
    img = image_pil.convert("RGB")
    w, h = img.size

    # Detection (Ultralytics accepts numpy/PIL)
    det = bundle.yolo(img, verbose=False)[0]

    bboxes = det.boxes.xyxy.tolist() if det.boxes is not None else []
    classes = det.boxes.cls.tolist() if det.boxes is not None else []
    confs = det.boxes.conf.tolist() if det.boxes is not None else []
    names = det.names

    order = list(range(len(bboxes)))
    order.sort(key=lambda i: confs[i] if confs else 0.0, reverse=True)
    order = order[:max_boxes]

    predictions: list[dict[str, Any]] = []
    for i in order:
        x1, y1, x2, y2 = _clip_bbox(bboxes[i], w, h)
        crop = img.crop((x1, y1, x2, y2))

        x = bundle.rec_transform(crop).unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            logits = bundle.crnn(x).detach().cpu()  # (T,B,C)
        tokens = logits.permute(1, 0, 2).argmax(2)  # (B,T)
        text = decode(tokens, bundle.idx_to_char)

        cls = int(classes[i]) if i < len(classes) else -1
        class_name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)

        predictions.append(
            {
                "bbox": [x1, y1, x2, y2],
                "conf": float(confs[i]) if i < len(confs) else 0.0,
                "class_name": class_name,
                "text": text,
            }
        )

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{p['text']} ({p['conf']:.2f})"
        draw.text((x1, max(0, y1 - 12)), label, fill="red")

    return annotated, predictions
