"""One-image CUDA smoke checks for the unchanged student and SAM ViT-H."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fecd.model_loading import build_boxpromptiml_with_local_tinyvit


TINYVIT_CHECKPOINT = Path("/home/weights/tiny_vit_5m_22kto1k_distill.pth")
SAM_CHECKPOINT = Path("/home/weights/sam_vit_h_4b8939.pth")
SMOKE_IMAGE = Path("/home/datasets/boxprompt-smoke/Tp/0000.jpg")


def student_smoke() -> dict:
    if not TINYVIT_CHECKPOINT.is_file():
        raise FileNotFoundError(TINYVIT_CHECKPOINT)
    model, load_info = build_boxpromptiml_with_local_tinyvit(TINYVIT_CHECKPOINT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224, device=device))
    finite = bool(torch.isfinite(output).all().item())
    result = {
        "device": str(device),
        "output_shape": list(output.shape),
        "finite": finite,
        "output_min": float(output.min().item()),
        "output_max": float(output.max().item()),
        "tinyvit": load_info,
    }
    if output.shape != (1, 1, 224, 224) or not finite:
        raise AssertionError(f"Student forward failed: {result}")
    del output, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def sam_smoke() -> dict:
    if not SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(SAM_CHECKPOINT)
    if not SMOKE_IMAGE.is_file():
        raise FileNotFoundError(SMOKE_IMAGE)
    from segment_anything import SamPredictor, sam_model_registry

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).to(device).eval()
    predictor = SamPredictor(model)
    bgr = cv2.imread(str(SMOKE_IMAGE), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode smoke image: {SMOKE_IMAGE}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    box = np.array([w * 0.15, h * 0.15, w * 0.85, h * 0.85], dtype=np.float32)
    predictor.set_image(rgb)
    with torch.inference_mode():
        logits, quality, _ = predictor.predict(box=box, multimask_output=False, return_logits=True)
    finite = bool(np.isfinite(logits).all())
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -80, 80)))
    result = {
        "device": str(device),
        "image_shape_hw": [h, w],
        "box_xyxy": box.tolist(),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_min": float(logits.min()),
        "logits_max": float(logits.max()),
        "logits_finite": finite,
        "sigmoid_probability_min": float(probability.min()),
        "sigmoid_probability_max": float(probability.max()),
        "quality": quality.tolist(),
        "return_logits": True,
    }
    if logits.shape != (1, h, w) or not finite:
        raise AssertionError(f"SAM prediction failed: {result}")
    del predictor, model, logits, probability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    report = {"student": student_smoke(), "sam": sam_smoke()}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
