"""Minimal deterministic coordinate helpers for FECD cache generation."""

from __future__ import annotations

import cv2
import numpy as np


def resize_probability(probability: np.ndarray, target_hw: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize a native-resolution probability map as float32 with linear sampling."""
    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got {value.shape}")
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Input must be finite sigmoid probabilities in [0, 1], not logits")
    height, width = target_hw
    out = cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out, dtype=np.float32)


def scale_box_xyxy(
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    source_hw: tuple[int, int],
    target_hw: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Scale a half-open native-resolution XYXY box to a target pixel grid."""
    h0, w0 = source_hw
    h1, w1 = target_hw
    if min(h0, w0, h1, w1) <= 0:
        raise ValueError("Image dimensions must be positive")
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"Expected four finite XYXY coordinates, got {box}")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w0 and 0 <= y1 < y2 <= h0):
        raise ValueError(f"Invalid half-open box {box.tolist()} for source {(h0, w0)}")
    return np.asarray([x1 * w1 / w0, y1 * h1 / h0, x2 * w1 / w0, y2 * h1 / h0], dtype=np.float32)


def rasterize_box_xyxy(
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    target_hw: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Rasterize a half-open continuous XYXY box using floor-start/ceil-end."""
    h, w = target_hw
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float64)
    left = int(np.clip(np.floor(x1), 0, w))
    top = int(np.clip(np.floor(y1), 0, h))
    right = int(np.clip(np.ceil(x2), 0, w))
    bottom = int(np.clip(np.ceil(y2), 0, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1
    return mask


def synchronized_rot90(image: np.ndarray, *maps: np.ndarray, k: int = 1) -> tuple[np.ndarray, ...]:
    """Rotate an image and same-grid maps by the same counter-clockwise quarter turns."""
    outputs = (image,) + maps
    shape = image.shape[:2]
    if any(item.shape[:2] != shape for item in maps):
        raise ValueError("Image and all cache maps must share the same spatial grid")
    return tuple(np.ascontiguousarray(np.rot90(item, k=k, axes=(0, 1))) for item in outputs)
