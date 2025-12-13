from __future__ import annotations

import cv2
import numpy as np

from src.pipeline.models import BoxResult


def create_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


def draw_boxes(image: np.ndarray, boxes: list[BoxResult], color=(0, 255, 0)) -> np.ndarray:
    out = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = b.as_tuple()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{b.prompt[:12]} {b.score:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def save_debug_image(enabled: bool, debug_dir, name: str, image: np.ndarray) -> None:
    if not enabled:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / name
    cv2.imwrite(str(path), image)

