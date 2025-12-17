from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np

from src.pipeline.models import BoxResult
from src.pipeline.device_utils import get_device

BASE_WEIGHTS = Path(__file__).resolve().parent / "weights"
DINO_CFG = BASE_WEIGHTS / "GroundingDINO_SwinT_OGC.py"
DINO_CKPT = BASE_WEIGHTS / "groundingdino_swint_ogc.pth"
SAM_CKPT = BASE_WEIGHTS / "sam_vit_h_4b8939.pth"


def load_grounding_dino(device: str | None = None):
    device = device or get_device()
    if not DINO_CFG.exists() or not DINO_CKPT.exists():
        raise FileNotFoundError(f"Missing GroundingDINO files at {DINO_CFG} and/or {DINO_CKPT}")
    try:
        from groundingdino.util.inference import Model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Please install groundingdino to use GroundingDINOModel") from exc
    return Model(model_config_path=str(DINO_CFG), model_checkpoint_path=str(DINO_CKPT), device=device)


def make_dino_predictor(model) -> Callable[[np.ndarray, str, float], List[BoxResult]]:
    def predict(image_bgr: np.ndarray, prompt: str, threshold: float) -> List[BoxResult]:
        # GroundingDINO expects RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict_with_classes(
            image=rgb,
            classes=[prompt],
            box_threshold=threshold,
            text_threshold=threshold,
        )
        results: List[BoxResult] = []
        if len(detections) > 0:
            boxes = detections.xyxy
            confidences = detections.confidence
            for (x1, y1, x2, y2), sc in zip(boxes, confidences):
                results.append(BoxResult(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=float(sc), prompt=prompt))
        return results

    return predict


def load_sam(device: str | None = None):
    device = device or get_device()
    if not SAM_CKPT.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint at {SAM_CKPT}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Please install segment-anything to use SAMModel") from exc
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(device)
    return SamPredictor(sam)


def make_sam_predictor(predictor) -> Callable[[np.ndarray, tuple[int, int, int, int], list | tuple], np.ndarray]:
    def predict(
        image_bgr: np.ndarray,
        box: tuple[int, int, int, int],
        points: list | tuple,
    ) -> np.ndarray:
        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pts = None
        labels = None
        mask_input = None
        if points:
            if isinstance(points, tuple) and len(points) == 3:
                coords, lbls, mask_input = points
                pts = np.array(coords) if len(coords) > 0 else None
                labels = np.array(lbls) if len(lbls) > 0 else None
                mask_input = None if mask_input is None else np.array(mask_input)
            elif isinstance(points, tuple) and len(points) == 2:
                coords, lbls = points
                pts = np.array(coords) if len(coords) > 0 else None
                labels = np.array(lbls) if len(lbls) > 0 else None
            else:
                pts = np.array(points)
                labels = np.ones(len(points))
        masks, _, _ = predictor.predict(
            box=np.array(box),
            point_coords=pts,
            point_labels=labels,
            multimask_output=False,
            mask_input=mask_input,
        )
        return (masks[0] > 0).astype("uint8")

    return predict

