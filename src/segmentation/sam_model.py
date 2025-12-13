from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.pipeline.model_registry import get_sam_model
from src.pipeline.models import BoxResult
from src.utils.logging_utils import get_logger

MaskPredictorFn = Callable[[np.ndarray, Tuple[int, int, int, int], List[Tuple[int, int]]], np.ndarray]


@dataclass
class SAMConfig:
    use_points: bool = True
    dilate_radius: int = 0
    variant: str = "sam2-large"


class SAMModel:
    """SAM wrapper to produce crack masks from detected boxes."""

    def __init__(
        self,
        predictor: Optional[MaskPredictorFn] = None,
        config: Optional[SAMConfig | dict] = None,
        model_loader: Optional[Callable[[], object]] = None,
    ) -> None:
        self.config = self._build_config(config)
        self.model = get_sam_model(model_loader)
        self.predictor = predictor or self._dummy_predictor
        self.logger = get_logger(self.__class__.__name__)

    def segment(self, image: np.ndarray, boxes: Sequence[BoxResult]) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        for box in boxes:
            points = self._points_from_box(box.as_tuple()) if self.config.use_points else []
            mask = self.predictor(image, box.as_tuple(), points)
            if self.config.dilate_radius > 0:
                mask = self._dilate(mask, self.config.dilate_radius)
            masks.append(mask)
        self.logger.debug("Segmented %d masks", len(masks))
        return masks

    def _points_from_box(self, box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if w >= h:
            step = max(1, h // 3)
            points = [(x1 + w // 4, cy), (cx, cy), (x2 - w // 4, cy)]
            points.append((cx, cy - step))
            points.append((cx, cy + step))
        else:
            step = max(1, w // 3)
            points = [(cx, y1 + h // 4), (cx, cy), (cx, y2 - h // 4)]
            points.append((cx - step, cy))
            points.append((cx + step, cy))
        return points

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask, kernel)

    def _dummy_predictor(self, image: np.ndarray, box: Tuple[int, int, int, int], points: List[Tuple[int, int]]) -> np.ndarray:
        x1, y1, x2, y2 = box
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 0
        return mask

    def _build_config(self, config: Optional[SAMConfig | dict]) -> SAMConfig:
        if isinstance(config, dict):
            return SAMConfig(
                use_points=config.get("use_points", True),
                dilate_radius=config.get("dilate_radius", 0),
                variant=config.get("variant", "sam2-large"),
            )
        return config or SAMConfig()

