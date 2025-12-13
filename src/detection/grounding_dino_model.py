from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.detection.sahi_tiler import SahiTiler, TilerConfig
from src.pipeline.model_registry import get_grounding_dino
from src.pipeline.models import BoxResult
from src.utils.logging_utils import get_logger

PredictorFn = Callable[[np.ndarray, str, float], Sequence[BoxResult]]


@dataclass
class GroundingDINOConfig:
    low_threshold: float = 0.1
    high_threshold: float = 0.35
    use_dynamic: bool = True
    quantile: float = 0.8
    dynamic_cap: float = 0.9
    retry_decay: float = 0.7
    min_boxes_for_quantile: int = 3
    iou_threshold: float = 0.5
    aspect_ratio_min: float = 2.0
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.25
    enable_retry: bool = True
    tiling_min_size: int = 1400
    tiler: Optional[TilerConfig] = None


class GroundingDINOModel:
    """GroundingDINO wrapper with dynamic thresholding and SAHI-style tiling."""

    def __init__(
        self,
        threshold_config: GroundingDINOConfig | dict | None = None,
        sahi_config: Optional[dict] = None,
        predictor: Optional[PredictorFn] = None,
        model_loader: Optional[Callable[[], object]] = None,
    ) -> None:
        self.config = self._build_config(threshold_config, sahi_config or {})
        self.model = get_grounding_dino(model_loader)
        self.predictor = predictor or self._dummy_predictor
        self.tiler = SahiTiler(self.config.tiler or TilerConfig())
        self.logger = get_logger(self.__class__.__name__)
        self.last_dynamic_threshold: Optional[float] = None

    def detect(self, image: np.ndarray, prompts: List[str]) -> List[BoxResult]:
        boxes: List[BoxResult] = []
        for prompt in prompts:
            prompt_boxes = self._detect_single_prompt(image, prompt)
            boxes.extend(prompt_boxes)
        merged = self._nms(boxes, self.config.iou_threshold)
        filtered = self._shape_filter(merged, image.shape[:2])
        self.logger.debug("Detected %d boxes after filtering", len(filtered))
        return filtered

    def _detect_single_prompt(self, image: np.ndarray, prompt: str) -> List[BoxResult]:
        use_tiling = self.config.tiler.use_tiling and max(image.shape[:2]) > self.config.tiling_min_size
        tiles_with_origins = self.tiler.tile(image) if use_tiling else [(image, (0, 0))]

        box_origin_pairs: List[Tuple[BoxResult, Tuple[int, int]]] = []
        for idx, (tile, origin) in enumerate(tiles_with_origins):
            raw_boxes = self.predictor(tile, prompt, self.config.low_threshold)
            for b in raw_boxes:
                b.tile_idx = idx
                box_origin_pairs.append((b, origin))

        scores = [b.score for b, _ in box_origin_pairs]
        dynamic_thr = self._choose_threshold(scores, self.config.high_threshold)
        self.last_dynamic_threshold = dynamic_thr
        filtered_pairs = [(b, o) for b, o in box_origin_pairs if b.score >= dynamic_thr]

        if self.config.enable_retry and not filtered_pairs and dynamic_thr > self.config.low_threshold / 2:
            retry_thr = max(self.config.low_threshold / 2, dynamic_thr * self.config.retry_decay)
            for idx, (tile, origin) in enumerate(tiles_with_origins):
                retry_boxes = self.predictor(tile, prompt, retry_thr)
                for b in retry_boxes:
                    b.tile_idx = idx
                    filtered_pairs.append((b, origin))

        if use_tiling and filtered_pairs:
            boxes = [b for b, _ in filtered_pairs]
            origins = [o for _, o in filtered_pairs]
            return self.tiler.merge_boxes(boxes, origins)
        return [b for b, _ in filtered_pairs]

    def _choose_threshold(self, scores: Sequence[float], default: float) -> float:
        if not self.config.use_dynamic or not scores or len(scores) < self.config.min_boxes_for_quantile:
            return default
        quant = float(np.quantile(np.array(scores), self.config.quantile))
        return max(self.config.low_threshold, min(quant, self.config.dynamic_cap))

    def _nms(self, boxes: List[BoxResult], iou_threshold: float) -> List[BoxResult]:
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: b.score, reverse=True)
        keep: List[BoxResult] = []
        while boxes_sorted:
            current = boxes_sorted.pop(0)
            keep.append(current)
            boxes_sorted = [b for b in boxes_sorted if self._iou(current.as_tuple(), b.as_tuple()) < iou_threshold]
        return keep

    def _shape_filter(self, boxes: List[BoxResult], image_shape: Tuple[int, int]) -> List[BoxResult]:
        h, w = image_shape
        img_area = h * w
        filtered: List[BoxResult] = []
        for b in boxes:
            x1, y1, x2, y2 = b.as_tuple()
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area = bw * bh
            aspect = max(bw, bh) / float(min(bw, bh))
            area_ratio = area / float(img_area)
            if aspect < self.config.aspect_ratio_min:
                continue
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                continue
            filtered.append(b)
        return filtered

    def _iou(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _dummy_predictor(self, image: np.ndarray, prompt: str, threshold: float) -> Sequence[BoxResult]:
        # Placeholder for real GroundingDINO inference.
        return []

    def _build_config(self, threshold_config: GroundingDINOConfig | dict | None, sahi_config: dict) -> GroundingDINOConfig:
        if isinstance(threshold_config, dict):
            cfg = GroundingDINOConfig(
                low_threshold=threshold_config.get("low", threshold_config.get("low_threshold", 0.1)),
                high_threshold=threshold_config.get("base", threshold_config.get("high_threshold", 0.35)),
                use_dynamic=threshold_config.get("use_dynamic", True),
                quantile=threshold_config.get("quantile", 0.8),
                dynamic_cap=threshold_config.get("dynamic_cap", 0.9),
                retry_decay=threshold_config.get("retry_decay", 0.7),
                min_boxes_for_quantile=threshold_config.get("min_boxes_for_quantile", 3),
                iou_threshold=threshold_config.get("iou_threshold", 0.5),
                aspect_ratio_min=threshold_config.get("aspect_ratio_min", 2.0),
                min_area_ratio=threshold_config.get("min_area_ratio", 0.0002),
                max_area_ratio=threshold_config.get("max_area_ratio", 0.25),
            )
        elif isinstance(threshold_config, GroundingDINOConfig):
            cfg = threshold_config
        else:
            cfg = GroundingDINOConfig()

        tile_size = sahi_config.get("tile_size", 512)
        overlap_cfg = sahi_config.get("overlap", 0)
        overlap_px = int(tile_size * overlap_cfg) if isinstance(overlap_cfg, float) and overlap_cfg <= 1 else int(overlap_cfg or 0)
        cfg.tiler = TilerConfig(tile_size=tile_size, overlap=overlap_px, use_tiling=sahi_config.get("enabled", True))
        cfg.tiling_min_size = sahi_config.get("min_image_size", cfg.tiling_min_size)
        return cfg

