from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.postprocess.crack_geometry import CrackGeometryFilter
from src.postprocess.refine import geometry_filter_mask
from src.postprocess.geometry_filter_v3 import filter_regions_hard_reject, GeometryRulesConfig
from src.utils.logging_utils import get_logger


@dataclass
class PostProcessConfig:
    min_component_size: int = 32
    morph_open: int = 3
    morph_close: int = 5
    dilate: int = 0
    edge_refine: bool = True
    canny_low: int = 50
    canny_high: int = 150
    fallback_canny_low: int = 30
    fallback_canny_high: int = 120


class PostProcessor:
    """Cleans and sharpens crack masks."""

    def __init__(self, config: PostProcessConfig | dict, geometry_config: dict | None = None) -> None:
        self.config = self._build_config(config)
        geometry_config = geometry_config or {}
        self.geometry_enabled = bool(geometry_config.get("enabled", False))
        self.geometry_filter = CrackGeometryFilter(geometry_config)
        self.logger = get_logger(self.__class__.__name__)

    def refine(self, image: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, str], np.ndarray, np.ndarray, List[Tuple[int, str]]]:
        meta: Dict[str, str] = {}
        if not masks:
            meta["fallback"] = "edge"
            fallback_mask = self.edge_based_fallback(image)
            self.logger.info("Fallback edge-based mask used")
            return fallback_mask, meta, fallback_mask, fallback_mask, []

        combined = self._combine_masks(masks)
        if not self.geometry_enabled:
            # Legacy path (area-based filtering still applies here).
            cleaned = self._remove_small(combined, self.config.min_component_size)
        else:
            # Geometry-enabled path: do not use min_region_area as primary criterion.
            cleaned = combined

        if self.config.morph_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_open, self.config.morph_open))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            meta["morph_open"] = str(self.config.morph_open)
        if self.config.morph_close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_close, self.config.morph_close))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            meta["morph_close"] = str(self.config.morph_close)
        if self.config.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.dilate, self.config.dilate))
            cleaned = cv2.dilate(cleaned, kernel)
            meta["dilate"] = str(self.config.dilate)
        if self.config.edge_refine:
            cleaned = self._edge_refine(image, cleaned)
            meta["edge_refine"] = "canny_intersect"

        geometry_input = cleaned
        if not self.geometry_enabled:
            return cleaned, meta, geometry_input, geometry_input, []

        # New hard reject on SAM regions (conservative)
        if hasattr(self.geometry_filter, "config") and isinstance(self.geometry_filter.config, dict):
            rules_cfg = self.geometry_filter.config.get("rules", {})
        else:
            rules_cfg = {}
        regions = [{"region_id": i + 1, "crop_mask": m} for i, m in enumerate(masks)]
        filtered_regions = filter_regions_hard_reject(regions, rules_cfg)
        kept_masks: List[np.ndarray] = []
        dropped_regions: List[Tuple[int, str]] = []
        for r in filtered_regions:
            rid = int(r.get("region_id", 0))
            kept = bool(r.get("kept", True))
            reason = r.get("dropped_reason")
            if kept:
                kmask = r.get("crop_mask")
                if kmask is not None:
                    kept_masks.append((kmask > 0).astype(np.uint8))
            else:
                if reason:
                    dropped_regions.append((rid, str(reason)))

        if kept_masks:
            merged = self._combine_masks(kept_masks)
        else:
            merged = np.zeros_like(cleaned)

        meta["dropped_regions"] = str(len(dropped_regions))
        meta["region_details"] = filtered_regions
        return merged, meta, geometry_input, merged, dropped_regions

    def _combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, (m > 0).astype(np.uint8))
        return combined

    def _remove_small(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 1
        return cleaned

    def _edge_refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=(mask > 0).astype(np.uint8))
        refined = cv2.bitwise_or(mask, (edges > 0).astype(np.uint8))
        return refined

    def edge_based_fallback(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.config.fallback_canny_low, self.config.fallback_canny_high)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and((edges > 0).astype(np.uint8), (otsu < 128).astype(np.uint8))
        mask = self._remove_small(mask, self.config.min_component_size)
        if self.config.morph_close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_close, self.config.morph_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _build_config(self, config: PostProcessConfig | dict | None) -> PostProcessConfig:
        if isinstance(config, dict):
            return PostProcessConfig(
                min_component_size=config.get("min_region_area", 32),
                morph_open=config.get("morph_open", 3),
                morph_close=config.get("morph_close", 5),
                dilate=config.get("dilate_iters", 0),
                edge_refine=config.get("edge_refine", True),
                canny_low=config.get("canny_low", 50),
                canny_high=config.get("canny_high", 150),
                fallback_canny_low=config.get("fallback_canny_low", 30),
                fallback_canny_high=config.get("fallback_canny_high", 120),
            )
        return config or PostProcessConfig()

