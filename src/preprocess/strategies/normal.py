from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class NormalStrategy:
    noise_filter: Optional[str]
    apply_clahe: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: Tuple[int, int]

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, str]]:
        meta: Dict[str, str] = {}

        denoised = self._denoise(image)
        if denoised is not None:
            meta["denoise"] = self.noise_filter or "none"
        else:
            denoised = image

        normalized = self._illumination_normalize(denoised)
        if self.apply_clahe:
            meta["illumination"] = "clahe"

        return normalized, meta

    def _denoise(self, image: np.ndarray) -> Optional[np.ndarray]:
        mode_raw = self.noise_filter
        if isinstance(mode_raw, bool):
            mode_raw = "bilateral" if mode_raw else None
        mode = (mode_raw or "").lower()
        if mode == "gaussian":
            return cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
        if mode == "median":
            return cv2.medianBlur(image, 3)
        if mode == "bilateral":
            return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
        return None

    def _illumination_normalize(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_clahe:
            return image
        if image.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
