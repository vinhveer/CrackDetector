from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class BlurStrategy:
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.5
    laplacian_weight: float = 0.15

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, str]]:
        meta: Dict[str, str] = {}

        sharpened = self._unsharp_mask(image, sigma=self.unsharp_sigma, amount=self.unsharp_amount)
        meta["unsharp"] = "1"

        boosted = self._edge_boost(sharpened, weight=self.laplacian_weight)
        meta["edge_boost"] = "laplacian"

        return boosted, meta

    def _unsharp_mask(self, image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    def _edge_boost(self, image: np.ndarray, weight: float) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap = cv2.normalize(lap, None, alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        img_f = image.astype(np.float32)
        if img_f.ndim == 2:
            out = img_f + (lap * 255.0) * weight
        else:
            out = img_f + (lap[..., None] * 255.0) * weight
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
