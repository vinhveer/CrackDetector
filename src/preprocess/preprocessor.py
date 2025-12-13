from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.logging_utils import get_logger


@dataclass
class PreprocessConfig:
    target_size: int = 1024
    noise_filter: Optional[str] = "bilateral"  # gaussian|median|bilateral|None
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    highpass: bool = False
    gabor: bool = False


class ImagePreprocessor:
    """Handles resizing, denoising, and illumination normalization."""

    def __init__(self, config: Optional[PreprocessConfig | dict] = None) -> None:
        self.config = self._build_config(config)
        self.logger = get_logger(self.__class__.__name__)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, str], float]:
        meta: Dict[str, str] = {}
        resized, scale = self._resize_keep_ratio(image, self.config.target_size)
        meta["scale"] = f"{scale:.4f}"

        denoised = self._denoise(resized)
        if denoised is not None:
            meta["denoise"] = self.config.noise_filter or "none"
        else:
            denoised = resized

        normalized = self._illumination_normalize(denoised)
        if self.config.apply_clahe:
            meta["illumination"] = "clahe"

        enhanced = self._highpass_or_gabor(normalized)
        if self.config.highpass or self.config.gabor:
            meta["edge_enhance"] = "highpass" if self.config.highpass else "gabor"

        self.logger.debug("Preprocess done with meta %s", meta)
        return enhanced, meta, scale

    def _resize_keep_ratio(self, image: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        if max(h, w) <= target:
            return image, 1.0
        scale = target / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _denoise(self, image: np.ndarray) -> Optional[np.ndarray]:
        mode_raw = self.config.noise_filter
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
        if not self.config.apply_clahe:
            return image
        if image.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=self.config.clahe_tile_grid_size)
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=self.config.clahe_tile_grid_size)
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _highpass_or_gabor(self, image: np.ndarray) -> np.ndarray:
        if self.config.highpass:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            return cv2.filter2D(image, -1, kernel)
        if self.config.gabor:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            theta = np.pi / 4
            kernel = cv2.getGaborKernel((7, 7), sigma=3, theta=theta, lambd=7, gamma=0.5, psi=0)
            gabor = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if image.ndim == 3:
                return cv2.cvtColor(gabor, cv2.COLOR_GRAY2BGR)
            return gabor
        return image

    def _build_config(self, config: Optional[PreprocessConfig | dict]) -> PreprocessConfig:
        if isinstance(config, dict):
            noise_filter = config.get("noise_filter", "bilateral")
            if noise_filter is True:
                noise_filter = "bilateral"
            if noise_filter is False:
                noise_filter = None
            return PreprocessConfig(
                target_size=config.get("target_size", 1024),
                noise_filter=noise_filter,
                apply_clahe=bool(config.get("clahe", True)),
                clahe_clip_limit=config.get("clahe_clip_limit", 2.0),
                clahe_tile_grid_size=tuple(config.get("clahe_tile_grid_size", (8, 8))),
                highpass=bool(config.get("highpass", False)),
                gabor=bool(config.get("gabor", False)),
            )
        return config or PreprocessConfig()

