from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.postprocess.crack_geometry import CrackGeometryFilter
from src.postprocess.refine import geometry_filter_mask


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def run_edge_first_fallback(
    image: np.ndarray,
    geometry_filter: CrackGeometryFilter,
    *,
    geometry_enabled: bool,
    canny_low: int = 30,
    canny_high: int = 120,
    grow_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, List[Tuple[int, str]]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    skel = _skeletonize(edges)
    grown = skel.copy()
    if grow_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(grow_iters):
            grown = cv2.dilate(grown, kernel)

    candidate = (grown > 0).astype(np.uint8)

    if not geometry_enabled:
        return candidate, candidate, []

    filtered, dropped = geometry_filter_mask(candidate, image, geometry_filter)
    return candidate, filtered, dropped
