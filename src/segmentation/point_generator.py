from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _morph_skeletonize(binary: np.ndarray) -> np.ndarray:
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


def generate_edge_points(image: np.ndarray, stride: int = 12, max_points: int = 800) -> List[Tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    skel = _morph_skeletonize(edges)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    coords = list(zip(xs.tolist(), ys.tolist()))
    coords.sort(key=lambda p: (p[1], p[0]))

    sampled: List[Tuple[int, int]] = []
    last_x, last_y = -10**9, -10**9
    for x, y in coords:
        if abs(x - last_x) + abs(y - last_y) < stride:
            continue
        sampled.append((x, y))
        last_x, last_y = x, y
        if len(sampled) >= max_points:
            break
    return sampled
