from __future__ import annotations

import cv2
import numpy as np


def compute_blur_score(gray_img: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
