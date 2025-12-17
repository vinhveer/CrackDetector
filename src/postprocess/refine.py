from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.postprocess.crack_geometry import CrackGeometryFilter
from src.postprocess.reasons import normalize_dropped_reason


def geometry_filter_mask(mask: np.ndarray, image: np.ndarray, geometry_filter: CrackGeometryFilter) -> tuple[np.ndarray, List[Tuple[int, str]]]:
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    kept = np.zeros_like(mask_bin, dtype=np.uint8)
    dropped: List[Tuple[int, str]] = []

    for region_id in range(1, num_labels):
        region_mask = (labels == region_id).astype(np.uint8)
        analysis = geometry_filter.analyze(region_mask, image)
        if bool(analysis.get("is_crack", False)):
            kept[labels == region_id] = 1
        else:
            dropped.append((region_id, normalize_dropped_reason(analysis.get("reason"))))

    return kept, dropped
