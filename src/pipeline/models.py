from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BoxResult:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    prompt: str
    is_weak: bool = False
    tile_idx: Optional[int] = None

    def as_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class CrackMetrics:
    num_regions: int
    crack_area_ratio: float
    avg_confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None


@dataclass
class CrackResult:
    final_mask: np.ndarray
    overlay_image: np.ndarray
    boxes: List[BoxResult]
    metrics: CrackMetrics
    used_prompts: List[str]
    fallback_used: bool

