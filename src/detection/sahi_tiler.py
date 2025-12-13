from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from src.pipeline.models import BoxResult


@dataclass
class TilerConfig:
    tile_size: int = 512
    overlap: int = 64
    use_tiling: bool = True


class SahiTiler:
    """Simple tiler to mimic SAHI-style sliding window inference."""

    def __init__(self, config: TilerConfig) -> None:
        self.config = config

    def tile(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        if not self.config.use_tiling:
            return [(image, (0, 0))]
        h, w = image.shape[:2]
        tiles: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        stride = self.config.tile_size - self.config.overlap
        for y in range(0, max(h - self.config.overlap, 1), stride):
            for x in range(0, max(w - self.config.overlap, 1), stride):
                y2 = min(y + self.config.tile_size, h)
                x2 = min(x + self.config.tile_size, w)
                tile = image[y:y2, x:x2]
                tiles.append((tile, (x, y)))
        return tiles

    def merge_boxes(self, boxes: Iterable[BoxResult], origins: Iterable[Tuple[int, int]]) -> List[BoxResult]:
        merged: List[BoxResult] = []
        for box, origin in zip(boxes, origins):
            dx, dy = origin
            merged.append(
                BoxResult(
                    x1=box.x1 + dx,
                    y1=box.y1 + dy,
                    x2=box.x2 + dx,
                    y2=box.y2 + dy,
                    score=box.score,
                    prompt=box.prompt,
                    is_weak=box.is_weak,
                    tile_idx=box.tile_idx,
                ),
            )
        return merged

