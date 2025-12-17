from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass
class CrackConfidenceAggregator:
    w_dino: float = 0.3
    w_geometry: float = 0.5
    w_continuity: float = 0.2

    def aggregate(self, dino_conf: float, geometry_score: float, continuity_score: float) -> float:
        d = _clamp01(dino_conf)
        g = _clamp01(geometry_score)
        c = _clamp01(continuity_score)
        return _clamp01(self.w_dino * d + self.w_geometry * g + self.w_continuity * c)
