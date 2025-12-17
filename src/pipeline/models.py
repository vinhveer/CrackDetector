from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    final_conf: Optional[float] = None


@dataclass
class CrackResult:
    final_mask: np.ndarray
    overlay_image: np.ndarray
    boxes: List[BoxResult]
    metrics: CrackMetrics
    used_prompts: List[str]
    fallback_used: bool


@dataclass
class RegionScores:
    dino_conf: Optional[float] = None
    final_conf: Optional[float] = None


@dataclass
class RegionGeometry:
    area: int = 0
    skeleton_length: float = 0.0
    endpoint_count: int = 0
    width_mean: float = 0.0
    width_var: float = 0.0
    length_area_ratio: float = 0.0
    curvature_var: float = 0.0


@dataclass
class RegionMeta:
    prompt_name: Optional[str] = None
    variant_name: Optional[str] = None
    damage_type: Optional[str] = None


@dataclass
class RegionResult:
    region_id: int
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    kept: bool
    dropped_reason: Optional[str]
    scores: RegionScores
    geometry: RegionGeometry
    meta: RegionMeta

    def to_dict(self, shallow: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "region_id": int(self.region_id),
            "bbox": tuple(int(v) for v in self.bbox),
            "kept": bool(self.kept),
            "dropped_reason": self.dropped_reason,
            "scores": asdict(self.scores),
            "geometry": asdict(self.geometry),
            "meta": asdict(self.meta),
        }
        if shallow:
            if isinstance(self.mask, np.ndarray):
                out["mask"] = {
                    "shape": list(self.mask.shape),
                    "dtype": str(self.mask.dtype),
                    "sum": int((self.mask > 0).sum()),
                }
            else:
                out["mask"] = None
        else:
            out["mask"] = self.mask
        return out


@dataclass
class PipelineResult:
    images: Dict[str, np.ndarray]
    regions: List[RegionResult]
    metrics: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    warnings: List[str]

    def to_dict(self, shallow: bool = True) -> Dict[str, Any]:
        images_out: Dict[str, Any] = {}
        if shallow:
            for k, img in (self.images or {}).items():
                if isinstance(img, np.ndarray):
                    images_out[k] = {"shape": list(img.shape), "dtype": str(img.dtype)}
                else:
                    images_out[k] = None
        else:
            images_out = dict(self.images or {})

        return {
            "images": images_out,
            "regions": [r.to_dict(shallow=shallow) for r in (self.regions or [])],
            "metrics": dict(self.metrics or {}),
            "config_snapshot": dict(self.config_snapshot or {}),
            "warnings": list(self.warnings or []),
        }
