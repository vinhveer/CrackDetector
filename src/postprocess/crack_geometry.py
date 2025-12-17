from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from src.postprocess.reasons import DroppedReason


@dataclass
class GeometryRuleConfig:
    length_min: float = 60.0
    ratio_min: float = 0.10
    width_mean_max: float = 18.0
    curvature_max: float = 8.0
    loop_length_min: float = 120.0
    ornament_length_min: float = 120.0
    curvature_variance_min: float = 0.005
    prune_iters: int = 10


class CrackGeometryFilter:
    def __init__(self, config: GeometryRuleConfig | dict | None = None) -> None:
        self.config = self._build_config(config)

    def analyze(self, mask: np.ndarray, image: np.ndarray | None = None) -> Dict[str, object]:
        mask_bin = (mask > 0).astype(np.uint8)
        area = int(mask_bin.sum())

        skeleton = self._skeletonize(mask_bin)
        skeleton = self._prune_skeleton(skeleton, iters=self.config.prune_iters)
        skeleton_length = float(int(skeleton.sum()))

        endpoints_mask = self._endpoints(skeleton)
        num_endpoints = int(endpoints_mask.sum())
        num_branch_points = int(self._branch_points(skeleton).sum())
        open_ratio = float(num_endpoints / skeleton_length) if skeleton_length > 0 else 0.0
        curvature_variance = float(self._estimate_curvature_variance(skeleton))

        length_area_ratio = float(skeleton_length / area) if area > 0 else 0.0

        width_mean, width_variance = self._estimate_width_stats(mask_bin, skeleton)
        curvature = self._estimate_curvature(skeleton)

        is_crack, reason = self._is_crack(
            skeleton_length=skeleton_length,
            area=area,
            length_area_ratio=length_area_ratio,
            width_mean=width_mean,
            curvature=curvature,
            num_endpoints=num_endpoints,
            open_ratio=open_ratio,
            curvature_variance=curvature_variance,
        )

        return {
            "skeleton_length": float(skeleton_length),
            "area": int(area),
            "length_area_ratio": float(length_area_ratio),
            "width_mean": float(width_mean),
            "width_variance": float(width_variance),
            "curvature": float(curvature),
            "curvature_variance": float(curvature_variance),
            "num_endpoints": int(num_endpoints),
            "num_branch_points": int(num_branch_points),
            "open_ratio": float(open_ratio),
            "is_crack": bool(is_crack),
            "reason": reason,
        }

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
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

    def _prune_skeleton(self, skel: np.ndarray, iters: int) -> np.ndarray:
        if iters <= 0:
            return (skel > 0).astype(np.uint8)
        out = (skel > 0).astype(np.uint8)
        for _ in range(iters):
            endpoints = self._endpoints(out)
            if endpoints.sum() == 0:
                break
            out[endpoints > 0] = 0
        return out

    def _endpoints(self, skel: np.ndarray) -> np.ndarray:
        s = (skel > 0).astype(np.uint8)
        k = np.array(
            [
                [1, 1, 1],
                [1, 10, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )
        neigh = cv2.filter2D(s, -1, k)
        # center=10, neighbors sum in [0..8]; endpoint => 10 + 1
        return ((neigh == 11) & (s == 1)).astype(np.uint8)

    def _neighbor_count(self, skel: np.ndarray) -> np.ndarray:
        s = (skel > 0).astype(np.uint8)
        k = np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )
        return cv2.filter2D(s, -1, k)

    def _branch_points(self, skel: np.ndarray) -> np.ndarray:
        s = (skel > 0).astype(np.uint8)
        deg = self._neighbor_count(s)
        return ((deg >= 3) & (s == 1)).astype(np.uint8)

    def _estimate_curvature_variance(self, skeleton: np.ndarray) -> float:
        s = (skeleton > 0).astype(np.uint8)
        if int(s.sum()) == 0:
            return 0.0

        deg = self._neighbor_count(s)
        ys, xs = np.where((s == 1) & (deg == 2))
        if ys.size == 0:
            return 0.0

        curvatures: list[float] = []
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        h, w = s.shape
        for y, x in zip(ys.tolist(), xs.tolist()):
            neighs: list[tuple[int, int]] = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and s[ny, nx] == 1:
                    neighs.append((ny, nx))
            if len(neighs) != 2:
                continue

            v1 = np.array([neighs[0][0] - y, neighs[0][1] - x], dtype=np.float32)
            v2 = np.array([neighs[1][0] - y, neighs[1][1] - x], dtype=np.float32)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-6 or n2 <= 1e-6:
                continue
            v1 /= n1
            v2 /= n2
            dot = float(np.clip(float(v1[0] * v2[0] + v1[1] * v2[1]), -1.0, 1.0))
            angle = float(np.arccos(dot))
            curvature = float(np.pi - angle)
            curvatures.append(curvature)

        if len(curvatures) < 2:
            return 0.0
        return float(np.var(np.array(curvatures, dtype=np.float32)))

    def _estimate_width_stats(self, mask: np.ndarray, skeleton: np.ndarray) -> Tuple[float, float]:
        if int(mask.sum()) == 0 or int(skeleton.sum()) == 0:
            return 0.0, 0.0
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        widths = dist[skeleton > 0] * 2.0
        if widths.size == 0:
            return 0.0, 0.0
        mean = float(np.mean(widths))
        var = float(np.var(widths))
        return mean, var

    def _estimate_curvature(self, skeleton: np.ndarray) -> float:
        # Simple tortuosity proxy: skeleton_length / straight_line_distance between farthest endpoints.
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 2:
            return 0.0

        endpoints = np.column_stack(np.where(self._endpoints(skeleton) > 0))
        if endpoints.shape[0] < 2:
            return 1.0

        # Find farthest pair among endpoints (O(n^2), n is small)
        max_d = 0.0
        p1 = endpoints[0]
        p2 = endpoints[1]
        for i in range(endpoints.shape[0]):
            for j in range(i + 1, endpoints.shape[0]):
                dy = float(endpoints[i, 0] - endpoints[j, 0])
                dx = float(endpoints[i, 1] - endpoints[j, 1])
                d = (dx * dx + dy * dy) ** 0.5
                if d > max_d:
                    max_d = d
                    p1, p2 = endpoints[i], endpoints[j]

        straight = float(max_d)
        length = float(int((skeleton > 0).sum()))
        if straight <= 1e-6:
            return float("inf")
        return float(length / straight)

    def _is_crack(
        self,
        *,
        skeleton_length: float,
        area: int,
        length_area_ratio: float,
        width_mean: float,
        curvature: float,
        num_endpoints: int,
        open_ratio: float,
        curvature_variance: float,
    ) -> Tuple[bool, str]:
        if open_ratio == 0.0 and num_endpoints == 0 and skeleton_length > 0:
            return False, DroppedReason.CLOSED_LOOP.value
        if curvature_variance < self.config.curvature_variance_min and skeleton_length >= self.config.ornament_length_min:
            return False, DroppedReason.ORNAMENT_LIKE.value
        if area <= 0:
            return False, DroppedReason.TOO_SHORT.value
        if skeleton_length < self.config.length_min:
            return False, DroppedReason.TOO_SHORT.value
        if length_area_ratio < self.config.ratio_min:
            return False, DroppedReason.BLOB_LIKE.value
        if width_mean > self.config.width_mean_max:
            return False, DroppedReason.TOO_THICK.value
        if curvature > self.config.curvature_max:
            return False, DroppedReason.DISCONTINUOUS.value
        return True, "ok"

    def _build_config(self, config: GeometryRuleConfig | dict | None) -> GeometryRuleConfig:
        if isinstance(config, dict):
            return GeometryRuleConfig(
                length_min=float(config.get("length_min", 60.0)),
                ratio_min=float(config.get("ratio_min", 0.10)),
                width_mean_max=float(config.get("width_mean_max", 18.0)),
                curvature_max=float(config.get("curvature_max", 8.0)),
                loop_length_min=float(config.get("loop_length_min", 120.0)),
                ornament_length_min=float(config.get("ornament_length_min", 120.0)),
                curvature_variance_min=float(config.get("curvature_variance_min", 0.005)),
                prune_iters=int(config.get("prune_iters", 10)),
            )
        return config or GeometryRuleConfig()
