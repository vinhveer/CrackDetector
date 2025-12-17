from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.pipeline.models import BoxResult

MaskPredictorFn = Callable[[np.ndarray, Tuple[int, int, int, int], List[Tuple[int, int]] | tuple], np.ndarray]


@dataclass
class SAMWrapperConfig:
    use_points: bool = True
    dilate_radius: int = 0


class SAMWrapper:
    def __init__(self, predictor: Optional[MaskPredictorFn] = None, config: Optional[SAMWrapperConfig | dict] = None) -> None:
        self.config = self._build_config(config)
        self.predictor = predictor or self._dummy_predictor

    def segment(
        self,
        image: np.ndarray,
        boxes: Optional[Sequence[BoxResult]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> List[np.ndarray]:
        # Legacy path disabled to avoid object-level masks; prefer segment_regions_tight_boxes.
        boxes = list(boxes or [])
        points = list(points or [])
        if not boxes:
            return []
        return [self._predict(image, box.as_tuple(), points or []) for box in boxes]

    def segment_regions(
        self,
        image: np.ndarray,
        boxes: Optional[Sequence[BoxResult]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        boxes_list = list(boxes or [])
        masks = self.segment(image, boxes=boxes_list, points=points)
        regions: List[Dict[str, Any]] = []
        for box, mask in zip(boxes_list, masks):
            regions.append({"bbox": box.as_tuple(), "mask": mask})
        return regions

    def _seed_mask_from_crop(self, crop: np.ndarray, kernel: int = 11, min_area: int = 12) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        kernel = max(3, int(kernel) | 1)
        bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
        thresh_val, seed = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if int(seed.sum()) == 0:
            # fallback: percentile
            val = np.percentile(blackhat, 90)
            _, seed = cv2.threshold(blackhat, val, 255, cv2.THRESH_BINARY)
        seed = seed.astype(np.uint8)
        if min_area > 1:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
            keep = np.zeros_like(seed)
            h, w = seed.shape[:2]
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                # drop components touching border
                if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                    continue
                aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) > 0 else 1.0
                comp_mask = (labels == i).astype(np.uint8)
                skel = self._skeletonize_binary(comp_mask)
                skel_len = int(skel.sum())
                elong_ok = aspect >= 3.0 or (skel_len > 0 and (skel_len / float(area)) >= 0.6)
                if elong_ok:
                    keep[labels == i] = 255
            seed = keep
        if seed.any():
            seed = self._skeletonize_binary(seed)
        return seed

    def _edge_and_lowgrad_points_from_crop(
        self,
        crop: np.ndarray,
        seed: Optional[np.ndarray],
        pos_count: int = 20,
        neg_count: int = 20,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        pos_points: List[Tuple[int, int]] = []
        if seed is not None and seed.any():
            ys, xs = np.where(seed > 0)
            if len(xs):
                coords = list(zip(xs.tolist(), ys.tolist()))
                pos_points = self._farthest_point_sample(coords, k=min(pos_count, len(coords)))
        else:
            flat = mag.reshape(-1)
            order = np.argsort(flat)[::-1]
            coords_all: List[Tuple[int, int]] = []
            for idx in order[: min(pos_count * 5, len(order))]:
                y = int(idx // mag.shape[1])
                x = int(idx % mag.shape[1])
                coords_all.append((x, y))
            pos_points = self._farthest_point_sample(coords_all, k=min(pos_count, len(coords_all)))

        # Negative sampling using farthest from positives to avoid overlap
        neg_points: List[Tuple[int, int]] = []
        flat = mag.reshape(-1)
        order_low = np.argsort(flat)
        desired_neg = min(neg_count, len(order_low))
        min_dist = int(max(3, round(min(gray.shape[:2]) * 0.05)))
        min_dist2 = min_dist * min_dist
        for idx in order_low:
            y = int(idx // mag.shape[1])
            x = int(idx % mag.shape[1])
            close = False
            for px, py in pos_points:
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy <= min_dist2:
                    close = True
                    break
            if close:
                continue
            for nx, ny in neg_points:
                dx = x - nx
                dy = y - ny
                if dx * dx + dy * dy <= min_dist2:
                    close = True
                    break
            if close:
                continue
            neg_points.append((x, y))
            if len(neg_points) >= desired_neg:
                break

        return pos_points, neg_points

    def _farthest_point_sample(self, coords: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
        if not coords or k <= 0:
            return []
        picked = [coords[0]]
        if k == 1:
            return picked
        remaining = coords[1:]
        while len(picked) < k and remaining:
            best_idx = -1
            best_dist = -1.0
            for i, c in enumerate(remaining):
                d = min((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 for p in picked)
                if d > best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx < 0:
                break
            picked.append(remaining.pop(best_idx))
        return picked

    def _skeletonize_binary(self, mask: np.ndarray) -> np.ndarray:
        """Simple morphological skeletonization without skimage."""
        img = (mask > 0).astype(np.uint8)
        size = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(img)
        done = False
        while not done:
            eroded = cv2.erode(img, size)
            temp = cv2.dilate(eroded, size)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            done = cv2.countNonZero(img) == 0
        return skel * 255

    def segment_regions_tight_boxes(
        self,
        image: np.ndarray,
        boxes: Optional[Sequence[BoxResult]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        boxes_list = list(boxes or [])
        pts_full = list(points or [])
        h, w = image.shape[:2]

        regions: List[Dict[str, Any]] = []
        for box in boxes_list:
            x1, y1, x2, y2 = box.as_tuple()
            x1 = int(max(0, min(x1, w - 1)))
            y1 = int(max(0, min(y1, h - 1)))
            x2 = int(max(0, min(x2, w - 1)))
            y2 = int(max(0, min(y2, h - 1)))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            crop = image[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            crop_box = (0, 0, int(crop_w - 1), int(crop_h - 1))
            crop_points_pos: List[Tuple[int, int]] = []
            crop_points_neg: List[Tuple[int, int]] = []
            if pts_full:
                for px, py in pts_full:
                    if int(px) < x1 or int(px) > x2 or int(py) < y1 or int(py) > y2:
                        continue
                    crop_points_pos.append((int(px) - x1, int(py) - y1))

            seed_mask = self._seed_mask_from_crop(crop, kernel=11, min_area=12)
            seed_bbox = None
            if seed_mask is not None and seed_mask.any():
                ys_seed, xs_seed = np.where(seed_mask > 0)
                sx1, sy1, sx2, sy2 = int(xs_seed.min()), int(ys_seed.min()), int(xs_seed.max()), int(ys_seed.max())
                pad = 16
                seed_bbox = (
                    max(0, sx1 - pad),
                    max(0, sy1 - pad),
                    min(crop_w - 1, sx2 + pad),
                    min(crop_h - 1, sy2 + pad),
                )
                crop_box = seed_bbox
                seed_area_ratio = float(seed_mask.sum()) / float(seed_mask.size) if seed_mask.size > 0 else 0.0
                if seed_area_ratio > 0.15:
                    seed_mask = np.zeros_like(seed_mask)
                    seed_bbox = None
                    crop_box = (0, 0, int(crop_w - 1), int(crop_h - 1))

            if not crop_points_pos:
                crop_points_pos, crop_points_neg = self._edge_and_lowgrad_points_from_crop(
                    crop,
                    seed_mask,
                    pos_count=20,
                    neg_count=20,
                )
            if not crop_points_pos and seed_mask is not None and seed_mask.any():
                ys, xs = np.where(seed_mask > 0)
                take = min(20, len(xs))
                for idx in range(take):
                    crop_points_pos.append((int(xs[idx]), int(ys[idx])))

            crop_points = crop_points_pos
            crop_labels: List[int] = [1] * len(crop_points_pos)
            if crop_points_neg:
                crop_points += crop_points_neg
                crop_labels += [0] * len(crop_points_neg)

            # ensure we do not call box-only; require at least points or seed as mask_input
            mask_input = seed_mask if seed_mask is not None and seed_mask.any() else None
            if not crop_points and mask_input is None:
                continue

            sam_points = (crop_points, crop_labels, mask_input) if crop_points or mask_input is not None else []

            mask_input_used = False
            try:
                crop_mask = self.predictor(crop, crop_box, sam_points)
                mask_input_used = mask_input is not None
            except TypeError:
                crop_mask = self.predictor(crop, crop_box, (crop_points, crop_labels) if crop_points else [])
                mask_input_used = False
            crop_mask = (crop_mask > 0).astype(np.uint8)

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1 : y2 + 1, x1 : x2 + 1] = crop_mask

            crop_img_bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            crop_overlay = crop_img_bgr.copy()
            crop_overlay[crop_mask > 0] = (crop_overlay[crop_mask > 0] * 0.5 + np.array((0, 0, 255)) * 0.5).astype(np.uint8)
            seed_vis = None
            if seed_mask is not None:
                seed_vis = cv2.cvtColor(seed_mask, cv2.COLOR_GRAY2BGR)
                seed_vis[seed_mask > 0] = (0, 255, 255)
                crop_overlay[seed_mask > 0] = (crop_overlay[seed_mask > 0] * 0.5 + np.array((0, 255, 255)) * 0.5).astype(np.uint8)
            for px, py in crop_points_pos:
                cv2.circle(crop_overlay, (int(px), int(py)), 2, (0, 255, 0), -1)
            for px, py in crop_points_neg:
                cv2.circle(crop_overlay, (int(px), int(py)), 2, (0, 0, 255), -1)

            regions.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "mask": full_mask,
                    "crop_image": crop_img_bgr,
                    "crop_mask": crop_mask,
                    "crop_overlay": crop_overlay,
                    "crop_seed_mask": seed_mask,
                    "seed_bbox": seed_bbox if seed_bbox is not None else (0, 0, crop_w - 1, crop_h - 1),
                    "mask_input_used": mask_input_used,
                    "crop_points_pos": crop_points_pos,
                    "crop_points_neg": crop_points_neg,
                },
            )

        return regions

    def _predict(self, image: np.ndarray, box: Tuple[int, int, int, int], points: List[Tuple[int, int]] | tuple) -> np.ndarray:
        mask = self.predictor(image, box, points)
        if self.config.dilate_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.dilate_radius * 2 + 1, self.config.dilate_radius * 2 + 1))
            mask = cv2.dilate(mask, kernel)
        return mask

    def _points_from_box(self, box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if w >= h:
            step = max(1, h // 3)
            pts = [(x1 + w // 4, cy), (cx, cy), (x2 - w // 4, cy), (cx, cy - step), (cx, cy + step)]
        else:
            step = max(1, w // 3)
            pts = [(cx, y1 + h // 4), (cx, cy), (cx, y2 - h // 4), (cx - step, cy), (cx + step, cy)]
        return pts

    def _box_from_points(self, points: List[Tuple[int, int]], image_shape: Tuple[int, int], pad: int = 24) -> Tuple[int, int, int, int]:
        h, w = image_shape
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w - 1, max(xs) + pad)
        y2 = min(h - 1, max(ys) + pad)
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return int(x1), int(y1), int(x2), int(y2)

    def _dummy_predictor(self, image: np.ndarray, box: Tuple[int, int, int, int], points: List[Tuple[int, int]]) -> np.ndarray:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def _build_config(self, config: Optional[SAMWrapperConfig | dict]) -> SAMWrapperConfig:
        if isinstance(config, dict):
            return SAMWrapperConfig(
                use_points=config.get("use_points", True),
                dilate_radius=config.get("dilate_radius", 0),
            )
        return config or SAMWrapperConfig()
