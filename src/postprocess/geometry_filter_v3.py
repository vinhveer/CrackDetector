from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GeometryRulesConfig:
    min_region_area: int = 30
    min_skeleton_len: int = 25
    min_len_loop: int = 40
    blob_area_ratio_max: float = 0.6
    len_area_min: float = 0.02
    t_border: float = 0.30
    t_width_var_low: float = 1.0
    t_width_mean_high: float = 10.0
    thin_width_max: float = 6.0
    len_area_keep_min: float = 0.06
    t_curv_var_low: float = 0.08
    ecc_straight_min: float = 8.0
    border_sides_large_area: int = 2
    border_band_frac: float = 0.03  # band thickness = max(3, round(min(w,h)*frac))
    pad: int = 1


def _binarize(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    img = _binarize(mask)
    if img.size == 0:
        return img
    padded = np.pad(img, 1, mode="constant", constant_values=0)
    prev = np.zeros_like(padded)
    cur = padded.copy()

    while True:
        if np.array_equal(prev, cur):
            break
        prev = cur.copy()

        p2 = cur[:-2, 1:-1]
        p3 = cur[:-2, 2:]
        p4 = cur[1:-1, 2:]
        p5 = cur[2:, 2:]
        p6 = cur[2:, 1:-1]
        p7 = cur[2:, :-2]
        p8 = cur[1:-1, :-2]
        p9 = cur[:-2, :-2]
        p1 = cur[1:-1, 1:-1]

        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        s = np.zeros_like(p1)
        for a, b in zip(seq[:-1], seq[1:]):
            s += ((a == 0) & (b == 1)).astype(np.uint8)

        m1 = (p1 == 1) & (n >= 2) & (n <= 6) & (s == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        cur[1:-1, 1:-1][m1] = 0

        p2 = cur[:-2, 1:-1]
        p3 = cur[:-2, 2:]
        p4 = cur[1:-1, 2:]
        p5 = cur[2:, 2:]
        p6 = cur[2:, 1:-1]
        p7 = cur[2:, :-2]
        p8 = cur[1:-1, :-2]
        p9 = cur[:-2, :-2]
        p1 = cur[1:-1, 1:-1]

        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        s = np.zeros_like(p1)
        for a, b in zip(seq[:-1], seq[1:]):
            s += ((a == 0) & (b == 1)).astype(np.uint8)

        m2 = (p1 == 1) & (n >= 2) & (n <= 6) & (s == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        cur[1:-1, 1:-1][m2] = 0

    return cur[1:-1, 1:-1].astype(np.uint8)


def _neighbors_8(y: int, x: int, h: int, w: int) -> List[Tuple[int, int]]:
    res = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                res.append((ny, nx))
    return res


def _endpoints_and_branchpoints(skel: np.ndarray) -> Tuple[int, int]:
    if skel.size == 0:
        return 0, 0
    sk = (skel > 0).astype(np.uint8)
    k = np.ones((3, 3), dtype=np.uint8)
    neigh = cv2.filter2D(sk, ddepth=cv2.CV_16S, kernel=k, borderType=cv2.BORDER_CONSTANT)
    neigh = neigh - sk
    endpoints = int(((sk == 1) & (neigh == 1)).sum())
    branches = int(((sk == 1) & (neigh >= 3)).sum())
    return endpoints, branches


def _skeleton_orientation_variance(skel: np.ndarray) -> float:
    ys, xs = np.where(skel > 0)
    if len(xs) < 3:
        return 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts_mean = pts.mean(axis=0, keepdims=True)
    pts_centered = pts - pts_mean
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    angles = np.arctan2(main_axis[1], main_axis[0])
    proj = pts_centered @ main_axis
    ortho = pts_centered @ np.array([-main_axis[1], main_axis[0]])
    # local orientation via derivative of polyline approx: use gradient of ortho vs proj
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]
    ortho_sorted = ortho[sort_idx]
    if len(proj_sorted) < 3:
        return 1.0
    dproj = np.diff(proj_sorted)
    dortho = np.diff(ortho_sorted)
    angles_local = np.arctan2(dortho + 1e-6, dproj + 1e-6)
    ang_var = float(np.var(angles_local))
    return ang_var


def _orientation_eccentricity(skel: np.ndarray) -> float:
    ys, xs = np.where(skel > 0)
    if len(xs) < 3:
        return 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts_mean = pts.mean(axis=0, keepdims=True)
    pts_centered = pts - pts_mean
    cov = np.cov(pts_centered.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)
    return float(eigvals[-1] / (eigvals[0] + 1e-6))


def _distance_width_stats(mask: np.ndarray, skel: np.ndarray) -> Dict[str, float]:
    if mask.size == 0 or mask.max() <= 0:
        return {"width_mean": 0.0, "width_var": 0.0, "width_std": 0.0, "width_p90": 0.0}
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return {"width_mean": 0.0, "width_var": 0.0, "width_std": 0.0, "width_p90": 0.0}
    widths = (dist[ys, xs] * 2.0).astype(np.float32)
    return {
        "width_mean": float(np.mean(widths)),
        "width_var": float(np.var(widths)),
        "width_std": float(np.std(widths)),
        "width_p90": float(np.percentile(widths, 90)),
    }


def _border_features(mask: np.ndarray, band_frac: float) -> Tuple[float, int, np.ndarray]:
    mask = _binarize(mask)
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0, np.zeros((0, 0), dtype=np.uint8)
    band = max(3, int(round(min(h, w) * band_frac)))
    border_band = np.zeros_like(mask, dtype=np.uint8)
    border_band[:band, :] = 1
    border_band[h - band :, :] = 1
    border_band[:, :band] = 1
    border_band[:, w - band :] = 1
    area = float(mask.sum())
    band_area = float((mask & border_band).sum())
    touch_ratio = band_area / (area + 1e-6)
    sides = 0
    if (mask[:band, :].sum()) > 0:
        sides += 1
    if (mask[h - band :, :].sum()) > 0:
        sides += 1
    if (mask[:, :band].sum()) > 0:
        sides += 1
    if (mask[:, w - band :].sum()) > 0:
        sides += 1
    return float(touch_ratio), int(sides), border_band


def _curvature_stats(skel: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ys, xs = np.where(skel > 0)
    if len(xs) < 5:
        return None, None
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    # order by x+y for a crude path ordering; good enough for small skeletons
    order = np.argsort(pts[:, 0] + pts[:, 1])
    pts = pts[order]
    deltas = np.diff(pts.astype(np.float32), axis=0)
    if deltas.shape[0] < 3:
        return None, None
    angles = np.arctan2(deltas[:, 1] + 1e-6, deltas[:, 0] + 1e-6)
    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    if dtheta.size == 0:
        return None, None
    return float(np.var(dtheta)), float(np.mean(np.abs(dtheta)))


def _straightness(skel: np.ndarray, endpoints: int) -> Optional[float]:
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return None
    if endpoints < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    # endpoints approximate as min/max of sum coord
    idx_min = np.argmin(xs + ys)
    idx_max = np.argmax(xs + ys)
    dist = float(np.linalg.norm(pts[idx_max] - pts[idx_min]))
    return float(len(xs) / (dist + 1e-6))


def _endpoint_distance(endpoints_xy: np.ndarray) -> Optional[float]:
    if endpoints_xy is None or endpoints_xy.size == 0 or endpoints_xy.shape[0] < 2:
        return None
    pts = endpoints_xy.astype(np.float32)
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    return float(dist.max())


def _endpoint_coords(skel: np.ndarray) -> np.ndarray:
    if skel.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    sk = (skel > 0).astype(np.uint8)
    k = np.ones((3, 3), dtype=np.uint8)
    neigh = cv2.filter2D(sk, ddepth=cv2.CV_16S, kernel=k, borderType=cv2.BORDER_CONSTANT)
    neigh = neigh - sk
    ys, xs = np.where((sk == 1) & (neigh == 1))
    return np.stack([xs, ys], axis=1).astype(np.int32) if len(xs) else np.zeros((0, 2), dtype=np.int32)


def _features_from_mask(mask: np.ndarray, rules: GeometryRulesConfig) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    mask_bin = _binarize(mask)
    h, w = mask_bin.shape[:2]
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return {}, {}
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox_w, bbox_h = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    area = float(mask_bin.sum())
    bbox_area = float(bbox_w * bbox_h)
    mask_area_ratio = float(area / (bbox_area + 1e-6))

    skel = _skeletonize_binary(mask_bin)
    skel_len = float(skel.sum())
    width_stats = _distance_width_stats(mask_bin, skel)

    touch_ratio, touch_sides, border_band = _border_features(mask_bin, rules.border_band_frac)
    ori_var = _skeleton_orientation_variance(skel)
    ori_ecc = _orientation_eccentricity(skel)
    endpoints, branchpoints = _endpoints_and_branchpoints(skel)
    closed_loop = (endpoints == 0) and (skel_len > rules.min_len_loop)
    endpoint_xy = _endpoint_coords(skel)
    end_dist = _endpoint_distance(endpoint_xy)
    straight = float(skel_len / (end_dist + 1e-6)) if (end_dist is not None and end_dist > 0) else None
    curv_var, curv_mean_abs = _curvature_stats(skel) if branchpoints == 0 else (None, None)
    len_area_ratio = float(skel_len / (area + 1e-6))

    feats: Dict[str, Any] = {
        "area": area,
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
        "bbox_x2": float(x2),
        "bbox_y2": float(y2),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_area": bbox_area,
        "mask_area_ratio": mask_area_ratio,
        "skeleton_length": skel_len,
        "len_area_ratio": len_area_ratio,
        "width_mean": width_stats["width_mean"],
        "width_variance": width_stats["width_var"],
        "width_std": width_stats["width_std"],
        "width_p90": width_stats["width_p90"],
        "touch_border_ratio": float(touch_ratio),
        "touch_border_sides": float(touch_sides),
        "orientation_variance": ori_var,
        "orientation_eccentricity": float(ori_ecc),
        "endpoints_count": float(endpoints),
        "endpoint_count": float(endpoints),
        "branchpoints_count": float(branchpoints),
        "closed_loop": bool(closed_loop),
        "straightness": straight,
        "curvature_variance": curv_var,
        "curvature_mean_abs": curv_mean_abs,
    }

    debug_imgs: Dict[str, np.ndarray] = {}
    skeleton_vis = np.zeros((h, w, 3), dtype=np.uint8)
    skeleton_vis[skel > 0] = (0, 255, 0)
    debug_imgs["skeleton_vis"] = skeleton_vis
    border_band_vis = np.zeros((h, w, 3), dtype=np.uint8)
    border_band_vis[border_band > 0] = (0, 255, 255)
    border_band_vis[mask_bin > 0] = (255, 255, 255)
    debug_imgs["border_band_vis"] = border_band_vis
    if mask_bin.max() > 0:
        dist = cv2.distanceTransform(mask_bin.astype(np.uint8), cv2.DIST_L2, 3)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        width_vis = cv2.cvtColor(dist_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        width_vis[skel > 0] = (0, 255, 0)
        debug_imgs["width_vis"] = width_vis

    return feats, debug_imgs


def hard_reject_region(mask: np.ndarray, rules: GeometryRulesConfig) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    feats, _ = _features_from_mask(mask, rules)
    if not feats:
        return False, None, feats

    area = float(feats.get("area", 0.0))
    skel_len = float(feats.get("skeleton_length", 0.0))
    mask_area_ratio = float(feats.get("mask_area_ratio", 0.0))
    width_mean = float(feats.get("width_mean", 0.0))
    touch_ratio = float(feats.get("touch_border_ratio", 0.0))
    ori_ecc = float(feats.get("orientation_eccentricity", 1.0))
    endpoints = int(feats.get("endpoints_count", 0))
    curv_var = feats.get("curvature_variance")

    len_area_ratio = float(feats.get("len_area_ratio", 0.0))

    # keep override for thin cracks
    keep_override = (
        width_mean <= rules.thin_width_max
        and len_area_ratio >= rules.len_area_keep_min
        and touch_ratio < 0.6
    )
    if keep_override:
        return False, None, feats

    # Reject ONLY if definitely not structural damage (PARSE 6)
    if feats.get("closed_loop", False):
        return True, "closed_loop", feats

    if area >= rules.min_region_area and skel_len > 0 and skel_len < rules.min_skeleton_len:
        # A short skeleton relative to its filled area is likely a blob.
        if mask_area_ratio > rules.blob_area_ratio_max or len_area_ratio < rules.len_area_min:
            return True, "blob_like", feats

    if width_mean > rules.t_width_mean_high:
        return True, "too_thick", feats

    if curv_var is not None and float(curv_var) < rules.t_curv_var_low and feats.get("width_variance", 0.0) < rules.t_width_var_low:
        return True, "ornament_like", feats

    return False, None, feats


def _extract_rules_dict(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if not config:
        return {}
    if isinstance(config.get("geometry_filter"), dict):
        gf = config.get("geometry_filter", {})
        if isinstance(gf.get("rules"), dict):
            return gf.get("rules", {}) or {}
    if isinstance(config.get("rules"), dict):
        return config.get("rules", {}) or {}
    return dict(config)


def filter_regions_geometry(regions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules_cfg = GeometryRulesConfig(**(_extract_rules_dict(config) or {}))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(regions):
        if "region_id" not in r:
            r["region_id"] = idx + 1

        mask = r.get("crop_mask")
        if mask is None:
            mask = r.get("mask")
        if mask is None:
            mask = r.get("full_mask")

        if mask is None:
            r["kept"] = True
            r["dropped_reason"] = None
            r.setdefault("features", {})
            out.append(r)
            continue

        mask_bin = _binarize(mask)
        r["crop_mask"] = mask_bin

        feats, dbg_imgs = _features_from_mask(mask_bin, rules_cfg)
        if not feats:
            r["kept"] = False
            r["dropped_reason"] = "too_small"
            r["features"] = {}
            out.append(r)
            continue

        bbox = r.get("bbox")
        if bbox is not None and isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)
                bbox_area = float(bbox_w * bbox_h)
                feats["bbox_w"] = float(bbox_w)
                feats["bbox_h"] = float(bbox_h)
                feats["bbox_area"] = bbox_area
                feats["mask_area_ratio"] = float(float(feats.get("area", 0.0)) / (bbox_area + 1e-6))
            except Exception:
                pass

        dropped, reason, _ = hard_reject_region(mask_bin, rules_cfg)
        r["kept"] = not dropped
        r["dropped_reason"] = reason
        r["features"] = feats

        r_debug = r.get("debug", {}) or {}
        r_debug["touch_border_ratio"] = feats.get("touch_border_ratio")
        r_debug["orientation_eccentricity"] = feats.get("orientation_eccentricity")
        r_debug["endpoints_count"] = feats.get("endpoints_count")
        r_debug["branchpoints_count"] = feats.get("branchpoints_count")
        r["debug"] = r_debug

        r["debug_images"] = dbg_imgs
        out.append(r)

    return out


def filter_regions_hard_reject(regions: List[Dict[str, object]], config: Dict[str, object]) -> List[Dict[str, object]]:
    return filter_regions_geometry(regions=[dict(r) for r in regions], config=dict(config or {}))
