from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.pipeline.models import BoxResult


def relax_box(box: BoxResult, ratio: float = 0.2, image_shape: Optional[Tuple[int, int]] = None) -> BoxResult:
    w = float(box.x2 - box.x1)
    h = float(box.y2 - box.y1)
    dx = w * ratio
    dy = h * ratio

    x1 = box.x1 - dx
    y1 = box.y1 - dy
    x2 = box.x2 + dx
    y2 = box.y2 + dy

    if image_shape is not None:
        ih, iw = image_shape
        x1 = max(0.0, min(x1, float(iw - 1)))
        y1 = max(0.0, min(y1, float(ih - 1)))
        x2 = max(0.0, min(x2, float(iw - 1)))
        y2 = max(0.0, min(y2, float(ih - 1)))

    return BoxResult(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=box.score,
        prompt=box.prompt,
        is_weak=box.is_weak,
        tile_idx=box.tile_idx,
    )


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def union_boxes_spatially(
    boxes: Sequence[BoxResult],
    iou_threshold: float = 0.0,
    image_shape: Optional[Tuple[int, int]] = None,
) -> List[BoxResult]:
    remaining = list(boxes)
    merged: List[BoxResult] = []

    while remaining:
        cluster = [remaining.pop(0)]

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(remaining):
                cand = remaining[i]
                cand_t = (float(cand.x1), float(cand.y1), float(cand.x2), float(cand.y2))
                overlaps = False
                for c in cluster:
                    c_t = (float(c.x1), float(c.y1), float(c.x2), float(c.y2))
                    if _iou(cand_t, c_t) >= float(iou_threshold) and _iou(cand_t, c_t) > 0.0:
                        overlaps = True
                        break
                if overlaps:
                    cluster.append(cand)
                    remaining.pop(i)
                    changed = True
                else:
                    i += 1

        x1 = min(float(b.x1) for b in cluster)
        y1 = min(float(b.y1) for b in cluster)
        x2 = max(float(b.x2) for b in cluster)
        y2 = max(float(b.y2) for b in cluster)
        if image_shape is not None:
            ih, iw = image_shape
            x1 = max(0.0, min(x1, float(iw - 1)))
            y1 = max(0.0, min(y1, float(ih - 1)))
            x2 = max(0.0, min(x2, float(iw - 1)))
            y2 = max(0.0, min(y2, float(ih - 1)))

        score = max(float(b.score) for b in cluster)
        prompt = cluster[0].prompt
        is_weak = any(bool(b.is_weak) for b in cluster)

        merged.append(
            BoxResult(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=score,
                prompt=prompt,
                is_weak=is_weak,
                tile_idx=None,
            ),
        )

    return merged
