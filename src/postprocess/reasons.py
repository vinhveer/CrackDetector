from __future__ import annotations

from enum import Enum


class DroppedReason(str, Enum):
    TOO_SHORT = "too_short"
    BLOB_LIKE = "blob_like"
    TOO_THICK = "too_thick"
    DISCONTINUOUS = "discontinuous"
    MERGED = "merged"
    CLOSED_LOOP = "closed_loop"
    ORNAMENT_LIKE = "ornament_like"


_ALLOWED_DROPPED_REASONS: set[str] = {r.value for r in DroppedReason}


def normalize_dropped_reason(reason: object) -> str:
    if reason is None:
        return DroppedReason.DISCONTINUOUS.value
    s = str(reason)
    if s in _ALLOWED_DROPPED_REASONS:
        return s
    if s == "low_continuity":
        return DroppedReason.DISCONTINUOUS.value
    if s == "rejected":
        return DroppedReason.DISCONTINUOUS.value
    return DroppedReason.DISCONTINUOUS.value
