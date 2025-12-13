from __future__ import annotations

from typing import Any, Callable, Optional

GroundingLoader = Callable[[], Any]
SamLoader = Callable[[], Any]

_GROUNDING_DINO: Any = None
_SAM_MODEL: Any = None


def get_grounding_dino(loader: Optional[GroundingLoader] = None) -> Any:
    global _GROUNDING_DINO
    if _GROUNDING_DINO is None and loader:
        _GROUNDING_DINO = loader()
    return _GROUNDING_DINO


def get_sam_model(loader: Optional[SamLoader] = None) -> Any:
    global _SAM_MODEL
    if _SAM_MODEL is None and loader:
        _SAM_MODEL = loader()
    return _SAM_MODEL


def reset_registry() -> None:
    global _GROUNDING_DINO, _SAM_MODEL
    _GROUNDING_DINO = None
    _SAM_MODEL = None

