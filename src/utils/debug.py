from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_debug_image(enabled: bool, debug_dir: Path, name: str, image: np.ndarray) -> None:
    if not enabled:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / name
    cv2.imwrite(str(path), image)
