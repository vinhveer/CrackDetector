from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticSample:
    name: str
    image_path: Path
    gt_mask: np.ndarray


def _draw_crack(mask: np.ndarray, points: List[Tuple[int, int]], thickness: int) -> None:
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], 1, thickness=thickness, lineType=cv2.LINE_AA)


def _make_base(h: int, w: int, texture: bool) -> np.ndarray:
    base = np.full((h, w, 3), 210, dtype=np.uint8)
    if not texture:
        return base
    noise = np.random.normal(0, 18, size=(h, w, 1)).astype(np.float32)
    base_f = base.astype(np.float32)
    base_f[..., 0] = np.clip(base_f[..., 0] + noise[..., 0], 0, 255)
    base_f[..., 1] = np.clip(base_f[..., 1] + noise[..., 0], 0, 255)
    base_f[..., 2] = np.clip(base_f[..., 2] + noise[..., 0], 0, 255)
    base = base_f.astype(np.uint8)
    return base


def _apply_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)


def build_synthetic_dataset(output_dir: Path, seed: int = 123) -> List[SyntheticSample]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = 720, 960

    samples: List[SyntheticSample] = []

    def make_one(name: str, blur: bool, texture: bool, thickness: int) -> None:
        img = _make_base(h, w, texture=texture)
        gt = np.zeros((h, w), dtype=np.uint8)

        x0 = int(rng.integers(80, 160))
        y0 = int(rng.integers(120, 220))
        pts = [(x0, y0)]
        for _ in range(10):
            x = int(np.clip(pts[-1][0] + rng.integers(50, 110), 0, w - 1))
            y = int(np.clip(pts[-1][1] + rng.integers(-80, 80), 0, h - 1))
            pts.append((x, y))

        _draw_crack(gt, pts, thickness=thickness)

        crack_color = (20, 20, 20)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], crack_color, thickness=thickness, lineType=cv2.LINE_AA)

        if blur:
            img = _apply_blur(img)

        path = output_dir / name
        cv2.imwrite(str(path), img)
        samples.append(SyntheticSample(name=name, image_path=path, gt_mask=(gt > 0).astype(np.uint8)))

    make_one("synthetic_sharp.jpg", blur=False, texture=False, thickness=2)
    make_one("synthetic_blur.jpg", blur=True, texture=False, thickness=2)
    make_one("synthetic_texture.jpg", blur=False, texture=True, thickness=1)

    return samples
