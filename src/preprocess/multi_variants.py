from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


def _clahe_lab(bgr: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, bb))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _bilateral(bgr: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def _guided_filter_gray(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    I_f = I.astype(np.float32) / 255.0
    p_f = p.astype(np.float32) / 255.0

    k = int(2 * int(radius) + 1)
    ksize = (k, k)

    mean_I = cv2.boxFilter(I_f, ddepth=-1, ksize=ksize)
    mean_p = cv2.boxFilter(p_f, ddepth=-1, ksize=ksize)
    mean_Ip = cv2.boxFilter(I_f * p_f, ddepth=-1, ksize=ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I_f * I_f, ddepth=-1, ksize=ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)

    q = mean_a * I_f + mean_b
    q = np.clip(q * 255.0, 0.0, 255.0).astype(np.uint8)
    return q


def _guided_filter_bgr(bgr: np.ndarray, radius: int, eps: float) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("guided filter expects BGR image")
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    channels = cv2.split(bgr)
    out = []
    for ch in channels:
        out.append(_guided_filter_gray(guide, ch, radius=radius, eps=eps))
    return cv2.merge(out)


def _unsharp_mask(bgr: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma))
    return cv2.addWeighted(bgr, 1.0 + float(amount), blurred, -float(amount), 0)


def _blackhat_ridge(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return bh


def _frangi_ridge(
    gray: np.ndarray,
    sigmas: list[float],
    beta1: float,
    beta2: float,
    ridge_type: str,
) -> np.ndarray:
    img = gray.astype(np.float32) / 255.0

    beta = float(beta1)
    c = float(beta2)

    vesselness = np.zeros_like(img, dtype=np.float32)

    for sigma in sigmas:
        s = float(sigma)
        if s <= 0:
            continue

        sm = cv2.GaussianBlur(img, (0, 0), sigmaX=s)
        Ixx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3) * (s * s)
        Iyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3) * (s * s)
        Ixy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3) * (s * s)

        tmp = np.sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy)
        l1 = 0.5 * (Ixx + Iyy - tmp)
        l2 = 0.5 * (Ixx + Iyy + tmp)

        abs_l1 = np.abs(l1)
        abs_l2 = np.abs(l2)
        swap = abs_l1 > abs_l2
        if np.any(swap):
            l1_s = l1.copy()
            l2_s = l2.copy()
            l1_s[swap] = l2[swap]
            l2_s[swap] = l1[swap]
            l1, l2 = l1_s, l2_s

        rb = (l1 / (l2 + 1e-12)) ** 2
        s2 = l1 * l1 + l2 * l2

        v = np.exp(-rb / (2.0 * beta * beta)) * (1.0 - np.exp(-s2 / (2.0 * c * c)))

        if ridge_type.lower() == "bright":
            v[l2 > 0] = 0.0
        else:
            v[l2 < 0] = 0.0

        vesselness = np.maximum(vesselness, v)

    vmin = float(vesselness.min())
    vmax = float(vesselness.max())
    if vmax - vmin <= 1e-12:
        out = np.zeros_like(gray, dtype=np.uint8)
    else:
        out = ((vesselness - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return out


def preprocess_base(image_bgr: np.ndarray, config: dict) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    clip = float(config.get("clahe_clip_limit", 2.0))
    tile = tuple(config.get("clahe_tile_grid_size", (8, 8)))

    out = _clahe_lab(image_bgr, clip_limit=clip, tile_grid_size=tile)

    denoise_method = str(config.get("denoise", config.get("noise_filter", "bilateral")) or "bilateral").lower()
    if denoise_method == "guided":
        radius = int(config.get("guided_radius", 8))
        eps = float(config.get("guided_eps", 1e-3))
        try:
            out = _guided_filter_bgr(out, radius=radius, eps=eps)
        except Exception:
            out = _bilateral(out, d=int(config.get("bilateral_d", 7)), sigma_color=float(config.get("bilateral_sigma_color", 50.0)), sigma_space=float(config.get("bilateral_sigma_space", 50.0)))
    else:
        out = _bilateral(out, d=int(config.get("bilateral_d", 7)), sigma_color=float(config.get("bilateral_sigma_color", 50.0)), sigma_space=float(config.get("bilateral_sigma_space", 50.0)))

    return out


def preprocess_blur_boost(image_bgr: np.ndarray, config: dict) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    sigma = float(config.get("unsharp_sigma", 1.0))
    amount = float(config.get("unsharp_amount", 0.6))
    clip = float(config.get("clahe_clip_limit", 2.0))
    tile = tuple(config.get("clahe_tile_grid_size", (8, 8)))

    out = _unsharp_mask(image_bgr, sigma=sigma, amount=amount)
    out = _clahe_lab(out, clip_limit=clip, tile_grid_size=tile)
    return out


def preprocess_ridge(image_bgr: np.ndarray, config: dict) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    method = str(config.get("ridge_method", "blackhat")).lower()

    if method == "frangi":
        sigmas_raw = config.get("frangi_sigmas", [1.0, 2.0, 3.0])
        sigmas = [float(s) for s in (list(sigmas_raw) if isinstance(sigmas_raw, (list, tuple)) else [sigmas_raw])]
        beta1 = float(config.get("frangi_beta1", 0.5))
        beta2 = float(config.get("frangi_beta2", 15.0))
        ridge_type = str(config.get("frangi_ridge_type", "dark"))
        ridge = _frangi_ridge(gray, sigmas=sigmas, beta1=beta1, beta2=beta2, ridge_type=ridge_type)
    else:
        k = int(config.get("blackhat_kernel", 9))
        ridge = _blackhat_ridge(gray, kernel_size=k)

    ridge_bgr = cv2.cvtColor(ridge, cv2.COLOR_GRAY2BGR)
    return ridge_bgr


def preprocess_variants(image_bgr: np.ndarray, config: dict | None = None) -> Dict[str, np.ndarray]:
    cfg = config or {}
    variants_cfg = cfg.get("variants", {}) if isinstance(cfg, dict) else {}

    variant_funcs: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
        "base": preprocess_base,
        "blur_boost": preprocess_blur_boost,
        "ridge": preprocess_ridge,
    }

    out: Dict[str, np.ndarray] = {}

    for name, fn in variant_funcs.items():
        vcfg = variants_cfg.get(name, {}) if isinstance(variants_cfg, dict) else {}
        enabled = bool(vcfg.get("enabled", name == "base"))
        if not enabled:
            continue

        merged: dict = {}
        if isinstance(cfg, dict):
            merged.update(cfg)
        if isinstance(vcfg, dict):
            merged.update(vcfg)

        out[name] = fn(image_bgr, merged)

    if not out:
        out["base"] = preprocess_base(image_bgr, cfg if isinstance(cfg, dict) else {})

    return out
