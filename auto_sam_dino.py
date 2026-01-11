import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.detection.box_utils import relax_box
from src.pipeline.pipeline import CrackDetectionPipeline
from src.pipeline.visualization import create_overlay, draw_boxes, visualize_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_name(p: Path) -> str:
    return p.stem.replace(" ", "_").replace(os.sep, "_")


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        return []
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file()]
    imgs = [p for p in paths if p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM (minimal runner). "
        "Fixes: BGR->RGB, prompt join, points=None, safe mask binarization."
    )
    ap.add_argument("--input-dir", default="img_input", help="Folder containing input images")
    ap.add_argument("--config", default=None, help="Path to YAML/JSON config")
    ap.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    ap.add_argument("--material-type", default=None, help="Material type override (e.g. concrete/asphalt)")
    ap.add_argument("--prompt-mode", default="disabled", help="Prompt mode: disabled | one_pass | multi_pass")
    ap.add_argument("--box-relax", type=float, default=0.2, help="Relax ratio for DINO boxes before SAM")
    ap.add_argument("--output-dir", default=None, help="Output folder (default: result_sam_dino_<ts>)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = no limit)")

    # Debug / forcing prompt behavior
    ap.add_argument(
        "--force-prompts",
        action="store_true",
        help="Use a built-in high-recall prompt list (ignores prompt_manager).",
    )
    ap.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print prompts (and joined text prompt) used for detection.",
    )
    ap.add_argument(
        "--save-rgb",
        action="store_true",
        help="Save rgb_input.jpg for debugging color pipeline.",
    )
    return ap.parse_args()


def _flatten_prompt_groups(prompt_groups: dict) -> list[str]:
    prompts: list[str] = []
    for ps in prompt_groups.values():
        if isinstance(ps, (list, tuple)):
            prompts.extend([str(x) for x in ps if str(x).strip()])
        else:
            s = str(ps).strip()
            if s:
                prompts.append(s)
    return prompts


def _default_high_recall_prompts(material_type: str) -> list[str]:
    m = (material_type or "asphalt").strip().lower()
    material_words = {
        "asphalt": ["asphalt", "road", "pavement"],
        "concrete": ["concrete", "cement", "pavement"],
        "wall": ["wall", "plaster wall", "painted wall"],
        "metal": ["metal", "steel", "metal surface"],
    }
    mats = material_words.get(m, [m]) or [m]

    # Very high recall phrasing for DINO (CLIP-style text)
    base = [
        "crack",
        "surface crack",
        "hairline crack",
        "thin crack line",
        "narrow linear crack",
        "long crack line",
        "dark crack line",
        "black crack line",
        "vertical crack line",
        "linear fracture",
        "surface fracture line",
        "surface defect line",
    ]
    # Expand with material variants
    expanded: list[str] = []
    for b in base:
        expanded.append(b)
        for mw in mats:
            expanded.append(f"{b} on {mw}")
    # De-dup keep order
    seen = set()
    out: list[str] = []
    for p in expanded:
        p2 = " ".join(p.split()).strip()
        if p2 and p2 not in seen:
            seen.add(p2)
            out.append(p2)
    return out


def _ensure_rgb_uint8(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    # cv2.imread gives BGR
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _join_prompts(prompts: Sequence[str]) -> str:
    # GroundingDINO wrappers often expect a single string prompt.
    # Joining with ". " is common and works well.
    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    return ". ".join(cleaned)


def _safe_masks_to_uint8_255(masks: Sequence[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Combine list of masks into a single uint8 mask (0/255).
    Handles bool, uint8 (0/1 or 0/255), float prob (0..1), and logits (neg/pos).
    """
    h, w = shape_hw
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)

    bin_masks: list[np.ndarray] = []
    for m in masks:
        if m is None:
            continue
        m = np.asarray(m)

        # Resize safeguard (if any wrapper returns crop-sized masks unexpectedly)
        if m.shape[:2] != (h, w):
            try:
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                # If resize fails, skip this mask
                continue

        if m.dtype == np.bool_:
            mb = m
        elif np.issubdtype(m.dtype, np.integer):
            # Could be 0/1 or 0/255
            mb = m > 0
        else:
            # float prob or logits
            mx = float(np.max(m))
            mn = float(np.min(m))
            # Heuristic:
            # - If in [0,1], treat as probability -> thr 0.5
            # - Else treat as logits -> thr 0.0
            if mn >= 0.0 and mx <= 1.0:
                mb = m > 0.5
            else:
                mb = m > 0.0

        bin_masks.append(mb.astype(np.uint8) * 255)

    if not bin_masks:
        return np.zeros((h, w), dtype=np.uint8)

    return np.maximum.reduce(bin_masks)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else Path(f"result_sam_dino_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    images = iter_images(input_dir, recursive=bool(args.recursive))
    if not images:
        print(f"No images found in: {input_dir.resolve()}")
        return

    # NOTE: CrackDetectionPipeline signature in your repo: CrackDetectionPipeline(config=None or path)
    pipeline = CrackDetectionPipeline(args.config)

    prompt_mode = str(args.prompt_mode or "").strip().lower()
    use_damage = bool(prompt_mode) and prompt_mode not in {"disabled", "off", "false", "0", "none"}

    limit = int(args.limit or 0)
    processed = 0

    for img_path in images:
        name = _safe_name(img_path)
        img_out_dir = out_root / name
        img_out_dir.mkdir(parents=True, exist_ok=True)

        original_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        # ✅ Convert to RGB for DINO/SAM (most wrappers expect RGB)
        image_rgb = _ensure_rgb_uint8(original_bgr)

        mat_type = args.material_type or pipeline.config.get("material_type", "concrete")

        # ---- prompts ----
        if args.force_prompts:
            prompts = _default_high_recall_prompts(mat_type)
        else:
            if use_damage:
                prompt_groups = pipeline.prompt_manager.damage_prompt_map(
                    material_type=mat_type,
                    mode=prompt_mode,
                )
                prompts = _flatten_prompt_groups(prompt_groups)
            else:
                # multi_prompt_set may return list[str] or something similar
                ps = pipeline.prompt_manager.multi_prompt_set(mat_type)
                if isinstance(ps, (list, tuple)):
                    prompts = [str(x) for x in ps if str(x).strip()]
                else:
                    s = str(ps).strip()
                    prompts = [s] if s else []

        # Ensure we always have at least something
        if not prompts:
            prompts = _default_high_recall_prompts(mat_type)

        # ✅ Many DINO wrappers expect a single text prompt string
        text_prompt = _join_prompts(prompts)

        if args.print_prompts:
            (img_out_dir / "prompts.txt").write_text(
                "\n".join(prompts) + "\n\nJOINED:\n" + text_prompt + "\n",
                encoding="utf-8",
            )
            print(f"[{img_path.name}] prompts={len(prompts)}")

        if args.save_rgb:
            cv2.imwrite(str(img_out_dir / "rgb_input.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        # ---- inference ----
        t0 = time.perf_counter()

        # DINO detect (try joined string first)
        try:
            boxes = pipeline.detector.detect(image_rgb, text_prompt)
            detect_call = "detect(image_rgb, text_prompt:str)"
        except TypeError:
            # Fallback: some implementations accept list[str]
            boxes = pipeline.detector.detect(image_rgb, prompts)
            detect_call = "detect(image_rgb, prompts:list[str])"

        # Relax boxes for SAM
        boxes_for_sam = [
            relax_box(b, ratio=float(args.box_relax), image_shape=image_rgb.shape[:2])
            for b in (boxes or [])
        ]

        # ✅ points must be None (NOT []) so SAM wrapper can do box-only or seed-first internally
        try:
            masks = pipeline.sam.segment(image_rgb, boxes=boxes_for_sam, points=None)
            sam_call = "segment(image_rgb, boxes, points=None)"
        except TypeError:
            # Some wrappers don't have points arg at all
            masks = pipeline.sam.segment(image_rgb, boxes=boxes_for_sam)
            sam_call = "segment(image_rgb, boxes)"

        infer_ms = (time.perf_counter() - t0) * 1000.0

        # ---- mask combine (safe) ----
        raw_mask_255 = _safe_masks_to_uint8_255(masks or [], shape_hw=image_rgb.shape[:2])

        # visualize_mask may already expect 0/255 or 0/1 depending on your implementation.
        # We pass through visualize_mask to keep your existing visuals consistent.
        mask_vis = visualize_mask(raw_mask_255)

        # ---- visuals (use BGR for OpenCV drawing/saving) ----
        boxes_img = draw_boxes(original_bgr, boxes_for_sam)
        overlay = create_overlay(original_bgr, mask_vis)

        cv2.imwrite(str(img_out_dir / "boxes.jpg"), boxes_img)
        cv2.imwrite(str(img_out_dir / "pred_mask.png"), mask_vis)
        cv2.imwrite(str(img_out_dir / "overlay.jpg"), overlay)

        # ---- meta ----
        (img_out_dir / "meta.txt").write_text(
            "\n".join(
                [
                    f"image={img_path.name}",
                    f"infer_ms={infer_ms:.2f}",
                    f"boxes_raw={len(boxes or [])}",
                    f"boxes_for_sam={len(boxes_for_sam)}",
                    f"material_type={mat_type}",
                    f"prompt_mode={prompt_mode}",
                    f"detect_call={detect_call}",
                    f"sam_call={sam_call}",
                    f"prompts={len(prompts)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"OK: {img_path.name} | boxes={len(boxes_for_sam)} | "
            f"infer_ms={infer_ms:.2f} | detect={detect_call} | sam={sam_call}"
        )

        processed += 1
        if limit > 0 and processed >= limit:
            break

    print(f"Done. Output: {out_root.resolve()}")


if __name__ == "__main__":
    main()