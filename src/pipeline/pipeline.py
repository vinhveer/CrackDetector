from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.detection.grounding_dino_model import GroundingDINOModel
from src.detection.prompt_manager import PromptManager
from src.models.loaders import load_grounding_dino, load_sam, make_dino_predictor, make_sam_predictor
from src.pipeline.device_utils import get_device
from src.pipeline.model_registry import get_grounding_dino, get_sam_model
from src.pipeline.models import BoxResult, CrackMetrics, CrackResult
from src.pipeline.visualization import create_overlay, draw_boxes, save_debug_image, visualize_mask
from src.postprocess.processor import PostProcessor
from src.preprocess.preprocessor import ImagePreprocessor
from src.segmentation.sam_model import SAMModel
from src.utils.config_loader import DEFAULT_CONFIG, load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.metrics_logger import MetricsLogger


class CrackDetectionPipeline:
    """End-to-end crack detection orchestrator."""

    def __init__(
        self,
        config: Optional[dict | str] = None,
        predictor=None,
        sam_predictor=None,
        model_loader=None,
        sam_loader=None,
    ) -> None:
        setup_logging()
        self.logger = get_logger(self.__class__.__name__)
        self.config = load_config(config)

        debug_cfg = self.config.get("debug", {})
        self.debug_enabled = debug_cfg.get("enabled", False)
        self.debug_dir = Path(debug_cfg.get("output_dir", "debug"))

        self.preprocessor = ImagePreprocessor(self.config.get("preprocess", DEFAULT_CONFIG["preprocess"]))
        self.prompt_manager = PromptManager(self.config.get("prompts", DEFAULT_CONFIG["prompts"]), material_type=self.config.get("material_type", "concrete"))
        device = get_device()

        # Auto-wire GroundingDINO if weights + deps are present
        if predictor is None:
            try:
                dino_model = get_grounding_dino(model_loader or (lambda: load_grounding_dino(device)))
                predictor = make_dino_predictor(dino_model)
            except Exception as exc:
                self.logger.warning("GroundingDINO not loaded (%s). Using dummy predictor.", exc)
                predictor = None

        # Auto-wire SAM if weights + deps are present
        if sam_predictor is None:
            try:
                sam_model = get_sam_model(sam_loader or (lambda: load_sam(device)))
                sam_predictor = make_sam_predictor(sam_model)
            except Exception as exc:
                self.logger.warning("SAM not loaded (%s). Using dummy predictor.", exc)
                sam_predictor = None

        self.detector = GroundingDINOModel(
            threshold_config=self.config.get("threshold"),
            sahi_config=self.config.get("sahi"),
            predictor=predictor,
            model_loader=model_loader,
        )
        self.sam = SAMModel(predictor=sam_predictor, config=self.config.get("sam"), model_loader=sam_loader)
        self.post = PostProcessor(self.config.get("postprocess"))

        self.metrics_logger = None
        if debug_cfg.get("log_csv", False):
            self.metrics_logger = MetricsLogger(str(self.debug_dir / "results.csv"))

    def run(self, image_path: str, material_type: str | None = None) -> CrackResult:
        start = time.time()
        original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        mat_type = material_type or self.config.get("material_type", "concrete")

        preprocessed, prep_meta, scale = self.preprocessor.preprocess(original)
        save_debug_image(self.debug_enabled, self.debug_dir, "step1_preprocessed.jpg", preprocessed)

        prompts = self.prompt_manager.multi_prompt_set(mat_type)
        boxes = self.detector.detect(preprocessed, prompts)
        debug_boxes_img = draw_boxes(preprocessed, boxes)
        save_debug_image(self.debug_enabled, self.debug_dir, "step2_boxes.jpg", debug_boxes_img)

        masks = self.sam.segment(preprocessed, boxes)
        if masks:
            raw_mask = visualize_mask(np.maximum.reduce(masks))
            save_debug_image(self.debug_enabled, self.debug_dir, "step3_masks.jpg", raw_mask)

        refined_mask, post_meta = self.post.refine(preprocessed, masks)
        save_debug_image(self.debug_enabled, self.debug_dir, "step4_refined_mask.jpg", visualize_mask(refined_mask))

        up_mask = self._resize_to_original(refined_mask, original.shape[:2])
        overlay = create_overlay(original, up_mask, color=(0, 0, 255), alpha=0.5)

        meta_boxes = self._scale_boxes(boxes, 1.0 / scale if scale > 0 else 1.0)
        metrics = self._compute_metrics(up_mask, meta_boxes)
        metrics.processing_time_ms = (time.time() - start) * 1000.0
        fallback_used = post_meta.get("fallback") is not None

        save_debug_image(self.debug_enabled, self.debug_dir, "step5_overlay.jpg", overlay)

        if self.metrics_logger:
            self.metrics_logger.log(Path(image_path).name, metrics, meta_boxes, fallback_used, prompts)

        self.logger.info("Pipeline finished in %.2f ms", metrics.processing_time_ms or 0)
        return CrackResult(
            final_mask=up_mask,
            overlay_image=overlay,
            boxes=meta_boxes,
            metrics=metrics,
            used_prompts=prompts,
            fallback_used=fallback_used,
        )

    def _resize_to_original(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        target_w, target_h = shape[1], shape[0]
        resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def _scale_boxes(self, boxes: list[BoxResult], inv_scale: float) -> list[BoxResult]:
        scaled: list[BoxResult] = []
        for b in boxes:
            scaled.append(
                BoxResult(
                    x1=b.x1 * inv_scale,
                    y1=b.y1 * inv_scale,
                    x2=b.x2 * inv_scale,
                    y2=b.y2 * inv_scale,
                    score=b.score,
                    prompt=b.prompt,
                    is_weak=b.is_weak,
                    tile_idx=b.tile_idx,
                ),
            )
        return scaled

    def _compute_metrics(self, final_mask: np.ndarray, boxes: list[BoxResult]) -> CrackMetrics:
        mask_bin = (final_mask > 0).astype(np.uint8)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        num_regions = max(0, num_labels - 1)
        total_pixels = final_mask.size if final_mask is not None else 0
        crack_pixels = int(mask_bin.sum()) if total_pixels else 0
        area_ratio = crack_pixels / total_pixels if total_pixels else 0.0
        avg_conf = sum(b.score for b in boxes) / len(boxes) if boxes else None
        return CrackMetrics(num_regions=num_regions, crack_area_ratio=area_ratio, avg_confidence=avg_conf)

