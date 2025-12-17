from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.detection.grounding_dino_model import GroundingDINOModel
from src.detection.box_utils import relax_box, union_boxes_spatially
from src.detection.prompt_manager import PromptManager
from src.models.loaders import load_grounding_dino, load_sam, make_dino_predictor, make_sam_predictor
from src.pipeline.device_utils import get_device
from src.pipeline.model_registry import get_grounding_dino, get_sam_model
from src.pipeline.models import (
    BoxResult,
    CrackMetrics,
    CrackResult,
    PipelineResult,
    RegionGeometry,
    RegionMeta,
    RegionResult,
    RegionScores,
)
from src.pipeline.confidence import CrackConfidenceAggregator
from src.pipeline.visualization import create_overlay, draw_boxes, visualize_mask
from src.postprocess.processor import PostProcessor
from src.postprocess.reasons import normalize_dropped_reason
from src.preprocess.preprocessor import ImagePreprocessor
from src.segmentation.point_generator import generate_edge_points
from src.segmentation.sam_wrapper import SAMWrapper
from src.pipeline.fallback import run_edge_first_fallback
from src.utils.config_loader import DEFAULT_CONFIG, load_config
from src.utils.debug import save_debug_image
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
        self.sam = SAMWrapper(predictor=sam_predictor, config=self.config.get("sam"))
        self.post = PostProcessor(self.config.get("postprocess"), geometry_config=self.config.get("geometry_filter"))

        self.metrics_logger = None
        if debug_cfg.get("log_csv", False):
            self.metrics_logger = MetricsLogger(str(self.debug_dir / "results.csv"))

    def run(self, image_path: str, material_type: str | None = None, runtime: dict | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        flags: dict[str, bool] = {"dino_failed": False}
        original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        warnings: list[str] = []

        orig_geometry_enabled = self.post.geometry_enabled
        if runtime is not None and "enable_geometry_filter" in runtime:
            self.post.geometry_enabled = bool(runtime.get("enable_geometry_filter"))

        debug_enabled = self.debug_enabled
        if isinstance(runtime, dict) and "debug_enabled" in runtime:
            debug_enabled = bool(runtime.get("debug_enabled"))

        mat_type = material_type or self.config.get("material_type", "concrete")

        try:
            prep_meta: dict[str, str] = {}
            selected_variant = "base"
            selected_variants = None
            if isinstance(runtime, dict) and isinstance(runtime.get("preprocess_variants"), list):
                selected_variants = [str(v) for v in runtime.get("preprocess_variants") if v]
            if selected_variants:
                variants = self.preprocessor.preprocess_variants(original)
                for name in selected_variants:
                    if name in variants:
                        selected_variant = name
                        break
                preprocessed = variants.get(selected_variant) or variants.get("base") or next(iter(variants.values()))
                h0, w0 = original.shape[:2]
                target = int(getattr(self.preprocessor.config, "target_size", 1024))
                scale = 1.0 if max(h0, w0) <= target else float(target / float(max(h0, w0)))
                prep_meta["variant"] = selected_variant
            else:
                preprocessed, prep_meta, scale = self.preprocessor.preprocess(original)
            save_debug_image(debug_enabled, self.debug_dir, "01_preprocess.jpg", preprocessed)

            prompt_mode = None
            if isinstance(runtime, dict) and runtime.get("damage_prompt_mode") is not None:
                prompt_mode = str(runtime.get("damage_prompt_mode") or "").strip().lower()
            use_damage = bool(prompt_mode) and prompt_mode not in {"disabled", "off", "false", "0", "none"}
            if use_damage:
                prompt_groups = self.prompt_manager.damage_prompt_map(material_type=mat_type, mode=prompt_mode)
                prompts: list[str] = []
                for ps in prompt_groups.values():
                    prompts.extend(ps)
            else:
                prompts = self.prompt_manager.multi_prompt_set(mat_type)

            boxes = self.detector.detect(preprocessed, prompts)
            if not boxes:
                flags["dino_failed"] = True
            boxes_for_sam = [relax_box(b, ratio=0.2, image_shape=preprocessed.shape[:2]) for b in boxes]
            debug_boxes_img = draw_boxes(preprocessed, boxes)
            save_debug_image(debug_enabled, self.debug_dir, "02_dino_boxes.jpg", debug_boxes_img)

            pipeline_cfg = self.config.get("pipeline", {})
            enable_edge_fallback = bool(pipeline_cfg.get("enable_edge_fallback", True))

            edge_points = generate_edge_points(preprocessed) if (not boxes_for_sam and enable_edge_fallback) else []
            masks = self.sam.segment(preprocessed, boxes=boxes_for_sam, points=edge_points)
            raw_mask = np.maximum.reduce(masks) if masks else np.zeros(preprocessed.shape[:2], dtype=np.uint8)
            sam_raw_mask_img = visualize_mask(raw_mask)
            save_debug_image(debug_enabled, self.debug_dir, "03_sam_raw_mask.jpg", sam_raw_mask_img)

            refined_mask, post_meta, geometry_input, geometry_filtered, dropped_regions = self.post.refine(preprocessed, masks)
            avg_conf_dino = sum(b.score for b in boxes_for_sam) / len(boxes_for_sam) if boxes_for_sam else None
            fallback_used = post_meta.get("fallback") is not None

            # PHASE 6: edge-first fallback when DINO fails badly.
            fallback_thr = 0.15
            should_fallback = bool(flags.get("dino_failed")) or (avg_conf_dino is not None and avg_conf_dino < fallback_thr)
            if should_fallback and enable_edge_fallback:
                fb_input, fb_filtered, fb_dropped = run_edge_first_fallback(
                    preprocessed,
                    self.post.geometry_filter,
                    geometry_enabled=self.post.geometry_enabled,
                )
                if int(fb_filtered.sum()) > 0:
                    geometry_input = fb_input
                    geometry_filtered = fb_filtered
                    refined_mask = fb_filtered
                    dropped_regions = fb_dropped
                    fallback_used = True

            save_debug_image(debug_enabled, self.debug_dir, "04_geometry_input.jpg", visualize_mask(geometry_input))
            save_debug_image(debug_enabled, self.debug_dir, "05_geometry_filtered.jpg", visualize_mask(geometry_filtered))

            up_mask = self._resize_to_original(refined_mask, original.shape[:2])
            overlay = create_overlay(original, up_mask, color=(0, 0, 255), alpha=0.5)

            meta_boxes = self._scale_boxes(boxes_for_sam, 1.0 / scale if scale > 0 else 1.0)
            metrics = self._compute_metrics(up_mask, meta_boxes)
            metrics.processing_time_ms = (time.perf_counter() - t0) * 1000.0

            # PHASE 7: confidence aggregation
            dino_conf = float(avg_conf_dino or 0.0)
            try:
                geom = self.post.geometry_filter.analyze(up_mask, original)
                length_area_ratio = float(geom.get("length_area_ratio", 0.0))
                is_crack = bool(geom.get("is_crack", False))
                curvature = float(geom.get("curvature", 0.0))
            except Exception:
                length_area_ratio = 0.0
                is_crack = False
                curvature = 0.0

            geometry_score = 1.0 if is_crack else min(1.0, length_area_ratio / 0.20) if length_area_ratio > 0 else 0.0
            continuity_regions = 1.0 / float(max(1, metrics.num_regions))
            continuity_curve = 1.0 / float(max(1.0, curvature))
            continuity_score = 0.5 * continuity_regions + 0.5 * min(1.0, continuity_curve)

            aggregator = CrackConfidenceAggregator()
            metrics.final_conf = aggregator.aggregate(dino_conf=dino_conf, geometry_score=geometry_score, continuity_score=continuity_score)

            save_debug_image(debug_enabled, self.debug_dir, "06_overlay.jpg", overlay)

            if self.metrics_logger:
                self.metrics_logger.log(Path(image_path).name, metrics, meta_boxes, fallback_used, prompts)
                self.metrics_logger.log_dropped_regions(Path(image_path).name, dropped_regions)

            self.logger.info("Pipeline finished in %.2f ms", metrics.processing_time_ms or 0)

            dropped_map = {rid: reason for rid, reason in dropped_regions}
            gi_bin = (geometry_input > 0).astype(np.uint8)
            gf_bin = (geometry_filtered > 0).astype(np.uint8)
            before_labels, before_lab, _, _ = cv2.connectedComponentsWithStats(gi_bin, connectivity=8)
            after_labels, _, _, _ = cv2.connectedComponentsWithStats(gf_bin, connectivity=8)
            region_details = post_meta.get("region_details")
            regions: list[RegionResult] = []
            kept_lengths: list[float] = []
            kept_widths: list[float] = []
            dropped_counts: dict[str, int] = {}

            if region_details:
                for i, r in enumerate(region_details):
                    rid = int(r.get("region_id", i + 1))
                    kept = bool(r.get("kept", True))
                    reason = r.get("dropped_reason")
                    if reason is not None:
                        reason = str(reason)
                        allowed = {"closed_loop", "blob_like", "too_thick", "ornament_like", "border_drop", "merged", "unknown"}
                        if reason not in allowed:
                            reason = "unknown"
                    feats = r.get("features")
                    if feats is None:
                        feats = {}

                    bbox_pre = None
                    if i < len(boxes_for_sam):
                        bbox_pre = boxes_for_sam[i].as_tuple()
                    bbox = bbox_pre
                    if bbox is None:
                        bbox = r.get("bbox")
                    if bbox is None:
                        bbox = (0, 0, 0, 0)
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    inv_scale = 1.0 / scale if scale > 0 else 1.0
                    bbox_orig = (int(x1 * inv_scale), int(y1 * inv_scale), int(x2 * inv_scale), int(y2 * inv_scale))

                    mask_pre = r.get("crop_mask")
                    if mask_pre is None and i < len(masks):
                        mask_pre = masks[i]
                    if mask_pre is None:
                        mask_pre = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
                    mask_pre = (mask_pre > 0).astype(np.uint8)
                    mask_orig = self._resize_to_original(mask_pre, original.shape[:2])

                    geom = RegionGeometry(
                        area=int(feats.get("area", 0)),
                        skeleton_length=float(feats.get("skeleton_length", 0.0)),
                        endpoint_count=int(feats.get("endpoint_count", feats.get("endpoints_count", 0) or 0)),
                        width_mean=float(feats.get("width_mean", 0.0)),
                        width_var=float(feats.get("width_variance", feats.get("width_var", 0.0) or 0.0)),
                        length_area_ratio=float(feats.get("len_area_ratio", feats.get("length_area_ratio", 0.0) or 0.0)),
                        curvature_var=float(feats.get("curvature_variance", 0.0) or 0.0),
                    )
                    scores = RegionScores(dino_conf=float(boxes_for_sam[i].score) if i < len(boxes_for_sam) else None, final_conf=None)
                    meta = RegionMeta(prompt_name=str(getattr(boxes_for_sam[i], "prompt", "")) if i < len(boxes_for_sam) else None, variant_name=prep_meta.get("variant"), damage_type=None)
                    regions.append(
                        RegionResult(
                            region_id=rid,
                            bbox=bbox_orig,
                            mask=mask_orig,
                            kept=bool(kept),
                            dropped_reason=reason,
                            scores=scores,
                            geometry=geom,
                            meta=meta,
                        ),
                    )
                    if kept:
                        kept_lengths.append(float(geom.skeleton_length))
                        kept_widths.append(float(geom.width_mean))
                    else:
                        if reason:
                            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1

                num_regions_before = len(region_details)
                num_regions_after = sum(1 for r in region_details if r.get("kept", True))
            else:
                num_regions_before = max(0, int(before_labels) - 1)
                num_regions_after = max(0, int(after_labels) - 1)

                for region_id in range(1, int(before_labels)):
                    region_mask = (before_lab == region_id).astype(np.uint8)
                    analysis = self.post.geometry_filter.analyze(region_mask, preprocessed)
                    dropped_reason = dropped_map.get(region_id)
                    if dropped_reason is None and int((gf_bin & region_mask).sum()) == 0:
                        dropped_reason = normalize_dropped_reason(analysis.get("reason"))
                    kept = dropped_reason is None
                    if dropped_reason is not None:
                        dropped_reason = normalize_dropped_reason(dropped_reason)
                        allowed = {"closed_loop", "blob_like", "too_thick", "ornament_like", "border_drop", "merged", "unknown"}
                        if str(dropped_reason) not in allowed:
                            dropped_reason = "unknown"
                        dropped_counts[dropped_reason] = dropped_counts.get(dropped_reason, 0) + 1

                    geom = RegionGeometry(
                        area=int(analysis.get("area", 0)),
                        skeleton_length=float(analysis.get("skeleton_length", 0.0)),
                        endpoint_count=int(analysis.get("endpoint_count", 0) or 0),
                        width_mean=float(analysis.get("width_mean", 0.0)),
                        width_var=float(analysis.get("width_variance", 0.0)),
                        length_area_ratio=float(analysis.get("length_area_ratio", 0.0)),
                        curvature_var=float(analysis.get("curvature_var", 0.0) or 0.0),
                    )
                    regions.append(
                        RegionResult(
                            region_id=int(region_id),
                            bbox=(0, 0, 0, 0),
                            mask=self._resize_to_original(region_mask, original.shape[:2]),
                            kept=bool(kept),
                            dropped_reason=str(dropped_reason) if dropped_reason is not None else None,
                            scores=RegionScores(dino_conf=None, final_conf=None),
                            geometry=geom,
                            meta=RegionMeta(),
                        ),
                    )

                    if kept:
                        kept_lengths.append(float(geom.skeleton_length))
                        kept_widths.append(float(geom.width_mean))

            avg_width = float(sum(kept_widths) / len(kept_widths)) if kept_widths else 0.0
            avg_length = float(sum(kept_lengths) / len(kept_lengths)) if kept_lengths else 0.0

            dropped_reason_counts = dict(dropped_counts)
            num_kept = sum(1 for r in regions if bool(r.kept))
            num_dropped = len(regions) - num_kept

            result_metrics: dict[str, object] = {
                "num_boxes": int(len(boxes_for_sam)),
                "num_regions_before": int(num_regions_before),
                "num_regions_after": int(num_regions_after),
                "num_kept": int(num_kept),
                "num_dropped": int(num_dropped),
                "dropped_reason_counts": dropped_reason_counts,
                "time_ms_total": float(metrics.processing_time_ms or 0.0),
                "fallback_used": bool(fallback_used),
                "avg_width": float(avg_width),
                "avg_length": float(avg_length),
                "final_confidence": float(metrics.final_conf or 0.0),
                "preprocess_scale": float(scale),
            }

            sam_raw_mask_bgr = cv2.cvtColor(sam_raw_mask_img, cv2.COLOR_GRAY2BGR) if sam_raw_mask_img.ndim == 2 else sam_raw_mask_img
            geom_in_viz = visualize_mask(geometry_input)
            geom_kept_viz = visualize_mask(geometry_filtered)
            geom_in_bgr = cv2.cvtColor(geom_in_viz, cv2.COLOR_GRAY2BGR) if geom_in_viz.ndim == 2 else geom_in_viz
            geom_kept_bgr = cv2.cvtColor(geom_kept_viz, cv2.COLOR_GRAY2BGR) if geom_kept_viz.ndim == 2 else geom_kept_viz
            geom_filt_bgr = geom_kept_bgr

            images: dict[str, np.ndarray] = {
                "input": original,
                "preprocessed": preprocessed,
                "dino_boxes_overlay": debug_boxes_img,
                "sam_raw_mask_viz": sam_raw_mask_bgr,
                "geometry_input_viz": geom_in_bgr,
                "geometry_kept_mask_viz": geom_kept_bgr,
                "final_overlay": overlay,
                "final_mask": (up_mask > 0).astype(np.uint8),
                "geometry_labels": before_lab.astype(np.int32),
                "dino_boxes": debug_boxes_img,
                "sam_raw_mask": sam_raw_mask_bgr,
                "geometry_filtered_mask": geom_filt_bgr,
            }

            config_snapshot = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}

            return PipelineResult(images=images, regions=regions, metrics=result_metrics, config_snapshot=config_snapshot, warnings=warnings)
        finally:
            self.post.geometry_enabled = orig_geometry_enabled

    def run_legacy(self, image_path: str, material_type: str | None = None, runtime: dict | None = None) -> CrackResult:
        result = self.run(image_path=image_path, material_type=material_type, runtime=runtime)
        final_mask = result.images.get("final_mask")
        if final_mask is None:
            viz = result.images.get("geometry_kept_mask_viz")
            if viz is None:
                final_mask = np.zeros_like(next(iter(result.images.values()))) if result.images else np.zeros((1, 1), dtype=np.uint8)
            else:
                final_mask = (viz > 0).astype(np.uint8)
        overlay = result.images.get("final_overlay")
        if overlay is None:
            inp = result.images.get("input")
            overlay = inp if inp is not None else (cv2.cvtColor(final_mask * 255, cv2.COLOR_GRAY2BGR) if final_mask.ndim == 2 else final_mask)

        mask_bin = (final_mask > 0).astype(np.uint8)
        total_pixels = int(mask_bin.size) if mask_bin is not None else 0
        crack_pixels = int(mask_bin.sum()) if total_pixels else 0
        area_ratio = float(crack_pixels / total_pixels) if total_pixels else 0.0

        metrics = CrackMetrics(
            num_regions=int(result.metrics.get("num_regions_after", 0) or 0),
            crack_area_ratio=area_ratio,
            avg_confidence=None,
            processing_time_ms=float(result.metrics.get("time_ms_total", 0.0) or 0.0),
            final_conf=float(result.metrics.get("final_confidence", 0.0) or 0.0),
        )
        return CrackResult(
            final_mask=mask_bin,
            overlay_image=overlay,
            boxes=[],
            metrics=metrics,
            used_prompts=[],
            fallback_used=bool(result.metrics.get("fallback_used", False)),
        )

    def propose_regions(self, image_path: str, material_type: str | None = None, runtime: dict | None = None) -> list[dict[str, object]]:
        original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        mat_type = material_type or self.config.get("material_type", "concrete")
        pipeline_cfg = self.config.get("pipeline", {})

        points = None
        if isinstance(runtime, dict) and isinstance(runtime.get("points"), list):
            points = runtime.get("points")

        variants = self.preprocessor.preprocess_variants(original)
        if not variants:
            base_img, _, _ = self.preprocessor.preprocess(original)
            variants = {"base": base_img}

        base_variant_img = variants.get("base") or next(iter(variants.values()))

        prompt_groups: dict[str, list[str]]
        if getattr(self.prompt_manager, "damage", None) is not None and bool(getattr(self.prompt_manager.damage, "enabled", False)):
            mode = str(getattr(self.prompt_manager.damage, "mode", "one_pass"))
            prompt_groups = self.prompt_manager.damage_prompt_map(material_type=mat_type, mode=mode)
        else:
            prompt_groups = {"crack": self.prompt_manager.multi_prompt_set(mat_type)}

        raw_boxes: list[BoxResult] = []
        for variant_name, variant_img in variants.items():
            for prompt_name, prompts in prompt_groups.items():
                for p in prompts:
                    for b in self.detector._detect_single_prompt(variant_img, p):
                        # Preserve the raw detection; encode origin for debugging (no score-based dedup here).
                        b.prompt = f"{variant_name}:{prompt_name}:{b.prompt}"
                        raw_boxes.append(b)

        merge_regions = True
        if isinstance(runtime, dict) and "merge_regions" in runtime:
            merge_regions = bool(runtime.get("merge_regions"))

        if merge_regions:
            union_iou = float(pipeline_cfg.get("proposal_union_iou", 0.0))
            boxes_for_sam = union_boxes_spatially(raw_boxes, iou_threshold=union_iou, image_shape=base_variant_img.shape[:2])
        else:
            boxes_for_sam = raw_boxes

        use_tight_boxes = True
        if isinstance(runtime, dict) and "use_tight_box_sam" in runtime:
            use_tight_boxes = bool(runtime.get("use_tight_box_sam"))

        if use_tight_boxes:
            regions = self.sam.segment_regions_tight_boxes(base_variant_img, boxes=boxes_for_sam, points=points)
        else:
            regions = self.sam.segment_regions(base_variant_img, boxes=boxes_for_sam, points=points)
        return regions

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

