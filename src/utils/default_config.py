from __future__ import annotations

from typing import Any, Dict

# EXTREME RECALL + MULTI PROMPT preset
# Goal: force GroundingDINO to produce at least SOME boxes
# even for single, obvious cracks touching image borders.
#
# False positives are expected and acceptable.

DEFAULT_CONFIG: Dict[str, Any] = {
    "material_type": "asphalt",

    "prompts": {
        # 🔥 Primary prompt (very explicit)
        "primary": "a dark thin crack line on asphalt pavement",

        # 🔥 Massive multi-prompt list to maximize recall
        "secondary": [
            # --- generic crack ---
            "crack on surface",
            "surface crack",
            "visible crack",
            "structural crack",
            "hairline crack",

            # --- single / dominant crack ---
            "single long crack",
            "one long crack line",
            "isolated crack",
            "dominant crack",

            # --- thin / linear ---
            "thin crack line",
            "narrow crack",
            "linear crack",
            "long linear crack",
            "vertical crack line",
            "straight crack line",

            # --- dark line like ---
            "dark crack",
            "black crack line",
            "dark line on surface",
            "black line defect",

            # --- pavement / road ---
            "crack on asphalt road",
            "crack on pavement",
            "road surface crack",
            "asphalt surface fracture",

            # --- concrete variants ---
            "crack on concrete",
            "concrete surface crack",
            "concrete pavement crack",

            # --- fallback language ---
            "surface defect line",
            "surface fracture line",
            "linear surface damage",
        ],

        "adaptive": True,

        # ✅ only one detection needed
        "min_detections_for_keep": 1,

        "damage": {
            "enabled": False,
            "mode": "one_pass",
            "one_pass": "surface damage on {material} surface",
            "types": [
                "crack",
                "spalling",
                "peeling",
                "surface deterioration",
                "stain",
            ],
        },
    },

    "threshold": {
        # 🔥 EXTREMELY LOW thresholds
        "base": 0.03,
        "low": 0.01,

        "use_dynamic": True,
        "quantile": 0.35,
        "dynamic_cap": 0.20,
        "retry_decay": 0.9,
        "min_boxes_for_quantile": 1,

        # merge aggressively
        "iou_threshold": 0.25,

        # 🔥 almost no filtering
        "aspect_ratio_min": 1.0,
        "min_area_ratio": 0.0000005,

        "max_area_ratio": 0.98,
        "object_max_area_ratio": 0.98,
        "square_area_ratio": 0.98,

        # 🔥 DO NOT drop border boxes
        "enable_border_drop": False,
        "border_area_ratio": 0.98,

        "debug_overlays": True,
    },

    "preprocess": {
        "enabled": True,
        "noise_filter": "bilateral",
        "clahe": True,
        "target_size": 1280,

        # strong enhancement for line-like structures
        "highpass": True,
        "gabor": False,

        "clahe_clip_limit": 3.0,
        "clahe_tile_grid_size": (8, 8),

        "variants": {
            "base": {
                "enabled": True,
                "denoise": "bilateral",
                "bilateral_d": 9,
                "bilateral_sigma_color": 80.0,
                "bilateral_sigma_space": 80.0,
                "guided_radius": 8,
                "guided_eps": 0.001,
            },

            # unsharp to make crack pop
            "blur_boost": {
                "enabled": True,
                "unsharp_sigma": 1.0,
                "unsharp_amount": 0.9,
            },

            # 🔥 ridge / blackhat for seed-first SAM
            "ridge": {
                "enabled": True,
                "ridge_method": "blackhat",
                "blackhat_kernel": 17,
                "frangi_sigmas": [0.8, 1.2, 2.0, 3.0],
                "frangi_beta1": 0.5,
                "frangi_beta2": 15.0,
                "frangi_ridge_type": "dark",
            },
        },
    },

    "postprocess": {
        # 🔥 keep everything
        "min_region_area": 3,

        "morph_open": 0,
        "morph_close": 2,
        "dilate_iters": 2,

        "edge_refine": True,
        "canny_low": 15,
        "canny_high": 60,
        "fallback_canny_low": 10,
        "fallback_canny_high": 50,
    },

    "sahi": {
        # 🔥 always tile
        "enabled": True,
        "tile_size": 320,
        "overlap": 0.40,
        "min_image_size": 0,
    },

    "sam": {
        "use_points": True,
        "dilate_radius": 1,
        "variant": "sam2-large",

        # if supported in code:
        # "pos_points": 40,
        # "neg_points": 40,
    },

    "pipeline": {
        "recall_first": True,
        "enable_edge_fallback": True,

        # if exists in code:
        # "force_edge_fallback": True,
    },

    # 🔥 COMPLETELY DISABLED
    "geometry_filter": {
        "enabled": False,
        "rules": {
            "min_region_area": 1,
            "min_skeleton_len": 1,
            "min_len_loop": 999,
            "blob_area_ratio_max": 1.0,
            "len_area_min": 0.0,
            "t_border": 1.0,
            "t_ori_var_low": 0.0,
            "t_width_var_low": 0.0,
            "t_width_mean_high": 999.0,
            "thin_width_max": 999.0,
            "len_area_keep_min": 0.0,
            "t_curv_var_low": 0.0,
            "border_sides_large_area": 4,
            "border_band_frac": 0.01,
        },
    },

    "debug": {
        "enabled": True,
        "output_dir": "debug",
        "log_csv": True,
    },
}