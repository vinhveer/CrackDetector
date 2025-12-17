from __future__ import annotations

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "material_type": "concrete",
    "prompts": {
        "primary": "hairline crack on concrete surface",
        "secondary": [
            "tiny surface fracture on concrete",
            "narrow linear crack on wall",
            "branching crack pattern",
        ],
        "adaptive": True,
        "min_detections_for_keep": 2,
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
        "base": 0.15,
        "low": 0.05,
        "use_dynamic": True,
        "quantile": 0.8,
        "dynamic_cap": 0.9,
        "retry_decay": 0.7,
        "min_boxes_for_quantile": 3,
        "iou_threshold": 0.5,
        "aspect_ratio_min": 2.0,
        "min_area_ratio": 0.0002,

        # 🔧 FIX: siết box to để tránh SAM segment object-level
        "max_area_ratio": 0.15,            # was 0.25
        "object_max_area_ratio": 0.12,     # was 0.20
        "square_area_ratio": 0.08,         # was 0.10 (vuông/to thường là object)

        "border_area_ratio": 0.10,         # was 0.12 (nhạy hơn với box chạm biên)
        "enable_border_drop": True,

        "debug_overlays": True,            # bật để debug nhanh boxes/filters
    },
    "preprocess": {
        "enabled": False,
        "noise_filter": "bilateral",
        "clahe": True,
        "target_size": 1024,
        "highpass": False,
        "gabor": False,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "variants": {
            "base": {
                "enabled": True,
                "denoise": "bilateral",
                "bilateral_d": 7,
                "bilateral_sigma_color": 50.0,
                "bilateral_sigma_space": 50.0,
                "guided_radius": 8,
                "guided_eps": 0.001,
            },

            # (optional) có thể bật nếu ảnh mờ nhiều
            "blur_boost": {
                "enabled": True,            # was False
                "unsharp_sigma": 1.0,
                "unsharp_amount": 0.6,
            },

            # 🔧 FIX: bật ridge để làm seed crack/damage (seed-first SAM)
            "ridge": {
                "enabled": True,            # was False
                "ridge_method": "blackhat", # keep blackhat first (ổn định)
                "blackhat_kernel": 15,      # was 9 (mạnh hơn cho crack mảnh)
                "frangi_sigmas": [1.0, 2.0, 3.0],
                "frangi_beta1": 0.5,
                "frangi_beta2": 15.0,
                "frangi_ridge_type": "dark",
            },
        },
    },
    "postprocess": {
        "min_region_area": 50,
        "morph_open": 3,
        "morph_close": 5,
        "dilate_iters": 1,
        "edge_refine": True,
        "canny_low": 50,
        "canny_high": 150,
        "fallback_canny_low": 30,
        "fallback_canny_high": 120,
    },
    "sahi": {
        "enabled": True,
        "tile_size": 512,
        "overlap": 0.2,
        "min_image_size": 1400,
    },
    "sam": {
        # giữ use_points=True, nhưng nhớ sửa code để lấy points từ ridge-seed
        "use_points": True,
        "dilate_radius": 0,
        "variant": "sam2-large",

        # (optional) nếu code support sampling counts
        # "pos_points": 20,
        # "neg_points": 20,
    },
    "pipeline": {
        "recall_first": True,
        "enable_edge_fallback": True,
    },

    # 🔧 Debug SAM trước: tắt geometry để không “drop sạch”
    "geometry_filter": {
        "enabled": False,
        "rules": {
            "min_region_area": 30,
            "min_skeleton_len": 25,
            "min_len_loop": 40,
            "blob_area_ratio_max": 0.6,
            "len_area_min": 0.02,
            "t_border": 0.30,
            "t_ori_var_low": 0.10,
            "t_width_var_low": 1.0,
            "t_width_mean_high": 10.0,
            "thin_width_max": 6.0,
            "len_area_keep_min": 0.06,
            "t_curv_var_low": 0.08,
            "border_sides_large_area": 2,
            "border_band_frac": 0.03,
        },
    },
    "debug": {
        "enabled": False,
        "output_dir": "debug",
        "log_csv": False,
    },
}
