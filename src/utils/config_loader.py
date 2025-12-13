from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

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
    },
    "threshold": {
        "base": 0.35,
        "low": 0.2,
        "use_dynamic": True,
        "quantile": 0.8,
        "dynamic_cap": 0.9,
        "retry_decay": 0.7,
        "min_boxes_for_quantile": 3,
        "iou_threshold": 0.5,
        "aspect_ratio_min": 2.0,
        "min_area_ratio": 0.0002,
        "max_area_ratio": 0.25,
    },
    "preprocess": {
        "noise_filter": "bilateral",
        "clahe": True,
        "target_size": 1024,
        "highpass": False,
        "gabor": False,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
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
        "use_points": True,
        "dilate_radius": 0,
        "variant": "sam2-large",
    },
    "debug": {
        "enabled": True,
        "output_dir": "debug",
        "log_csv": False,
    },
}


def load_config(config: Optional[Dict[str, Any] | str] = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return cfg
    if isinstance(config, dict):
        return _deep_update(cfg, config)
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Please install PyYAML to load YAML configs.") from exc
        with config_path.open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
    else:
        with config_path.open("r") as f:
            user_cfg = json.load(f)
    return _deep_update(cfg, user_cfg or {})


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

