from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.default_config import DEFAULT_CONFIG


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

