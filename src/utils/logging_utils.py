from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger("crack_pipeline")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "crack_pipeline")

