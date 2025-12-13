from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.pipeline.models import BoxResult, CrackMetrics


class MetricsLogger:
    def __init__(self, output_csv: str) -> None:
        self.output_csv = Path(output_csv)
        if not self.output_csv.exists():
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.output_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "num_regions", "area_ratio", "avg_conf", "time_ms", "fallback_used", "prompts"])

    def log(self, image_name: str, metrics: CrackMetrics, boxes: List[BoxResult], fallback_used: bool, prompts: List[str]) -> None:
        avg_conf = metrics.avg_confidence if metrics.avg_confidence is not None else ""
        time_ms = metrics.processing_time_ms if metrics.processing_time_ms is not None else ""
        with self.output_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    image_name,
                    metrics.num_regions,
                    f"{metrics.crack_area_ratio:.6f}",
                    avg_conf,
                    time_ms,
                    int(fallback_used),
                    "|".join(prompts),
                ],
            )

