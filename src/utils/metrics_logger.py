from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from src.pipeline.models import BoxResult, CrackMetrics


class MetricsLogger:
    def __init__(self, output_csv: str) -> None:
        self.output_csv = Path(output_csv)
        self.dropped_csv = self.output_csv.with_name("dropped_regions.csv")
        if not self.output_csv.exists():
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.output_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "num_regions", "area_ratio", "avg_conf", "final_conf", "time_ms", "fallback_used", "prompts"])

        if not self.dropped_csv.exists():
            self.dropped_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.dropped_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["region_id", "dropped_reason"])

    def log(self, image_name: str, metrics: CrackMetrics, boxes: List[BoxResult], fallback_used: bool, prompts: List[str]) -> None:
        avg_conf = metrics.avg_confidence if metrics.avg_confidence is not None else ""
        final_conf = metrics.final_conf if getattr(metrics, "final_conf", None) is not None else ""
        time_ms = metrics.processing_time_ms if metrics.processing_time_ms is not None else ""
        with self.output_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    image_name,
                    metrics.num_regions,
                    f"{metrics.crack_area_ratio:.6f}",
                    avg_conf,
                    final_conf,
                    time_ms,
                    int(fallback_used),
                    "|".join(prompts),
                ],
            )

    def log_dropped_regions(self, image_name: str, dropped: List[Tuple[int, str]]) -> None:
        if not dropped:
            return
        with self.dropped_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            for region_id, reason in dropped:
                writer.writerow([f"{image_name}#{region_id}", reason])

