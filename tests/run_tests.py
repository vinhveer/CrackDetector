import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    print("Missing dependency: cv2")
    print("Install dependencies first, e.g.:")
    print("  python3 -m pip install -r requirements.txt")
    raise SystemExit(1)

from src.pipeline.pipeline import CrackDetectionPipeline
from tests.synthetic_dataset import build_synthetic_dataset


def _binary_iou(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np

    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter / union) if union > 0 else 0.0


def _precision_recall(pred: "np.ndarray", gt: "np.ndarray") -> tuple[float, float]:
    import numpy as np

    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    tp = int((p & g).sum())
    fp = int((p & (1 - g)).sum())
    fn = int(((1 - p) & g).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall

def main() -> None:
    pipe = CrackDetectionPipeline()
    base = ROOT / "tests" / "data"
    samples = build_synthetic_dataset(base)

    out_csv = base / "benchmark_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "num_regions",
                "area_ratio",
                "avg_conf",
                "final_conf",
                "time_ms",
                "fallback_used",
                "iou",
                "precision",
                "recall",
            ],
        )

        for s in samples:
            result = pipe.run(str(s.image_path))
            pred = result.images.get("geometry_filtered_mask")
            if pred is None:
                raise RuntimeError("Missing geometry_filtered_mask in pipeline result")
            iou = _binary_iou(pred, s.gt_mask)
            precision, recall = _precision_recall(pred, s.gt_mask)

            print(f"=== {s.name} ===")
            print("num_regions_before:", result.metrics.get("num_regions_before"))
            print("num_regions_after:", result.metrics.get("num_regions_after"))
            print("avg_width:", result.metrics.get("avg_width"))
            print("avg_length:", result.metrics.get("avg_length"))
            print("final_confidence:", result.metrics.get("final_confidence"))
            print("fallback_used:", result.metrics.get("fallback_used"))
            print("iou:", f"{iou:.4f}")
            print("precision:", f"{precision:.4f}")
            print("recall:", f"{recall:.4f}")

            writer.writerow(
                [
                    s.name,
                    result.metrics.get("num_regions_after"),
                    "",
                    "",
                    result.metrics.get("final_confidence") if result.metrics.get("final_confidence") is not None else "",
                    "",
                    int(bool(result.metrics.get("fallback_used"))),
                    f"{iou:.6f}",
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                ],
            )


if __name__ == "__main__":
    main()

