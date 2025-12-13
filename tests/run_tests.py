from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import CrackDetectionPipeline

# Updated test set using newly added images
TEST_IMAGES = [
    ("crack1.jpg", "expect_some"),
    ("crack2.jpg", "expect_some"),
    ("crack3.jpg", "expect_some"),
]

def main() -> None:
    pipe = CrackDetectionPipeline()
    base = ROOT / "tests" / "data"
    for name, expectation in TEST_IMAGES:
        img_path = base / name
        if not img_path.exists():
            print(f"[WARN] Missing test image: {img_path}")
            continue
        result = pipe.run(str(img_path))
        print(f"=== {name} ({expectation}) ===")
        print("num_regions:", result.metrics.num_regions)
        print("area_ratio:", result.metrics.crack_area_ratio)
        print("avg_conf:", result.metrics.avg_confidence)
        print("time_ms:", result.metrics.processing_time_ms)
        print("fallback_used:", result.fallback_used)


if __name__ == "__main__":
    main()

