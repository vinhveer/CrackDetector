import argparse

from src.pipeline.pipeline import CrackDetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crack detection pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--config", help="Path to YAML/JSON config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = CrackDetectionPipeline(args.config)
    result = pipeline.run(args.image)
    print("finished:", args.image)
    print("num_regions:", result.metrics.num_regions)
    print("area_ratio:", result.metrics.crack_area_ratio)
    print("avg_conf:", result.metrics.avg_confidence)
    print("time_ms:", result.metrics.processing_time_ms)
    print("fallback_used:", result.fallback_used)


if __name__ == "__main__":
    main()

