import argparse

from src.pipeline.pipeline import CrackDetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crack detection pipeline")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--config", help="Path to YAML/JSON config", default=None)
    parser.add_argument("--ui", action="store_true", help="Launch UI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ui or args.image is None:
        from src.ui.app import main as ui_main

        ui_main()
        return

    pipeline = CrackDetectionPipeline(args.config)
    result = pipeline.run(args.image)
    print("finished:", args.image)
    print("num_regions_before:", result.metrics.get("num_regions_before"))
    print("num_regions_after:", result.metrics.get("num_regions_after"))
    print("avg_width:", result.metrics.get("avg_width"))
    print("avg_length:", result.metrics.get("avg_length"))
    print("final_confidence:", result.metrics.get("final_confidence"))
    print("fallback_used:", result.metrics.get("fallback_used"))


if __name__ == "__main__":
    main()

