import argparse
import os
from pathlib import Path

from roboflow import Roboflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Roboflow eyeglasses dataset and print its data.yaml path"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="yolo-practice-xgsmc",
        help="Roboflow workspace slug",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="glasses-iy1og",
        help="Roboflow project slug",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version number on Roboflow",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov11",
        help="Export format (e.g. yolov11, yolov8, yolov5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Roboflow API key is required. "
            "Pass --api-key or set ROBOFLOW_API_KEY environment variable."
        )

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format)

    dataset_dir = Path(dataset.location)
    data_yaml = dataset_dir / "data.yaml"

    print(f"Dataset downloaded to: {dataset_dir}")
    if data_yaml.is_file():
        print(f"Found data.yaml at: {data_yaml}")
        print("\nYou can now train with:")
        print(
            f"  python eyeglasses_project/train_eyeglasses.py "
            f"--data \"{data_yaml}\""
        )
    else:
        print(
            "data.yaml not found automatically. Check the downloaded folder "
            "and pass the correct path via --data."
        )


if __name__ == "__main__":
    main()

