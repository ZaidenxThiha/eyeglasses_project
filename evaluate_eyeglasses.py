import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate eyeglasses YOLO model (mAP, precision, recall)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt",
        help="Path to trained eyeglasses model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="eyeglasses_project/data/eyeglasses.yaml",
        help="Path to data.yaml for eyeglasses dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device, e.g. 'cpu', '0'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
    )

    print("Evaluation metrics for eyeglasses detection:")
    print(f"  mAP50:   {metrics.box.map50:.4f}")
    print(f"  mAP50-95:{metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.mp:.4f}")
    print(f"  Recall:  {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

