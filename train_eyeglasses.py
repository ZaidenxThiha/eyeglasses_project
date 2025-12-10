import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 model for eyeglasses detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO11 model checkpoint to start from (e.g. yolo11n.pt, yolo11s.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="eyeglasses_project/data/eyeglasses.yaml",
        help="Path to data.yaml for eyeglasses dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size (higher for small objects)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device, e.g. 'cpu', '0', '0,1'",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/eyeglasses_train",
        help="Directory to save training runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11_eyeglasses",
        help="Run name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=50,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        scale=0.7,
        mosaic=1.0,
    )
    # Best weights will be saved as best.pt in runs/eyeglasses_train/...


if __name__ == "__main__":
    main()

