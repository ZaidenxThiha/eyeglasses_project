import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.solutions import object_counter


@dataclass
class CountEvent:
    timestamp: str
    frame_idx: int
    track_id: int
    direction: str
    total_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count eyeglasses in images, videos, or streams using YOLO11 + ObjectCounter"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt",
        help="Path to trained eyeglasses model weights (best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image/video, webcam index (e.g. 0), or RTSP URL",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["image", "video", "stream", "auto"],
        help="Processing mode: image, video, stream, or auto",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker type/config (e.g. bytetrack.yaml)",
    )
    parser.add_argument(
        "--roi_type",
        type=str,
        default="line",
        choices=["line", "polygon"],
        help="Region of interest type for counting",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="ROI definition. For line: x1,y1,x2,y2. For polygon: x1,y1,x2,y2,...",
    )
    parser.add_argument(
        "--once_per_id",
        action="store_true",
        help="If set, increment count only once per unique track ID",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Show video window during processing",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save output video with overlays",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/eyeglasses_count",
        help="Output directory for logs and videos",
    )
    parser.add_argument(
        "--log_format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="Log format for count events",
    )
    return parser.parse_args()


def parse_roi(roi_str: str, roi_type: str) -> Optional[Union[List[Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]]:
    if not roi_str:
        return None
    parts = [int(p) for p in roi_str.split(",")]
    if roi_type == "line" and len(parts) == 4:
        return ( (parts[0], parts[1]), (parts[2], parts[3]) )
    if roi_type == "polygon" and len(parts) >= 6 and len(parts) % 2 == 0:
        pts = []
        for i in range(0, len(parts), 2):
            pts.append((parts[i], parts[i + 1]))
        return pts
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_events(
    events: List[CountEvent],
    out_dir: str,
    base_name: str,
    fmt: str = "csv",
) -> None:
    ensure_dir(out_dir)
    if fmt == "json":
        out_path = os.path.join(out_dir, f"{base_name}_counts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in events], f, indent=2)
    else:
        out_path = os.path.join(out_dir, f"{base_name}_counts.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame_idx", "track_id", "direction", "total_count"])
            for e in events:
                writer.writerow([e.timestamp, e.frame_idx, e.track_id, e.direction, e.total_count])


def is_image_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def line_position(point: Tuple[float, float], line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def process_image(
    model: YOLO,
    source: str,
    conf: float,
) -> int:
    results: List[Results] = model.predict(source=source, conf=conf, verbose=False)
    if not results:
        return 0
    res = results[0]
    if res.boxes is None:
        return 0
    return len(res.boxes)


def process_video_or_stream(
    model: YOLO,
    source: Union[str, int],
    conf: float,
    tracker: str,
    roi_type: str,
    roi,
    once_per_id: bool,
    view: bool,
    save_video: bool,
    out_dir: str,
    log_format: str,
) -> int:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    ensure_dir(out_dir)
    base_name = "stream" if isinstance(source, int) or str(source).startswith("rtsp") else os.path.splitext(os.path.basename(str(source)))[0]

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(out_dir, f"{base_name}_out.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    counter = object_counter.ObjectCounter()
    class_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    reg_pts = None
    if roi_type == "line" and isinstance(roi, tuple):
        reg_pts = [roi[0], roi[1]]
    elif roi_type == "polygon" and isinstance(roi, list):
        reg_pts = roi

    counter.set_args(
        view_img=view,
        reg_pts=reg_pts,
        classes_names=class_names,
        draw_tracks=True,
    )

    total_count = 0
    events: List[CountEvent] = []
    last_positions: Dict[int, float] = {}
    counted_ids: Dict[int, bool] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.track(
            source=frame,
            conf=conf,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        if not results:
            if view:
                cv2.imshow("Eyeglasses Count", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if writer is not None:
                writer.write(frame)
            continue

        res = results[0]
        frame = counter.start_counting(frame, results)

        if roi_type == "line" and isinstance(roi, tuple):
            for box, track_id_tensor in zip(res.boxes.xyxy, res.boxes.id or []):
                track_id = int(track_id_tensor)
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pos = line_position((cx, cy), roi)

                prev_pos = last_positions.get(track_id)
                last_positions[track_id] = pos

                if prev_pos is None:
                    continue

                if pos == 0 or prev_pos == 0:
                    continue

                if pos * prev_pos < 0:
                    if once_per_id and counted_ids.get(track_id):
                        continue

                    direction = "forward" if pos > 0 else "backward"
                    total_count += 1
                    counted_ids[track_id] = True
                    events.append(
                        CountEvent(
                            timestamp=datetime.utcnow().isoformat(),
                            frame_idx=frame_idx,
                            track_id=track_id,
                            direction=direction,
                            total_count=total_count,
                        )
                    )

        if view:
            cv2.imshow("Eyeglasses Count", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    if view:
        cv2.destroyAllWindows()

    save_events(events, out_dir=out_dir, base_name=base_name, fmt=log_format)
    return total_count


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    mode = args.mode
    if mode == "auto":
        if args.source.isdigit() or args.source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            mode = "stream"
        elif is_image_file(args.source):
            mode = "image"
        else:
            mode = "video"

    total_count = 0
    if mode == "image":
        count = process_image(model, args.source, args.conf)
        print(f"Eyeglasses count in image: {count}")
        total_count = count
    else:
        roi = parse_roi(args.roi, args.roi_type)
        total_count = process_video_or_stream(
            model=model,
            source=int(args.source) if mode == "stream" and args.source.isdigit() else args.source,
            conf=args.conf,
            tracker=args.tracker,
            roi_type=args.roi_type,
            roi=roi,
            once_per_id=args.once_per_id,
            view=args.view,
            save_video=args.save_video,
            out_dir=args.out_dir,
            log_format=args.log_format,
        )
        print(f"Total eyeglasses counted: {total_count}")


if __name__ == "__main__":
    main()

