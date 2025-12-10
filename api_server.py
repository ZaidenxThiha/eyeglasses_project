import io
import os
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .count_eyeglasses import (
    CountEvent,
    parse_roi,
    process_video_or_stream,
)


DEFAULT_WEIGHTS = os.environ.get(
    "EYEGLASSES_WEIGHTS",
    "runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt",
)

app = FastAPI(title="Eyeglasses Detection and Counting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(DEFAULT_WEIGHTS)


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class DetectResponse(BaseModel):
    count: int
    boxes: List[BBox]


class CountRequest(BaseModel):
    source: str
    mode: str = "auto"
    conf: float = 0.25
    tracker: str = "bytetrack.yaml"
    roi_type: str = "line"
    roi: str = ""
    once_per_id: bool = True
    log_format: str = "csv"


class CountResponse(BaseModel):
    total_count: int
    log_path: Optional[str] = None


def results_to_bboxes(results: Results) -> List[BBox]:
    boxes = []
    names = results.names
    if results.boxes is None:
        return boxes
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else 0
        track_id = int(box.id[0]) if box.id is not None else None
        boxes.append(
            BBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=conf,
                class_id=cls_id,
                class_name=names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                track_id=track_id,
            )
        )
    return boxes


@app.post("/detect", response_model=DetectResponse)
async def detect_eyeglasses(file: UploadFile = File(...), conf: float = 0.25) -> DetectResponse:
    data = await file.read()
    image_stream = io.BytesIO(data)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results_list = model.predict(source=img, conf=conf, verbose=False)
    if not results_list:
        return DetectResponse(count=0, boxes=[])
    results = results_list[0]
    boxes = results_to_bboxes(results)
    return DetectResponse(count=len(boxes), boxes=boxes)


@app.post("/count", response_model=CountResponse)
async def count_eyeglasses_endpoint(req: CountRequest) -> CountResponse:
    roi = parse_roi(req.roi, req.roi_type)
    out_dir = "runs/eyeglasses_api"
    os.makedirs(out_dir, exist_ok=True)

    total_count = process_video_or_stream(
        model=model,
        source=req.source,
        conf=req.conf,
        tracker=req.tracker,
        roi_type=req.roi_type,
        roi=roi,
        once_per_id=req.once_per_id,
        view=False,
        save_video=False,
        out_dir=out_dir,
        log_format=req.log_format,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"{ts}_counts.{req.log_format}")

    return CountResponse(total_count=total_count, log_path=log_path)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "weights": DEFAULT_WEIGHTS}

