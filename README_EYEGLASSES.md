# Eyeglasses Detection and Counting – Quick Guide

This file only shows how to **train** the eyeglasses model and **run** it.

Assumption: your Roboflow glasses dataset `data.yaml` is at:
- `eyeglasses_project/data/data.yaml`

## 1. Install dependencies

```bash
pip install ultralytics fastapi uvicorn opencv-python
```

## 2. Train the eyeglasses model

From the repository root:

```bash
python eyeglasses_project/train_eyeglasses.py \
  --model yolo11n.pt \
  --data eyeglasses_project/data/data.yaml
```

This will create:

```text
runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt
```

Use `best.pt` for all inference and counting.

## 3. Run counting on images

Count eyeglasses in a single image:

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source path/to/image.jpg \
  --mode image
```

## 4. Run counting on videos / streams

### 4.1. Video file with line-based counting

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source path/to/video.mp4 \
  --mode video \
  --roi_type line \
  --roi 100,200,500,200 \
  --once_per_id \
  --view \
  --save_video
```

- `--roi` is the line (x1,y1,x2,y2) where counts are incremented when eyeglasses cross it.
- Results (logs + annotated video) go to `runs/eyeglasses_count/`.

### 4.2. Webcam / RTSP stream

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source 0 \
  --mode stream \
  --roi_type line \
  --roi 100,200,500,200 \
  --once_per_id
```

Replace `0` with an RTSP URL if needed.

## 5. (Optional) Run API server

If you want a simple HTTP API:

```bash
uvicorn eyeglasses_project.api_server:app --host 0.0.0.0 --port 8000
```

Then you can:

- `POST /detect` with an image file to get boxes + count.
- `POST /count` with a JSON body to run video/stream counting.
