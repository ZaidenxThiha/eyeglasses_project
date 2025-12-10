# Eyeglasses Detection and Counting with Ultralytics YOLO

This project is an end-to-end pipeline to **detect and count eyeglasses** in images and video streams using **Ultralytics YOLO11**.

It includes:

- Custom dataset configuration for eyeglasses.
- Training scripts for YOLO11.
- Inference and counting (images, videos, RTSP) using `ObjectCounter` and custom logic.
- FastAPI inference/counting API.
- Docker-based deployment (CPU/GPU/Jetson).
- Evaluation utilities (mAP, precision, recall).

The code assumes **Python 3.10+**.

---

## 1. Dataset Setup

### 1.1. Recommended Folder Structure

Create a dataset root, for example:

```text
datasets/
  eyeglasses/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Each image in `images/*` should have a corresponding YOLO TXT label file in `labels/*` with the same base filename.

Example:

- `images/train/img_001.jpg`
- `labels/train/img_001.txt`

`eyeglasses_project/data/eyeglasses.yaml` assumes the dataset root is:

```text
datasets/eyeglasses/
```

You can move or modify this path as needed.

### 1.2. Data Configuration File

Config file: `eyeglasses_project/data/eyeglasses.yaml`

- `path`: dataset root
- `train`, `val`, `test`: relative paths to images
- `names`: list of class names (single class: `eyeglasses`)

### 1.3. Collecting and Labeling Data

You can collect images from:

- Public datasets (face images with eyeglasses).
- Your own photos / surveillance streams.
- Web-crawled data (respecting licenses and privacy).

Then label them using tools such as:

- **Roboflow**
  - Upload your images.
  - Create an object detection project.
  - Label eyeglasses as a single class: `eyeglasses`.
  - Export in **YOLO** format.
  - Download and place the exported `train/val/test` images and labels into the folder structure above.

- **CVAT**
  - Create a task and upload images.
  - Use polygon/box annotations for eyeglasses.
  - Export in **YOLO** format.
  - Organize the exported data into `images/*` and `labels/*`.

### 1.4. Recommended Dataset Size

For a single class like eyeglasses:

- **Minimum**: ~500–1,000 labeled images.
- **Good**: 2,000–5,000 images.
- **Robust**: 5,000+ images with diverse lighting, poses, and backgrounds.

Ensure variation in:

- Face orientation (frontal, profile, tilted).
- Eyeglass types (thin frames, thick frames, rimless).
- Distances (small objects in the frame).
- Backgrounds (indoor/outdoor, different environments).

### 1.5. Augmentation Strategy (Recommended)

For small objects like eyeglasses, use:

- **Geometric**
  - Random scaling (slightly zoom in/out).
  - Random rotation (small angles to avoid label distortion).
  - Random horizontal flip.
- **Photometric**
  - Random brightness/contrast.
  - Color jitter (hue, saturation, value).
  - Gaussian noise.
- **Degradation**
  - Motion blur / Gaussian blur (to simulate motion).
  - Low-light simulation.
- **Occlusion**
  - Random cutout/patches on the face region to simulate occlusions.

Ultralytics YOLO already includes built-in augmentations; we only tweak their intensity via training hyperparameters where needed (see training section).

### 1.6. Using the Roboflow *Glasses* Dataset

You can use the Roboflow dataset:

- URL: https://universe.roboflow.com/yolo-practice-xgsmc/glasses-iy1og

Two simple options:

**Option A – Manual export**

1. Open the dataset URL above.
2. Click *Download Dataset* and choose a YOLO format (YOLOv8/YOLO11 PyTorch).
3. Download and unzip into `datasets/eyeglasses/` so that you get:

```text
datasets/eyeglasses/
  images/train ...
  images/val ...
  images/test ...
  labels/train ...
  labels/val ...
  labels/test ...
```

If the export includes its own `data.yaml`, you can either:

- Use it directly with `--data path/to/data.yaml`, or
- Keep using `eyeglasses_project/data/eyeglasses.yaml` if the folder structure matches the layout above.

**Option B – Python download via Roboflow**

```bash
pip install roboflow
```

Then in a small helper script or notebook:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("yolo-practice-xgsmc").project("glasses-iy1og")
dataset = project.version(1).download("yolov11")  # or "yolov8"

print("Dataset at:", dataset.location)  # use this path with --data
```

You can then:

- Point training directly to Roboflow’s `data.yaml`:
  - `python eyeglasses_project/train_eyeglasses.py --data path/to/roboflow/data.yaml`
- Or move/rename the downloaded folder to `datasets/eyeglasses/` so it matches `eyeglasses_project/data/eyeglasses.yaml`.

---

## 2. Model Training

### 2.1. Shell Command (Recommended)

From the repo root:

```bash
python eyeglasses_project/train_eyeglasses.py \
  --model yolo11n.pt \
  --data eyeglasses_project/data/eyeglasses.yaml \
  --epochs 100 \
  --imgsz 1280 \
  --batch 16
```

This will create a run directory similar to:

```text
runs/eyeglasses_train/yolo11_eyeglasses/
  weights/
    best.pt
    last.pt
```

`best.pt` is the model used for inference and deployment.

### 2.2. Hyperparameters for Small-Object Detection

The training script configures:

- Larger input size: `imgsz=1280` (helps small objects).
- Adequate epochs: `epochs=100` (with `patience=50`).
- Optimizer: `AdamW` with cosine learning rate.
- Data augmentations:
  - `mosaic=1.0`, `close_mosaic=10`.
  - `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`.
  - `fliplr=0.5`, `scale=0.7`.

You can adjust these in `eyeglasses_project/train_eyeglasses.py`.

---

## 3. Inference and Object Counting

### 3.1. Script Overview

Main counting script: `eyeglasses_project/count_eyeglasses.py`

It supports:

- Single image eyeglasses counting.
- Video file eyeglasses counting.
- RTSP/webcam stream counting.
- Two modes:
  - **Per-frame counting** (images or simple box count).
  - **Line-based counting** (objects counted when crossing a region).

It uses:

- Ultralytics `ObjectCounter` solution for visualization and basic counting.
- Custom logic to:
  - Count only when crossing a line region.
  - Avoid duplicate counts using tracker IDs.
  - Optionally enforce “increment only once per person” (per track ID).
  - Log all events to CSV or JSON.

### 3.2. Basic Usage

**Single image (per-frame count):**

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source path/to/image.jpg \
  --mode image \
  --conf 0.25
```

**Video file with line-based counting:**

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source path/to/video.mp4 \
  --mode video \
  --conf 0.25 \
  --tracker bytetrack.yaml \
  --roi_type line \
  --roi 100,200,500,200 \
  --once_per_id \
  --view \
  --save_video \
  --log_format csv
```

**RTSP or webcam stream:**

```bash
python eyeglasses_project/count_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --source 0 \
  --mode stream \
  --roi_type line \
  --roi 100,200,500,200 \
  --once_per_id
```

### 3.3. Configuration Parameters

- `--conf`: detection confidence threshold.
- `--tracker`: tracker configuration (e.g. `bytetrack.yaml`).
- `--roi_type`:
  - `line`: region defined by two points.
  - `polygon`: region defined by multiple points (for visualization / advanced logic).
- `--roi`:
  - Line format: `x1,y1,x2,y2`.
  - Polygon format: `x1,y1,x2,y2,x3,y3,...`.
- `--once_per_id`:
  - When enabled, each tracked person/eyeglasses pair increments the count at most once.
- `--view`: show live video with overlays.
- `--save_video`: save annotated video.
- `--out_dir`: where logs and videos are stored.
- `--log_format`: `csv` or `json`.

### 3.4. Custom Counting Logic

The script:

- Uses tracker IDs from YOLO tracking to maintain identity.
- Tracks each object’s position relative to a **line ROI**.
- Detects **crossing events** when the sign of the point-to-line distance changes between frames.
- For each crossing:
  - Increments the total count.
  - Logs an event with:
    - UTC timestamp.
    - frame index.
    - tracker ID.
    - direction (forward/backward).
    - cumulative total.
- Uses `--once_per_id` to:
  - Enforce “increment only once per person” mode.
  - Avoid duplicate counts when an object crosses back and forth.

Events are saved to:

- CSV: `*_counts.csv` (one row per event).
- JSON: `*_counts.json` (array of event objects).

---

## 4. FastAPI Inference API

### 4.1. Endpoints

API module: `eyeglasses_project/api_server.py`

Run locally:

```bash
uvicorn eyeglasses_project.api_server:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /detect`
  - Input: image file (`multipart/form-data`, field name `file`).
  - Optional query param: `conf` (confidence threshold).
  - Output:
    - `count`: number of eyeglasses detected.
    - `boxes`: list of bounding boxes with:
      - `x1, y1, x2, y2`
      - `confidence`
      - `class_id`, `class_name`
      - `track_id` (if available).

- `POST /count`
  - Input JSON:
    - `source`: path/URL or camera index.
    - `mode`: `auto` / `video` / `stream`.
    - `conf`: detection confidence.
    - `tracker`: tracker config name (e.g. `bytetrack.yaml`).
    - `roi_type`: `line` or `polygon`.
    - `roi`: ROI definition string.
    - `once_per_id`: boolean.
    - `log_format`: `csv` or `json`.
  - Output:
    - `total_count`: total eyeglasses counted.
    - `log_path`: path where the count log is stored.

- `GET /health`
  - Returns basic health info and which weights are loaded.

### 4.2. CORS Support

The API enables CORS with `allow_origins=["*"]` so that web frontends can call it directly from the browser.

---

## 5. Deployment

### 5.1. Docker (CPU)

Dockerfile: `Dockerfile` at repo root.

Build:

```bash
docker build -t eyeglasses-api .
```

Run (CPU):

```bash
docker run --rm -p 8000:8000 eyeglasses-api
```

The API will be available at `http://localhost:8000`.

### 5.2. Docker (GPU with NVIDIA)

Requires NVIDIA Container Toolkit.

Build:

```bash
docker build -t eyeglasses-api-gpu .
```

Run:

```bash
docker run --rm -p 8000:8000 --gpus all eyeglasses-api-gpu
```

Inside the container, Ultralytics will automatically use `device=0` when CUDA is available.

### 5.3. NVIDIA Jetson Example

On Jetson devices:

- Use a Jetson-compatible base image (e.g. `nvcr.io/nvidia/l4t-ml`).
- Install Python 3.10+ and Ultralytics.
- Copy this project and run:

```bash
uvicorn eyeglasses_project.api_server:app --host 0.0.0.0 --port 8000
```

For better performance:

- Use a smaller model (e.g. `yolo11n.pt`).
- Reduce `imgsz` during training and inference if needed.

### 5.4. Scaling for Multiple Camera Streams

Options:

- **Multiple processes/containers**:
  - Run one container per camera stream.
  - Use a load balancer or simple coordinator to assign streams.
- **Threaded workers in a single service**:
  - Spawn multiple worker processes, each handling a different RTSP source.
  - Ensure each worker loads its own model instance for best throughput.
- **GPU considerations**:
  - Limit per-process batch sizes.
  - Use smaller models on constrained hardware.

---

## 6. Evaluation

### 6.1. Evaluation Script

Script: `eyeglasses_project/evaluate_eyeglasses.py`

Example usage:

```bash
python eyeglasses_project/evaluate_eyeglasses.py \
  --weights runs/eyeglasses_train/yolo11_eyeglasses/weights/best.pt \
  --data eyeglasses_project/data/eyeglasses.yaml \
  --split val
```

This reports:

- `mAP50`, `mAP50-95` for eyeglasses.
- Mean precision and recall.

### 6.2. Validating Model Performance

To validate performance specifically for eyeglasses:

- Ensure the **validation/test sets** contain:
  - Diverse face poses and eyeglass types.
  - Challenging lighting (indoor, outdoor, low light).
  - Small eyeglasses far from the camera.
- Compare:
  - Precision: avoid false positives (non-eyeglass objects).
  - Recall: capture as many true eyeglasses as possible.
- Perform visual inspection:
  - Run inference on representative videos.
  - Check for:
    - Missed detections when glasses are small or partially occluded.
    - False detections on facial features (e.g. eyebrows).
- Adjust:
  - Confidence threshold (`--conf`).
  - Training hyperparameters and augmentations.
  - Dataset quality (more/cleaner labels).

---

## 7. Summary of Components

- **Dataset config**: `eyeglasses_project/data/eyeglasses.yaml`
- **Training**: `eyeglasses_project/train_eyeglasses.py`
- **Counting & logging**: `eyeglasses_project/count_eyeglasses.py`
- **API server**: `eyeglasses_project/api_server.py`
- **Evaluation**: `eyeglasses_project/evaluate_eyeglasses.py`
- **Deployment**:
  - Docker: `Dockerfile`
  - Run API: `uvicorn eyeglasses_project.api_server:app --host 0.0.0.0 --port 8000`
