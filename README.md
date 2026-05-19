# HW2 Task 2 - Scene Object Detection and Multi-Object Tracking

This folder contains a YOLOv8-based implementation for task 2:

- train a road-vehicle detector on a YOLO-format Road Vehicle Images Dataset;
- run YOLOv8 detection plus ByteTrack multi-object tracking on `traffic.mp4`;
- export an annotated video, per-frame bounding boxes with tracking IDs, sample frames, and summary statistics.

## Environment

```powershell
python -m pip install -r requirements.txt
```

## 1. Train on Road Vehicle Images Dataset

Edit `configs/road_vehicle.yaml` so `path`, split folders, and `names` match your dataset.
The public Road Vehicle Images Dataset is commonly distributed with train/valid splits and 21 classes; the YAML file is a starter template, so keep the class order identical to the dataset labels before training.
The expected label format is the standard YOLO text format:

```text
class_id x_center y_center width height
```

Run:

```powershell
wandb login
yolo settings wandb=True
python src/train_yolov8_road_vehicle.py --data configs/road_vehicle.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --lr0 0.01 --optimizer SGD
```

The trained checkpoint is usually saved under:

```text
runs/task2_train/yolov8_road_vehicle/weights/best.pt
```

## 2. Track the first 20 seconds of traffic.mp4

With the pretrained COCO YOLOv8n vehicle classes:

```powershell
python src/track_traffic.py --source traffic.mp4 --weights yolov8n.pt --seconds 20 --process-width 1280
```

With your trained road-vehicle checkpoint:

```powershell
python src/track_traffic.py --source traffic.mp4 --weights runs/task2_train/yolov8_road_vehicle/weights/best.pt --seconds 20 --process-width 1280 --classes
```

Passing `--classes` without class names keeps all classes from the custom model.

## Outputs

The default run writes to `runs/task2/traffic_20s/`:

- `tracked.mp4`: annotated video with bounding boxes, class names, confidence, and Tracking ID;
- `tracks.csv`: per-frame detection and tracking records;
- `summary.json`: detection counts and unique Tracking ID counts by class;
- `sample_frames/`: four annotated frames for report screenshots and ID analysis.

Useful summary fields:

- `detections_by_class`: total detections over all processed frames;
- `unique_track_id_count_by_class`: number of different track IDs seen per class;
- `total_unique_track_id_count`: total number of tracked object identities.

