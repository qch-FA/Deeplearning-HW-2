from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

# Windows conda environments can load more than one OpenMP runtime when
# torch, OpenCV, SciPy/lap, and Ultralytics are imported together.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 detection + multi-object tracking on a video segment."
    )
    parser.add_argument("--source", type=Path, default=Path("traffic.mp4"))
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLOv8 weights. Use your trained best.pt here after training.",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_VEHICLE_CLASSES),
        help=(
            "Class names to keep. Default keeps COCO vehicle classes. "
            "Pass no values after --classes to keep all classes."
        ),
    )
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--line",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help=(
            "Virtual counting line in processed/output pixel coordinates. "
            "If omitted, --line-ratio is used."
        ),
    )
    parser.add_argument(
        "--line-ratio",
        nargs=4,
        type=float,
        default=[0.0, 0.30, 1.0, 0.30],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalized counting line coordinates used when --line is omitted.",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=1280,
        help="Resize video frames to this width before inference/output. Use 0 for original size.",
    )
    parser.add_argument("--trail", type=int, default=24, help="Trajectory length per ID.")
    parser.add_argument("--sample-frames", type=int, default=4)
    parser.add_argument(
        "--focus-start",
        type=float,
        default=None,
        help="Save consecutive focus frames starting at this source-video timestamp.",
    )
    parser.add_argument("--focus-frames", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/task2"))
    parser.add_argument("--name", default="traffic_20s")
    parser.add_argument("--device", default=None, help="cuda, 0, cpu, or omit for auto.")
    parser.add_argument("--show", action="store_true", help="Display frames while running.")
    return parser.parse_args()


def normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def resolve_class_filter(
    model_names: dict[int, str], requested: Iterable[str] | None
) -> list[int] | None:
    requested_list = list(requested or [])
    if not requested_list:
        return None

    lower_to_ids: dict[str, list[int]] = defaultdict(list)
    for class_id, name in model_names.items():
        lower_to_ids[name.lower()].append(class_id)

    resolved: list[int] = []
    missing: list[str] = []
    for item in requested_list:
        item_lower = item.lower()
        if item_lower.isdigit():
            resolved.append(int(item_lower))
        elif item_lower in lower_to_ids:
            resolved.extend(lower_to_ids[item_lower])
        else:
            missing.append(item)

    if missing:
        available = ", ".join(model_names[i] for i in sorted(model_names))
        raise ValueError(
            f"Unknown class name(s): {', '.join(missing)}. Available classes: {available}"
        )
    return sorted(set(resolved))


def choose_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_size = (target_width, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (180, 180, 180)
    # Deterministic bright colors in BGR.
    rng = np.random.default_rng(track_id + 2024)
    color = rng.integers(64, 256, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(
    frame: np.ndarray,
    x1: int,
    y1: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, frame.shape[1] / 2200)
    thickness = max(1, int(round(frame.shape[1] / 900)))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y_text = max(th + baseline + 4, y1)
    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline - 6),
        (x1 + tw + 8, y_text + 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def build_counting_line(
    output_width: int, output_height: int, line: list[float] | None, line_ratio: list[float]
) -> tuple[int, int, int, int]:
    if line is not None:
        x1, y1, x2, y2 = line
    else:
        rx1, ry1, rx2, ry2 = line_ratio
        x1, y1 = rx1 * output_width, ry1 * output_height
        x2, y2 = rx2 * output_width, ry2 * output_height
    return tuple(int(round(v)) for v in (x1, y1, x2, y2))


def signed_line_side(
    center: tuple[float, float], line: tuple[int, int, int, int], eps: float = 1.0
) -> int:
    x1, y1, x2, y2 = line
    px, py = center
    signed_area = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if signed_area > eps:
        return 1
    if signed_area < -eps:
        return -1
    return 0


def draw_counting_line(
    frame: np.ndarray,
    line: tuple[int, int, int, int],
    crossing_count: int,
) -> None:
    x1, y1, x2, y2 = line
    color = (40, 220, 255)
    thickness = max(2, int(round(frame.shape[1] / 500)))
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label = f"Line crossings: {crossing_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.65, frame.shape[1] / 1800)
    text_thickness = max(2, int(round(frame.shape[1] / 900)))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad = 8
    box_x1, box_y1 = 16, 16
    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x1 + tw + 2 * pad, box_y1 + th + baseline + 2 * pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (box_x1 + pad, box_y1 + pad + th),
        font,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_tracks(
    frame: np.ndarray,
    boxes,
    class_names: dict[int, str],
    histories: dict[int, deque[tuple[int, int]]],
    trail_len: int,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    if boxes is None or len(boxes) == 0:
        return rows

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        class_id = int(cls[i])
        class_name = class_names.get(class_id, str(class_id))
        confidence = float(conf[i])
        track_id = int(ids[i]) if ids is not None else -1

        ix1, iy1, ix2, iy2 = map(lambda v: int(round(float(v))), (x1, y1, x2, y2))
        color = color_for_id(track_id)
        thickness = max(2, int(round(frame.shape[1] / 640)))
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, thickness)
        draw_label(frame, ix1, iy1, f"ID {track_id} {class_name} {confidence:.2f}", color)

        cx = int(round((ix1 + ix2) / 2))
        cy = int(round((iy1 + iy2) / 2))
        cv2.circle(frame, (cx, cy), max(3, thickness), color, -1, cv2.LINE_AA)
        if track_id >= 0:
            histories[track_id].append((cx, cy))
            if len(histories[track_id]) > trail_len:
                histories[track_id].popleft()
            if len(histories[track_id]) >= 2:
                pts = np.array(histories[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, max(1, thickness - 1), cv2.LINE_AA)

        rows.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
            }
        )
    return rows


def make_sample_indices(total_frames: int, sample_count: int) -> set[int]:
    if total_frames <= 0 or sample_count <= 0:
        return set()
    if sample_count == 1:
        return {0}
    return {
        min(total_frames - 1, int(round(i * (total_frames - 1) / (sample_count - 1))))
        for i in range(sample_count)
    }


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Video not found: {args.source}")

    run_dir = args.project / args.name
    frames_dir = run_dir / "sample_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "tracks.csv"
    summary_path = run_dir / "summary.json"
    output_video_path = run_dir / "tracked.mp4"

    device = choose_device(args.device)
    model = YOLO(args.weights)
    class_names = normalize_names(model.names)
    class_filter = resolve_class_filter(class_names, args.classes)

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        raise RuntimeError("Could not read video FPS.")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(math.floor(args.start * fps)))
    max_frames = max(1, int(round(args.seconds * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at start={args.start}s.")

    first_frame = resize_frame(first_frame, args.process_width)
    output_height, output_width = first_frame.shape[:2]
    counting_line = build_counting_line(output_width, output_height, args.line, args.line_ratio)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "line_side",
        "crossed_line",
    ]

    histories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    last_line_side_by_id: dict[int, int] = {}
    crossed_track_ids: set[int] = set()
    crossed_track_classes: dict[int, str] = {}
    line_cross_count_by_class: Counter[str] = Counter()
    unique_ids_by_class: dict[str, set[int]] = defaultdict(set)
    classes_by_track_id: dict[int, Counter[str]] = defaultdict(Counter)
    track_frame_ranges: dict[int, list[int]] = {}
    detections_by_class: Counter[str] = Counter()
    frames_with_tracks = 0
    sample_indices = make_sample_indices(max_frames, args.sample_frames)
    focus_indices: set[int] = set()
    focus_dir = run_dir / "occlusion_frames"
    if args.focus_start is not None and args.focus_frames > 0:
        focus_start_frame = max(0, int(round(args.focus_start * fps)) - start_frame)
        focus_indices = set(range(focus_start_frame, focus_start_frame + args.focus_frames))
        focus_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    print(
        f"Tracking {args.source} from {args.start:.2f}s for {args.seconds:.2f}s "
        f"({max_frames} frames @ {fps:.3f} FPS) on {device}."
    )
    print(
        f"Source: {source_width}x{source_height}; output/process: "
        f"{output_width}x{output_height}; classes: "
        f"{'all' if class_filter is None else [class_names[i] for i in class_filter]}"
    )
    print(f"Counting line in output pixels: {counting_line}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()

        current_frame = first_frame
        processed = 0
        while processed < max_frames:
            if processed > 0:
                ok, frame = cap.read()
                if not ok:
                    break
                current_frame = resize_frame(frame, args.process_width)

            frame_index = start_frame + processed
            timestamp = frame_index / fps
            result = model.track(
                current_frame,
                persist=True,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=class_filter,
                device=device,
                verbose=False,
            )[0]

            annotated = current_frame.copy()
            rows = draw_tracks(annotated, result.boxes, class_names, histories, args.trail)
            if rows:
                frames_with_tracks += 1
            for row in rows:
                track_id = int(row["track_id"])
                side = signed_line_side(
                    (float(row["center_x"]), float(row["center_y"])), counting_line
                )
                crossed_line = 0
                if track_id >= 0 and side != 0:
                    previous_side = last_line_side_by_id.get(track_id)
                    if (
                        previous_side is not None
                        and previous_side != side
                        and track_id not in crossed_track_ids
                    ):
                        crossed_line = 1
                        crossed_track_ids.add(track_id)
                        class_name_for_count = str(row["class_name"])
                        crossed_track_classes[track_id] = class_name_for_count
                        line_cross_count_by_class[class_name_for_count] += 1
                    last_line_side_by_id[track_id] = side
                row["line_side"] = side
                row["crossed_line"] = crossed_line
                row_with_frame = {"frame": frame_index, "timestamp": round(timestamp, 3), **row}
                writer_csv.writerow(row_with_frame)
                detections_by_class[str(row["class_name"])] += 1
                if track_id >= 0:
                    class_name = str(row["class_name"])
                    unique_ids_by_class[class_name].add(track_id)
                    classes_by_track_id[track_id][class_name] += 1
                    if track_id not in track_frame_ranges:
                        track_frame_ranges[track_id] = [frame_index, frame_index]
                    else:
                        track_frame_ranges[track_id][1] = frame_index

            draw_counting_line(annotated, counting_line, len(crossed_track_ids))
            writer.write(annotated)
            if processed in sample_indices:
                frame_path = frames_dir / f"frame_{frame_index:06d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_path), annotated)
            if processed in focus_indices:
                frame_path = focus_dir / f"frame_{frame_index:06d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_path), annotated)

            if args.show:
                cv2.imshow("YOLOv8 tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if processed % 30 == 0 or processed == max_frames:
                elapsed = time.perf_counter() - started_at
                speed = processed / elapsed if elapsed else 0
                print(f"Processed {processed}/{max_frames} frames ({speed:.2f} FPS).")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    unique_ids_summary = {
        class_name: sorted(ids) for class_name, ids in sorted(unique_ids_by_class.items())
    }
    total_unique_ids = sorted({track_id for ids in unique_ids_by_class.values() for track_id in ids})
    classes_by_track_id_summary = {
        str(track_id): dict(class_counter)
        for track_id, class_counter in sorted(classes_by_track_id.items())
    }
    track_ids_with_multiple_classes = {
        track_id: class_counts
        for track_id, class_counts in classes_by_track_id_summary.items()
        if len(class_counts) > 1
    }
    summary = {
        "source": str(args.source),
        "weights": str(args.weights),
        "tracker": args.tracker,
        "device": device,
        "start_seconds": args.start,
        "duration_seconds_requested": args.seconds,
        "frames_processed": processed,
        "source_fps": fps,
        "source_resolution": [source_width, source_height],
        "output_resolution": [output_width, output_height],
        "classes": "all" if class_filter is None else [class_names[i] for i in class_filter],
        "frames_with_tracks": frames_with_tracks,
        "counting_line_output_pixels": list(counting_line),
        "line_cross_count": len(crossed_track_ids),
        "line_cross_count_by_class": dict(sorted(line_cross_count_by_class.items())),
        "line_crossed_track_ids": sorted(crossed_track_ids),
        "line_crossed_track_classes": {
            str(track_id): crossed_track_classes[track_id]
            for track_id in sorted(crossed_track_classes)
        },
        "detections_by_class": dict(sorted(detections_by_class.items())),
        "unique_track_ids_by_class": unique_ids_summary,
        "unique_track_id_count_by_class": {
            class_name: len(ids) for class_name, ids in unique_ids_summary.items()
        },
        "classes_by_track_id": classes_by_track_id_summary,
        "track_ids_with_multiple_classes": track_ids_with_multiple_classes,
        "track_frame_ranges": {
            str(track_id): {"first_frame": bounds[0], "last_frame": bounds[1]}
            for track_id, bounds in sorted(track_frame_ranges.items())
        },
        "total_unique_track_ids": total_unique_ids,
        "total_unique_track_id_count": len(total_unique_ids),
        "outputs": {
            "video": str(output_video_path),
            "tracks_csv": str(csv_path),
            "summary_json": str(summary_path),
            "sample_frames": str(frames_dir),
            "occlusion_frames": str(focus_dir) if focus_indices else None,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Output video: {output_video_path}")
    print(f"Tracks CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(
        "Unique IDs by class: "
        + json.dumps(summary["unique_track_id_count_by_class"], ensure_ascii=False)
    )
    print(f"Line crossings: {summary['line_cross_count']}")


if __name__ == "__main__":
    main()
