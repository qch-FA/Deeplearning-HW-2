from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


COCO_TO_LOCAL = {
    2: 0,  # car
    5: 1,  # bus
    7: 2,  # truck
    3: 3,  # motorcycle
    1: 4,  # bicycle
}

LOCAL_NAMES = ["car", "bus", "truck", "motorcycle", "bicycle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small YOLO-format pseudo-label dataset from traffic.mp4."
    )
    parser.add_argument("--source", type=Path, default=Path("traffic.mp4"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--output", type=Path, default=Path("datasets/traffic_pseudo_yolo"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--train", type=int, default=64, help="Number of train frames.")
    parser.add_argument("--val", type=int, default=16, help="Number of validation frames.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--process-width", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--keep-coco-ids",
        action="store_true",
        help="Keep original COCO class IDs/names instead of remapping vehicles to 0..4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output dataset directory before writing.",
    )
    return parser.parse_args()


def ensure_empty_dataset(root: Path, overwrite: bool) -> None:
    if root.exists() and overwrite:
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def frame_indices(total_frames: int, fps: float, seconds: float, count: int) -> list[int]:
    usable_frames = min(total_frames, max(1, int(round(fps * seconds))))
    if count <= 1:
        return [0]
    return sorted(
        {
            min(usable_frames - 1, round(i * (usable_frames - 1) / (count - 1)))
            for i in range(count)
        }
    )


def resize_frame(frame, process_width: int):
    height, width = frame.shape[:2]
    if process_width <= 0 or width <= process_width:
        return frame
    new_height = round(height * process_width / width)
    return cv2.resize(frame, (process_width, new_height), interpolation=cv2.INTER_AREA)


def write_yaml(root: Path, names: list[str]) -> None:
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_text = (
        "# YOLO dataset generated from traffic.mp4 with pseudo labels.\n"
        "# Labels were produced by the COCO-pretrained yolov8n.pt detector.\n"
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        "names:\n"
        f"{names_block}\n"
    )
    (root / "traffic_pseudo.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Video not found: {args.source}")

    ensure_empty_dataset(args.output, args.overwrite)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("Could not read FPS/frame count from the input video.")

    total_samples = args.train + args.val
    indices = frame_indices(total_frames, fps, args.seconds, total_samples)
    train_cut = min(args.train, len(indices))
    model = YOLO(args.weights)
    summary = {
        "source": str(args.source),
        "weights": args.weights,
        "fps": fps,
        "requested_seconds": args.seconds,
        "sampled_frames": len(indices),
        "train_images": 0,
        "val_images": 0,
        "labels_by_class": {name: 0 for name in LOCAL_NAMES},
    }

    for sample_idx, frame_idx in enumerate(indices):
        split = "train" if sample_idx < train_cut else "val"
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        frame = resize_frame(frame, args.process_width)
        image_name = f"traffic_{split}_{sample_idx:04d}.jpg"
        label_name = Path(image_name).with_suffix(".txt").name
        image_path = args.output / "images" / split / image_name
        label_path = args.output / "labels" / split / label_name
        cv2.imwrite(str(image_path), frame)

        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=list(COCO_TO_LOCAL.keys()),
            verbose=False,
        )[0]

        lines: list[str] = []
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().tolist()
            xywhn = result.boxes.xywhn.cpu().tolist()
            for coco_cls, box in zip(classes, xywhn):
                coco_cls = int(coco_cls)
                local_cls = COCO_TO_LOCAL.get(coco_cls)
                if local_cls is None:
                    continue
                label_cls = coco_cls if args.keep_coco_ids else local_cls
                x_center, y_center, width, height = box
                lines.append(
                    f"{label_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                summary["labels_by_class"][LOCAL_NAMES[local_cls]] += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        summary[f"{split}_images"] += 1

    cap.release()
    model_names = [str(model.names[i]) for i in sorted(model.names)] if args.keep_coco_ids else LOCAL_NAMES
    write_yaml(args.output, model_names)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
