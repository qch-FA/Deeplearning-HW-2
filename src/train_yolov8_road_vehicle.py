from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on a Road Vehicle Images Dataset in YOLO format."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/road_vehicle.yaml"),
        help="Path to YOLO dataset yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 checkpoint to fine-tune, e.g. yolov8n.pt/yolov8s.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor.")
    parser.add_argument(
        "--optimizer",
        default="SGD",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda, 0, cpu, or omit for auto.")
    parser.add_argument("--project", default="runs/task2_train")
    parser.add_argument("--name", default="yolov8_road_vehicle")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the previous Ultralytics training run.",
    )
    return parser.parse_args()


def validate_dataset_yaml(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", ".")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    missing: list[Path] = []
    for key in ("train", "val"):
        rel = cfg.get(key)
        if not rel:
            raise ValueError(f"Dataset yaml is missing required key: {key}")
        split_path = Path(rel)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            missing.append(split_path)

    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Dataset paths do not exist. Edit configs/road_vehicle.yaml first:\n"
            f"{formatted}"
        )


def main() -> None:
    args = parse_args()
    validate_dataset_yaml(args.data)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
