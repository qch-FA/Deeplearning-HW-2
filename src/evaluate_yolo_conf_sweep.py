from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO checkpoint at multiple confidence thresholds."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--conf",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/detect/hw2_task2_full_gpu_val")
    parser.add_argument("--name-prefix", default="val_conf")
    parser.add_argument("--output", type=Path, default=Path("runs/detect/hw2_task2_full_gpu_val/conf_sweep.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    rows: list[dict[str, float]] = []
    for conf in args.conf:
        result = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=conf,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=args.project,
            name=f"{args.name_prefix}_{int(conf * 100):03d}",
            exist_ok=True,
        )
        box = result.box
        row = {
            "conf": conf,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        }
        rows.append(row)
        print(
            "conf={conf:.2f} precision={precision:.4f} recall={recall:.4f} "
            "mAP50={map50:.4f} mAP50-95={map50_95:.4f}".format(**row),
            flush=True,
        )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf", "precision", "recall", "map50", "map50_95"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
