from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_COLUMNS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]

METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot YOLO training curves and optionally upload them to wandb."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to Ultralytics results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/figures"),
        help="Directory for output PNG files.",
    )
    parser.add_argument("--prefix", default="wandb")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Training results not found: {path}")

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                clean_key = key.strip()
                clean_value = (value or "").strip()
                if not clean_key or not clean_value:
                    continue
                try:
                    row[clean_key] = float(clean_value)
                except ValueError:
                    pass
            if row:
                rows.append(row)
    return rows


def values(rows: list[dict[str, float]], column: str) -> list[float]:
    return [row[column] for row in rows if column in row and math.isfinite(row[column])]


def epochs(rows: list[dict[str, float]], column: str) -> list[int]:
    return [int(row.get("epoch", index + 1)) for index, row in enumerate(rows) if column in row]


def plot_losses(rows: list[dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharex=True)
    pairs = [
        ("box", "train/box_loss", "val/box_loss"),
        ("classification", "train/cls_loss", "val/cls_loss"),
        ("DFL", "train/dfl_loss", "val/dfl_loss"),
    ]
    for ax, (title, train_col, val_col) in zip(axes, pairs):
        if train_col in rows[0]:
            ax.plot(epochs(rows, train_col), values(rows, train_col), marker="o", label="train")
        if val_col in rows[0]:
            ax.plot(epochs(rows, val_col), values(rows, val_col), marker="o", label="validation")
        ax.set_title(f"{title} loss")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("loss")
    fig.suptitle("Training and Validation Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95",
    }
    for column in METRIC_COLUMNS:
        if column in rows[0]:
            ax.plot(epochs(rows, column), values(rows, column), marker="o", label=labels[column])
    ax.set_title("Validation Detection Metrics")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def upload_to_wandb(
    rows: list[dict[str, float]],
    project: str,
    run_name: str | None,
    loss_path: Path,
    metric_path: Path,
    results_path: Path,
) -> None:
    import wandb

    run = wandb.init(project=project, name=run_name, job_type="curve-export")
    try:
        for row in rows:
            log_row = {
                key: value
                for key, value in row.items()
                if key in LOSS_COLUMNS or key in METRIC_COLUMNS
            }
            if "epoch" in row:
                log_row["epoch"] = int(row["epoch"])
            if log_row:
                wandb.log(log_row, step=int(row.get("epoch", len(log_row))))
        wandb.log(
            {
                "loss_curves": wandb.Image(str(loss_path)),
                "metric_curves": wandb.Image(str(metric_path)),
            }
        )
        artifact = wandb.Artifact(
            name=f"{run.name or 'training'}-curves", type="training-results"
        )
        artifact.add_file(str(results_path))
        artifact.add_file(str(loss_path))
        artifact.add_file(str(metric_path))
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    rows = read_results(args.results)
    if not rows:
        raise RuntimeError(f"No rows found in {args.results}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = args.output_dir / f"{args.prefix}_loss_curves.png"
    metric_path = args.output_dir / f"{args.prefix}_metric_curves.png"
    plot_losses(rows, loss_path)
    plot_metrics(rows, metric_path)

    if args.wandb_project:
        upload_to_wandb(
            rows,
            args.wandb_project,
            args.wandb_name,
            loss_path,
            metric_path,
            args.results,
        )

    print(f"Wrote {loss_path}")
    print(f"Wrote {metric_path}")


if __name__ == "__main__":
    main()
