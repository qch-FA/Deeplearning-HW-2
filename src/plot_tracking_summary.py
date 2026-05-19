from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tracking and line-crossing summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))

    detections = data.get("detections_by_class", {})
    unique_ids = data.get("unique_track_id_count_by_class", {})
    line_counts = data.get("line_cross_count_by_class", {})

    classes = sorted(set(detections) | set(unique_ids), key=lambda c: detections.get(c, 0), reverse=True)
    det_values = [detections.get(c, 0) for c in classes]
    id_values = [unique_ids.get(c, 0) for c in classes]

    line_classes = sorted(line_counts, key=lambda c: line_counts.get(c, 0), reverse=True)
    line_values = [line_counts.get(c, 0) for c in line_classes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.4, 1.0]})

    x = range(len(classes))
    width = 0.42
    axes[0].bar([i - width / 2 for i in x], det_values, width=width, label="Per-frame detections")
    axes[0].bar([i + width / 2 for i in x], id_values, width=width, label="Unique track IDs")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(classes, rotation=35, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tracking Summary")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    if line_classes:
        axes[1].bar(line_classes, line_values, color="#e67e22")
        axes[1].set_title(f"Line Crossings: {data.get('line_cross_count', 0)}")
        axes[1].set_ylabel("Count")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No line crossings", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
