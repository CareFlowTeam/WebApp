from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prediction outputs for review and batch inspection.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV or JSON output from predict_convnext.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                {
                    "image_path": row["image_path"],
                    "relative_path": row.get("relative_path", ""),
                    "top1_class": row["top1_class"],
                    "top1_label_text": row.get("top1_label_text", ""),
                    "top1_probability": float(row["top1_probability"]),
                }
            )
        return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.predictions)
    if not rows:
        raise ValueError("No prediction rows found.")

    class_counter = Counter(row["top1_class"] for row in rows)
    low_confidence = [row for row in rows if float(row["top1_probability"]) < args.low_confidence_threshold]
    low_confidence.sort(key=lambda row: float(row["top1_probability"]))

    top_classes = [
        {"class_name": class_name, "count": count}
        for class_name, count in class_counter.most_common(args.top_n)
    ]

    summary = {
        "predictions": str(args.predictions),
        "images": len(rows),
        "unique_top1_classes": len(class_counter),
        "low_confidence_threshold": args.low_confidence_threshold,
        "low_confidence_count": len(low_confidence),
        "top_n": args.top_n,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "top_predicted_classes.csv", ["class_name", "count"], top_classes)
    write_csv(
        args.output_dir / "low_confidence_predictions.csv",
        ["image_path", "relative_path", "top1_class", "top1_label_text", "top1_probability"],
        low_confidence[: args.top_n],
    )

    print(f"Images summarized: {len(rows)}")
    print(f"Low-confidence predictions: {len(low_confidence)}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()