# 기존의 sys.path.append 등을 모두 지우고 아래 내용으로 교체하세요
from __future__ import annotations
import argparse
import csv
import json

# 항상 scripts 패키지로 import (main.py에서 sys.path를 올바르게 설정해야 함)
from scripts.medicine_utils import is_image_file, normalize_label_text

import torch
from PIL import Image
from timm import create_model
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained ConvNeXt medicine classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--input-path", type=Path, required=True, help="Single image or folder of images")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Optional image limit for quick checks. 0 means all images.")
    return parser.parse_args()


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def collect_images(input_path: Path, max_images: int) -> list[Path]:
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    images = sorted([path for path in input_path.rglob("*") if is_image_file(path)], key=lambda path: str(path))
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise ValueError(f"No supported images found under: {input_path}")
    return images


def build_relative_path(image_path: Path, input_path: Path) -> str:
    if input_path.is_file():
        return image_path.name
    return str(image_path.relative_to(input_path))


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    classes = [None] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        classes[class_index] = class_name

    model = create_model(checkpoint["model_name"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    img_size = checkpoint.get("args", {}).get("img_size", 224)
    return model, classes, build_eval_transform(img_size)


def predict_batch(
    model,
    classes: list[str],
    image_paths: list[Path],
    input_root: Path,
    transform,
    device: torch.device,
    top_k: int,
) -> list[dict]:
    batch_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        batch_tensors.append(transform(image))

    inputs = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)

    results = []
    for image_path, probs, indices in zip(image_paths, top_probs.cpu(), top_indices.cpu()):
        predictions = []
        for probability, index in zip(probs.tolist(), indices.tolist()):
            class_name = classes[index]
            predictions.append(
                {
                    "class_name": class_name,
                    "label_text": normalize_label_text(class_name),
                    "probability": probability,
                }
            )

        results.append(
            {
                "image_path": str(image_path),
                "relative_path": build_relative_path(image_path, input_root),
                "top1_class": predictions[0]["class_name"],
                "top1_label_text": predictions[0]["label_text"],
                "top1_probability": predictions[0]["probability"],
                "predictions": predictions,
            }
        )
    return results


def write_csv(path: Path, rows: list[dict], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "relative_path", "top1_label_text"]
    for rank in range(1, top_k + 1):
        fieldnames.extend([f"top{rank}_class", f"top{rank}_probability"])

    csv_rows = []
    for row in rows:
        record = {
            "image_path": row["image_path"],
            "relative_path": row["relative_path"],
            "top1_label_text": row["top1_label_text"],
        }
        for rank, prediction in enumerate(row["predictions"], start=1):
            record[f"top{rank}_class"] = prediction["class_name"]
            record[f"top{rank}_probability"] = prediction["probability"]
        csv_rows.append(record)

    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, classes, transform = load_model(args.checkpoint, device)
    image_paths = collect_images(args.input_path, args.max_images)

    results: list[dict] = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        results.extend(predict_batch(model, classes, batch_paths, args.input_path, transform, device, args.top_k))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_csv is not None:
        write_csv(args.output_csv, results, args.top_k)

    if len(results) == 1:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    else:
        print(f"Images processed: {len(results)}")
        print(f"First image: {results[0]['image_path']}")
        print(f"Top-1: {results[0]['top1_class']} ({results[0]['top1_probability']:.4f})")
        if args.output_csv is not None:
            print(f"Saved CSV: {args.output_csv}")
        if args.output_json is not None:
            print(f"Saved JSON: {args.output_json}")


def predict_single_image(image_path: Path, checkpoint_path: Path, device: str = "cpu", top_k: int = 5) -> dict:
    device_obj = torch.device(device)
    model, classes, transform = load_model(checkpoint_path, device_obj)
    results = predict_batch(
        model,
        classes,
        [image_path],
        image_path.parent,
        transform,
        device_obj,
        top_k,
    )
    return results[0] if results else {}

if __name__ == "__main__":
    main()