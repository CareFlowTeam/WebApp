from __future__ import annotations

import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_class_directories(source_dir: Path) -> list[Path]:
    return sorted([path for path in source_dir.iterdir() if path.is_dir()], key=lambda path: path.name)


def list_images(class_dir: Path) -> list[Path]:
    return sorted([path for path in class_dir.iterdir() if is_image_file(path)], key=lambda path: path.name)


def strip_label_prefix(class_name: str) -> str:
    return re.sub(r"^\d+_", "", class_name)


def normalize_label_text(class_name: str) -> str:
    text = strip_label_prefix(class_name)
    text = text.replace("_x000D_", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ocr_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "밀리그램": "mg",
        "밀리그람": "mg",
        "그램": "g",
        "마이크로그램": "mcg",
        "마이크로그람": "mcg",
        "마이크로": "micro",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^0-9a-z가-힣.%]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"(\d)\s+(mg|g|mcg|ml)", r"\1\2", normalized)
    return normalized


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_ocr_text(text)
    return [token for token in normalized.split() if token]


def extract_dosage_tokens(text: str) -> list[str]:
    normalized = normalize_ocr_text(text)
    matches = re.findall(r"\d+(?:\.\d+)?(?:mg|g|mcg|ml|%)", normalized)
    return matches


def extract_numeric_tokens(text: str) -> list[str]:
    normalized = normalize_ocr_text(text)
    return re.findall(r"\d+(?:\.\d+)?", normalized)


def build_label_token_profile(class_name: str) -> dict[str, list[str]]:
    label_text = normalize_label_text(class_name)
    return {
        "normalized": normalize_ocr_text(label_text),
        "tokens": tokenize_text(label_text),
        "dosages": extract_dosage_tokens(label_text),
        "numbers": extract_numeric_tokens(label_text),
    }


def split_export_name(text: str) -> tuple[str, str | None]:
    parts = re.split(r"수출명", text, maxsplit=1)
    primary = parts[0].strip()
    export_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return primary, export_name


def build_clip_prompts(class_name: str) -> list[str]:
    text = normalize_label_text(class_name)
    primary, export_name = split_export_name(text)
    prompts = OrderedDict()

    base_candidates = [primary, text]
    if export_name:
        base_candidates.append(export_name)

    for candidate in base_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        prompts[f"a photo of a Korean medicine package labeled {candidate}"] = None
        prompts[f"a pharmaceutical package of {candidate}"] = None
        prompts[f"a product photo of medicine {candidate}"] = None

    if export_name:
        prompts[f"a medicine package with text {primary} and export name {export_name}"] = None

    return list(prompts.keys())


def compute_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(f"Each class needs at least 3 images to make a train/val/test split, got {total}.")

    raw_counts = [total * train_ratio, total * val_ratio, total * test_ratio]
    base_counts = [math.floor(value) for value in raw_counts]
    remainder = total - sum(base_counts)

    fractional = sorted(
        enumerate([raw - base for raw, base in zip(raw_counts, base_counts)]),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, _ in fractional[:remainder]:
        base_counts[index] += 1

    for index in range(3):
        if base_counts[index] == 0:
            donor = max(range(3), key=lambda item: base_counts[item])
            if base_counts[donor] <= 1:
                raise ValueError(f"Unable to create a non-empty split for class with {total} images.")
            base_counts[index] += 1
            base_counts[donor] -= 1

    return tuple(base_counts)


def topk_accuracy(logits, targets, ks=(1, 5)):
    import torch

    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    batch_size = targets.size(0)
    values = {}
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        values[k] = float(correct_k.item() / batch_size)
    return values


def build_confusion_class_weights(
    confusion_csv: Path,
    class_to_idx: dict[str, int],
    strength: float,
) -> list[float] | None:
    if strength <= 0 or not confusion_csv.exists():
        return None

    scores = [0.0] * len(class_to_idx)
    with confusion_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                count = float(row.get("count", 0) or 0)
                pair_error_rate = float(row.get("pair_error_rate_within_true", 0) or 0)
            except ValueError:
                continue

            score = count * max(pair_error_rate, 1e-6)
            true_class = row.get("true_class")
            predicted_class = row.get("predicted_class")

            if true_class in class_to_idx:
                scores[class_to_idx[true_class]] += score
            if predicted_class in class_to_idx:
                scores[class_to_idx[predicted_class]] += score * 0.5

    max_score = max(scores, default=0.0)
    if max_score <= 0:
        return None

    return [1.0 + strength * (score / max_score) for score in scores]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")