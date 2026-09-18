"""Evaluate EasyOCR on annotated real SIDTD fields from the manifest.

This is a standalone debugging script. It reads the manifest and images only;
all crops are kept in memory.
"""

import json
import re
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
REAL_IMAGES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sidtd"
    / "templates"
    / "templates"
    / "Images"
    / "reals"
)
SAMPLE_COUNT = 10


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    """Load non-empty JSONL rows without altering the manifest."""
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        return [json.loads(line) for line in manifest_file if line.strip()]


def select_samples(rows: list[dict], limit: int) -> list[tuple[dict, str, dict]]:
    """Select fields with values, favoring unused images and field names."""
    candidates = []
    required_bbox_keys = {"x", "y", "width", "height"}

    for row in rows:
        if row.get("label") != "real":
            continue

        image_name = row.get("image")
        fields = row.get("fields")
        if not image_name or not isinstance(fields, dict):
            continue

        for field_name, annotation in fields.items():
            if not isinstance(annotation, dict):
                continue
            value = annotation.get("value")
            if value is None or not str(value).strip():
                continue
            if not required_bbox_keys.issubset(annotation):
                continue
            if not (REAL_IMAGES_DIR / image_name).is_file():
                continue
            candidates.append((row, field_name, annotation))

    selected = []
    used_images = set()
    used_fields = set()
    remaining = candidates.copy()

    while remaining and len(selected) < limit:
        # A lower score gives priority to both a new image and a new field.
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                remaining[index][0]["image"] in used_images,
                remaining[index][1] in used_fields,
                remaining[index][0]["image"],
                remaining[index][1],
            ),
        )
        row, field_name, annotation = remaining.pop(best_index)
        selected.append((row, field_name, annotation))
        used_images.add(row["image"])
        used_fields.add(field_name)

    return selected


def levenshtein_distance(left: str, right: str) -> int:
    """Return the character-level Levenshtein edit distance."""
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current_row[-1] + 1
            deletion = previous_row[right_index] + 1
            substitution = previous_row[right_index - 1] + (
                left_character != right_character
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def similarity(ground_truth: str, prediction: str) -> float:
    """Return normalized character similarity, including the empty-string case."""
    max_length = max(len(ground_truth), len(prediction))
    if max_length == 0:
        return 1.0
    return 1 - levenshtein_distance(ground_truth, prediction) / max_length


def normalize_for_optional_comparison(text: str) -> str:
    """Case-fold and collapse whitespace for the separately displayed comparison."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def crop_field(image_path: Path, bbox: dict) -> Image.Image:
    """Crop a manifest bbox and return an RGB image suitable for EasyOCR."""
    x, y = bbox["x"], bbox["y"]
    width, height = bbox["width"], bbox["height"]

    with Image.open(image_path) as image:
        crop = image.crop((x, y, x + width, y + height))
        return crop.convert("RGB") if crop.mode != "RGB" else crop.copy()


def main() -> None:
    samples = select_samples(load_manifest_rows(MANIFEST_PATH), SAMPLE_COUNT)
    if not samples:
        raise LookupError("No eligible real-image fields with non-empty values found.")

    # Initialize once for the entire evaluation, rather than once per sample.
    reader = easyocr.Reader(["en"])

    raw_exact_matches = 0
    similarities = []

    for index, (row, field_name, bbox) in enumerate(samples, start=1):
        image_name = row["image"]
        ground_truth = str(bbox["value"])
        crop = crop_field(REAL_IMAGES_DIR / image_name, bbox)

        # This list is printed unchanged so the raw EasyOCR result is visible.
        raw_detected_text = reader.readtext(np.array(crop), detail=0)
        prediction = " ".join(raw_detected_text)
        raw_similarity = similarity(ground_truth, prediction)
        normalized_ground_truth = normalize_for_optional_comparison(ground_truth)
        normalized_prediction = normalize_for_optional_comparison(prediction)
        raw_exact_match = ground_truth == prediction

        similarities.append(raw_similarity)
        raw_exact_matches += raw_exact_match

        print(f"\nSample {index}")
        print(f"Image: {image_name}")
        print(f"Field: {field_name}")
        print(f"Ground truth (raw): {ground_truth!r}")
        print(f"Raw EasyOCR detected text: {raw_detected_text!r}")
        print(f"Prediction used for raw comparison: {prediction!r}")
        print(f"Raw exact match: {raw_exact_match}")
        print(f"Similarity (raw): {raw_similarity:.4f}")
        print(f"Ground truth (normalized): {normalized_ground_truth!r}")
        print(f"Prediction (normalized): {normalized_prediction!r}")
        print(
            "Normalized exact match: "
            f"{normalized_ground_truth == normalized_prediction}"
        )
        print(
            "Similarity (normalized): "
            f"{similarity(normalized_ground_truth, normalized_prediction):.4f}"
        )

    sample_total = len(samples)
    average_similarity = sum(similarities) / sample_total
    exact_percentage = 100 * raw_exact_matches / sample_total

    print("\nSummary")
    print(f"Number of samples: {sample_total}")
    print(f"Average character similarity (raw): {average_similarity:.4f}")
    print(f"Number of exact matches (raw): {raw_exact_matches}")
    print(f"Exact-match percentage (raw): {exact_percentage:.2f}%")


if __name__ == "__main__":
    main()
