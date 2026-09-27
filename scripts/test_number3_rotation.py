"""Test rotation and resizing variants for tall/narrow ``number3`` crops only.

This is an isolated diagnostic experiment. It reads the manifest and uses the
same EasyOCR construction as the project baseline (``easyocr.Reader(["en"])``)
without modifying the evaluator, manifest, or OCR configuration.
"""

import json
from collections import defaultdict
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
REAL_IMAGES_DIR = (
    PROJECT_ROOT / "datasets" / "sidtd" / "templates" / "templates" / "Images" / "reals"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "number3_rotation_test"
FIELD_NAME = "number3"
TALL_NARROW_RATIO = 2.0
MAX_EXAMPLES = 5

ROTATIONS = {
    "original": None,
    "clockwise_90": Image.Transpose.ROTATE_270,
    "counter_clockwise_90": Image.Transpose.ROTATE_90,
}
SCALES = (1, 2, 3)
INTERPOLATIONS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def levenshtein_distance(left: str, right: str) -> int:
    """Return the same character-level edit distance used by the evaluator."""
    if len(left) < len(right):
        left, right = right, left
    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current_row.append(min(
                current_row[-1] + 1,
                previous_row[right_index] + 1,
                previous_row[right_index - 1] + (left_character != right_character),
            ))
        previous_row = current_row
    return previous_row[-1]


def similarity(ground_truth: str, prediction: str) -> float:
    """Return evaluator-compatible normalized Levenshtein similarity."""
    maximum_length = max(len(ground_truth), len(prediction))
    return 1.0 if maximum_length == 0 else 1 - levenshtein_distance(ground_truth, prediction) / maximum_length


def numeric_ground_truth(text: str) -> str | None:
    """Return digit-only truth only when removing spaces leaves digits alone."""
    without_spaces = text.replace(" ", "")
    if without_spaces and without_spaces.isdigit():
        return without_spaces
    return None


def select_examples() -> list[dict]:
    """Select the first five real ``number3`` annotations meeting the geometry rule."""
    selected = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != "real" or not isinstance(row.get("fields"), dict):
                continue
            annotation = row["fields"].get(FIELD_NAME)
            image_name = row.get("image")
            if not isinstance(annotation, dict) or not isinstance(image_name, str):
                continue
            try:
                x, y, width, height = (annotation[key] for key in ("x", "y", "width", "height"))
            except KeyError:
                continue
            if not all(isinstance(value, (int, float)) for value in (x, y, width, height)) or width <= 0 or height < 0:
                continue
            if height / width < TALL_NARROW_RATIO:
                continue
            selected.append({
                "image": image_name, "x": x, "y": y, "width": width, "height": height,
                "ground_truth": str(annotation.get("value", "")),
            })
            if len(selected) == MAX_EXAMPLES:
                break
    return selected


def load_crop(example: dict) -> Image.Image:
    """Load the exact annotated crop as RGB."""
    image_path = REAL_IMAGES_DIR / example["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.crop((example["x"], example["y"], example["x"] + example["width"], example["y"] + example["height"])).convert("RGB")


def prepare_variant(crop: Image.Image, rotation: str, scale: int, interpolation: Image.Resampling) -> Image.Image:
    """Apply exactly one rotation and one resize operation."""
    transpose = ROTATIONS[rotation]
    if transpose is not None:
        crop = crop.transpose(transpose)
    return crop.resize((crop.width * scale, crop.height * scale), interpolation)


def compare(ground_truth: str, prediction: str) -> tuple[str, str, int, float, bool]:
    """Use digits-only comparison only for purely numeric (apart from spaces) truth."""
    digit_truth = numeric_ground_truth(ground_truth)
    if digit_truth is not None:
        normalized_prediction = "".join(character for character in prediction if character.isdigit())
        return digit_truth, normalized_prediction, levenshtein_distance(digit_truth, normalized_prediction), similarity(digit_truth, normalized_prediction), True
    return ground_truth, "n/a (raw comparison)", levenshtein_distance(ground_truth, prediction), similarity(ground_truth, prediction), False


def main() -> None:
    examples = select_examples()
    if not examples:
        raise LookupError(f"No real {FIELD_NAME} examples meet height / width >= {TALL_NARROW_RATIO}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"])
    aggregate = defaultdict(list)

    for index, example in enumerate(examples, start=1):
        crop = load_crop(example)
        print(f"\nExample {index}")
        print(f"Image: {example['image']}")
        print(f"Ground truth: {example['ground_truth']}")
        if numeric_ground_truth(example["ground_truth"]) is None:
            print("Comparison: raw (ground truth contains meaningful non-digit characters)")
        else:
            print("Comparison: normalized digits (spaces removed from ground truth)")
        print("Variant | Raw OCR | Normalized OCR | Distance")
        print("--- | --- | --- | ---:")

        image_stem = Path(example["image"]).stem
        for rotation in ROTATIONS:
            for scale in SCALES:
                for interpolation_name, interpolation in INTERPOLATIONS.items():
                    variant = prepare_variant(crop, rotation, scale, interpolation)
                    variant_name = f"{rotation}_{scale}x_{interpolation_name}"
                    output_path = OUTPUT_DIR / f"number3_{image_stem}_{variant_name}.png"
                    variant.save(output_path)
                    prediction = " ".join(reader.readtext(np.array(variant), detail=0))
                    target, normalized_prediction, distance, character_similarity, numeric = compare(example["ground_truth"], prediction)
                    print(f"{variant_name} | {prediction or '(empty)'} | {normalized_prediction} | {distance}")
                    aggregate[variant_name].append({
                        "exact": (normalized_prediction == target) if numeric else (prediction == target),
                        "distance": distance,
                        "similarity": character_similarity,
                    })

    print("\n## Aggregate results")
    print("Variant | Examples | Exact matches | Average distance | Average similarity")
    print("--- | ---: | ---: | ---: | ---:")
    summaries = []
    for variant_name, results in aggregate.items():
        exact_matches = sum(result["exact"] for result in results)
        average_distance = sum(result["distance"] for result in results) / len(results)
        average_similarity = sum(result["similarity"] for result in results) / len(results)
        summaries.append((variant_name, exact_matches, len(results), average_distance, average_similarity))
    summaries.sort(key=lambda result: (-result[1], result[3], -result[4], result[0]))
    for variant_name, exact, total, avg_distance, avg_similarity in summaries:
        print(f"{variant_name} | {total} | {exact} | {avg_distance:.3f} | {avg_similarity:.4f}")

    best_name, best_exact, best_total, best_distance, _ = summaries[0]
    equivalent_best = [
        variant_name
        for variant_name, exact, total, avg_distance, avg_similarity in summaries
        if (exact, total, avg_distance, avg_similarity)
        == (best_exact, best_total, best_distance, summaries[0][4])
    ]
    if len(equivalent_best) > 1:
        print("\nTied best configurations: " + ", ".join(equivalent_best))
    rotation, scale, interpolation = best_name.rsplit("_", 2)
    print("\n## BEST CONFIGURATION")
    print(f"Rotation: {rotation}")
    print(f"Scale: {scale}")
    print(f"Interpolation: {interpolation}")
    print(f"Exact matches: {best_exact}")
    print(f"Total examples: {best_total}")
    print(f"Average distance: {best_distance:.3f}")
    print(f"Processed crops saved under: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
