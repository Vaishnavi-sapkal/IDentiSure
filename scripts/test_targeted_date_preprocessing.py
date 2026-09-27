"""Compare small, field-specific preprocessing variants for two SIDTD dates."""

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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "targeted_date_preprocessing"
TARGET_FIELDS = ("birth_date_22", "birth_date_2")
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")


def find_first_real_fields() -> dict[str, tuple[str, dict]]:
    """Find the same first real examples used by the earlier test scripts."""
    found = {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != "real":
                continue

            image_name = row.get("image")
            annotations = row.get("fields")
            if not isinstance(image_name, str) or not isinstance(annotations, dict):
                continue

            for field_name in TARGET_FIELDS:
                annotation = annotations.get(field_name)
                if field_name not in found and isinstance(annotation, dict):
                    found[field_name] = (image_name, annotation)

            if len(found) == len(TARGET_FIELDS):
                break
    return found


def crop_annotation(image_name: str, annotation: dict) -> Image.Image:
    """Load an exact manifest crop as RGB pixels."""
    missing_keys = [key for key in REQUIRED_BBOX_KEYS if key not in annotation]
    if missing_keys:
        raise ValueError(f"Missing bbox key(s): {', '.join(missing_keys)}")

    image_path = REAL_IMAGES_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, y = annotation["x"], annotation["y"]
    width, height = annotation["width"], annotation["height"]
    with Image.open(image_path) as image:
        crop = image.crop((x, y, x + width, y + height))
        return crop.convert("RGB") if crop.mode != "RGB" else crop.copy()


def upscale(image: Image.Image, factor: int, resample: Image.Resampling) -> Image.Image:
    """Resize an image using a named Pillow resampling method."""
    return image.resize((image.width * factor, image.height * factor), resample)


def normalize_digits(text: str) -> str:
    """Keep only decimal digits for a date-specific OCR comparison."""
    return re.sub(r"\D", "", text)


def levenshtein_distance(left: str, right: str) -> int:
    """Return a character-level edit distance."""
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[right_index] + 1,
                    previous_row[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous_row = current_row

    return previous_row[-1]


def build_variants(field_name: str, original: Image.Image) -> dict[str, Image.Image]:
    """Build only the requested preprocessing variants for one target field."""
    variants = {
        "original": original,
        "3x_bilinear": upscale(original, 3, Image.Resampling.BILINEAR),
        "3x_bicubic": upscale(original, 3, Image.Resampling.BICUBIC),
        "3x_lanczos": upscale(original, 3, Image.Resampling.LANCZOS),
    }
    if field_name == "birth_date_22":
        clockwise = original.transpose(Image.Transpose.ROTATE_270)
        variants.update(
            {
                "cw_3x_bilinear": upscale(
                    clockwise, 3, Image.Resampling.BILINEAR
                ),
                "cw_3x_bicubic": upscale(clockwise, 3, Image.Resampling.BICUBIC),
                "cw_3x_lanczos": upscale(clockwise, 3, Image.Resampling.LANCZOS),
            }
        )
    else:
        variants.update(
            {
                "4x_bicubic": upscale(original, 4, Image.Resampling.BICUBIC),
                "4x_lanczos": upscale(original, 4, Image.Resampling.LANCZOS),
            }
        )
    return variants


def main() -> None:
    found = find_first_real_fields()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"])

    for field_name in TARGET_FIELDS:
        match = found.get(field_name)
        if match is None:
            print(f"Field: {field_name}\nNot found in a real manifest row.\n")
            continue

        image_name, annotation = match
        try:
            ground_truth = str(annotation.get("value", ""))
            normalized_ground_truth = normalize_digits(ground_truth)
            variants = build_variants(
                field_name, crop_annotation(image_name, annotation)
            )
            results = []
            for name, image in variants.items():
                image.save(OUTPUT_DIR / f"{field_name}_{name}.png")
                raw_ocr = " ".join(reader.readtext(np.array(image), detail=0))
                normalized_ocr = normalize_digits(raw_ocr)
                distance = levenshtein_distance(
                    normalized_ground_truth, normalized_ocr
                )
                results.append((name, raw_ocr, normalized_ocr, distance))

            print(f"Field: {field_name}")
            print(f"Ground truth: {ground_truth}")
            print("Variant | Raw OCR | Normalized OCR | Character distance")
            print("-" * 78)
            for name, raw_ocr, normalized_ocr, distance in results:
                print(f"{name} | {raw_ocr} | {normalized_ocr} | {distance}")

            best_name, best_raw, best_normalized, best_distance = min(
                results, key=lambda result: result[3]
            )
            print(
                f"Best variant: {best_name} | Raw OCR: {best_raw} | "
                f"Normalized OCR: {best_normalized} | "
                f"Character distance: {best_distance}\n"
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Field: {field_name}\nCould not run test: {error}\n")


if __name__ == "__main__":
    main()
