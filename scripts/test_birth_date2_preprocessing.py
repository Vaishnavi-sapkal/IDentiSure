"""Diagnose EasyOCR on the small, vertical SIDTD ``birth_date_2`` field."""

import json
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageEnhance


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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "birth_date2_preprocessing"
FIELD_NAME = "birth_date_2"
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")


def find_first_real_field() -> tuple[str, dict]:
    """Find the same first real ``birth_date_2`` annotation as the rotation test."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue

            row = json.loads(line)
            if row.get("label") != "real":
                continue

            image_name = row.get("image")
            annotations = row.get("fields")
            annotation = annotations.get(FIELD_NAME) if isinstance(annotations, dict) else None
            if isinstance(image_name, str) and isinstance(annotation, dict):
                return image_name, annotation

    raise LookupError(f"No real manifest annotation found for '{FIELD_NAME}'.")


def crop_annotation(image_name: str, annotation: dict) -> Image.Image:
    """Load the exact annotation bbox as an RGB Pillow image."""
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
    """Resize an image by an integer factor using the supplied Pillow method."""
    return image.resize((image.width * factor, image.height * factor), resample)


def threshold(image: Image.Image, cutoff: int = 128) -> Image.Image:
    """Convert a grayscale image into a black-and-white image."""
    return image.point(lambda pixel: 255 if pixel >= cutoff else 0)


def levenshtein_distance(left: str, right: str) -> int:
    """Return character-level edit distance for ranking OCR candidates."""
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


def main() -> None:
    image_name, annotation = find_first_real_field()
    ground_truth = str(annotation.get("value", ""))
    original = crop_annotation(image_name, annotation)
    lanczos_6x = upscale(original, 6, Image.Resampling.LANCZOS)
    grayscale_6x = lanczos_6x.convert("L")
    contrast_6x = ImageEnhance.Contrast(grayscale_6x).enhance(2.0)
    threshold_6x = threshold(grayscale_6x)

    variants = {
        "a_original": original,
        "b_3x_nearest": upscale(original, 3, Image.Resampling.NEAREST),
        "c_3x_bilinear": upscale(original, 3, Image.Resampling.BILINEAR),
        "d_3x_bicubic": upscale(original, 3, Image.Resampling.BICUBIC),
        "e_4x_bicubic": upscale(original, 4, Image.Resampling.BICUBIC),
        "f_4x_lanczos": upscale(original, 4, Image.Resampling.LANCZOS),
        "g_6x_lanczos": lanczos_6x,
        "h_6x_lanczos_grayscale": grayscale_6x,
        "i_6x_lanczos_grayscale_contrast": contrast_6x,
        "j_6x_lanczos_grayscale_threshold": threshold_6x,
    }
    # Repeat the highest-resolution variants in their readable orientation.
    for name in ("g_6x_lanczos", "h_6x_lanczos_grayscale", "i_6x_lanczos_grayscale_contrast", "j_6x_lanczos_grayscale_threshold"):
        variants[f"{name}_cw"] = variants[name].transpose(Image.Transpose.ROTATE_270)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"])
    results = []
    for name, image in variants.items():
        output_path = OUTPUT_DIR / f"birth_date_2_{name}.png"
        image.save(output_path)
        prediction = " ".join(reader.readtext(np.array(image), detail=0))
        results.append((name, prediction, levenshtein_distance(ground_truth, prediction)))

    print(f"Image: {image_name}")
    print(f"Ground truth: {ground_truth}\n")
    print("Variant | OCR result")
    print("-" * 72)
    for name, prediction, _ in results:
        print(f"{name} | {prediction}")

    closest_name, closest_prediction, closest_distance = min(
        results, key=lambda result: result[2]
    )
    print(
        f"\nClosest result: {closest_name} | {closest_prediction} "
        f"(Levenshtein distance: {closest_distance})"
    )
    print(f"Saved variants: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
