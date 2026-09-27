"""Compare EasyOCR output before and after rotating weak vertical SIDTD fields."""

import json
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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "rotation_test"
TARGET_FIELDS = ("birth_date_22", "birth_date_2")
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")
UPSCALE_FACTOR = 3


def find_first_real_fields() -> dict[str, tuple[str, dict]]:
    """Return the first real manifest annotation found for each target field."""
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
    """Load an annotation crop and return it as an RGB Pillow image."""
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


def upscale(image: Image.Image) -> Image.Image:
    """Enlarge a crop while preserving readable character edges."""
    return image.resize(
        (image.width * UPSCALE_FACTOR, image.height * UPSCALE_FACTOR),
        Image.Resampling.LANCZOS,
    )


def read_text(reader: easyocr.Reader, image: Image.Image) -> str:
    """Run EasyOCR and join its text lines in the baseline's format."""
    return " ".join(reader.readtext(np.array(image), detail=0))


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
            original = crop_annotation(image_name, annotation)
            clockwise = upscale(original.transpose(Image.Transpose.ROTATE_270))
            counter_clockwise = upscale(original.transpose(Image.Transpose.ROTATE_90))

            original.save(OUTPUT_DIR / f"{field_name}_original.png")
            clockwise.save(OUTPUT_DIR / f"{field_name}_cw.png")
            counter_clockwise.save(OUTPUT_DIR / f"{field_name}_ccw.png")

            print(
                f"Field: {field_name}\n"
                f"Ground truth: {annotation.get('value', '')}\n"
                f"Original OCR: {read_text(reader, original)}\n"
                f"Clockwise OCR: {read_text(reader, clockwise)}\n"
                f"Counter-clockwise OCR: {read_text(reader, counter_clockwise)}\n"
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Field: {field_name}\nCould not run rotation test: {error}\n")


if __name__ == "__main__":
    main()
