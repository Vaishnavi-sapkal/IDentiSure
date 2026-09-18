"""Crop an annotated SIDTD field from a real image using the manifest."""

import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
IMAGE_NAME = "alb_id_00.jpg"
FIELD_NAME = "birth_date"
IMAGE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "sidtd"
    / "templates"
    / "templates"
    / "Images"
    / "reals"
    / IMAGE_NAME
)
OUTPUT_PATH = (
    PROJECT_ROOT / "docs" / "crops" / "alb_id_00_birth_date.png"
)


def find_manifest_row(manifest_path: Path, image_name: str) -> dict:
    """Return the manifest row matching ``image_name`` without modifying it."""
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            if row.get("image") == image_name:
                return row

    raise LookupError(f"Image '{image_name}' was not found in {manifest_path}")


def main() -> None:
    row = find_manifest_row(MANIFEST_PATH, IMAGE_NAME)

    try:
        bbox = row["fields"][FIELD_NAME]
        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]
    except KeyError as error:
        raise KeyError(
            f"Missing field or bbox key {error!s} for '{IMAGE_NAME}'"
        ) from error

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(IMAGE_PATH) as image:
        crop = image.crop((x, y, x + width, y + height))
        if crop.mode == "CMYK":
            crop = crop.convert("RGB")
        crop.save(OUTPUT_PATH)

    print(f"Image: {IMAGE_NAME}")
    print(f"Field: {FIELD_NAME}")
    print(f"BBox: {{'x': {x}, 'y': {y}, 'width': {width}, 'height': {height}}}")
    print(f"Crop size: {crop.size}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
