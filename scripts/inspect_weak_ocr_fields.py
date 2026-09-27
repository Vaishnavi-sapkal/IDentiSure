"""Save one real-image crop for each weak EasyOCR field in the SIDTD manifest."""

import json
from pathlib import Path

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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "weak_fields"
TARGET_FIELDS = ("birth_date_22", "birth_date_2", "mrz_line1", "birth_place")
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")


def find_first_real_fields() -> dict[str, tuple[str, dict]]:
    """Return the first real manifest annotation found for every target field."""
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


def crop_field(field_name: str, image_name: str, annotation: dict) -> Path:
    """Crop an annotation from its real image and save an RGB PNG."""
    missing_keys = [key for key in REQUIRED_BBOX_KEYS if key not in annotation]
    if missing_keys:
        raise ValueError(f"Missing bbox key(s): {', '.join(missing_keys)}")

    x, y = annotation["x"], annotation["y"]
    width, height = annotation["width"], annotation["height"]
    image_path = REAL_IMAGES_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = OUTPUT_DIR / f"{field_name}_{Path(image_name).stem}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        crop = image.crop((x, y, x + width, y + height))
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        crop.save(output_path)

    return output_path


def main() -> None:
    found = find_first_real_fields()

    for field_name in TARGET_FIELDS:
        match = found.get(field_name)
        if match is None:
            print(f"Field: {field_name}\nNot found in a real manifest row.\n")
            continue

        image_name, annotation = match
        try:
            output_path = crop_field(field_name, image_name, annotation)
            print(
                f"Field: {field_name}\n"
                f"Image: {image_name}\n"
                f"Ground truth: {annotation.get('value', '')}\n"
                f"BBox: x={annotation['x']}, y={annotation['y']}, "
                f"width={annotation['width']}, height={annotation['height']}\n"
                f"Crop: {output_path.relative_to(PROJECT_ROOT)}\n"
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"Field: {field_name}\nCould not create crop: {error}\n")


if __name__ == "__main__":
    main()
