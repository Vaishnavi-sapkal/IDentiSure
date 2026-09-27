"""Export representative rotation-candidate crops for visual inspection only.

This script reads the manifest and crops annotated real-image bounding boxes.
It deliberately does not import, initialize, or run any OCR engine.
"""

import json
import re
from collections import defaultdict
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
OUTPUT_DIR = PROJECT_ROOT / "docs" / "crops" / "rotation_candidate_validation"

FIELDS_TO_INSPECT = ("birth_date_22", "number3", "number", "gender")
TALL_NARROW_RATIO = 2.0
EXAMPLES_PER_CATEGORY = 3


def safe_stem(value: str) -> str:
    """Return a Windows-safe, readable filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).stem).strip("._") or "image"


def load_examples() -> dict[str, dict[str, list[dict]]]:
    """Collect real examples for each requested field, split by geometry."""
    examples = {
        field_name: {"tall": [], "normal": []}
        for field_name in FIELDS_TO_INSPECT
    }
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != "real" or not isinstance(row.get("fields"), dict):
                continue
            image_name = row.get("image")
            if not isinstance(image_name, str):
                continue

            for field_name in FIELDS_TO_INSPECT:
                annotation = row["fields"].get(field_name)
                if not isinstance(annotation, dict):
                    continue
                try:
                    x, y = annotation["x"], annotation["y"]
                    width, height = annotation["width"], annotation["height"]
                except KeyError:
                    continue
                if not all(isinstance(value, (int, float)) for value in (x, y, width, height)):
                    print(f"Skipping non-numeric bbox at manifest line {line_number}: {field_name}")
                    continue
                if width <= 0 or height < 0:
                    print(f"Skipping invalid bbox at manifest line {line_number}: {field_name}")
                    continue
                aspect_ratio = height / width
                category = "tall" if aspect_ratio >= TALL_NARROW_RATIO else "normal"
                examples[field_name][category].append(
                    {
                        "field_name": field_name,
                        "image": image_name,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio,
                        "ground_truth": str(annotation.get("value", "")),
                        "category": category,
                    }
                )

    # Keep selection deterministic. Prefer the most strongly tall/narrow examples
    # and normal examples closest to the threshold for useful visual comparison.
    for field_examples in examples.values():
        field_examples["tall"].sort(key=lambda item: (-item["aspect_ratio"], item["image"]))
        field_examples["normal"].sort(key=lambda item: (-item["aspect_ratio"], item["image"]))
    return examples


def crop_and_print(example: dict) -> None:
    """Save one exact bbox crop and print its annotation details."""
    image_path = REAL_IMAGES_DIR / example["image"]
    if not image_path.is_file():
        print(f"Skipping missing image: {image_path}")
        return

    filename = (
        f"{safe_stem(example['field_name'])}_{safe_stem(example['image'])}"
        f"_{example['category']}.png"
    )
    output_path = OUTPUT_DIR / filename
    with Image.open(image_path) as image:
        crop = image.crop((
            example["x"], example["y"],
            example["x"] + example["width"], example["y"] + example["height"],
        ))
        crop.convert("RGB").save(output_path)

    print(f"Field: {example['field_name']}")
    print(f"Image: {example['image']}")
    print(f"BBox: {{'x': {example['x']}, 'y': {example['y']}, "
          f"'width': {example['width']}, 'height': {example['height']}}}")
    print(f"Width: {example['width']}")
    print(f"Height: {example['height']}")
    print(f"Aspect ratio: {example['aspect_ratio']:.3f}")
    print(f"Ground truth: {example['ground_truth']}")
    print(f"Orientation category: {example['category']}")
    print(f"Saved crop: {output_path.relative_to(PROJECT_ROOT)}\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_examples()
    generated = defaultdict(int)

    for field_name in FIELDS_TO_INSPECT:
        print(f"\n## {field_name}")
        for category in ("tall", "normal"):
            selected = examples[field_name][category][:EXAMPLES_PER_CATEGORY]
            if not selected:
                print(f"No {category} examples available.\n")
                continue
            for example in selected:
                crop_and_print(example)
                generated[field_name] += 1

    print("## Examples generated")
    for field_name in FIELDS_TO_INSPECT:
        print(f"{field_name}: {generated[field_name]}")
    print(f"Crops saved under: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
