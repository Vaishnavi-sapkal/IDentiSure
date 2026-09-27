"""Find tall/narrow annotated text fields that may merit rotation experiments.

This is a geometry-only diagnostic.  A tall/narrow box is reported as a
"rotation candidate"; it is not evidence that its text is actually rotated.
The manifest and EasyOCR evaluator are read only.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "easyocr_full_report.json"
REAL_IMAGES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sidtd"
    / "templates"
    / "templates"
    / "Images"
    / "reals"
)
CSV_PATH = PROJECT_ROOT / "docs" / "rotation_candidates.csv"
CROPS_DIR = PROJECT_ROOT / "docs" / "crops" / "rotation_candidates"

NON_TEXT_FIELDS = {"photo", "signature", "face"}
TALL_NARROW_RATIO = 2.0
VERY_LOW_EXACT_MATCH_PERCENTAGE = 25.0
EXAMPLES_PER_FIELD = 3


def load_real_text_fields() -> list[dict]:
    """Return evaluator-compatible real text annotations with valid geometry."""
    records = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != "real":
                continue

            image_name = row.get("image")
            fields = row.get("fields")
            if not isinstance(image_name, str) or not isinstance(fields, dict):
                continue

            for field_name, annotation in fields.items():
                if field_name in NON_TEXT_FIELDS or not isinstance(annotation, dict):
                    continue
                if "value" not in annotation:
                    continue
                try:
                    x = annotation["x"]
                    y = annotation["y"]
                    width = annotation["width"]
                    height = annotation["height"]
                except KeyError:
                    continue
                if not all(isinstance(value, (int, float)) for value in (x, y, width, height)):
                    print(f"Skipping non-numeric bbox at manifest line {line_number}: {field_name}")
                    continue
                if width <= 0 or height < 0:
                    print(f"Skipping invalid bbox at manifest line {line_number}: {field_name}")
                    continue

                ground_truth = str(annotation["value"])
                aspect_ratio = height / width
                records.append(
                    {
                        "field_name": field_name,
                        "image": image_name,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": width,
                        "bbox_height": height,
                        "aspect_ratio": aspect_ratio,
                        "ground_truth": ground_truth,
                        "text_length": len(ground_truth),
                    }
                )
    return records


def load_per_field_metrics() -> dict:
    """Load a precomputed report when it has the expected reusable structure."""
    if not REPORT_PATH.is_file():
        return {}
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as report_file:
            metrics = json.load(report_file).get("per_field_metrics", {})
    except (json.JSONDecodeError, OSError) as error:
        print(f"Skipping OCR-performance comparison: cannot read {REPORT_PATH}: {error}")
        return {}
    return metrics if isinstance(metrics, dict) else {}


def write_csv(candidates: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "field_name", "image", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "aspect_ratio", "ground_truth", "text_length",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                **candidate,
                "aspect_ratio": f"{candidate['aspect_ratio']:.6f}",
            })


def safe_stem(value: str) -> str:
    """Make an informative filename component that is portable on Windows."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).stem).strip("._") or "image"


def save_example_crops(ranked_groups: list[tuple[str, list[dict]]]) -> None:
    """Save at most three highest-aspect candidate crops for every candidate field."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for field_name, candidates in ranked_groups:
        for candidate in sorted(candidates, key=lambda item: item["aspect_ratio"], reverse=True)[:EXAMPLES_PER_FIELD]:
            image_path = REAL_IMAGES_DIR / candidate["image"]
            if not image_path.is_file():
                print(f"Crop skipped; image not found: {image_path}")
                continue
            output_name = f"{safe_stem(field_name)}__{safe_stem(candidate['image'])}.png"
            output_path = CROPS_DIR / output_name
            with Image.open(image_path) as image:
                crop = image.crop((
                    candidate["bbox_x"], candidate["bbox_y"],
                    candidate["bbox_x"] + candidate["bbox_width"],
                    candidate["bbox_y"] + candidate["bbox_height"],
                ))
                crop.convert("RGB").save(output_path)
            print(f"  example: {candidate['image']} -> {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    records = load_real_text_fields()
    candidates = [record for record in records if record["aspect_ratio"] >= TALL_NARROW_RATIO]
    candidates.sort(key=lambda record: (record["field_name"], record["image"]))
    write_csv(candidates)

    totals = defaultdict(int)
    grouped_candidates = defaultdict(list)
    for record in records:
        totals[record["field_name"]] += 1
    for candidate in candidates:
        grouped_candidates[candidate["field_name"]].append(candidate)

    ranked_groups = sorted(
        grouped_candidates.items(),
        key=lambda item: (-len(item[1]) / totals[item[0]], -len(item[1]), item[0]),
    )
    metrics = load_per_field_metrics()

    print("\n## Rotation candidate fields")
    print("field_name | candidate_count | total_count | candidate_percentage | avg_aspect_ratio | min_aspect_ratio | max_aspect_ratio")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for field_name, group in ranked_groups:
        aspects = [item["aspect_ratio"] for item in group]
        percentage = 100 * len(group) / totals[field_name]
        print(
            f"{field_name} | {len(group)} | {totals[field_name]} | {percentage:.2f}% | "
            f"{sum(aspects) / len(aspects):.3f} | {min(aspects):.3f} | {max(aspects):.3f}"
        )

    low_performance = []
    for field_name, group in ranked_groups:
        field_metrics = metrics.get(field_name)
        if not isinstance(field_metrics, dict):
            continue
        exact_percentage = field_metrics.get("exact_match_percentage")
        if isinstance(exact_percentage, (int, float)) and exact_percentage <= VERY_LOW_EXACT_MATCH_PERCENTAGE:
            low_performance.append((field_name, len(group), exact_percentage, field_metrics.get("average_character_similarity")))
    if metrics:
        print(f"\n## Rotation candidates with very low OCR performance (raw exact match <= {VERY_LOW_EXACT_MATCH_PERCENTAGE:.0f}%)")
        if low_performance:
            print("field_name | candidate_count | raw_exact_match_percentage | average_character_similarity")
            print("--- | ---: | ---: | ---:")
            for field_name, count, exact, similarity in low_performance:
                similarity_text = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else "n/a"
                print(f"{field_name} | {count} | {exact:.2f}% | {similarity_text}")
        else:
            print("None.")

    print(f"\n## Example crops (up to {EXAMPLES_PER_FIELD} per candidate field)")
    save_example_crops(ranked_groups)
    print(f"\nReal annotated text fields examined: {len(records)}")
    print(f"Rotation candidates (height / width >= {TALL_NARROW_RATIO:.1f}): {len(candidates)}")
    print(f"CSV report: {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Crop directory: {CROPS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
