"""Run a resumable EasyOCR baseline over every annotated real SIDTD field.

The manifest is read only. Results are appended one field at a time so a
stopped run can continue from ``docs/easyocr_full_results.jsonl``.
"""

import json
import re
from collections import defaultdict
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
RESULTS_PATH = PROJECT_ROOT / "docs" / "easyocr_full_results.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "easyocr_full_report.json"
PROGRESS_EVERY = 100
REQUIRED_BBOX_KEYS = {"x", "y", "width", "height"}


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
    """Use the same normalized Levenshtein similarity as the test script."""
    max_length = max(len(ground_truth), len(prediction))
    if max_length == 0:
        return 1.0
    return 1 - levenshtein_distance(ground_truth, prediction) / max_length


def normalize_for_optional_comparison(text: str) -> str:
    """Keep raw output intact; use this only for the separate normalized metric."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_manifest_rows() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        return [json.loads(line) for line in manifest_file if line.strip()]


def valid_real_fields(rows: list[dict]) -> list[tuple[str, str, dict]]:
    """Return every real annotation with a bbox and an explicit ground truth."""
    fields = []
    for row in rows:
        if row.get("label") != "real":
            continue

        image_name = row.get("image")
        annotations = row.get("fields")
        if not image_name or not isinstance(annotations, dict):
            continue

        for field_name, annotation in annotations.items():
            if not isinstance(annotation, dict):
                continue
            if "value" not in annotation:
                continue
            if not REQUIRED_BBOX_KEYS.issubset(annotation):
                continue
            fields.append((image_name, field_name, annotation))
    return fields


def result_key(result: dict) -> tuple[str, str] | None:
    image_name = result.get("image")
    field_name = result.get("field")
    if isinstance(image_name, str) and isinstance(field_name, str):
        return image_name, field_name
    return None


def load_completed_results() -> dict[tuple[str, str], dict]:
    """Load the last saved result for each field; tolerate a partial final line."""
    completed = {}
    if not RESULTS_PATH.exists():
        return completed

    with RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring malformed prior result at line {line_number}.")
                continue
            key = result_key(result)
            if key is not None:
                completed[key] = result
    return completed


def crop_field(image_path: Path, bbox: dict) -> Image.Image:
    """Load a real image, crop one bbox, and return a Pillow RGB image."""
    x, y = bbox["x"], bbox["y"]
    width, height = bbox["width"], bbox["height"]
    if width < 0 or height < 0:
        raise ValueError(f"Negative bbox dimensions: width={width}, height={height}")

    with Image.open(image_path) as image:
        crop = image.crop((x, y, x + width, y + height))
        return crop.convert("RGB") if crop.mode != "RGB" else crop.copy()


def evaluate_field(
    reader: easyocr.Reader | None,
    reader_error: str | None,
    image_name: str,
    field_name: str,
    annotation: dict,
) -> dict:
    """Evaluate one field and convert image/OCR errors into saved results."""
    ground_truth = str(annotation["value"])
    result = {
        "image": image_name,
        "field": field_name,
        "ground_truth": ground_truth,
        "raw_ocr": [],
        "prediction": "",
        "raw_exact_match": False,
        "normalized_exact_match": False,
        "character_similarity": 0.0,
    }

    try:
        if reader_error is not None:
            raise RuntimeError(reader_error)
        if reader is None:
            raise RuntimeError("EasyOCR Reader was not initialized.")

        crop = crop_field(REAL_IMAGES_DIR / image_name, annotation)
        # Preserve EasyOCR's returned list. Comparison follows the test script.
        raw_ocr = reader.readtext(np.array(crop), detail=0)
        prediction = " ".join(raw_ocr)
        normalized_ground_truth = normalize_for_optional_comparison(ground_truth)
        normalized_prediction = normalize_for_optional_comparison(prediction)

        result.update(
            raw_ocr=raw_ocr,
            prediction=prediction,
            raw_exact_match=(ground_truth == prediction),
            normalized_exact_match=(
                normalized_ground_truth == normalized_prediction
            ),
            character_similarity=similarity(ground_truth, prediction),
        )
    except Exception as error:  # Continue and record errors for all difficult fields.
        result["error"] = f"{type(error).__name__}: {error}"

    return result


def append_result(results_file, result: dict) -> None:
    results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    results_file.flush()


def summarize(results: list[dict], real_image_count: int, expected_fields: int) -> dict:
    """Build overall and per-field metrics from all completed expected fields."""
    field_totals = defaultdict(
        lambda: {
            "samples": 0,
            "raw_exact_matches": 0,
            "normalized_exact_matches": 0,
            "similarity_total": 0.0,
            "ocr_failures": 0,
        }
    )
    raw_exact_matches = 0
    normalized_exact_matches = 0
    similarity_total = 0.0
    failures = 0

    for result in results:
        field_total = field_totals[result["field"]]
        field_total["samples"] += 1
        field_total["raw_exact_matches"] += bool(result.get("raw_exact_match"))
        field_total["normalized_exact_matches"] += bool(
            result.get("normalized_exact_match")
        )
        field_total["similarity_total"] += float(
            result.get("character_similarity", 0.0)
        )
        field_total["ocr_failures"] += "error" in result

        raw_exact_matches += bool(result.get("raw_exact_match"))
        normalized_exact_matches += bool(result.get("normalized_exact_match"))
        similarity_total += float(result.get("character_similarity", 0.0))
        failures += "error" in result

    total_fields = len(results)
    def percent(count: int, total: int) -> float:
        return 100 * count / total if total else 0.0

    per_field = {}
    for field_name in sorted(field_totals):
        metrics = field_totals[field_name]
        sample_count = metrics["samples"]
        per_field[field_name] = {
            "samples": sample_count,
            "exact_matches": metrics["raw_exact_matches"],
            "exact_match_percentage": percent(
                metrics["raw_exact_matches"], sample_count
            ),
            "average_character_similarity": (
                metrics["similarity_total"] / sample_count if sample_count else 0.0
            ),
            "normalized_exact_matches": metrics["normalized_exact_matches"],
            "normalized_exact_match_percentage": percent(
                metrics["normalized_exact_matches"], sample_count
            ),
            "ocr_failures": metrics["ocr_failures"],
        }

    return {
        "dataset_totals": {
            "real_images_in_manifest": real_image_count,
            "valid_annotated_real_fields": expected_fields,
            "fields_processed": total_fields,
        },
        "overall_metrics": {
            "exact_matches": raw_exact_matches,
            "exact_match_accuracy": percent(raw_exact_matches, total_fields),
            "average_character_similarity": (
                similarity_total / total_fields if total_fields else 0.0
            ),
            "normalized_exact_matches": normalized_exact_matches,
            "normalized_exact_match_accuracy": percent(
                normalized_exact_matches, total_fields
            ),
        },
        "ocr_failure_count": failures,
        "per_field_metrics": per_field,
    }


def print_summary_table(report: dict) -> None:
    print("\nPer-field summary (raw comparison)")
    print(f"{'Field':<24} {'Samples':>8} {'Exact':>8} {'Exact %':>10} {'Avg sim.':>10} {'Failures':>10}")
    print("-" * 78)
    for field_name, metrics in report["per_field_metrics"].items():
        print(
            f"{field_name:<24} {metrics['samples']:>8} "
            f"{metrics['exact_matches']:>8} "
            f"{metrics['exact_match_percentage']:>9.2f}% "
            f"{metrics['average_character_similarity']:>10.4f} "
            f"{metrics['ocr_failures']:>10}"
        )

    overall = report["overall_metrics"]
    print("\nOverall")
    print(f"Fields processed: {report['dataset_totals']['fields_processed']}")
    print(f"Raw exact-match accuracy: {overall['exact_match_accuracy']:.2f}%")
    print(f"Average character similarity: {overall['average_character_similarity']:.4f}")
    print(f"OCR failures: {report['ocr_failure_count']}")


def main() -> None:
    rows = load_manifest_rows()
    real_rows = [row for row in rows if row.get("label") == "real"]
    expected = valid_real_fields(real_rows)
    expected_keys = {(image_name, field_name) for image_name, field_name, _ in expected}

    completed = load_completed_results()
    pending = [
        field for field in expected if (field[0], field[1]) not in completed
    ]
    total_fields = len(expected)
    print(
        f"Real images: {len(real_rows)} | Valid annotated fields: {total_fields} | "
        f"Already completed: {total_fields - len(pending)} | Pending: {len(pending)}"
    )

    reader = None
    reader_error = None
    if pending:
        try:
            # Exactly one Reader instance is used for this entire run.
            reader = easyocr.Reader(["en"])
        except Exception as error:
            reader_error = f"EasyOCR initialization failed: {type(error).__name__}: {error}"
            print(reader_error)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as results_file:
        for pending_index, (image_name, field_name, annotation) in enumerate(
            pending, start=1
        ):
            result = evaluate_field(
                reader, reader_error, image_name, field_name, annotation
            )
            append_result(results_file, result)
            completed[(image_name, field_name)] = result

            processed = total_fields - len(pending) + pending_index
            if processed % PROGRESS_EVERY == 0 or processed == total_fields:
                print(f"Processed {processed}/{total_fields} fields...")

    # Limit the report to fields in the current real manifest, even after resumes.
    report_results = [
        completed[key] for key in expected_keys if key in completed
    ]
    report = summarize(report_results, len(real_rows), total_fields)
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    print_summary_table(report)
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Aggregate report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
