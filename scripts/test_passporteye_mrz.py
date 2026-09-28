"""Benchmark PassportEye on the exact passport sample set used by MRZ tests.

This standalone experiment reads prior MRZ reports/results but does not alter
them, does not initialize EasyOCR, and never changes the production pipeline.
"""

import json
import time
import calendar
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = PROJECT_ROOT / "docs" / "mrz_validation_experiment.json"
NORMALIZATION_PATH = PROJECT_ROOT / "docs" / "mrz_normalization_experiment.json"
RESULTS_PATH = PROJECT_ROOT / "docs" / "easyocr_full_results.jsonl"
REAL_IMAGES_DIR = PROJECT_ROOT / "datasets" / "sidtd" / "templates" / "templates" / "Images" / "reals"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "passporteye_mrz_experiment.json"
MRZ_FIELDS = ("mrz_line0", "mrz_line1")
MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)
CHECKS = {
    "passport_number": (list(range(0, 9)), 9),
    "date_of_birth": (list(range(13, 19)), 19),
    "date_of_expiry": (list(range(21, 27)), 27),
    "optional_data": (list(range(28, 42)), 42),
    "composite": (list(range(0, 10)) + list(range(13, 20)) + list(range(21, 43)), 43),
}
PROGRESS_EVERY = 10


def normalize_conservative(text: str) -> str:
    """Use exactly the previously approved whitespace/case/filler handling."""
    text = "".join(character for character in text if not character.isspace()).upper()
    return text.replace("=", "<").replace("-", "<")


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    maximum = max(len(left), len(right))
    return 1.0 if maximum == 0 else 1 - levenshtein_distance(left, right) / maximum


def mrz_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    if character == "<":
        return 0
    raise ValueError(f"Invalid MRZ character: {character!r}")


def check_digit(data: str) -> str:
    return str(sum(mrz_value(character) * WEIGHTS[index % 3] for index, character in enumerate(data)) % 10)


def validate_checks(line1: str | None) -> dict:
    """Mirror the TD3 line-2 ICAO validation used by test_mrz_validation.py."""
    if line1 is None or len(line1) != 44:
        return {name: False for name in CHECKS}
    outcomes = {}
    for name, (positions, check_position) in CHECKS.items():
        try:
            outcomes[name] = line1[check_position].isdigit() and check_digit("".join(line1[position] for position in positions)) == line1[check_position]
        except ValueError:
            outcomes[name] = False
    return outcomes


def valid_date(value: str) -> bool:
    """Use the same century-agnostic YYMMDD format check as MRZ validation."""
    if len(value) != 6 or not value.isdigit():
        return False
    year, month, day = int(value[:2]), int(value[2:4]), int(value[4:])
    return 1 <= month <= 12 and any(day <= calendar.monthrange(century + year, month)[1] for century in (1900, 2000))


def valid_td3_structure(line0: str, line1: str) -> bool:
    """Mirror the TD3 structural requirements in test_mrz_validation.py."""
    if len(line0) != 44 or len(line1) != 44 or not set(line0 + line1) <= MRZ_ALLOWED:
        return False
    return (
        all(character.isalnum() or character == "<" for character in line0[:2])
        and line0[2:5].isalpha() and line0[2:5].isupper()
        and all(character.isalpha() or character == "<" for character in line0[5:])
        and all(character.isalnum() or character == "<" for character in line1[:9])
        and line1[9].isdigit()
        and line1[10:13].isalpha() and line1[10:13].isupper()
        and valid_date(line1[13:19]) and line1[19].isdigit()
        and line1[20] in "MF<"
        and valid_date(line1[21:27]) and line1[27].isdigit()
        and all(character.isalnum() or character == "<" for character in line1[28:42])
        and line1[42].isdigit() and line1[43].isdigit()
    )


def split_td3(raw_text: str) -> tuple[str, str | None, str | None, bool, bool]:
    """Normalize output and split only genuine 2x44 representations.

    A combined 88-character string can be split; all other malformed output is
    retained verbatim but never padded/truncated to manufacture TD3 lines.
    """
    normalized = normalize_conservative(raw_text)
    normalized_lines = [normalize_conservative(line) for line in raw_text.splitlines() if normalize_conservative(line)]
    line0 = line1 = None
    if len(normalized_lines) == 2:
        line0, line1 = normalized_lines
    elif len(normalized) == 88:
        line0, line1 = normalized[:44], normalized[44:]
    valid_length = line0 is not None and len(line0) == 44 and len(line1) == 44
    invalid_characters = bool(normalized) and any(character not in MRZ_ALLOWED for character in normalized)
    return normalized, line0, line1, valid_length, invalid_characters


def load_dataset() -> tuple[list[dict], dict]:
    """Inspect previous report schemas and join the exact validation image set to truth."""
    with VALIDATION_PATH.open("r", encoding="utf-8") as validation_file:
        validation = json.load(validation_file)
    with NORMALIZATION_PATH.open("r", encoding="utf-8") as normalization_file:
        normalization = json.load(normalization_file)
    normalized_samples = validation.get("normalized_ocr", {}).get("samples")
    if not isinstance(normalized_samples, list):
        raise ValueError("mrz_validation_experiment.json lacks normalized_ocr.samples.")
    source_relative = validation.get("input_schema_detected", {}).get("source_results_file")
    if not isinstance(source_relative, str):
        raise ValueError("Validation report lacks its source_results_file.")
    latest = {}
    with (PROJECT_ROOT / source_relative).open("r", encoding="utf-8") as results_file:
        for line in results_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("field") in MRZ_FIELDS and isinstance(record.get("image"), str):
                latest[(record["image"], record["field"])] = record
    dataset = []
    for sample in normalized_samples:
        image = sample.get("filename")
        if not isinstance(image, str):
            continue
        truth0 = latest.get((image, "mrz_line0"), {}).get("ground_truth")
        truth1 = latest.get((image, "mrz_line1"), {}).get("ground_truth")
        easy0, easy1 = sample.get("line0"), sample.get("line1")
        if not all(isinstance(value, str) for value in (truth0, truth1, easy0, easy1)):
            continue
        dataset.append({
            "image": image,
            "ground_truth_line0": normalize_conservative(truth0),
            "ground_truth_line1": normalize_conservative(truth1),
            "easyocr_line0": easy0,
            "easyocr_line1": easy1,
        })
    schema = {
        "validation_report_top_level_keys": sorted(validation) if isinstance(validation, dict) else [],
        "validation_normalized_ocr_keys": sorted(validation.get("normalized_ocr", {})) if isinstance(validation, dict) else [],
        "normalization_report_top_level_keys": sorted(normalization) if isinstance(normalization, dict) else [],
        "source_results_file": source_relative,
    }
    return dataset, schema


def extract_raw_text(mrz_result) -> tuple[str, dict]:
    """Extract PassportEye's raw text across supported MRZ object versions."""
    details = mrz_result.to_dict() if hasattr(mrz_result, "to_dict") else {}
    raw_text = details.get("raw_text") if isinstance(details, dict) else None
    if not isinstance(raw_text, str):
        raw_text = getattr(mrz_result, "raw_text", "")
    return (raw_text if isinstance(raw_text, str) else ""), (dict(details) if isinstance(details, dict) else {})


def summarize(records: list[dict], prefix: str) -> dict:
    """Calculate exact, similarity, TD3, and check-digit metrics for one system."""
    total = len(records)
    detected = sum(record.get("passporteye_detected", True) for record in records) if prefix == "passporteye" else total
    line0_exact = sum(record[f"{prefix}_line0"] == record["ground_truth_line0"] for record in records)
    line1_exact = sum(record[f"{prefix}_line1"] == record["ground_truth_line1"] for record in records)
    full_exact = sum(
        record[f"{prefix}_line0"] == record["ground_truth_line0"] and record[f"{prefix}_line1"] == record["ground_truth_line1"]
        for record in records
    )
    full_similarity = sum(similarity(record[f"{prefix}_full"], record["ground_truth_full"]) for record in records) / total
    line0_similarity = sum(similarity(record[f"{prefix}_line0"], record["ground_truth_line0"]) for record in records) / total
    line1_similarity = sum(similarity(record[f"{prefix}_line1"], record["ground_truth_line1"]) for record in records) / total
    valid_td3 = sum(record[f"{prefix}_valid_td3"] for record in records)
    invalid_length = sum(not record[f"{prefix}_valid_length"] for record in records)
    invalid_characters = sum(record[f"{prefix}_invalid_characters"] for record in records)
    checks = {name: sum(record[f"{prefix}_checks"][name] for record in records) for name in CHECKS}
    all_checks = sum(all(record[f"{prefix}_checks"].values()) for record in records)
    return {
        "samples": total,
        "detection_success": detected,
        "detection_success_percentage": 100 * detected / total,
        "detection_failures": total - detected,
        "full_mrz_exact": full_exact,
        "full_mrz_exact_percentage": 100 * full_exact / total,
        "line0_exact": line0_exact,
        "line0_exact_percentage": 100 * line0_exact / total,
        "line1_exact": line1_exact,
        "line1_exact_percentage": 100 * line1_exact / total,
        "average_full_mrz_similarity": full_similarity,
        "average_line0_similarity": line0_similarity,
        "average_line1_similarity": line1_similarity,
        "valid_td3_pairs": valid_td3,
        "invalid_length_pairs": invalid_length,
        "invalid_character_pairs": invalid_characters,
        "valid_passport_number_checks": checks["passport_number"],
        "valid_dob_checks": checks["date_of_birth"],
        "valid_expiry_checks": checks["date_of_expiry"],
        "valid_optional_data_checks": checks["optional_data"],
        "valid_composite_checks": checks["composite"],
        "all_five_checks_valid": all_checks,
    }


def failure_reason(record: dict) -> str:
    if not record["passporteye_detected"]:
        return record.get("passporteye_error") or "no MRZ detected"
    if not record["passporteye_valid_length"]:
        return "invalid TD3 length or line count"
    if record["passporteye_invalid_characters"]:
        return "invalid MRZ characters"
    if not record["passporteye_line0_exact"] and not record["passporteye_line1_exact"]:
        return "wrong characters in both lines"
    if not record["passporteye_line0_exact"]:
        return "wrong characters in line 0"
    return "wrong characters in line 1"


def main() -> None:
    try:
        from passporteye import read_mrz
    except ImportError as error:
        report = {"error": f"PassportEye import unavailable: {type(error).__name__}: {error}"}
        OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report["error"])
        print(f"Report: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        return

    dataset, schema = load_dataset()
    records = []
    start = time.perf_counter()
    for index, item in enumerate(dataset, start=1):
        image_path = REAL_IMAGES_DIR / item["image"]
        item_start = time.perf_counter()
        raw_text, details, error = "", {}, None
        detected = False
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            result = read_mrz(str(image_path))
            if result is not None:
                raw_text, details = extract_raw_text(result)
                detected = bool(raw_text)
            if not detected:
                error = "no MRZ detected"
        except Exception as exception:  # Individual detection failures must not stop the benchmark.
            error = f"{type(exception).__name__}: {exception}"
        elapsed = time.perf_counter() - item_start
        normalized, line0, line1, valid_length, invalid_characters = split_td3(raw_text)
        passport_checks = validate_checks(line1)
        easy0, easy1 = item["easyocr_line0"], item["easyocr_line1"]
        truth0, truth1 = item["ground_truth_line0"], item["ground_truth_line1"]
        record = {
            **item,
            "ground_truth_full": truth0 + truth1,
            "easyocr_full": easy0 + easy1,
            "easyocr_valid_length": len(easy0) == 44 and len(easy1) == 44,
            "easyocr_invalid_characters": any(character not in MRZ_ALLOWED for character in easy0 + easy1),
            "easyocr_valid_td3": valid_td3_structure(easy0, easy1),
            "easyocr_checks": validate_checks(easy1),
            "passporteye_detected": detected,
            "passporteye_error": error,
            "passporteye_raw": raw_text,
            "passporteye_details": details,
            "passporteye_normalized": normalized,
            "passporteye_line0": line0 or "",
            "passporteye_line1": line1 or "",
            "passporteye_full": normalized,
            "passporteye_valid_length": valid_length,
            "passporteye_invalid_characters": invalid_characters,
            "passporteye_valid_td3": valid_td3_structure(line0, line1) if line0 is not None and line1 is not None else False,
            "passporteye_checks": passport_checks,
            "processing_time_seconds": elapsed,
        }
        record["passporteye_line0_exact"] = record["passporteye_line0"] == truth0
        record["passporteye_line1_exact"] = record["passporteye_line1"] == truth1
        records.append(record)
        if index % PROGRESS_EVERY == 0 or index == len(dataset):
            print(f"Processed {index}/{len(dataset)} images...")

    total_time = time.perf_counter() - start
    passporteye_metrics = summarize(records, "passporteye")
    easyocr_metrics = summarize(records, "easyocr")
    exact_examples = [record for record in records if record["passporteye_line0_exact"] and record["passporteye_line1_exact"]][:5]
    failure_examples = [record for record in records if not (record["passporteye_line0_exact"] and record["passporteye_line1_exact"])][:5]
    for record in failure_examples:
        record["failure_reason"] = failure_reason(record)
    report = {
        "configuration": {
            "passporteye_call": "passporteye.read_mrz(image_path)",
            "normalization": "remove whitespace; uppercase; =/- to < only",
            "td3_requirement": "two 44-character lines; no padding or truncation",
            "sample_source": "mrz_validation_experiment.json normalized_ocr.samples",
        },
        "input_schema_detected": schema,
        "dataset_sample_count": len(dataset),
        "detection_metrics": {key: passporteye_metrics[key] for key in ("samples", "detection_success", "detection_success_percentage", "detection_failures")},
        "passporteye_metrics": passporteye_metrics,
        "easyocr_normalized_metrics": easyocr_metrics,
        "comparison_metrics": {"easyocr_normalized": easyocr_metrics, "passporteye": passporteye_metrics},
        "timing": {
            "total_processing_time_seconds": total_time,
            "average_time_per_image_seconds": total_time / len(dataset),
            "average_time_per_successful_detection_seconds": sum(record["processing_time_seconds"] for record in records if record["passporteye_detected"]) / passporteye_metrics["detection_success"] if passporteye_metrics["detection_success"] else None,
        },
        "representative_exact_matches": exact_examples,
        "representative_failures": failure_examples,
        "per_image_results": records,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print("\n## PassportEye MRZ Benchmark")
    for key in (
        "samples", "detection_success", "detection_failures", "full_mrz_exact", "full_mrz_exact_percentage",
        "line0_exact", "line0_exact_percentage", "line1_exact", "line1_exact_percentage",
        "average_full_mrz_similarity", "average_line0_similarity", "average_line1_similarity",
        "valid_td3_pairs", "invalid_length_pairs", "invalid_character_pairs", "valid_passport_number_checks",
        "valid_dob_checks", "valid_expiry_checks", "valid_optional_data_checks", "valid_composite_checks", "all_five_checks_valid",
    ):
        print(f"{key}: {passporteye_metrics[key]}")
    print(f"total_processing_time_seconds: {total_time:.3f}")
    print(f"average_time_per_image_seconds: {total_time / len(dataset):.3f}")
    print(f"Report: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
