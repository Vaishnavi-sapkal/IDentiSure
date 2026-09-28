"""Validate saved EasyOCR TD3 MRZ predictions without rerunning OCR.

The normalization experiment report is inspected first for its source file and
variant schema. Because that report stores aggregate metrics rather than every
normalized prediction, this script reads its declared source JSONL and rebuilds
only the report's conservative Variant 3 string normalization.
"""

import calendar
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORMALIZATION_REPORT = PROJECT_ROOT / "docs" / "mrz_normalization_experiment.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mrz_validation_experiment.json"
MRZ_FIELDS = ("mrz_line0", "mrz_line1")
MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)
CHECKS = {
    "passport_number": (0, 9, 9),
    "date_of_birth": (13, 19, 19),
    "date_of_expiry": (21, 27, 27),
    "optional_data": (28, 42, 42),
    "composite": (None, None, 43),
}
AMBIGUITIES = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S"}
STRUCTURAL_FAILURE_TYPES = (
    "line_count", "line0_length", "line1_length", "line0_invalid_characters", "line1_invalid_characters",
    "document_code_format", "issuing_state_format", "name_field_format", "passport_number_format",
    "passport_number_check_digit_format", "nationality_format", "date_of_birth_format",
    "date_of_birth_check_digit_format", "sex_format", "date_of_expiry_format",
    "date_of_expiry_check_digit_format", "optional_data_format", "optional_data_check_digit_format",
    "composite_check_digit_format",
)


def normalize_conservative(text: str) -> str:
    """Exactly reproduce Variant 3: whitespace removal, uppercase, =/- filler."""
    text = "".join(character for character in text if not character.isspace()).upper()
    return text.replace("=", "<").replace("-", "<")


def character_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    if character == "<":
        return 0
    raise ValueError(f"Invalid ICAO MRZ character: {character!r}")


def check_digit(data: str) -> str:
    """Calculate the ICAO 9303 check digit for valid MRZ data."""
    return str(sum(character_value(character) * WEIGHTS[index % 3] for index, character in enumerate(data)) % 10)


def is_date_format_valid(value: str) -> bool:
    """Accept a six-digit YYMMDD value only when month/day form a calendar date.

    The century is not encoded in TD3 data. Both 1900+YY and 2000+YY are
    checked so leap-day format remains valid without guessing the century.
    """
    if len(value) != 6 or not value.isdigit():
        return False
    year, month, day = int(value[:2]), int(value[2:4]), int(value[4:])
    if not 1 <= month <= 12:
        return False
    return any(day <= calendar.monthrange(century + year, month)[1] for century in (1900, 2000))


def structural_failures(line0: str | None, line1: str | None) -> list[str]:
    """Return every TD3 structural failure, retaining malformed samples."""
    failures = []
    if line0 is None or line1 is None:
        return ["line_count"]
    for label, line in (("line0", line0), ("line1", line1)):
        if len(line) != 44:
            failures.append(f"{label}_length")
        invalid = sorted(set(line) - MRZ_ALLOWED)
        if invalid:
            failures.append(f"{label}_invalid_characters")
    if len(line0) == 44:
        if not all(character.isalnum() or character == "<" for character in line0[:2]):
            failures.append("document_code_format")
        if not line0[2:5].isalpha() or not line0[2:5].isupper():
            failures.append("issuing_state_format")
        if not all(character.isalpha() or character == "<" for character in line0[5:]):
            failures.append("name_field_format")
    if len(line1) == 44:
        if not all(character.isalnum() or character == "<" for character in line1[:9]):
            failures.append("passport_number_format")
        if not line1[9].isdigit():
            failures.append("passport_number_check_digit_format")
        if not line1[10:13].isalpha() or not line1[10:13].isupper():
            failures.append("nationality_format")
        if not is_date_format_valid(line1[13:19]):
            failures.append("date_of_birth_format")
        if not line1[19].isdigit():
            failures.append("date_of_birth_check_digit_format")
        if line1[20] not in "MF<":
            failures.append("sex_format")
        if not is_date_format_valid(line1[21:27]):
            failures.append("date_of_expiry_format")
        if not line1[27].isdigit():
            failures.append("date_of_expiry_check_digit_format")
        if not all(character.isalnum() or character == "<" for character in line1[28:42]):
            failures.append("optional_data_format")
        if not line1[42].isdigit():
            failures.append("optional_data_check_digit_format")
        if not line1[43].isdigit():
            failures.append("composite_check_digit_format")
    return failures


def check_data(line: str, check_name: str) -> str:
    """Return the TD3 data positions protected by one named check digit."""
    if check_name == "composite":
        return line[0:10] + line[13:20] + line[21:43]
    start, end, _ = CHECKS[check_name]
    return line[start:end]


def validate_checks(line1: str | None) -> dict:
    """Validate each TD3 line-2 digit whenever a 44-char source permits it."""
    results = {}
    if line1 is None or len(line1) != 44:
        return {name: {"evaluated": False, "valid": False, "reason": "invalid_length"} for name in CHECKS}
    for name, (_, _, digit_position) in CHECKS.items():
        data = check_data(line1, name)
        expected = line1[digit_position]
        try:
            calculated = check_digit(data)
        except ValueError:
            results[name] = {"evaluated": False, "valid": False, "reason": "invalid_character", "expected_check_digit": expected}
            continue
        results[name] = {
            "evaluated": expected.isdigit(),
            "valid": expected.isdigit() and calculated == expected,
            "calculated_check_digit": calculated,
            "expected_check_digit": expected,
        }
    return results


def check_positions(check_name: str) -> list[int]:
    """Return zero-based line-2 data positions relevant to one check."""
    if check_name == "composite":
        return list(range(0, 10)) + list(range(13, 20)) + list(range(21, 43))
    start, end, _ = CHECKS[check_name]
    return list(range(start, end))


def ambiguity_candidates(line1: str | None, checks: dict) -> list[dict]:
    """Report, but never apply, one-character ambiguity hypotheses."""
    if line1 is None or len(line1) != 44:
        return []
    candidates = []
    for check_name, result in checks.items():
        if not result.get("evaluated") or result.get("valid"):
            continue
        expected = result["expected_check_digit"]
        for position in check_positions(check_name):
            original = line1[position]
            replacement = AMBIGUITIES.get(original)
            if replacement is None:
                continue
            changed = line1[:position] + replacement + line1[position + 1:]
            try:
                after = check_digit(check_data(changed, check_name))
            except ValueError:
                continue
            candidates.append({
                "field": check_name,
                "position": position + 1,
                "ocr_character": original,
                "possible_character": replacement,
                "current_calculated_check_digit": result.get("calculated_check_digit"),
                "expected_check_digit": expected,
                "check_digit_after_substitution": after,
                "matches_expected": after == expected,
            })
    return candidates


def load_pairs() -> tuple[list[dict], dict]:
    """Inspect the normalization-report schema and load its declared OCR source."""
    with NORMALIZATION_REPORT.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict) or not isinstance(report.get("results"), dict):
        raise ValueError("Normalization report does not contain a 'results' object.")
    variant = report["results"].get("variant_3_mrz_punctuation")
    if not isinstance(variant, dict):
        raise KeyError("Normalization report does not contain conservative 'variant_3_mrz_punctuation'.")
    input_file = report.get("input_file")
    if not isinstance(input_file, str):
        raise KeyError("Normalization report does not identify its input_file.")
    source_path = PROJECT_ROOT / input_file
    latest = {}
    with source_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("field") in MRZ_FIELDS and isinstance(record.get("image"), str):
                if not isinstance(record.get("prediction"), str):
                    raise ValueError(f"MRZ prediction missing at source line {line_number}.")
                latest[(record["image"], record["field"])] = record
    images = sorted({image for image, _ in latest})
    pairs = []
    for image in images:
        line0 = latest.get((image, "mrz_line0"))
        line1 = latest.get((image, "mrz_line1"))
        pairs.append({
            "image": image,
            "raw_line0": line0.get("prediction") if line0 else None,
            "raw_line1": line1.get("prediction") if line1 else None,
            "normalized_line0": normalize_conservative(line0["prediction"]) if line0 else None,
            "normalized_line1": normalize_conservative(line1["prediction"]) if line1 else None,
        })
    return pairs, {
        "normalization_report_top_level_keys": sorted(report),
        "conservative_variant_keys": sorted(variant),
        "source_results_file": input_file,
    }


def evaluate_version(pairs: list[dict], prefix: str) -> tuple[dict, list[dict]]:
    """Evaluate one raw or conservative-normalized MRZ representation."""
    failures = Counter({failure_type: 0 for failure_type in STRUCTURAL_FAILURE_TYPES})
    check_counts = Counter()
    examples = []
    samples = []
    for pair in pairs:
        line0, line1 = pair[f"{prefix}_line0"], pair[f"{prefix}_line1"]
        structure = structural_failures(line0, line1)
        checks = validate_checks(line1)
        candidates = ambiguity_candidates(line1, checks) if prefix == "normalized" else []
        valid_checks = {name: result.get("valid", False) for name, result in checks.items()}
        for failure in structure:
            failures[failure] += 1
        for name, valid in valid_checks.items():
            check_counts[f"{name}_valid"] += valid
        all_checks_valid = all(valid_checks.values())
        check_counts["all_checks_valid"] += all_checks_valid
        samples.append({
            "filename": pair["image"], "line0": line0, "line1": line1,
            "structural_failures": structure, "structurally_valid": not structure,
            "check_digits": checks, "all_check_digits_valid": all_checks_valid,
            "ambiguity_candidates": candidates,
        })
    total = len(samples)
    metrics = {
        "total_samples": total,
        "structurally_valid_samples": sum(sample["structurally_valid"] for sample in samples),
        "invalid_length_samples": sum(any(failure.endswith("_length") for failure in sample["structural_failures"]) for sample in samples),
        "invalid_character_samples": sum(any(failure.endswith("_invalid_characters") for failure in sample["structural_failures"]) for sample in samples),
        "structural_failure_counts": dict(sorted(failures.items())),
        "samples_with_exactly_one_failed_check_digit": sum(
            sum(not result.get("valid", False) for result in sample["check_digits"].values()) == 1
            for sample in samples
        ),
    }
    for key in ("passport_number", "date_of_birth", "date_of_expiry", "optional_data", "composite"):
        metrics[f"valid_{key}_check_digit"] = check_counts[f"{key}_valid"]
    metrics["all_check_digits_valid"] = check_counts["all_checks_valid"]
    percentage_metrics = (
        "structurally_valid_samples", "invalid_length_samples", "invalid_character_samples",
        "valid_passport_number_check_digit", "valid_date_of_birth_check_digit",
        "valid_date_of_expiry_check_digit", "valid_optional_data_check_digit",
        "valid_composite_check_digit", "all_check_digits_valid",
    )
    for key in percentage_metrics:
        metrics[f"{key}_percentage"] = 100 * metrics[key] / total if total else 0.0
    return metrics, samples


def representative_examples(samples: list[dict]) -> list[dict]:
    """Select useful all-pass, one-failure, multi-failure, ambiguity examples."""
    selected, seen = [], set()
    categories = (
        lambda sample: sample["all_check_digits_valid"],
        lambda sample: sum(not result.get("valid", False) for result in sample["check_digits"].values()) == 1,
        lambda sample: sum(not result.get("valid", False) for result in sample["check_digits"].values()) > 1,
        lambda sample: any(candidate["matches_expected"] for candidate in sample["ambiguity_candidates"]),
    )
    for category in categories:
        for sample in samples:
            if category(sample) and sample["filename"] not in seen:
                selected.append(sample)
                seen.add(sample["filename"])
                break
    for sample in samples:
        if sample["filename"] not in seen:
            selected.append(sample)
            seen.add(sample["filename"])
            if len(selected) == 15:
                break
    return selected


def print_summary(raw: dict, normalized: dict, normalized_samples: list[dict], examples: list[dict]) -> None:
    print("## MRZ validation experiment")
    print("Metric | Raw OCR | Normalized OCR")
    print("--- | ---: | ---:")
    metrics = (
        "total_samples", "structurally_valid_samples", "invalid_length_samples", "invalid_character_samples",
        "valid_passport_number_check_digit", "valid_date_of_birth_check_digit",
        "valid_date_of_expiry_check_digit", "valid_optional_data_check_digit",
        "valid_composite_check_digit", "all_check_digits_valid",
    )
    for metric in metrics:
        raw_value, normalized_value = raw[metric], normalized[metric]
        if metric != "total_samples":
            print(f"{metric} | {raw_value} ({raw[metric + '_percentage']:.2f}%) | {normalized_value} ({normalized[metric + '_percentage']:.2f}%)")
        else:
            print(f"{metric} | {raw_value} | {normalized_value}")
    failed = [sample for sample in normalized_samples if not sample["all_check_digits_valid"]]
    possible = [sample for sample in failed if any(candidate["matches_expected"] for candidate in sample["ambiguity_candidates"])]
    multiple = [sample for sample in failed if sum(candidate["matches_expected"] for candidate in sample["ambiguity_candidates"]) > 1]
    print(f"\nNormalized samples with all check digits valid: {normalized['all_check_digits_valid']}")
    print(f"Normalized samples with at least one check-digit failure: {len(failed)}")
    print(f"Failed samples with a possible single-character ambiguity fix: {len(possible)}")
    print(f"Failed samples with multiple possible corrections: {len(multiple)}")
    print(f"Normalized samples with exactly one failed check digit: {normalized['samples_with_exactly_one_failed_check_digit']}")
    print("\n## Representative normalized line-2 examples")
    for sample in examples:
        failed_checks = [name for name, result in sample["check_digits"].items() if not result.get("valid", False)]
        candidate = next((item for item in sample["ambiguity_candidates"] if item["matches_expected"]), None)
        print(f"{sample['filename']}\n  line2: {sample['line1']}\n  failed checks: {', '.join(failed_checks) or 'none'}")
        if candidate:
            print(f"  candidate: {candidate['field']} pos {candidate['position']} {candidate['ocr_character']}->{candidate['possible_character']} (calculated {candidate['current_calculated_check_digit']} -> {candidate['check_digit_after_substitution']}, expected {candidate['expected_check_digit']})")
    if not normalized["samples_with_exactly_one_failed_check_digit"]:
        print("No normalized sample had exactly one failed check digit; representative failures therefore show multiple-check cases.")


def main() -> None:
    pairs, schema = load_pairs()
    raw_metrics, raw_samples = evaluate_version(pairs, "raw")
    normalized_metrics, normalized_samples = evaluate_version(pairs, "normalized")
    examples = representative_examples(normalized_samples)
    report = {
        "input_schema_detected": schema,
        "scope": "ICAO 9303 TD3 passports: two 44-character MRZ lines",
        "normalization": "whitespace removal, uppercase, and conservative =/- to < filler mapping only",
        "raw_ocr": {"metrics": raw_metrics, "samples": raw_samples},
        "normalized_ocr": {"metrics": normalized_metrics, "samples": normalized_samples},
        "representative_normalized_examples": examples,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print_summary(raw_metrics, normalized_metrics, normalized_samples, examples)
    print(f"\nComplete results: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
