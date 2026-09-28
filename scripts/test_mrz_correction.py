"""Test bounded, hypothetical TD3 check-digit ambiguity corrections.

This diagnostic reads the existing MRZ validation report and its declared OCR
source. It never changes a prediction: every proposed correction is recorded
as a separate hypothesis limited to O/0, I/1, B/8, and S/5 substitutions.
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = PROJECT_ROOT / "docs" / "mrz_validation_experiment.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mrz_correction_experiment.json"
MRZ_FIELDS = ("mrz_line0", "mrz_line1")
MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)
AMBIGUITIES = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S"}
CHECKS = {
    "passport_number": (list(range(0, 9)), 9),
    "date_of_birth": (list(range(13, 19)), 19),
    "date_of_expiry": (list(range(21, 27)), 27),
    "optional_data": (list(range(28, 42)), 42),
    "composite": (list(range(0, 10)) + list(range(13, 20)) + list(range(21, 43)), 43),
}


def mrz_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    if character == "<":
        return 0
    raise ValueError(f"Invalid MRZ character: {character!r}")


def calculate_digit(data: str) -> str:
    return str(sum(mrz_value(character) * WEIGHTS[index % 3] for index, character in enumerate(data)) % 10)


def check_data(line: str, check_name: str) -> str:
    positions, _ = CHECKS[check_name]
    return "".join(line[position] for position in positions)


def validate_line2(line: str | None) -> dict:
    """Evaluate each TD3 check digit; malformed lines remain explicit failures."""
    if line is None or len(line) != 44:
        return {name: {"valid": False, "evaluated": False, "reason": "invalid_length"} for name in CHECKS}
    results = {}
    for name, (_, check_position) in CHECKS.items():
        expected = line[check_position]
        try:
            calculated = calculate_digit(check_data(line, name))
        except ValueError:
            results[name] = {"valid": False, "evaluated": False, "reason": "invalid_character", "expected_check_digit": expected}
            continue
        results[name] = {
            "valid": expected.isdigit() and calculated == expected,
            "evaluated": expected.isdigit(),
            "calculated_check_digit": calculated,
            "expected_check_digit": expected,
        }
    return results


def all_checks_valid(results: dict) -> bool:
    return all(result.get("valid", False) for result in results.values())


def failed_checks(results: dict) -> list[str]:
    return [name for name, result in results.items() if not result.get("valid", False)]


def relevant_positions(failed: list[str]) -> list[int]:
    """Restrict hypotheses to positions influencing an actually failed check."""
    positions = set()
    for name in failed:
        data_positions, check_position = CHECKS[name]
        positions.update(data_positions)
        positions.add(check_position)
    return sorted(positions)


def apply_changes(line: str, changes: list[tuple[int, str]]) -> str:
    characters = list(line)
    for position, replacement in changes:
        characters[position] = replacement
    return "".join(characters)


def line2_characters_valid(line: str) -> bool:
    """Ensure hypotheses never introduce a non-MRZ character or length change."""
    return len(line) == 44 and set(line) <= MRZ_ALLOWED


def candidate_record(filename: str, original: str, changes: list[tuple[int, str]], original_checks: dict) -> dict | None:
    """Build one hypothesis and retain it only if all TD3 checks become valid."""
    proposed = apply_changes(original, changes)
    if not line2_characters_valid(proposed):
        return None
    proposed_checks = validate_line2(proposed)
    if not all_checks_valid(proposed_checks):
        return None
    changes_detail = []
    for position, replacement in changes:
        source = original[position]
        affected = [name for name, (data_positions, check_position) in CHECKS.items() if position in data_positions or position == check_position]
        changes_detail.append({
            "position": position + 1,
            "original_character": source,
            "proposed_character": replacement,
            "affected_fields": affected,
        })
    original_composite = original_checks["composite"]
    proposed_composite = proposed_checks["composite"]
    failed = failed_checks(original_checks)
    return {
        "filename": filename,
        "mrz_line": "mrz_line1",
        "original_normalized_mrz": original,
        "proposed_normalized_mrz": proposed,
        "changes": changes_detail,
        "failed_checks_before": failed,
        "original_check_digits": original_checks,
        "proposed_check_digits": proposed_checks,
        "original_composite_result": original_composite,
        "proposed_composite_result": proposed_composite,
        "all_relevant_checks_pass": all(proposed_checks[name]["valid"] for name in failed),
        "all_check_digits_pass": True,
    }


def possible_changes(line: str, positions: list[int]) -> list[tuple[int, str]]:
    return [(position, AMBIGUITIES[line[position]]) for position in positions if line[position] in AMBIGUITIES]


def conservative_normalize(text: str) -> str:
    text = "".join(character for character in text if not character.isspace()).upper()
    return text.replace("=", "<").replace("-", "<")


def load_samples() -> tuple[list[dict], dict]:
    """Inspect report schema and join saved normalized samples to ground truth."""
    with VALIDATION_PATH.open("r", encoding="utf-8") as validation_file:
        validation = json.load(validation_file)
    normalized = validation.get("normalized_ocr")
    schema = {
        "validation_report_top_level_keys": sorted(validation) if isinstance(validation, dict) else [],
        "normalized_ocr_keys": sorted(normalized) if isinstance(normalized, dict) else [],
    }
    if not isinstance(normalized, dict) or not isinstance(normalized.get("samples"), list):
        raise ValueError("Validation report does not contain normalized_ocr.samples.")
    source_relative = validation.get("input_schema_detected", {}).get("source_results_file")
    if not isinstance(source_relative, str):
        raise ValueError("Validation report does not identify source_results_file.")
    schema["source_results_file"] = source_relative
    truths = {}
    with (PROJECT_ROOT / source_relative).open("r", encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("field") in MRZ_FIELDS and isinstance(record.get("image"), str):
                truths[(record["image"], record["field"])] = str(record.get("ground_truth", ""))
    samples = []
    for sample in normalized["samples"]:
        filename = sample.get("filename")
        line0, line1 = sample.get("line0"), sample.get("line1")
        if not isinstance(filename, str) or not isinstance(line1, str):
            continue
        truth0 = truths.get((filename, "mrz_line0"), "")
        truth1 = truths.get((filename, "mrz_line1"), "")
        samples.append({
            "filename": filename,
            "line0": line0,
            "line1": line1,
            "truth_line0": conservative_normalize(truth0),
            "truth_line1": conservative_normalize(truth1),
        })
    return samples, schema


def evaluate_sample(sample: dict) -> dict:
    """Classify one line-2 prediction using bounded one/two-change searches."""
    original = sample["line1"]
    original_checks = validate_line2(original)
    original_valid = all_checks_valid(original_checks)
    exact_before = sample["line0"] == sample["truth_line0"] and original == sample["truth_line1"]
    result = {
        "filename": sample["filename"], "normalized_line0": sample["line0"],
        "original_normalized_mrz": original, "ground_truth_line1": sample["truth_line1"],
        "all_checks_valid_before": original_valid, "exact_mrz_match_before": exact_before,
        "failed_checks_before": failed_checks(original_checks), "original_check_digits": original_checks,
        "single_candidates": [], "two_character_candidates": [], "two_character_combinations_tested": 0,
    }
    if original_valid:
        result["classification"] = "D. Already valid after normalization"
        result["correction_size"] = 0
        result["hypothetical_all_checks_valid"] = True
        result["hypothetical_exact_mrz_match"] = exact_before
        return result
    if len(original) != 44:
        result["classification"] = "A. No correction found"
        result["correction_size"] = None
        result["hypothetical_all_checks_valid"] = False
        result["hypothetical_exact_mrz_match"] = False
        return result

    changes = possible_changes(original, relevant_positions(result["failed_checks_before"]))
    for change in changes:
        candidate = candidate_record(sample["filename"], original, [change], original_checks)
        if candidate is not None:
            result["single_candidates"].append(candidate)
    if len(result["single_candidates"]) == 1:
        result["classification"] = "B. One unique correction candidate"
        result["correction_size"] = 1
        chosen = result["single_candidates"][0]["proposed_normalized_mrz"]
        result["hypothetical_all_checks_valid"] = True
        result["hypothetical_exact_mrz_match"] = sample["line0"] == sample["truth_line0"] and chosen == sample["truth_line1"]
        return result
    if len(result["single_candidates"]) > 1:
        result["classification"] = "C. Multiple correction candidates"
        result["correction_size"] = 1
        result["hypothetical_all_checks_valid"] = False
        result["hypothetical_exact_mrz_match"] = False
        return result

    # All currently failed samples have multiple failed checks in this data;
    # test only two distinct ambiguity positions, never unrestricted edits.
    for pair in combinations(changes, 2):
        result["two_character_combinations_tested"] += 1
        candidate = candidate_record(sample["filename"], original, list(pair), original_checks)
        if candidate is not None:
            result["two_character_candidates"].append(candidate)
    if len(result["two_character_candidates"]) == 1:
        result["classification"] = "B. One unique correction candidate"
        result["correction_size"] = 2
        chosen = result["two_character_candidates"][0]["proposed_normalized_mrz"]
        result["hypothetical_all_checks_valid"] = True
        result["hypothetical_exact_mrz_match"] = sample["line0"] == sample["truth_line0"] and chosen == sample["truth_line1"]
    elif len(result["two_character_candidates"]) > 1:
        result["classification"] = "C. Multiple correction candidates"
        result["correction_size"] = 2
        result["hypothetical_all_checks_valid"] = False
        result["hypothetical_exact_mrz_match"] = False
    else:
        result["classification"] = "A. No correction found"
        result["correction_size"] = None
        result["hypothetical_all_checks_valid"] = False
        result["hypothetical_exact_mrz_match"] = False
    return result


def select_examples(samples: list[dict], classifications: tuple[str, ...], limit: int = 5) -> list[dict]:
    return [sample for sample in samples if sample["classification"] in classifications][:limit]


def print_examples(title: str, examples: list[dict]) -> None:
    print(f"\n## {title}")
    if not examples:
        print("No examples in this category.")
        return
    print(f"Showing {len(examples)} example(s).")
    for sample in examples:
        candidates = sample["single_candidates"] or sample["two_character_candidates"]
        changes = "; ".join(
            ", ".join(f"pos {change['position']} {change['original_character']}->{change['proposed_character']}" for change in candidate["changes"])
            for candidate in candidates[:3]
        )
        print(f"{sample['filename']} | {sample['classification']}\n  MRZ: {sample['original_normalized_mrz']}\n  Failed: {', '.join(sample['failed_checks_before'])}\n  Candidates: {changes or 'none'}")


def main() -> None:
    source_samples, schema = load_samples()
    samples = [evaluate_sample(sample) for sample in source_samples]
    total = len(samples)
    failed = [sample for sample in samples if not sample["all_checks_valid_before"]]
    counts = Counter(sample["classification"] for sample in samples)
    unique_single = sum(sample["classification"] == "B. One unique correction candidate" and sample["correction_size"] == 1 for sample in samples)
    multiple_single = sum(sample["classification"] == "C. Multiple correction candidates" and sample["correction_size"] == 1 for sample in samples)
    unique_two = sum(sample["classification"] == "B. One unique correction candidate" and sample["correction_size"] == 2 for sample in samples)
    multiple_two = sum(sample["classification"] == "C. Multiple correction candidates" and sample["correction_size"] == 2 for sample in samples)
    no_correction = counts["A. No correction found"]
    corrected_unique = [sample for sample in samples if sample["classification"] == "B. One unique correction candidate"]
    metrics = {
        "total_normalized_mrz_samples": total,
        "samples_passing_all_checks_before_correction": total - len(failed),
        "samples_failing_at_least_one_check": len(failed),
        "unique_single_character_correction": unique_single,
        "multiple_single_character_corrections": multiple_single,
        "no_single_character_correction": len(failed) - unique_single - multiple_single,
        "unique_two_character_correction": unique_two,
        "multiple_two_character_corrections": multiple_two,
        "no_correction_within_tested_set": no_correction,
        "two_character_combinations_tested": sum(sample["two_character_combinations_tested"] for sample in samples),
        "failed_samples_recoverable_by_one_unique_correction_percentage": 100 * unique_single / len(failed) if failed else 0.0,
        "failed_samples_with_one_or_more_single_corrections_percentage": 100 * (unique_single + multiple_single) / len(failed) if failed else 0.0,
        "failed_samples_recoverable_by_unique_two_character_correction_percentage": 100 * unique_two / len(failed) if failed else 0.0,
        "failed_samples_with_one_or_more_two_character_corrections_percentage": 100 * (unique_two + multiple_two) / len(failed) if failed else 0.0,
        "failed_samples_with_any_tested_correction_percentage": 100 * (unique_single + multiple_single + unique_two + multiple_two) / len(failed) if failed else 0.0,
        "original_normalized_all_checks_valid": total - len(failed),
        "original_normalized_exact_mrz_match": sum(sample["exact_mrz_match_before"] for sample in samples),
        "hypothetical_corrected_all_checks_valid": (total - len(failed)) + len(corrected_unique),
        "hypothetical_corrected_exact_mrz_match": sum(sample["exact_mrz_match_before"] or sample["hypothetical_exact_mrz_match"] for sample in samples),
    }
    report = {"input_schema_detected": schema, "scope": "TD3 line 2; bounded O/0, I/1, B/8, S/5 one- and two-character hypotheses", "metrics": metrics, "samples": samples}
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print("## MRZ correction experiment")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print_examples("Unique correction examples", select_examples(samples, ("B. One unique correction candidate",)))
    print_examples("Ambiguous correction examples", select_examples(samples, ("C. Multiple correction candidates",)))
    print_examples("No-correction examples", select_examples(samples, ("A. No correction found",)))
    print(f"\nComplete results: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
