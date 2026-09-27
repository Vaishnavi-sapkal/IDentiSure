"""Analyze raw character-level OCR errors for MRZ fields without rerunning OCR.

The input schema is detected from ``easyocr_full_results.jsonl``.  The script
uses the latest record per image/field, matching the evaluator's resume
semantics, and writes a read-only diagnostic report.
"""

import json
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "easyocr_full_results.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mrz_error_analysis.json"
MRZ_FIELDS = ("mrz_line0", "mrz_line1")
REPRESENTATIVE_EXAMPLES = 10


def select_key(records: list[dict], candidates: tuple[str, ...], purpose: str) -> str:
    """Find a schema key shared by records, or explain why analysis cannot run."""
    for key in candidates:
        if any(key in record for record in records):
            return key
    raise KeyError(f"No {purpose} key found. Checked: {', '.join(candidates)}")


def load_latest_mrz_records() -> tuple[list[dict], dict]:
    """Read JSONL and retain the last occurrence of each requested image/field."""
    records = []
    schema_keys = set()
    with RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at line {line_number}: {error}") from error
            if not isinstance(record, dict):
                continue
            schema_keys.update(record)
            if record.get("field") in MRZ_FIELDS:
                records.append(record)

    if not records:
        raise LookupError(f"No records found for: {', '.join(MRZ_FIELDS)}")
    schema = {
        "all_observed_top_level_keys": sorted(schema_keys),
        "ground_truth_key": select_key(records, ("ground_truth", "truth", "target"), "ground-truth"),
        "prediction_key": select_key(records, ("prediction", "ocr_prediction", "ocr_text"), "prediction"),
        "exact_match_key": select_key(records, ("raw_exact_match", "exact_match"), "exact-match"),
        "similarity_key": select_key(records, ("character_similarity", "similarity"), "character-similarity"),
    }
    latest = {}
    for record in records:
        image = record.get("image")
        field = record.get("field")
        if isinstance(image, str) and isinstance(field, str):
            latest[(image, field)] = record
    return list(latest.values()), schema


def align_strings(ground_truth: str, prediction: str) -> list[dict]:
    """Globally align strings and label every non-match edit operation.

    Unit-cost Levenshtein alignment is used. On equal-cost alternatives, a
    substitution is preferred, then deletion, then insertion; this makes the
    resulting error categories deterministic.
    """
    rows, columns = len(ground_truth) + 1, len(prediction) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
    for column in range(1, columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            if ground_truth[row - 1] == prediction[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
            else:
                costs[row][column] = 1 + min(
                    costs[row - 1][column - 1], costs[row - 1][column], costs[row][column - 1]
                )

    aligned = []
    row, column = len(ground_truth), len(prediction)
    while row or column:
        if row and column and ground_truth[row - 1] == prediction[column - 1] and costs[row][column] == costs[row - 1][column - 1]:
            aligned.append({"operation": "match", "ground_truth": ground_truth[row - 1], "prediction": prediction[column - 1]})
            row, column = row - 1, column - 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            aligned.append({"operation": "substitution", "ground_truth": ground_truth[row - 1], "prediction": prediction[column - 1]})
            row, column = row - 1, column - 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            aligned.append({"operation": "deletion", "ground_truth": ground_truth[row - 1], "prediction": ""})
            row -= 1
        else:
            aligned.append({"operation": "insertion", "ground_truth": "", "prediction": prediction[column - 1]})
            column -= 1
    return list(reversed(aligned))


def format_alignment(alignment: list[dict]) -> dict:
    """Preserve a human-readable alignment alongside structured operations."""
    return {
        "ground_truth_aligned": "".join(item["ground_truth"] or "-" for item in alignment),
        "prediction_aligned": "".join(item["prediction"] or "-" for item in alignment),
        "operations": [item for item in alignment if item["operation"] != "match"],
    }


def percentage(part: int, whole: int) -> float:
    return 100 * part / whole if whole else 0.0


def summarize(records: list[dict], schema: dict) -> tuple[dict, list[dict]]:
    """Build per-field and combined raw-string error statistics."""
    truth_key = schema["ground_truth_key"]
    prediction_key = schema["prediction_key"]
    exact_key = schema["exact_match_key"]
    similarity_key = schema["similarity_key"]
    per_field = {}
    all_failed = []

    for field_name in MRZ_FIELDS:
        field_records = [record for record in records if record.get("field") == field_name]
        exact_records = [record for record in field_records if record.get(exact_key) is True]
        failed_records = [record for record in field_records if record.get(exact_key) is False]
        substitutions, insertions, deletions, pairs = 0, 0, 0, Counter()
        same_length, different_length = 0, 0
        failed_details = []
        for record in failed_records:
            ground_truth = str(record.get(truth_key, ""))
            prediction = str(record.get(prediction_key, ""))
            alignment = align_strings(ground_truth, prediction)
            errors = [item for item in alignment if item["operation"] != "match"]
            substitutions += sum(item["operation"] == "substitution" for item in errors)
            insertions += sum(item["operation"] == "insertion" for item in errors)
            deletions += sum(item["operation"] == "deletion" for item in errors)
            for item in errors:
                if item["operation"] == "substitution":
                    pairs[(item["ground_truth"], item["prediction"])] += 1
            if len(ground_truth) == len(prediction):
                same_length += 1
            else:
                different_length += 1
            failed_details.append({
                "field": field_name,
                "image": record.get("image"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "ground_truth_length": len(ground_truth),
                "prediction_length": len(prediction),
                "alignment": format_alignment(alignment),
                "error_count": len(errors),
            })
        total_edits = substitutions + insertions + deletions
        per_field[field_name] = {
            "total_samples": len(field_records),
            "exact_matches": len(exact_records),
            "failed_samples": len(failed_records),
            "exact_match_percentage": percentage(len(exact_records), len(field_records)),
            "average_character_similarity": mean(float(record.get(similarity_key, 0.0)) for record in field_records) if field_records else 0.0,
            "same_length_failed_samples": same_length,
            "different_length_failed_samples": different_length,
            "total_substitutions": substitutions,
            "total_insertions": insertions,
            "total_deletions": deletions,
            "substitution_percentage_of_edits": percentage(substitutions, total_edits),
            "insertion_percentage_of_edits": percentage(insertions, total_edits),
            "deletion_percentage_of_edits": percentage(deletions, total_edits),
            "substitution_pairs": [
                {"ground_truth": source, "prediction": target, "count": count}
                for (source, target), count in pairs.most_common()
            ],
        }
        all_failed.extend(failed_details)

    combined = {
        "total_samples": sum(item["total_samples"] for item in per_field.values()),
        "exact_matches": sum(item["exact_matches"] for item in per_field.values()),
        "failed_samples": sum(item["failed_samples"] for item in per_field.values()),
        "same_length_failed_samples": sum(item["same_length_failed_samples"] for item in per_field.values()),
        "different_length_failed_samples": sum(item["different_length_failed_samples"] for item in per_field.values()),
        "total_substitutions": sum(item["total_substitutions"] for item in per_field.values()),
        "total_insertions": sum(item["total_insertions"] for item in per_field.values()),
        "total_deletions": sum(item["total_deletions"] for item in per_field.values()),
    }
    combined["exact_match_percentage"] = percentage(combined["exact_matches"], combined["total_samples"])
    combined["average_character_similarity"] = mean(
        float(record.get(similarity_key, 0.0)) for record in records
    ) if records else 0.0
    total_edits = combined["total_substitutions"] + combined["total_insertions"] + combined["total_deletions"]
    combined.update({
        "substitution_percentage_of_edits": percentage(combined["total_substitutions"], total_edits),
        "insertion_percentage_of_edits": percentage(combined["total_insertions"], total_edits),
        "deletion_percentage_of_edits": percentage(combined["total_deletions"], total_edits),
    })
    merged_pairs = Counter()
    for field_summary in per_field.values():
        for pair in field_summary["substitution_pairs"]:
            merged_pairs[(pair["ground_truth"], pair["prediction"])] += pair["count"]
    combined["substitution_pairs"] = [
        {"ground_truth": source, "prediction": target, "count": count}
        for (source, target), count in merged_pairs.most_common()
    ]
    all_failed.sort(key=lambda item: (-item["error_count"], item["field"], str(item["image"])))
    return {"per_field": per_field, "combined": combined}, all_failed


def print_summary(analysis: dict, examples: list[dict]) -> None:
    """Print concise evidence without choosing an MRZ strategy."""
    print("## MRZ OCR error analysis")
    print("Field | Samples | Exact | Failed | Exact % | Avg similarity | Substitutions | Insertions | Deletions")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for field_name, summary in analysis["per_field"].items():
        print(f"{field_name} | {summary['total_samples']} | {summary['exact_matches']} | {summary['failed_samples']} | {summary['exact_match_percentage']:.2f}% | {summary['average_character_similarity']:.4f} | {summary['total_substitutions']} | {summary['total_insertions']} | {summary['total_deletions']}")
    combined = analysis["combined"]
    print(f"combined | {combined['total_samples']} | {combined['exact_matches']} | {combined['failed_samples']} | {combined['exact_match_percentage']:.2f}% | {combined['average_character_similarity']:.4f} | {combined['total_substitutions']} | {combined['total_insertions']} | {combined['total_deletions']}")
    print(f"\nFailed samples with same raw length: {combined['same_length_failed_samples']}")
    print(f"Failed samples with different raw lengths: {combined['different_length_failed_samples']}")
    print(f"Edit mix: substitutions {combined['substitution_percentage_of_edits']:.2f}%, insertions {combined['insertion_percentage_of_edits']:.2f}%, deletions {combined['deletion_percentage_of_edits']:.2f}%")
    print("\n## Most common substitution pairs")
    print("Ground truth -> OCR | Count")
    print("--- | ---:")
    for pair in combined["substitution_pairs"][:20]:
        print(f"{pair['ground_truth']} -> {pair['prediction']} | {pair['count']}")
    print("\n## Representative failed examples")
    for example in examples:
        operations = ", ".join(
            f"{item['operation']}({item['ground_truth'] or '-'}->{item['prediction'] or '-'})"
            for item in example["alignment"]["operations"]
        )
        print(f"{example['field']} | {example['image']}\n  GT:   {example['ground_truth']}\n  OCR:  {example['prediction']}\n  Edit: {operations}")


def main() -> None:
    records, schema = load_latest_mrz_records()
    analysis, failed_examples = summarize(records, schema)
    report = {
        "input_file": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "schema_detected": schema,
        "alignment_method": "unit-cost global Levenshtein alignment; deterministic substitution, deletion, insertion tie-break",
        **analysis,
        "failed_samples": failed_examples,
        "representative_failed_examples": failed_examples[:REPRESENTATIVE_EXAMPLES],
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print_summary(analysis, report["representative_failed_examples"])
    print(f"\nDetailed analysis: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
