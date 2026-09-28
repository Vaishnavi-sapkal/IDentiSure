"""Measure conservative MRZ string normalization against saved EasyOCR results.

This experiment reads existing OCR results only. It never initializes EasyOCR,
changes a production preprocessing rule, or writes to the source JSONL.
"""

import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "easyocr_full_results.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mrz_normalization_experiment.json"
MRZ_FIELDS = ("mrz_line0", "mrz_line1")
EXAMPLE_LIMIT = 10


def levenshtein_distance(left: str, right: str) -> int:
    """Return unit-cost character edit distance."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    """Use the evaluator's normalized Levenshtein similarity definition."""
    maximum_length = max(len(left), len(right))
    return 1.0 if maximum_length == 0 else 1 - levenshtein_distance(left, right) / maximum_length


def remove_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def mrz_punctuation(text: str) -> str:
    """Map only punctuation visibly used as an MRZ filler by OCR output.

    ``=`` and ``-`` are not valid ICAO MRZ characters. Here they are treated
    as a filler only after whitespace removal and uppercasing; no letter/digit
    characters are modified in this conservative structural step.
    """
    return text.replace("=", "<").replace("-", "<")


def structural_normalize(text: str, level: int) -> str:
    """Apply raw, whitespace, case, or conservative punctuation normalization."""
    if level == 0:
        return text
    text = remove_whitespace(text)
    if level == 1:
        return text
    text = text.upper()
    if level == 2:
        return text
    return mrz_punctuation(text)


def make_variants() -> OrderedDict[str, dict]:
    """Define structural variants plus one directional confusion test at a time."""
    variants = OrderedDict([
        ("variant_0_raw", {"description": "No normalization.", "structural_level": 0, "replacement": None}),
        ("variant_1_whitespace", {"description": "Remove whitespace from both compared strings.", "structural_level": 1, "replacement": None}),
        ("variant_2_case", {"description": "Remove whitespace and uppercase both compared strings.", "structural_level": 2, "replacement": None}),
        ("variant_3_mrz_punctuation", {"description": "Remove whitespace, uppercase, and map OCR filler punctuation =/- to <.", "structural_level": 3, "replacement": None}),
    ])
    for source, target in (("O", "0"), ("0", "O"), ("I", "1"), ("1", "I"), ("B", "8"), ("8", "B"), ("S", "5"), ("5", "S")):
        variants[f"variant_4_{source}_to_{target}"] = {
            "description": f"Variant 3 plus OCR-only {source} -> {target}; no reverse mapping.",
            "structural_level": 3,
            "replacement": (source, target),
        }
    return variants


def normalize_pair(ground_truth: str, prediction: str, config: dict) -> tuple[str, str]:
    """Normalize the reference structurally and OCR structurally plus one test map."""
    normalized_truth = structural_normalize(ground_truth, config["structural_level"])
    normalized_prediction = structural_normalize(prediction, config["structural_level"])
    if config["replacement"] is not None:
        source, target = config["replacement"]
        normalized_prediction = normalized_prediction.replace(source, target)
    return normalized_truth, normalized_prediction


def load_records() -> tuple[list[dict], list[str]]:
    """Inspect the JSONL schema and retain the latest record for each MRZ sample."""
    observed_keys = set()
    latest = {}
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
            observed_keys.update(record)
            if record.get("field") in MRZ_FIELDS and isinstance(record.get("image"), str):
                latest[(record["image"], record["field"])] = record
    required = {"image", "field", "ground_truth", "prediction"}
    missing = required - observed_keys
    if missing:
        raise KeyError(f"Results schema lacks required key(s): {', '.join(sorted(missing))}")
    records = list(latest.values())
    if not records:
        raise LookupError(f"No records found for {', '.join(MRZ_FIELDS)}")
    return records, sorted(observed_keys)


def calculate_metrics(records: list[dict], config: dict) -> tuple[dict, list[dict]]:
    """Calculate raw-string metrics after one explicitly named normalization."""
    per_field = {}
    improved_examples = []
    for field_name in MRZ_FIELDS:
        field_records = [record for record in records if record["field"] == field_name]
        comparisons = []
        for record in field_records:
            raw_truth, raw_prediction = str(record["ground_truth"]), str(record["prediction"])
            truth, prediction = normalize_pair(raw_truth, raw_prediction, config)
            raw_exact = raw_truth == raw_prediction
            exact = truth == prediction
            comparisons.append((truth, prediction, exact))
            if not raw_exact and exact:
                improved_examples.append({
                    "field": field_name,
                    "image": record["image"],
                    "raw_ground_truth": raw_truth,
                    "raw_prediction": raw_prediction,
                    "normalized_ground_truth": truth,
                    "normalized_prediction": prediction,
                })
        total = len(comparisons)
        exact_matches = sum(item[2] for item in comparisons)
        failed = [item for item in comparisons if not item[2]]
        per_field[field_name] = {
            "total_samples": total,
            "exact_matches": exact_matches,
            "exact_match_percentage": 100 * exact_matches / total if total else 0.0,
            "average_character_similarity": mean(similarity(item[0], item[1]) for item in comparisons) if comparisons else 0.0,
            "failed_samples": len(failed),
            "same_length_failures": sum(len(item[0]) == len(item[1]) for item in failed),
            "different_length_failures": sum(len(item[0]) != len(item[1]) for item in failed),
        }
    all_comparisons = []
    for record in records:
        truth, prediction = normalize_pair(str(record["ground_truth"]), str(record["prediction"]), config)
        all_comparisons.append((truth, prediction, truth == prediction))
    total = len(all_comparisons)
    exact_matches = sum(item[2] for item in all_comparisons)
    failed = [item for item in all_comparisons if not item[2]]
    combined = {
        "total_samples": total,
        "exact_matches": exact_matches,
        "exact_match_percentage": 100 * exact_matches / total if total else 0.0,
        "average_character_similarity": mean(similarity(item[0], item[1]) for item in all_comparisons),
        "failed_samples": len(failed),
        "same_length_failures": sum(len(item[0]) == len(item[1]) for item in failed),
        "different_length_failures": sum(len(item[0]) != len(item[1]) for item in failed),
    }
    return {**per_field, "combined": combined}, improved_examples


def add_improvements(results: dict) -> None:
    """Attach raw and structural-normalization deltas for each line."""
    baseline = results["variant_0_raw"]["metrics"]
    structural = results["variant_3_mrz_punctuation"]["metrics"]
    for variant in results.values():
        for line in (*MRZ_FIELDS, "combined"):
            metrics = variant["metrics"][line]
            raw = baseline[line]
            structural_metrics = structural[line]
            metrics["exact_match_percentage_improvement_vs_raw"] = metrics["exact_match_percentage"] - raw["exact_match_percentage"]
            metrics["average_similarity_improvement_vs_raw"] = metrics["average_character_similarity"] - raw["average_character_similarity"]
            metrics["exact_match_percentage_improvement_vs_variant_3"] = metrics["exact_match_percentage"] - structural_metrics["exact_match_percentage"]
            metrics["average_similarity_improvement_vs_variant_3"] = metrics["average_character_similarity"] - structural_metrics["average_character_similarity"]


def print_summary(results: dict) -> None:
    """Print compact results and examples without prescribing a production action."""
    print("## MRZ normalization experiment")
    print("Variant | Line | Exact % | Avg similarity | Exact % improvement | Similarity improvement")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for variant_name, result in results.items():
        for line in (*MRZ_FIELDS, "combined"):
            metrics = result["metrics"][line]
            print(f"{variant_name} | {line} | {metrics['exact_match_percentage']:.2f}% | {metrics['average_character_similarity']:.4f} | {metrics['exact_match_percentage_improvement_vs_raw']:+.2f} pp | {metrics['average_similarity_improvement_vs_raw']:+.4f}")
    whitespace = results["variant_1_whitespace"]["metrics"]["combined"]
    case = results["variant_2_case"]["metrics"]["combined"]
    punctuation = results["variant_3_mrz_punctuation"]["metrics"]["combined"]
    print("\n## Incremental combined exact-match evidence")
    print(f"Whitespace (Variant 1 vs raw): {whitespace['exact_match_percentage_improvement_vs_raw']:+.2f} pp")
    print(f"Case (Variant 2 vs Variant 1): {case['exact_match_percentage'] - whitespace['exact_match_percentage']:+.2f} pp")
    print(f"Filler punctuation (Variant 3 vs Variant 2): {punctuation['exact_match_percentage'] - case['exact_match_percentage']:+.2f} pp")
    print("Individual global mappings (Variant 4 vs Variant 3):")
    for variant_name, result in results.items():
        if not variant_name.startswith("variant_4_"):
            continue
        change = result["metrics"]["combined"]["exact_match_percentage_improvement_vs_variant_3"]
        print(f"  {variant_name}: {change:+.2f} pp")
    examples = []
    for variant_name, result in results.items():
        if variant_name == "variant_0_raw":
            continue
        for example in result["raw_failures_made_exact"]:
            examples.append((variant_name, example))
    print("\n## Failed raw predictions made exact by normalization")
    for variant_name, example in examples[:EXAMPLE_LIMIT]:
        print(f"{variant_name} | {example['field']} | {example['image']}\n  GT:  {example['raw_ground_truth']}\n  OCR: {example['raw_prediction']}\n  Normalized: {example['normalized_prediction']}")
    if not examples:
        print("None.")


def main() -> None:
    records, observed_keys = load_records()
    variants = make_variants()
    results = OrderedDict()
    for variant_name, config in variants.items():
        metrics, improved_examples = calculate_metrics(records, config)
        results[variant_name] = {
            "description": config["description"],
            "metrics": metrics,
            "raw_failures_made_exact": improved_examples,
        }
    add_improvements(results)
    report = {
        "input_file": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "observed_jsonl_top_level_keys": observed_keys,
        "records_used_latest_per_image_field": len(records),
        "mrz_fields": list(MRZ_FIELDS),
        "results": results,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print_summary(results)
    print(f"\nComplete results: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
