"""Compare stored EasyOCR and PassportEye MRZ outputs with validation-only selection.

No OCR is run. Selection sees only each candidate's TD3 structure and ICAO
check-digit outcomes; ground truth is used only after selection for metrics.
"""

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = PROJECT_ROOT / "docs" / "mrz_validation_experiment.json"
PASSPORTEYE_PATH = PROJECT_ROOT / "docs" / "passporteye_mrz_experiment.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "hybrid_mrz_experiment.json"
CHECKS = ("passport_number", "date_of_birth", "date_of_expiry", "optional_data", "composite")


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, start=1):
        current = [index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (character != right_character)))
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    maximum = max(len(left), len(right))
    return 1.0 if maximum == 0 else 1 - levenshtein_distance(left, right) / maximum


def score(candidate: dict) -> int:
    """Explicit combined score: 100 for valid TD3 structure plus 0--5 checks."""
    return (100 if candidate["valid_td3"] else 0) + candidate["valid_check_count"]


def choose(left: dict, right: dict, rule: str) -> str:
    """Return easyocr/passporteye/tie without consulting ground truth."""
    if rule == "more_check_digits":
        left_key, right_key = left["valid_check_count"], right["valid_check_count"]
    elif rule == "structure_first":
        left_key = (int(left["valid_td3"]), left["valid_check_count"])
        right_key = (int(right["valid_td3"]), right["valid_check_count"])
    else:
        left_key, right_key = score(left), score(right)
    return "easyocr" if left_key > right_key else "passporteye" if right_key > left_key else "tie"


def build_candidate(record: dict, prefix: str, truth0: str, truth1: str) -> dict:
    checks = record[f"{prefix}_checks"]
    valid_checks = {name: bool(checks.get(name, False)) for name in CHECKS}
    line0, line1 = record[f"{prefix}_line0"], record[f"{prefix}_line1"]
    return {
        "line0": line0,
        "line1": line1,
        "full": record[f"{prefix}_full"],
        "valid_td3": bool(record[f"{prefix}_valid_td3"]),
        "valid_length": bool(record[f"{prefix}_valid_length"]),
        "invalid_characters": bool(record[f"{prefix}_invalid_characters"]),
        "checks": valid_checks,
        "valid_check_count": sum(valid_checks.values()),
        "full_exact": line0 == truth0 and line1 == truth1,
        "line0_exact": line0 == truth0,
        "line1_exact": line1 == truth1,
        "full_similarity": similarity(record[f"{prefix}_full"], truth0 + truth1),
        "line0_similarity": similarity(line0, truth0),
        "line1_similarity": similarity(line1, truth1),
    }


def metrics(candidates: list[dict]) -> dict:
    """Calculate identical evaluation fields for an engine or a selected subset."""
    total = len(candidates)
    if not total:
        return {"samples": 0, "full_mrz_exact": 0, "full_mrz_exact_percentage": 0.0}
    output = {
        "samples": total,
        "full_mrz_exact": sum(item["full_exact"] for item in candidates),
        "line0_exact": sum(item["line0_exact"] for item in candidates),
        "line1_exact": sum(item["line1_exact"] for item in candidates),
        "average_full_mrz_similarity": sum(item["full_similarity"] for item in candidates) / total,
        "average_line0_similarity": sum(item["line0_similarity"] for item in candidates) / total,
        "average_line1_similarity": sum(item["line1_similarity"] for item in candidates) / total,
        "valid_td3_pairs": sum(item["valid_td3"] for item in candidates),
        "invalid_length": sum(not item["valid_length"] for item in candidates),
        "invalid_characters": sum(item["invalid_characters"] for item in candidates),
        "all_five_checks_valid": sum(item["valid_check_count"] == 5 for item in candidates),
    }
    for name in CHECKS:
        output[f"valid_{name}"] = sum(item["checks"][name] for item in candidates)
    for name in ("full_mrz_exact", "line0_exact", "line1_exact"):
        output[f"{name}_percentage"] = 100 * output[name] / total
    return output


def hybrid_metrics(records: list[dict], rule: str, tie_policy: str | None = None) -> dict:
    selected = []
    decisions = Counter()
    for record in records:
        choice = record["choices"][rule]
        decisions[choice] += 1
        if choice == "tie" and tie_policy is None:
            continue
        selected.append(record[tie_policy if choice == "tie" else choice])
    output = metrics(selected)
    output.update({"easyocr_selected": decisions["easyocr"], "passporteye_selected": decisions["passporteye"], "ties": decisions["tie"], "evaluated_non_tied_samples": len(selected)})
    return output


def example(record: dict) -> dict:
    return {
        "image": record["image"], "ground_truth": record["truth0"] + record["truth1"],
        "easyocr": {key: record["easyocr"][key] for key in ("full", "valid_td3", "valid_check_count", "checks", "full_exact")},
        "passporteye": {key: record["passporteye"][key] for key in ("full", "valid_td3", "valid_check_count", "checks", "full_exact")},
        "combined_scores": {"easyocr": score(record["easyocr"]), "passporteye": score(record["passporteye"])},
        "hybrid_choice": record["choices"]["combined_score"],
    }


def main() -> None:
    with VALIDATION_PATH.open("r", encoding="utf-8") as validation_file:
        validation = json.load(validation_file)
    with PASSPORTEYE_PATH.open("r", encoding="utf-8") as passporteye_file:
        passporteye_report = json.load(passporteye_file)
    validation_samples = validation.get("normalized_ocr", {}).get("samples")
    passporteye_records = passporteye_report.get("per_image_results")
    if not isinstance(validation_samples, list) or not isinstance(passporteye_records, list):
        raise ValueError("Required validation/report record arrays are missing.")
    # Schema is inspected and PassportEye's stored joined truth is used only for evaluation.
    schema = {
        "validation_top_level_keys": sorted(validation),
        "validation_normalized_ocr_keys": sorted(validation.get("normalized_ocr", {})),
        "passporteye_top_level_keys": sorted(passporteye_report),
        "passporteye_per_image_keys": sorted(passporteye_records[0]) if passporteye_records else [],
    }
    validation_by_image = {sample["filename"]: sample for sample in validation_samples if isinstance(sample.get("filename"), str)}
    records = []
    for source in passporteye_records:
        image = source.get("image")
        validation_sample = validation_by_image.get(image)
        if validation_sample is None:
            continue
        truth0, truth1 = source.get("ground_truth_line0"), source.get("ground_truth_line1")
        if not all(isinstance(value, str) for value in (truth0, truth1)):
            continue
        # Reuse EasyOCR strings from PassportEye's stored same-sample join;
        # verify they agree with the validation report's normalized values.
        if source.get("easyocr_line0") != validation_sample.get("line0") or source.get("easyocr_line1") != validation_sample.get("line1"):
            raise ValueError(f"Stored EasyOCR output mismatch for {image}.")
        easyocr = build_candidate(source, "easyocr", truth0, truth1)
        passporteye = build_candidate(source, "passporteye", truth0, truth1)
        records.append({
            "image": image, "truth0": truth0, "truth1": truth1,
            "easyocr": easyocr, "passporteye": passporteye,
            "choices": {rule: choose(easyocr, passporteye, rule) for rule in ("more_check_digits", "structure_first", "combined_score")},
        })
    if len(records) != len(validation_samples):
        raise ValueError(f"Expected {len(validation_samples)} matched samples; found {len(records)}.")

    easyocr_result = metrics([record["easyocr"] for record in records])
    passporteye_result = metrics([record["passporteye"] for record in records])
    hybrid_results = {rule: hybrid_metrics(records, rule) for rule in ("more_check_digits", "structure_first", "combined_score")}
    hybrid_tie_easyocr = {rule: hybrid_metrics(records, rule, tie_policy="easyocr") for rule in hybrid_results}
    agreement = sum(record["easyocr"]["full"] == record["passporteye"]["full"] for record in records)
    combined_choices = Counter(record["choices"]["combined_score"] for record in records)
    validation_signal = Counter()
    selected_exact = Counter()
    for record in records:
        choice = record["choices"]["combined_score"]
        if choice == "tie":
            continue
        chosen_exact = record[choice]["full_exact"]
        selected_exact[f"{choice}_selected_exact"] += chosen_exact
        weaker = "passporteye" if choice == "easyocr" else "easyocr"
        validation_signal["stronger_exact"] += chosen_exact
        validation_signal["weaker_exact"] += record[weaker]["full_exact"]
        validation_signal["compared"] += 1
    cases = {
        "easyocr_exact_passporteye_wrong": [example(record) for record in records if record["easyocr"]["full_exact"] and not record["passporteye"]["full_exact"]][:5],
        "passporteye_exact_easyocr_wrong": [example(record) for record in records if record["passporteye"]["full_exact"] and not record["easyocr"]["full_exact"]][:5],
        "both_wrong": [example(record) for record in records if not record["easyocr"]["full_exact"] and not record["passporteye"]["full_exact"]][:5],
        "both_exact": [example(record) for record in records if record["easyocr"]["full_exact"] and record["passporteye"]["full_exact"]][:5],
        "validation_selects_wrong_engine": [example(record) for record in records if record["choices"]["combined_score"] != "tie" and not record[record["choices"]["combined_score"]]["full_exact"] and record["passporteye" if record["choices"]["combined_score"] == "easyocr" else "easyocr"]["full_exact"]][:5],
    }
    matrix = Counter()
    for record in records:
        left, right = record["easyocr"], record["passporteye"]
        relation = "equal" if score(left) == score(right) else "easyocr_stronger" if score(left) > score(right) else "passporteye_stronger"
        left_status = f"{'valid TD3' if left['valid_td3'] else 'invalid TD3'} + {left['valid_check_count']}/5 checks"
        right_status = f"{'valid TD3' if right['valid_td3'] else 'invalid TD3'} + {right['valid_check_count']}/5 checks"
        matrix[(left_status, right_status, record["choices"]["combined_score"])] += 1
    report = {
        "configuration": {"uses_existing_results_only": True, "no_ground_truth_in_selection": True, "combined_score_formula": "100 if valid TD3 structure, plus valid check-digit count (0 through 5)", "tie_policy_main": "tie; no candidate selected", "tie_policy_secondary": "EasyOCR only for a labeled secondary analysis"},
        "input_schema_detected": schema,
        "samples": len(records),
        "strategies": {"easyocr": easyocr_result, "passporteye": passporteye_result, "hybrid": hybrid_results, "hybrid_tie_to_easyocr_secondary": hybrid_tie_easyocr},
        "decision_statistics": {"combined_score": {"easyocr_selected": combined_choices["easyocr"], "passporteye_selected": combined_choices["passporteye"], "ties": combined_choices["tie"], "engine_full_output_agreement": agreement, "hybrid_changed_from_easyocr": combined_choices["passporteye"], "hybrid_changed_from_passporteye": combined_choices["easyocr"], "easyocr_selected_exact": selected_exact["easyocr_selected_exact"], "passporteye_selected_exact": selected_exact["passporteye_selected_exact"], "hybrid_exact_non_ties": selected_exact["easyocr_selected_exact"] + selected_exact["passporteye_selected_exact"]}, "validation_signal_on_non_ties": dict(validation_signal)},
        "decision_matrix": [{"easyocr_validation": left_status, "passporteye_validation": right_status, "hybrid_choice": choice, "count": count} for (left_status, right_status, choice), count in sorted(matrix.items())],
        "disagreement_examples": cases,
        "per_image_results": records,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print("## Hybrid MRZ Benchmark")
    print(f"samples: {len(records)}")
    for name, result in (("EasyOCR", easyocr_result), ("PassportEye", passporteye_result)):
        print(f"\n=== {name} ===\nfull_mrz_exact: {result['full_mrz_exact']}\nfull_mrz_exact_percentage: {result['full_mrz_exact_percentage']:.2f}\navg_full_similarity: {result['average_full_mrz_similarity']:.4f}\nvalid_td3_pairs: {result['valid_td3_pairs']}\nall_five_checks_valid: {result['all_five_checks_valid']}")
    for name, rule in (("More Check Digits", "more_check_digits"), ("Structure First", "structure_first"), ("Combined Score", "combined_score")):
        result = hybrid_results[rule]
        print(f"\n=== Hybrid: {name} (non-ties only) ===\nfull_mrz_exact: {result['full_mrz_exact']}\nfull_mrz_exact_percentage: {result['full_mrz_exact_percentage']:.2f}")
    print(f"\n=== Decisions (Combined Score) ===\neasyocr_selected: {combined_choices['easyocr']}\npassporteye_selected: {combined_choices['passporteye']}\nties: {combined_choices['tie']}\neasyocr_passporteye_agreement: {agreement}\nReport: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"easyocr_selected_exact: {selected_exact['easyocr_selected_exact']}\npassporteye_selected_exact: {selected_exact['passporteye_selected_exact']}\nhybrid_exact_non_ties: {selected_exact['easyocr_selected_exact'] + selected_exact['passporteye_selected_exact']}")
    for category, examples in cases.items():
        print(f"\n{category}: {len(examples)} representative example(s)")
        for item in examples:
            print(f"  {item['image']} | choice={item['hybrid_choice']} | EasyOCR exact={item['easyocr']['full_exact']} | PassportEye exact={item['passporteye']['full_exact']}")


if __name__ == "__main__":
    main()
