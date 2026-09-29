"""Analyze existing validation-score ties without rerunning either OCR system."""

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HYBRID_PATH = PROJECT_ROOT / "docs" / "hybrid_mrz_experiment.json"
VALIDATION_PATH = PROJECT_ROOT / "docs" / "mrz_validation_experiment.json"
NORMALIZATION_PATH = PROJECT_ROOT / "docs" / "mrz_normalization_experiment.json"
PASSPORTEYE_PATH = PROJECT_ROOT / "docs" / "passporteye_mrz_experiment.json"
RESULTS_PATH = PROJECT_ROOT / "docs" / "easyocr_full_results.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mrz_tie_breaker_analysis.json"
MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
CHECKS = ("passport_number", "date_of_birth", "date_of_expiry", "optional_data", "composite")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pattern(candidate: dict) -> str:
    return "".join("V" if candidate["checks"][name] else "I" for name in CHECKS)


def structural_properties(candidate: dict, raw: str) -> dict:
    """Calculate only observable string/structure quality signals."""
    line0, line1, normalized = candidate["line0"], candidate["line1"], candidate["full"]
    line0_length, line1_length = len(line0), len(line1)
    raw_non_mrz = sum(character not in MRZ_ALLOWED and not character.isspace() for character in raw.upper())
    return {
        "total_mrz_length": len(normalized), "line_count": 2 if line0 or line1 else 0,
        "line0_length": line0_length, "line1_length": line1_length,
        "line0_is_44": line0_length == 44, "line1_is_44": line1_length == 44,
        "valid_td3_length": candidate["valid_length"], "valid_td3_structure": candidate["valid_td3"],
        "invalid_character_count": sum(character not in MRZ_ALLOWED for character in normalized),
        "length_deviation_from_88": abs(len(normalized) - 88),
        "line_length_deviation": abs(line0_length - 44) + abs(line1_length - 44),
        "filler_count": normalized.count("<"), "spaces_after_normalization": normalized.count(" "),
        "non_mrz_characters_before_normalization": raw_non_mrz, "both_lines_present": bool(line0 and line1),
    }


def position_agreement(left: str, right: str) -> dict:
    """Compare only aligned positions, retaining length mismatch explicitly."""
    overlap = min(len(left), len(right))
    matches = sum(left[index] == right[index] for index in range(overlap))
    return {"overlap_length": overlap, "differing_characters": overlap - matches, "agreement_percentage": 100 * matches / overlap if overlap else 0.0, "lengths_allow_full_position_comparison": len(left) == len(right)}


def rule_choice(left: dict, right: dict, rule: str) -> str:
    """Return a deterministic hypothetical decision or tie; never use truth."""
    if rule == "A_structural_validity":
        return "easyocr" if left["structure"]["valid_td3_structure"] and not right["structure"]["valid_td3_structure"] else "passporteye" if right["structure"]["valid_td3_structure"] and not left["structure"]["valid_td3_structure"] else "tie"
    if rule == "B_invalid_characters":
        # Per specification, only use invalid count after equal structure status.
        if left["structure"]["valid_td3_structure"] != right["structure"]["valid_td3_structure"]:
            return "tie"
        return "easyocr" if left["structure"]["invalid_character_count"] < right["structure"]["invalid_character_count"] else "passporteye" if right["structure"]["invalid_character_count"] < left["structure"]["invalid_character_count"] else "tie"
    if rule == "C_check_status_dominance":
        left_checks, right_checks = left["checks"], right["checks"]
        left_dominates = all(not right_checks[name] or left_checks[name] for name in CHECKS) and any(left_checks[name] and not right_checks[name] for name in CHECKS)
        right_dominates = all(not left_checks[name] or right_checks[name] for name in CHECKS) and any(right_checks[name] and not left_checks[name] for name in CHECKS)
        return "easyocr" if left_dominates else "passporteye" if right_dominates else "tie"
    if rule == "D_combined_score":
        left_score = (100 if left["structure"]["valid_td3_structure"] else 0) + left["valid_check_count"]
        right_score = (100 if right["structure"]["valid_td3_structure"] else 0) + right["valid_check_count"]
        return "easyocr" if left_score > right_score else "passporteye" if right_score > left_score else "tie"
    # Rule E: lexicographic format quality: valid TD3, lower length deviation,
    # lower invalid count, lower line deviation, lower pre-normalization noise.
    left_key = (int(left["structure"]["valid_td3_structure"]), -left["structure"]["length_deviation_from_88"], -left["structure"]["invalid_character_count"], -left["structure"]["line_length_deviation"], -left["structure"]["non_mrz_characters_before_normalization"])
    right_key = (int(right["structure"]["valid_td3_structure"]), -right["structure"]["length_deviation_from_88"], -right["structure"]["invalid_character_count"], -right["structure"]["line_length_deviation"], -right["structure"]["non_mrz_characters_before_normalization"])
    return "easyocr" if left_key > right_key else "passporteye" if right_key > left_key else "tie"


def evaluate_rule(ties: list[dict], rule: str) -> dict:
    choices = Counter(rule_choice(item["easyocr"], item["passporteye"], rule) for item in ties)
    chosen = [(item, rule_choice(item["easyocr"], item["passporteye"], rule)) for item in ties]
    resolved = [(item, choice) for item, choice in chosen if choice != "tie"]
    exact_easy = sum(item["easyocr"]["full_exact"] for item, choice in resolved if choice == "easyocr")
    exact_passport = sum(item["passporteye"]["full_exact"] for item, choice in resolved if choice == "passporteye")
    exact = exact_easy + exact_passport
    return {"resolved_cases": len(resolved), "easyocr_selected": choices["easyocr"], "passporteye_selected": choices["passporteye"], "remaining_ties": choices["tie"], "easyocr_selected_exact": exact_easy, "passporteye_selected_exact": exact_passport, "resolved_exact": exact, "resolved_exact_percentage": 100 * exact / len(resolved) if resolved else 0.0}


def representative(ties: list[dict], predicate) -> list[dict]:
    return [{"image": item["image"], "ground_truth": item["ground_truth"], "easyocr": item["easyocr"], "passporteye": item["passporteye"]} for item in ties if predicate(item)][:5]


def main() -> None:
    hybrid, validation, normalization, passporteye = (read_json(path) for path in (HYBRID_PATH, VALIDATION_PATH, NORMALIZATION_PATH, PASSPORTEYE_PATH))
    records = hybrid.get("per_image_results")
    if not isinstance(records, list):
        raise ValueError("Hybrid report lacks per_image_results.")
    passport_by_image = {record.get("image"): record for record in passporteye.get("per_image_results", []) if isinstance(record.get("image"), str)}
    # Latest saved EasyOCR raw predictions by image/line; no OCR call occurs.
    easy_raw = {}
    with RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        for line in results_file:
            if line.strip():
                result = json.loads(line)
                if result.get("field") in ("mrz_line0", "mrz_line1") and isinstance(result.get("image"), str):
                    easy_raw[(result["image"], result["field"])] = str(result.get("prediction", ""))
    tie_records = [record for record in records if record.get("choices", {}).get("combined_score") == "tie"]
    if len(tie_records) != 147:
        raise ValueError(f"Expected exactly 147 combined-score ties; found {len(tie_records)}.")
    ties = []
    for record in tie_records:
        image = record["image"]
        stored = passport_by_image.get(image)
        if stored is None:
            raise ValueError(f"PassportEye stored result missing for tie image: {image}")
        truth0, truth1 = record["truth0"], record["truth1"]
        engines = {}
        for name in ("easyocr", "passporteye"):
            original = record[name]
            raw = (easy_raw.get((image, "mrz_line0"), "") + "\n" + easy_raw.get((image, "mrz_line1"), "")) if name == "easyocr" else str(stored.get("passporteye_raw", ""))
            candidate = {"raw_mrz": raw, "normalized_mrz": original["full"], "line0": original["line0"], "line1": original["line1"], "checks": original["checks"], "valid_check_count": original["valid_check_count"], "full_exact": original["full_exact"]}
            candidate["passporteye_valid_score"] = stored.get("passporteye_details", {}).get("valid_score") if name == "passporteye" else None
            candidate["structure"] = structural_properties(original, raw)
            engines[name] = candidate
        left, right = engines["easyocr"], engines["passporteye"]
        exact_agreement = left["normalized_mrz"] == right["normalized_mrz"]
        line0_same, line1_same = left["line0"] == right["line0"], left["line1"] == right["line1"]
        ties.append({"image": image, "ground_truth": truth0 + truth1, **engines, "check_pattern_same": pattern(left) == pattern(right), "structural_status_same": left["structure"]["valid_td3_structure"] == right["structure"]["valid_td3_structure"], "mrz_agreement": {"identical": exact_agreement, "full": position_agreement(left["normalized_mrz"], right["normalized_mrz"]), "line0": position_agreement(left["line0"], right["line0"]), "line1": position_agreement(left["line1"], right["line1"]), "category": "identical" if exact_agreement else "line0_only" if not line0_same and line1_same else "line1_only" if line0_same and not line1_same else "both_lines"}})

    same_pattern = sum(item["check_pattern_same"] for item in ties)
    same_count_different_pattern = sum(item["easyocr"]["valid_check_count"] == item["passporteye"]["valid_check_count"] and not item["check_pattern_same"] for item in ties)
    same_structure_and_pattern = sum(item["check_pattern_same"] and item["structural_status_same"] for item in ties)
    outcomes = Counter((item["easyocr"]["full_exact"], item["passporteye"]["full_exact"]) for item in ties)
    agreement = Counter(item["mrz_agreement"]["category"] for item in ties)
    signal_comparisons = {}
    signals = {"td3_structural_validity": lambda x: x["structure"]["valid_td3_structure"], "invalid_character_count": lambda x: x["structure"]["invalid_character_count"], "length_deviation": lambda x: x["structure"]["length_deviation_from_88"], "check_status_pattern": pattern, "filler_count": lambda x: x["structure"]["filler_count"], "line0_structural_validity": lambda x: x["structure"]["line0_is_44"], "line1_structural_validity": lambda x: x["structure"]["line1_is_44"], "normalized_output_agreement": lambda x: x["normalized_mrz"], "passporteye_valid_score": lambda x: x.get("passporteye_valid_score")}
    for name, getter in signals.items():
        differing = sum(getter(item["easyocr"]) != getter(item["passporteye"]) for item in ties)
        signal_comparisons[name] = {"differs": differing, "equal": len(ties) - differing, "stronger_engine": "not_applicable_for_pattern_or_agreement" if name in {"check_status_pattern", "normalized_output_agreement"} else "reported per-image; no correctness inference"}
    signal_comparisons["passporteye_valid_score"] = {"differs": "not comparable: PassportEye-only quality field", "equal": "not comparable", "available_samples": sum(item["passporteye"].get("passporteye_valid_score") is not None for item in ties), "stronger_engine": "not comparable to EasyOCR; no saved EasyOCR confidence"}
    # PassportEye exposes ``valid_score`` in its saved details. It is recorded
    # as a quality value, not assumed to be a calibrated OCR confidence.
    confidence = {"passporteye_confidence_available": any(item["passporteye"].get("passporteye_valid_score") is not None for item in ties), "passporteye_valid_score_available_in_stored_details": sum(item["passporteye"].get("passporteye_valid_score") is not None for item in ties), "easyocr_confidence_available": False}
    rules = {name: evaluate_rule(ties, name) for name in ("A_structural_validity", "B_invalid_characters", "C_check_status_dominance", "D_combined_score", "E_lexical_format_quality")}
    oracle = sum(item["easyocr"]["full_exact"] or item["passporteye"]["full_exact"] for item in ties)
    examples = {"easyocr_exact_passporteye_wrong": representative(ties, lambda x: x["easyocr"]["full_exact"] and not x["passporteye"]["full_exact"]), "passporteye_exact_easyocr_wrong": representative(ties, lambda x: x["passporteye"]["full_exact"] and not x["easyocr"]["full_exact"]), "both_exact": representative(ties, lambda x: x["easyocr"]["full_exact"] and x["passporteye"]["full_exact"]), "both_wrong": representative(ties, lambda x: not x["easyocr"]["full_exact"] and not x["passporteye"]["full_exact"]), "different_check_status_patterns": representative(ties, lambda x: not x["check_pattern_same"]), "same_check_status_patterns": representative(ties, lambda x: x["check_pattern_same"])}
    report = {"configuration": {"analysis_only": True, "tie_definition": "existing hybrid per_image_results choices.combined_score == tie", "ground_truth_used_only_after_signal_calculation": True, "no_ocr_rerun": True}, "input_schema_detected": {"hybrid_keys": sorted(hybrid), "validation_keys": sorted(validation), "normalization_keys": sorted(normalization), "passporteye_keys": sorted(passporteye)}, "total_samples": len(records), "tie_cases": len(ties), "per_image_tie_analysis": ties, "validation_pattern_statistics": {"same_check_count_different_pattern": same_count_different_pattern, "same_check_status_pattern": same_pattern, "same_structural_status_and_check_pattern": same_structure_and_pattern, "different_structural_status": sum(not item["structural_status_same"] for item in ties)}, "mrz_agreement_statistics": {"identical_normalized_mrz": agreement["identical"], "different_line0_only": agreement["line0_only"], "different_line1_only": agreement["line1_only"], "different_both_lines": agreement["both_lines"]}, "signal_comparisons": signal_comparisons, "confidence_availability": confidence, "retrospective_outcomes": {"easyocr_exact_passporteye_wrong": outcomes[(True, False)], "passporteye_exact_easyocr_wrong": outcomes[(False, True)], "both_exact": outcomes[(True, True)], "both_wrong": outcomes[(False, False)], "easyocr_exact_percentage": 100 * (outcomes[(True, False)] + outcomes[(True, True)]) / len(ties), "passporteye_exact_percentage": 100 * (outcomes[(False, True)] + outcomes[(True, True)]) / len(ties)}, "hypothetical_tie_breakers": rules, "oracle_upper_bound_not_available_to_production": {"best_possible_exact_count": oracle, "best_possible_exact_percentage": 100 * oracle / len(ties)}, "representative_examples": examples, "overall_finding": {"useful_ground_truth_independent_signal": "yes" if any(rules[name]["resolved_exact_percentage"] >= 80 and rules[name]["resolved_cases"] >= 20 for name in rules) else "no", "basis": "A signal is marked yes only when it resolves at least 20 ties with >=80% retrospective exactness; this is a reported threshold, not a production decision."}}
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    o = report["retrospective_outcomes"]; p = report["validation_pattern_statistics"]; a = report["mrz_agreement_statistics"]
    print("## MRZ Tie-Breaker Analysis\n")
    print(f"total_samples: {len(records)}\ntie_cases: {len(ties)}")
    print(f"\n=== Tie Outcomes ===\neasyocr_exact: {o['easyocr_exact_passporteye_wrong'] + o['both_exact']}\npassporteye_exact: {o['passporteye_exact_passporteye_wrong'] if 'passporteye_exact_passporteye_wrong' in o else o['passporteye_exact_easyocr_wrong'] + o['both_exact']}\nboth_exact: {o['both_exact']}\nboth_wrong: {o['both_wrong']}")
    print(f"\n=== Validation Pattern ===\nsame_check_count_different_pattern: {p['same_check_count_different_pattern']}\nsame_check_status_pattern: {p['same_check_status_pattern']}\ndifferent_structural_status: {p['different_structural_status']}")
    print(f"\n=== MRZ Agreement ===\nidentical_normalized_mrz: {a['identical_normalized_mrz']}\ndifferent_line0_only: {a['different_line0_only']}\ndifferent_line1_only: {a['different_line1_only']}\ndifferent_both_lines: {a['different_both_lines']}")
    print("\n=== Hypothetical Tie-Breakers ===")
    for label, name in (("A — Structural validity", "A_structural_validity"), ("B — Invalid character count", "B_invalid_characters"), ("C — Check-status dominance", "C_check_status_dominance"), ("D — Combined score", "D_combined_score")):
        r = rules[name]; print(f"\nRule {label}\nresolved: {r['resolved_cases']}\nremaining: {r['remaining_ties']}\nexact: {r['resolved_exact']}\nexact_percentage: {r['resolved_exact_percentage']:.2f}")
    print(f"\n=== Overall Finding ===\nuseful_ground_truth_independent_signal:\n{report['overall_finding']['useful_ground_truth_independent_signal']}\n\nReport:\n{OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("\n## Representative examples")
    for category, items in examples.items():
        print(f"{category}: {len(items)}")
        for item in items:
            print(f"  {item['image']} | EasyOCR checks={pattern(item['easyocr'])} | PassportEye checks={pattern(item['passporteye'])}")


if __name__ == "__main__":
    main()
