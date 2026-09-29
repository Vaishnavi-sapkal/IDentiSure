"""Build a grouped, leakage-safe SIDTD train/validation/test manifest split.

Each fake's primary source is recovered from its SIDTD filename and its optional
secondary source is ``donor_image``.  Source/donor links form connected
components: this is stricter than assigning a Crop-and-Replace fake to just
one source, and ensures neither referenced real can leak into another split.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "splits"
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
FORGERY_TYPES = ("Inpaint_and_Rewrite", "Crop_and_Replace")
FAKE_SOURCE_PATTERN = re.compile(r"^(?P<source>.+)_fake_\d+_\d+\.[^.]+$")


class UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def load_rows() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def primary_source(fake_image: str) -> str:
    """Recover the real filename from the project SIDTD fake-image convention."""
    match = FAKE_SOURCE_PATTERN.match(fake_image)
    if match is None:
        raise ValueError(f"Cannot infer primary source from fake filename: {fake_image}")
    return f"{match.group('source')}{Path(fake_image).suffix}"


def build_groups(rows: list[dict]) -> tuple[list[dict], dict[str, set[str]]]:
    """Return connected real-source groups and lineage sources for every fake."""
    real_images = {row["image"] for row in rows if row.get("label") == "real" and isinstance(row.get("image"), str)}
    union_find = UnionFind(real_images)
    fake_sources = {}
    for row in rows:
        if row.get("label") != "fake":
            continue
        image = row.get("image")
        if not isinstance(image, str):
            raise ValueError("Fake row has no string image filename.")
        sources = {primary_source(image)}
        donor = row.get("donor_image")
        if isinstance(donor, str) and donor:
            sources.add(donor)
        unknown = sources - real_images
        if unknown:
            raise ValueError(f"Fake {image} references missing real image(s): {sorted(unknown)}")
        source_list = sorted(sources)
        for source in source_list[1:]:
            union_find.union(source_list[0], source)
        fake_sources[image] = sources

    members = defaultdict(set)
    for real_image in real_images:
        members[union_find.find(real_image)].add(real_image)
    groups = {root: {"reals": reals, "rows": [], "fake_types": Counter()} for root, reals in members.items()}
    for row in rows:
        if row.get("label") == "real":
            root = union_find.find(row["image"])
        else:
            sources = fake_sources[row["image"]]
            roots = {union_find.find(source) for source in sources}
            if len(roots) != 1:
                raise AssertionError(f"Unresolved source component for fake {row['image']}")
            root = roots.pop()
            groups[root]["fake_types"][row.get("forgery_type")] += 1
        groups[root]["rows"].append(row)
    return list(groups.values()), fake_sources


def group_features(group: dict) -> Counter:
    counts = Counter({"groups": 1, "rows": len(group["rows"]), "reals": len(group["reals"]), "fakes": 0})
    for row in group["rows"]:
        if row.get("label") == "fake":
            counts["fakes"] += 1
    for forgery_type in FORGERY_TYPES:
        counts[forgery_type] = group["fake_types"][forgery_type]
    return counts


def assign_groups(groups: list[dict], seed: int) -> dict[str, list[dict]]:
    """Greedily minimize split-size and fake-type target deviations by group."""
    totals = sum((group_features(group) for group in groups), Counter())
    targets = {split: {key: totals[key] * RATIOS[split] for key in totals} for split in SPLITS}
    assignment = {split: [] for split in SPLITS}
    running = {split: Counter() for split in SPLITS}
    rng = random.Random(seed)
    ordered = groups[:]
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: (group_features(group)["fakes"], group_features(group)["rows"], len(group["reals"])), reverse=True)

    for index, group in enumerate(ordered):
        features = group_features(group)
        # Keep each split non-empty if there are enough groups.
        empty = [split for split in SPLITS if not assignment[split]]
        candidates = empty if len(ordered) - index == len(empty) else SPLITS
        def cost(split: str) -> float:
            # Score the *whole* provisional allocation, not just the candidate
            # split. This prevents a large train target from greedily absorbing
            # most groups before validation/test receive their quotas.
            provisional = {name: running[name] + (features if name == split else Counter()) for name in SPLITS}
            return sum(
                weight * ((provisional[name][key] - targets[name][key]) / max(1, targets[name][key])) ** 2
                for name in SPLITS
                for key, weight in (("Inpaint_and_Rewrite", 6), ("Crop_and_Replace", 6), ("fakes", 2), ("rows", 1), ("groups", 1), ("reals", 1))
            )
        chosen = min(candidates, key=cost)
        assignment[chosen].append(group)
        running[chosen] += features
    return assignment


def verify(rows: list[dict], fake_sources: dict[str, set[str]], row_splits: dict[str, str]) -> None:
    """Fail loudly if any source real or its derived fake crosses a split."""
    if len(row_splits) != len(rows):
        raise AssertionError("Some manifest rows were not assigned to a split.")
    for fake_image, sources in fake_sources.items():
        fake_split = row_splits[fake_image]
        for source in sources:
            if row_splits[source] != fake_split:
                raise AssertionError(f"Leakage: {fake_image} is in {fake_split}, but source {source} is in {row_splits[source]}.")


def split_summary(groups_by_split: dict[str, list[dict]]) -> dict:
    summary = {}
    for split, groups in groups_by_split.items():
        rows = [row for group in groups for row in group["rows"]]
        fakes = [row for row in rows if row.get("label") == "fake"]
        fake_types = Counter(row.get("forgery_type") for row in fakes)
        summary[split] = {
            "groups": len(groups), "rows": len(rows), "real_images": sum(row.get("label") == "real" for row in rows),
            "fakes": len(fakes), "inpaint_and_rewrite": fake_types["Inpaint_and_Rewrite"],
            "crop_and_replace": fake_types["Crop_and_Replace"],
            "crop_and_replace_percentage_of_fakes": 100 * fake_types["Crop_and_Replace"] / len(fakes) if fakes else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260929, help="Deterministic split seed.")
    args = parser.parse_args()
    rows = load_rows()
    groups, fake_sources = build_groups(rows)
    groups_by_split = assign_groups(groups, args.seed)
    row_splits = {row["image"]: split for split, groups in groups_by_split.items() for group in groups for row in group["rows"]}
    verify(rows, fake_sources, row_splits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, groups in groups_by_split.items():
        with (OUTPUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for group in groups:
                for row in group["rows"]:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = split_summary(groups_by_split)
    report = {"manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)), "seed": args.seed, "ratios": RATIOS, "grouping": "connected components over inferred fake primary source and donor_image secondary source", "verification": {"passed": True, "real_image_and_all_derived_fakes_co_located": True}, "summary": summary}
    with (OUTPUT_DIR / "split_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print("Leakage-safe grouped split verified: no source real or derived fake crosses splits.")
    print("split | groups | rows | reals | fakes | Inpaint_and_Rewrite | Crop_and_Replace | Crop % of fakes")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for split in SPLITS:
        item = summary[split]
        print(f"{split} | {item['groups']} | {item['rows']} | {item['real_images']} | {item['fakes']} | {item['inpaint_and_rewrite']} | {item['crop_and_replace']} | {item['crop_and_replace_percentage_of_fakes']:.2f}%")
    print(f"\nOutputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
