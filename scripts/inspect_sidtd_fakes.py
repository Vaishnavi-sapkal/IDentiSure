import json
from collections import Counter
from pathlib import Path


FAKE_ANNOTATIONS = Path(
    "datasets/sidtd/templates/templates/Annotations/fakes"
)

OUTPUT_FILE = Path("docs/sidtd-manifest-notes.md")


ctype_counts = Counter()
field_counts = Counter()

second_src_count = 0
second_field_count = 0
total_files = 0

second_src_examples = []
second_field_examples = []


for json_file in sorted(FAKE_ANNOTATIONS.glob("*_fake_*.json")):
    total_files += 1

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ctype = data.get("ctype", "MISSING")
    field = data.get("field", "None")
    second_src = data.get("second_src", "None")
    second_field = data.get("second_field", "None")

    ctype_counts[ctype] += 1
    field_counts[field] += 1

    if second_src != "None":
        second_src_count += 1

        if len(second_src_examples) < 5:
            second_src_examples.append(
                (json_file.name, second_src)
            )

    if second_field != "None":
        second_field_count += 1

        if len(second_field_examples) < 5:
            second_field_examples.append(
                (json_file.name, second_field)
            )


# -------------------------
# Console output
# -------------------------

print("=" * 60)
print("SIDTD FAKE ANNOTATION ANALYSIS")
print("=" * 60)

print(f"\nTotal fake JSON files: {total_files}")

print("\nForgery types (ctype):")
for ctype, count in ctype_counts.most_common():
    print(f"  {ctype}: {count}")

print("\nManipulated fields:")
for field, count in field_counts.most_common():
    print(f"  {field}: {count}")

print("\nSecond source usage:")
print(f"  second_src != None: {second_src_count}")

print("\nSecond field usage:")
print(f"  second_field != None: {second_field_count}")

if second_src_examples:
    print("\nExample second_src records:")
    for filename, value in second_src_examples:
        print(f"  {filename} -> {value}")

if second_field_examples:
    print("\nExample second_field records:")
    for filename, value in second_field_examples:
        print(f"  {filename} -> {value}")


# -------------------------
# Markdown report
# -------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("# SIDTD Dataset Analysis\n\n")

    f.write("## Fake Annotation Summary\n\n")
    f.write(f"- Total fake annotation files: **{total_files}**\n")
    f.write(
        f"- Records with `second_src != None`: "
        f"**{second_src_count}**\n"
    )
    f.write(
        f"- Records with `second_field != None`: "
        f"**{second_field_count}**\n\n"
    )

    f.write("## Forgery Types (`ctype`)\n\n")
    f.write("| Forgery Type | Count |\n")
    f.write("|---|---:|\n")

    for ctype, count in ctype_counts.most_common():
        f.write(f"| `{ctype}` | {count} |\n")

    f.write("\n## Manipulated Fields\n\n")
    f.write("| Field | Count |\n")
    f.write("|---|---:|\n")

    for field, count in field_counts.most_common():
        f.write(f"| `{field}` | {count} |\n")

    f.write("\n## Second Source Usage\n\n")

    if second_src_count == 0:
        f.write(
            "No fake annotations contain a `second_src` value "
            "other than `None`.\n"
        )
    else:
        f.write(
            f"{second_src_count} fake annotations contain a "
            "`second_src` value.\n\n"
        )

        f.write("Examples:\n\n")

        for filename, value in second_src_examples:
            f.write(f"- `{filename}` → `{value}`\n")

    f.write("\n## Second Field Usage\n\n")

    if second_field_count == 0:
        f.write(
            "No fake annotations contain a `second_field` value "
            "other than `None`.\n"
        )
    else:
        f.write(
            f"{second_field_count} fake annotations contain a "
            "`second_field` value.\n\n"
        )

        f.write("Examples:\n\n")

        for filename, value in second_field_examples:
            f.write(f"- `{filename}` → `{value}`\n")


print(f"\nReport written to: {OUTPUT_FILE}")