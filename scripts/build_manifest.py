import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIDTD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sidtd"
    / "templates"
    / "templates"
)

REAL_IMAGES_DIR = SIDTD_ROOT / "Images" / "reals"
FAKE_IMAGES_DIR = SIDTD_ROOT / "Images" / "fakes"

REAL_ANNOTATIONS_DIR = SIDTD_ROOT / "Annotations" / "reals"
FAKE_ANNOTATIONS_DIR = SIDTD_ROOT / "Annotations" / "fakes"

OUTPUT_FILE = PROJECT_ROOT / "docs" / "manifest.jsonl"
VALIDATION_FILE = PROJECT_ROOT / "docs" / "manifest_validation.json"


# ============================================================
# HELPERS
# ============================================================

def normalize_real_filename(document_type, filename):
    """
    SIDTD real annotation files store filenames such as:

        23.jpg

    while the actual image is:

        alb_id_23.jpg

    So we stitch the annotation file's template prefix
    back onto the filename.
    """

    return f"{document_type}_{filename}"


def bbox_from_region(region):
    """
    Extract a bounding box from a VIA annotation region.
    """

    shape = region.get("shape_attributes", {})

    if shape.get("name") != "rect":
        return None

    required = ["x", "y", "width", "height"]

    if not all(key in shape for key in required):
        return None

    return {
        "x": shape["x"],
        "y": shape["y"],
        "width": shape["width"],
        "height": shape["height"],
    }


# ============================================================
# STEP 1
# BUILD REAL ANNOTATION LOOKUP
# ============================================================

def build_real_lookup():
    """
    Build:

        {
            "alb_id_23.jpg": {
                "name": {
                    "x": 704,
                    "y": 365,
                    "width": 160,
                    "height": 48,
                    "value": "Arbër"
                }
            }
        }
    """

    lookup = {}

    annotation_files = sorted(REAL_ANNOTATIONS_DIR.glob("*.json"))

    print("=" * 70)
    print("STEP 1: BUILDING REAL ANNOTATION LOOKUP")
    print("=" * 70)

    print(f"Real annotation files: {len(annotation_files)}")

    if not annotation_files:
        raise FileNotFoundError(
            f"No real annotation files found in: {REAL_ANNOTATIONS_DIR}"
        )

    for annotation_file in annotation_files:

        # Example:
        # alb_id.json -> alb_id
        document_type = annotation_file.stem

        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        metadata = data.get("_via_img_metadata", {})

        for record in metadata.values():

            original_filename = record.get("filename")

            if not original_filename:
                continue

            actual_filename = normalize_real_filename(
                document_type,
                original_filename
            )

            fields = lookup.setdefault(actual_filename, {})

            for region in record.get("regions", []):

                attributes = region.get("region_attributes", {})

                field_name = attributes.get("field_name")

                if not field_name:
                    continue

                bbox = bbox_from_region(region)

                if bbox is None:
                    continue

                bbox["value"] = attributes.get("value")

                fields[field_name] = bbox

    print(f"Real images represented in lookup: {len(lookup)}")

    total_fields = sum(len(fields) for fields in lookup.values())

    print(f"Total annotated fields: {total_fields}")

    return lookup


# ============================================================
# STEP 2
# PROCESS FAKE ANNOTATIONS
# ============================================================

def build_fake_rows(real_lookup):
    """
    Convert all 1222 fake annotation JSON files into manifest rows.
    """

    print()
    print("=" * 70)
    print("STEP 2: PROCESSING FAKE ANNOTATIONS")
    print("=" * 70)

    fake_annotation_files = sorted(FAKE_ANNOTATIONS_DIR.glob("*.json"))

    print(f"Fake annotation files: {len(fake_annotation_files)}")

    rows = []

    errors = []

    forgery_counter = Counter()

    for fake_annotation_file in fake_annotation_files:

        try:
            with fake_annotation_file.open(
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            fake_name = data.get("name")
            forgery_type = data.get("ctype")
            source_image = data.get("src")
            source_field = data.get("field")

            if not fake_name:
                errors.append(
                    f"{fake_annotation_file.name}: missing name"
                )
                continue

            if not forgery_type:
                errors.append(
                    f"{fake_annotation_file.name}: missing ctype"
                )
                continue

            if not source_image:
                errors.append(
                    f"{fake_annotation_file.name}: missing src"
                )
                continue

            if not source_field:
                errors.append(
                    f"{fake_annotation_file.name}: missing field"
                )
                continue

            fake_image = f"{fake_name}.jpg"

            source_fields = real_lookup.get(source_image)

            if source_fields is None:

                # The fake metadata normally contains the actual
                # image name, e.g. alb_id_23.jpg.
                #
                # If not found directly, try extracting the
                # document prefix from the source image.

                errors.append(
                    f"{fake_annotation_file.name}: "
                    f"source image not found in real lookup: "
                    f"{source_image}"
                )
                continue

            source_bbox = source_fields.get(source_field)

            if source_bbox is None:

                errors.append(
                    f"{fake_annotation_file.name}: "
                    f"source field '{source_field}' not found "
                    f"for {source_image}"
                )
                continue

            row = {
                "image": fake_image,
                "label": "fake",
                "forgery_type": forgery_type,
                "manipulated_field": source_field,
                "source_bbox": {
                    key: value
                    for key, value in source_bbox.items()
                    if key != "value"
                },
                "donor_image": None,
                "donor_field": None,
                "donor_bbox": None,
            }

            # ----------------------------------------------------
            # Crop_and_Replace
            # ----------------------------------------------------

            if forgery_type == "Crop_and_Replace":

                donor_image = data.get("second_src")
                donor_field = data.get("second_field")

                if not donor_image:
                    errors.append(
                        f"{fake_annotation_file.name}: "
                        "Crop_and_Replace missing second_src"
                    )
                    continue

                if not donor_field:
                    errors.append(
                        f"{fake_annotation_file.name}: "
                        "Crop_and_Replace missing second_field"
                    )
                    continue

                donor_fields = real_lookup.get(donor_image)

                if donor_fields is None:
                    errors.append(
                        f"{fake_annotation_file.name}: "
                        f"donor image not found in real lookup: "
                        f"{donor_image}"
                    )
                    continue

                donor_bbox = donor_fields.get(donor_field)

                if donor_bbox is None:
                    errors.append(
                        f"{fake_annotation_file.name}: "
                        f"donor field '{donor_field}' not found "
                        f"for {donor_image}"
                    )
                    continue

                row["donor_image"] = donor_image
                row["donor_field"] = donor_field
                row["donor_bbox"] = {
                    key: value
                    for key, value in donor_bbox.items()
                    if key != "value"
                }

            rows.append(row)
            forgery_counter[forgery_type] += 1

        except Exception as error:
            errors.append(
                f"{fake_annotation_file.name}: {error}"
            )

    print(f"Valid fake rows: {len(rows)}")
    print()
    print("Forgery types:")

    for forgery_type, count in forgery_counter.items():
        print(f"  {forgery_type}: {count}")

    print(f"Fake processing errors: {len(errors)}")

    return rows, errors


# ============================================================
# STEP 3
# ADD REAL IMAGES
# ============================================================

def build_real_rows(real_lookup):
    """
    Add one manifest row for every real image.
    """

    print()
    print("=" * 70)
    print("STEP 3: ADDING REAL IMAGES")
    print("=" * 70)

    image_files = sorted(
        REAL_IMAGES_DIR.glob("*")
    )

    image_files = [
        file
        for file in image_files
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    rows = []

    for image_file in image_files:

        rows.append({
            "image": image_file.name,
            "label": "real",
            "forgery_type": None,
            "manipulated_field": None,
            "source_bbox": None,
            "donor_image": None,
            "donor_field": None,
            "donor_bbox": None,
            "fields": real_lookup.get(image_file.name, {}),
        })

    print(f"Real image rows: {len(rows)}")

    return rows


# ============================================================
# STEP 4
# WRITE MANIFEST
# ============================================================

def write_manifest(rows):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print("STEP 4: MANIFEST CREATED")
    print("=" * 70)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows written: {len(rows)}")


# ============================================================
# STEP 5
# SANITY CHECK
# ============================================================

def validate_manifest(rows, errors):

    print()
    print("=" * 70)
    print("STEP 5: SANITY CHECK")
    print("=" * 70)

    total_rows = len(rows)

    real_rows = [
        row for row in rows
        if row["label"] == "real"
    ]

    fake_rows = [
        row for row in rows
        if row["label"] == "fake"
    ]

    crop_replace_rows = [
        row for row in fake_rows
        if row["forgery_type"] == "Crop_and_Replace"
    ]

    inpaint_rows = [
        row for row in fake_rows
        if row["forgery_type"] == "Inpaint_and_Rewrite"
    ]

    missing_source_bbox = [
        row for row in fake_rows
        if row["source_bbox"] is None
    ]

    missing_donor_bbox = [
        row for row in crop_replace_rows
        if row["donor_bbox"] is None
    ]

    checks = {
        "total_rows": total_rows,
        "expected_rows": 2222,
        "real_rows": len(real_rows),
        "expected_real_rows": 1000,
        "fake_rows": len(fake_rows),
        "expected_fake_rows": 1222,
        "inpaint_and_rewrite_rows": len(inpaint_rows),
        "expected_inpaint_and_rewrite": 1077,
        "crop_and_replace_rows": len(crop_replace_rows),
        "expected_crop_and_replace": 145,
        "missing_source_bbox": len(missing_source_bbox),
        "missing_donor_bbox_crop_and_replace": len(missing_donor_bbox),
        "processing_errors": len(errors),
    }

    for key, value in checks.items():
        print(f"{key}: {value}")

    passed = (
        total_rows == 2222
        and len(real_rows) == 1000
        and len(fake_rows) == 1222
        and len(inpaint_rows) == 1077
        and len(crop_replace_rows) == 145
        and len(missing_source_bbox) == 0
        and len(missing_donor_bbox) == 0
        and len(errors) == 0
    )

    validation_result = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "errors": errors,
    }

    with VALIDATION_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            validation_result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()

    if passed:
        print("STATUS: PASS")
        print("SIDTD manifest validation successful.")
    else:
        print("STATUS: FAIL")
        print("Review docs/manifest_validation.json")

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 70)
    print("# SIDTD MANIFEST BUILDER")
    print("#" * 70)

    # Step 1
    real_lookup = build_real_lookup()

    # Step 2
    fake_rows, errors = build_fake_rows(real_lookup)

    # Step 3
    real_rows = build_real_rows(real_lookup)

    # Combine
    all_rows = real_rows + fake_rows

    # Step 4
    write_manifest(all_rows)

    # Step 5
    passed = validate_manifest(
        all_rows,
        errors
    )

    print()
    print("#" * 70)

    if passed:
        print("# COMPLETE: MANIFEST READY")
    else:
        print("# COMPLETE: MANIFEST CREATED BUT VALIDATION FAILED")

    print("#" * 70)


if __name__ == "__main__":
    main()

