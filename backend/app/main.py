"""FastAPI endpoints for manifest-backed SIDTD OCR extraction."""

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import easyocr
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "docs" / "manifest.jsonl"
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")

app = FastAPI()


def levenshtein_distance(left: str, right: str) -> int:
    """Return the character-level Levenshtein edit distance."""
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current_row[-1] + 1
            deletion = previous_row[right_index] + 1
            substitution = previous_row[right_index - 1] + (
                left_character != right_character
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def character_similarity(ground_truth: str, prediction: str) -> float:
    """Use the normalized Levenshtein similarity from the test OCR script."""
    maximum_length = max(len(ground_truth), len(prediction))
    if maximum_length == 0:
        return 1.0
    return 1 - levenshtein_distance(ground_truth, prediction) / maximum_length


@lru_cache(maxsize=1)
def load_manifest_by_image() -> dict[str, dict]:
    """Read the manifest once and index its rows by image filename."""
    rows_by_image = {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in manifest line {line_number}."
                ) from error

            image_name = row.get("image")
            if isinstance(image_name, str):
                rows_by_image[image_name] = row

    return rows_by_image


@lru_cache(maxsize=1)
def get_ocr_reader() -> easyocr.Reader:
    """Create one EasyOCR Reader and reuse it for all extraction requests."""
    return easyocr.Reader(["en"])


def get_bbox(annotation: dict) -> dict:
    """Validate and return an x/y/width/height bbox from a field annotation."""
    if not isinstance(annotation, dict):
        raise ValueError("field annotation is not an object")

    missing_keys = [key for key in REQUIRED_BBOX_KEYS if key not in annotation]
    if missing_keys:
        raise ValueError(f"field bbox is missing: {', '.join(missing_keys)}")

    bbox = {key: annotation[key] for key in REQUIRED_BBOX_KEYS}
    if not all(isinstance(value, (int, float)) for value in bbox.values()):
        raise ValueError("field bbox values must be numeric")
    if bbox["width"] < 0 or bbox["height"] < 0:
        raise ValueError("field bbox width and height must not be negative")
    return bbox


def crop_to_rgb(image: Image.Image, bbox: dict) -> Image.Image:
    """Crop a field in memory and return RGB pixels suitable for EasyOCR."""
    x, y = bbox["x"], bbox["y"]
    right = x + bbox["width"]
    bottom = y + bbox["height"]
    crop = image.crop((x, y, right, bottom))
    return crop.convert("RGB") if crop.mode != "RGB" else crop.copy()


def error_field_result(annotation: object, error: Exception) -> dict:
    """Return a stable field response when one annotation or OCR call fails."""
    ground_truth = ""
    bbox = None
    if isinstance(annotation, dict) and "value" in annotation:
        ground_truth = str(annotation["value"])
    if isinstance(annotation, dict):
        try:
            bbox = get_bbox(annotation)
        except ValueError:
            pass

    return {
        "bbox": bbox,
        "ground_truth": ground_truth,
        "raw_ocr": [],
        "prediction": "",
        "exact_match": False,
        "character_similarity": 0.0,
        "error": f"{type(error).__name__}: {error}",
    }


def extract_field(reader: easyocr.Reader, image: Image.Image, annotation: dict) -> dict:
    """OCR one field crop while retaining raw EasyOCR output separately."""
    bbox = get_bbox(annotation)
    ground_truth = str(annotation.get("value", ""))
    crop = crop_to_rgb(image, bbox)

    # This retains EasyOCR's returned list. Joining follows the test script.
    raw_ocr = reader.readtext(np.array(crop), detail=0)
    prediction = " ".join(raw_ocr)

    return {
        "bbox": bbox,
        "ground_truth": ground_truth,
        "raw_ocr": raw_ocr,
        "prediction": prediction,
        "exact_match": ground_truth == prediction,
        "character_similarity": character_similarity(ground_truth, prediction),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/extract")
async def extract_manifest_fields(file: UploadFile = File(...)):
    """Extract every annotated field for one uploaded real SIDTD image."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    image_name = Path(file.filename).name
    if not image_name:
        raise HTTPException(status_code=400, detail="Uploaded file has an invalid filename.")

    try:
        manifest_row = load_manifest_by_image().get(image_name)
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=500, detail=f"Unable to load manifest: {error}"
        ) from error

    if manifest_row is None:
        raise HTTPException(
            status_code=404, detail=f"Image '{image_name}' was not found in the manifest."
        )
    if manifest_row.get("label") != "real":
        raise HTTPException(
            status_code=422,
            detail="Only real SIDTD images are supported by this endpoint.",
        )

    annotations = manifest_row.get("fields")
    if not isinstance(annotations, dict) or not annotations:
        raise HTTPException(
            status_code=422,
            detail=f"Real manifest row '{image_name}' has no annotated fields.",
        )

    try:
        upload_bytes = await file.read()
    finally:
        await file.close()

    try:
        uploaded_image = Image.open(BytesIO(upload_bytes))
        uploaded_image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file is not a readable image: {error}"
        ) from error

    try:
        reader = get_ocr_reader()
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"EasyOCR could not be initialized: {error}"
        ) from error

    extracted_fields = {}
    try:
        for field_name, annotation in annotations.items():
            try:
                extracted_fields[field_name] = extract_field(
                    reader, uploaded_image, annotation
                )
            except Exception as error:
                extracted_fields[field_name] = error_field_result(annotation, error)
    finally:
        uploaded_image.close()

    return {
        "image": image_name,
        "field_count": len(extracted_fields),
        "fields": extracted_fields,
    }
