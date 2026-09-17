import json

path = r"datasets\sidtd\templates\templates\Annotations\reals\alb_id.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

metadata = data["_via_img_metadata"]

target_files = {"23.jpg", "87.jpg"}

for record in metadata.values():
    if record.get("filename") in target_files:
        print("\n" + "=" * 70)
        print(record["filename"])
        print("=" * 70)

        for region in record.get("regions", []):
            attributes = region.get("region_attributes", {})

            if attributes.get("field_name") == "name":
                print(json.dumps(region, indent=2, ensure_ascii=False))