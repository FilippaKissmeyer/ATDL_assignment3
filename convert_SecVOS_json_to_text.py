import json
import os

# Path to SeCVOS JSON metadata
json_path = "SeCVOS/meta_expressions.json"

# Output TXT file compatible with SAM2
output_txt = "SeCVOS/ImageSets/val.txt"

# Make sure output folder exists
os.makedirs(os.path.dirname(output_txt), exist_ok=True)

# Load SeCVOS metadata
with open(json_path, "r") as f:
    data = json.load(f)

# Extract video IDs from the "videos" key
video_ids = list(data["videos"].keys())

# Write to TXT
with open(output_txt, "w") as f:
    for vid in video_ids:
        f.write(f"{vid}\n")

print(f"DAVIS-style TXT file created at: {output_txt}")