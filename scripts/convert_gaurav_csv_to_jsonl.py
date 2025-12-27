import csv
import json
from pathlib import Path

csv_path = Path("data/raw/gauravduttakiit__resume-dataset/UpdatedResumeDataSet.csv")
jsonl_path = Path("training-data/cleaned/gauravduttakiit__resume-dataset.jsonl")
jsonl_path.parent.mkdir(parents=True, exist_ok=True)

with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    with jsonl_path.open("w", encoding="utf-8") as out:
        for row in reader:
            # Use the resume text as 'output', category as 'category'
            obj = {
                "output": row.get("Resume", "").strip(),
                "category": row.get("Category", "").strip()
            }
            if obj["output"]:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"Wrote JSONL to {jsonl_path}")
