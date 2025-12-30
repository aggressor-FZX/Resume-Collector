import json
from pathlib import Path

jsonl_path = Path("training-data/cleaned/gauravduttakiit__resume-dataset__cleaned.jsonl")
out_path = Path("training-data/cleaned/gauravduttakiit__resume-dataset__cleaned_for_formatter.json")

resumes = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Map to expected keys for formatter
        resumes.append({
            "original_bullet": obj.get("output", ""),
            "context": obj.get("category", "")
        })

with out_path.open("w", encoding="utf-8") as f:
    json.dump({"resumes": resumes}, f, ensure_ascii=False, indent=2)
print(f"Wrote formatter input to {out_path}")
