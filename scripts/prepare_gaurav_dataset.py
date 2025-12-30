import json

def convert_jsonl_to_json(jsonl_path, json_path):
    resumes = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            resumes.append({
                "original_bullet": data.get("output", ""),
                "context": data.get("category", "")
            })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"resumes": resumes}, f, indent=2)

if __name__ == "__main__":
    convert_jsonl_to_json(
        'training-data/cleaned/gauravduttakiit__resume-dataset.jsonl',
        'training-data/raw/gaurav_resumes.json'
    )
