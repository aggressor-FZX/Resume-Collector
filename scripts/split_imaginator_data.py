#!/usr/bin/env python3
"""
split_imaginator_data.py
Split cleaned imaginator JSONL files into two datasets:
- skyfall_dataset.jsonl: Records where model_used == "thedrummer/skyfall-36b-v2"
- other_models_dataset.jsonl: All other records
"""

import json
from pathlib import Path

def main():
    folder = Path('training-data/imaginator_generated')
    skyfall_file = folder / 'skyfall_dataset.jsonl'
    other_file = folder / 'other_models_dataset.jsonl'

    with open(skyfall_file, 'w', encoding='utf-8') as f_skyfall, open(other_file, 'w', encoding='utf-8') as f_other:
        jsonl_files = list(folder.glob('*.jsonl'))
        for filepath in jsonl_files:
            if filepath.name in ['skyfall_dataset.jsonl', 'other_models_dataset.jsonl']:
                continue  # Skip the output files if they exist
            print(f"Processing {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        record = json.loads(line.strip())
                        model_used = record.get('model_used', '')
                        if model_used == 'thedrummer/skyfall-36b-v2':
                            f_skyfall.write(line)
                        else:
                            f_other.write(line)
                    except json.JSONDecodeError:
                        print(f"Error parsing line in {filepath}")

    print(f"Split complete. Skyfall: {skyfall_file}, Others: {other_file}")

if __name__ == '__main__':
    main()