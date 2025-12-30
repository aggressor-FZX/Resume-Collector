#!/usr/bin/env python3
"""
Inspects and reports on the suitability of formatted JSONL data for fine-tuning.

This script performs the following steps:
1.  Identifies the most suitable JSONL files for training (preferring `__rewritten.jsonl`).
2.  Loads all records from these files into memory.
3.  Randomly samples a subset of these records.
4.  Performs a series of validation checks on each sampled record.
5.  Generates a summary report of the findings, including any problematic records.

Usage:
    python scripts/inspect_and_report.py [--sample-size 100]
"""

import json
import random
import re
from pathlib import Path
import argparse

# --- Configuration ---
DATA_DIR = Path('training-data/formatted')
SAMPLE_SIZE = 100
MIN_CONTENT_LENGTH = 15  # Minimum number of characters for user/assistant content

# --- Regular Expressions for Basic Checks ---
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# Simple check for long digit sequences that might be phone numbers or IDs
PHONE_ID_RE = re.compile(r'\b\d{9,}\b')
UNICODE_REPLACEMENT_CHAR = "�"

def find_training_files(data_dir: Path):
    """
    Finds the best available .jsonl files for training.
    It prefers `__rewritten.jsonl` over the base `.jsonl` file.
    """
    all_files = list(data_dir.glob("*.jsonl"))
    base_names = set()
    for f in all_files:
        if '__rewritten' in f.name:
            base = f.name.split('__rewritten')[0]
            base_names.add(base)
        elif '__' in f.name:
            base = f.name.split('__')[0]
            base_names.add(base)
        else:
             base_names.add(f.stem)

    chosen_files = []
    for base in base_names:
        rewritten_file = data_dir / f"{base}__rewritten.jsonl"
        base_file_pattern = data_dir / f"{base}*.jsonl"
        
        # Find the original base file, avoiding ones with extra suffixes like '__pairs'
        potential_bases = [p for p in list(data_dir.glob(f"{base}*.jsonl")) if '__' not in p.name.replace(base, '') or p.name.endswith('__rewritten.jsonl')]
        
        if rewritten_file.exists():
            chosen_files.append(rewritten_file)
        elif potential_bases:
            # Heuristic: pick the shortest name as the most likely "base" file
            potential_bases.sort(key=lambda x: len(x.name))
            base_file = potential_bases[0]
            # Ensure it's not an intermediate file
            if '__pairs' not in base_file.name and '__cleaned' not in base_file.name:
                 chosen_files.append(base_file)

    # Deduplicate and ensure files exist
    final_files = sorted([f for f in set(chosen_files) if f.exists()])
    print(f"Found {len(final_files)} primary training files to inspect.")
    return final_files

def validate_record(record_str: str, line_num: int, file_path: Path):
    """Performs a series of validation checks on a single JSONL record string."""
    errors = []
    
    # 1. JSON Parsing Check
    try:
        data = json.loads(record_str)
    except json.JSONDecodeError:
        errors.append("Invalid JSON format")
        return errors, None

    # 2. Schema Validation
    if 'messages' not in data or not isinstance(data['messages'], list) or len(data['messages']) < 2:
        errors.append("Missing or invalid 'messages' array (must be a list with at least 2 entries)")
        return errors, data
    
    roles = [msg.get('role') for msg in data['messages']]
    if not all(role in ['system', 'user', 'assistant'] for role in roles):
        errors.append(f"Invalid or missing roles. Found: {roles}")

    for i, msg in enumerate(data['messages']):
        if 'role' not in msg or 'content' not in msg:
            errors.append(f"Message at index {i} is missing 'role' or 'content'")
        
        # 3. Content Length Check
        content = msg.get('content', '')
        if not isinstance(content, str):
             errors.append(f"Content for role '{msg.get('role')}' is not a string.")
             content = str(content) # Coerce for next checks

        if msg.get('role') in ['user', 'assistant'] and len(content) < MIN_CONTENT_LENGTH:
            errors.append(f"Content for role '{msg.get('role')}' is too short (len: {len(content)})")
        
        # 4. PII and Encoding Checks
        if EMAIL_RE.search(content):
            errors.append(f"Potential email found in '{msg.get('role')}' content")
        if PHONE_ID_RE.search(content):
            errors.append(f"Potential long number (ID/phone) in '{msg.get('role')}' content")
        if UNICODE_REPLACEMENT_CHAR in content:
            errors.append(f"Unicode replacement character '�' found in '{msg.get('role')}' content, indicating possible encoding issue.")

    return errors, data

def main():
    """Main function to run the inspection and reporting."""
    parser = argparse.ArgumentParser(description="Inspect and report on finetuning data readiness.")
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE, help="Number of records to randomly sample.")
    args = parser.parse_args()

    files_to_scan = find_training_files(DATA_DIR)
    if not files_to_scan:
        print("No training files found. Exiting.")
        return

    print("\n--- Loading Records ---")
    all_records = []
    for file_path in files_to_scan:
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                all_records.append({'line_content': line.strip(), 'file': file_path.name, 'line_num': i + 1})
    
    total_records = len(all_records)
    if total_records == 0:
        print("No records found in the identified files. Exiting.")
        return

    sample_size = min(args.sample_size, total_records)
    print(f"Loaded {total_records} records from {len(files_to_scan)} files. Sampling {sample_size} records for inspection.")
    
    sampled_records = random.sample(all_records, sample_size)
    
    print("\n--- Inspection Report ---")
    
    error_counts = {
        "Invalid JSON": 0,
        "Schema Invalid": 0,
        "Content Too Short": 0,
        "PII/Encoding Issues": 0
    }
    
    failed_records = []

    for record_info in sampled_records:
        errors, data = validate_record(record_info['line_content'], record_info['line_num'], record_info['file'])
        
        if errors:
            record_info['errors'] = errors
            failed_records.append(record_info)
            
            if "Invalid JSON" in " ".join(errors):
                error_counts["Invalid JSON"] += 1
            if "Missing or invalid 'messages'" in " ".join(errors) or "Invalid or missing roles" in " ".join(errors):
                error_counts["Schema Invalid"] += 1
            if "too short" in " ".join(errors):
                error_counts["Content Too Short"] += 1
            if "Potential" in " ".join(errors) or "Unicode" in " ".join(errors):
                error_counts["PII/Encoding Issues"] += 1

    # --- Print Summary ---
    print("\n--- Summary Statistics ---")
    print(f"Total Records Sampled: {sample_size}")
    passed_count = sample_size - len(failed_records)
    passed_pct = (passed_count / sample_size) * 100 if sample_size > 0 else 0
    print(f"Records Passed Validation: {passed_count} ({passed_pct:.2f}%)")
    print(f"Records Failed Validation: {len(failed_records)}")

    if failed_records:
        print("\n--- Failure Category Breakdown ---")
        for category, count in error_counts.items():
            if count > 0:
                pct = (count / len(failed_records)) * 100
                print(f"- {category}: {count} ({pct:.2f}% of failures)")
    
        print("\n--- Details on Failed Records (up to 5) ---")
        for i, failure in enumerate(failed_records[:5]):
            print(f"\nRecord {i+1}:")
            print(f"  File: {failure['file']}, Line: {failure['line_num']}")
            print(f"  Content: {failure['line_content'][:200]}...")
            print(f"  Errors:")
            for err in failure['errors']:
                print(f"    - {err}")

if __name__ == "__main__":
    main()
