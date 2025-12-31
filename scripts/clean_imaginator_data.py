#!/usr/bin/env python3
"""
clean_imaginator_data.py
Post-process imaginator-generated JSONL files to clean up issues:
- Strip conversational chatter after final bullet points
- Add disclaimers for hallucinated entities
- Attempt basic typo fixes (optional, as it's tricky)
"""

import json
import re
import os
from pathlib import Path
from typing import List

def clean_resume_text(text: str) -> str:
    """
    Clean the resume_text by:
    1. Stripping everything after the last bullet point
    2. Adding a disclaimer about placeholders
    """
    if not text:
        return text

    # Split into lines
    lines = text.split('\n')

    # Find the last line that looks like a bullet point
    bullet_patterns = [
        re.compile(r'^\s*[-•]\s'),  # - or •
        re.compile(r'^\s*\d+\.\s'),  # 1. 2. etc.
        re.compile(r'^\s*[a-zA-Z]\.\s'),  # a. b. etc.
    ]

    last_bullet_idx = -1
    for i, line in enumerate(lines):
        for pattern in bullet_patterns:
            if pattern.match(line):
                last_bullet_idx = i
                break

    if last_bullet_idx >= 0:
        # Keep up to and including the last bullet and a few lines after if they seem part of it
        # But to be safe, keep up to the last bullet line
        cleaned_lines = lines[:last_bullet_idx + 1]
        # Check if the next lines are continuations (indented)
        for j in range(last_bullet_idx + 1, len(lines)):
            if lines[j].startswith(' ') or lines[j].startswith('\t') or not lines[j].strip():
                cleaned_lines.append(lines[j])
            else:
                break
        text = '\n'.join(cleaned_lines).rstrip()

    # Add disclaimer
    disclaimer = "\n\n*Note: Company names and specific details are AI-generated placeholders for training purposes.*"
    if not text.endswith(disclaimer):
        text += disclaimer

    return text

def fix_typos(text: str) -> str:
    """
    Attempt to fix common typos. This is basic and may not catch all.
    """
    # Common fixes
    fixes = {
        r'\bomprehensively\b': 'comprehensively',
        r'\b(%geregory)\b': 'Gregory',  # Assuming it's a name
        # Add more as needed
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def process_file(filepath: Path) -> None:
    """
    Process a single JSONL file, cleaning each record.
    """
    temp_file = filepath.with_suffix('.tmp')
    with open(filepath, 'r', encoding='utf-8') as f_in, open(temp_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
                if 'resume_text' in record:
                    record['resume_text'] = clean_resume_text(record['resume_text'])
                    record['resume_text'] = fix_typos(record['resume_text'])
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {filepath}: {e}")
                f_out.write(line)  # Write as is

    # Replace original
    temp_file.replace(filepath)
    print(f"Processed {filepath}")

def main():
    """
    Process all .jsonl files in training-data/imaginator_generated/
    """
    folder = Path('training-data/imaginator_generated')
    if not folder.exists():
        print(f"Folder {folder} does not exist.")
        return

    jsonl_files = list(folder.glob('*.jsonl'))
    if not jsonl_files:
        print("No .jsonl files found.")
        return

    for filepath in jsonl_files:
        process_file(filepath)

    print("All files processed.")

if __name__ == '__main__':
    main()