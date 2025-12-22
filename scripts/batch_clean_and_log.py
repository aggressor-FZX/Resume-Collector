#!/usr/bin/env python3
"""Batch clean all *_pairs.jsonl files, log results, and prep for LLM heuristics.
- For each file: run cleaning, produce cleaned file, report, and samples.
- Log all processing in a master log (JSONL or CSV).
- Optionally, split into two parts for very large datasets.
"""
import json
import glob
from pathlib import Path
import subprocess

import argparse
DATA_DIR = Path('training-data/formatted')
CLEANED_DIR = Path('training-data/cleaned')
LOG_PATH = Path('training-data/cleaning_log.jsonl')
SAMPLE_N = 20
KEEP_N = 10

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--llm-live', action='store_true', help='Run LLM annotations live (default: dry-run)')
parser.add_argument('--cleaned-dir', type=str, default=str(CLEANED_DIR), help='Directory to copy cleaned outputs and samples into')
parser.add_argument('--start', type=int, default=0, help='Start index for processing files (for partial runs)')
parser.add_argument('--end', type=int, default=None, help='End index (exclusive) for processing files (for partial runs)')
args = parser.parse_args()

if not Path(args.cleaned_dir).exists():
    Path(args.cleaned_dir).mkdir(parents=True, exist_ok=True)

# Find all *_pairs.jsonl files (excluding already cleaned)
pair_files = [f for f in DATA_DIR.glob('*_pairs.jsonl') if '__cleaned' not in str(f)]
# allow slicing for partial runs
pair_files = pair_files[args.start:args.end]

log_entries = []

for i, f in enumerate(pair_files):
    print(f'[{i+1}/{len(pair_files)}] Processing {f.name}')
    # Build cleaning command
    import sys
    cmd = [
        sys.executable, 'scripts/clean_and_sample.py',
        '--input', str(f),
        '--keep', str(KEEP_N),
        '--sample', str(SAMPLE_N),
        '--annotate'
    ]
    # default to dry-run unless --llm-live was passed
    if not args.llm_live:
        cmd.append('--dry-run')

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse report file
    report_path = Path(str(f).replace('.jsonl','__clean_report.json'))
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception as e:
            report = {'error': str(e)}

    # move cleaned and sample artifacts into cleaned/<basename>/ for easier review
    src_output = Path(report.get('output_file')) if report.get('output_file') else None
    base = f.stem.replace('__pairs','')
    dest_dir = Path(args.cleaned_dir) / base
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved_files = {}
    if src_output and src_output.exists():
        dst = dest_dir / src_output.name
        src_output.replace(dst)
        moved_files['cleaned'] = str(dst)

    # move samples and annotated artifacts if present
    candidates = [
        str(f).replace('.jsonl','__cleaned__kept_sample.jsonl'),
        str(f).replace('.jsonl','__cleaned__removed_sample.jsonl'),
        str(f).replace('.jsonl','__cleaned__kept_sample_annotated_llm.jsonl'),
        str(f).replace('.jsonl','__cleaned__removed_sample_annotated_llm.jsonl'),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            dst = dest_dir / p.name
            p.replace(dst)
            moved_files[p.name] = str(dst)

    # Log entry
    entry = {
        'input_file': str(f),
        'output_file': report.get('output_file'),
        'report': report,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'llm_annotated': True,
        'llm_mode': 'live' if args.llm_live else 'dry-run',
        'kept_sample': moved_files.get(f.name.replace('.jsonl','__cleaned__kept_sample.jsonl')) or str(Path(str(f).replace('.jsonl','__cleaned__kept_sample.jsonl'))),
        'removed_sample': moved_files.get(f.name.replace('.jsonl','__cleaned__removed_sample.jsonl')) or str(Path(str(f).replace('.jsonl','__cleaned__removed_sample.jsonl'))),
        'kept_sample_annotated': moved_files.get(f.name.replace('.jsonl','__cleaned__kept_sample_annotated_llm.jsonl')) or str(Path(str(f).replace('.jsonl','__cleaned__kept_sample_annotated.jsonl'))),
        'removed_sample_annotated': moved_files.get(f.name.replace('.jsonl','__cleaned__removed_sample_annotated_llm.jsonl')) or str(Path(str(f).replace('.jsonl','__cleaned__removed_sample_annotated.jsonl'))),
        'cleaned_dir': str(dest_dir)
    }
    log_entries.append(entry)
    # Optionally, split into two parts if too large (not implemented here, but can be added)

# Write master log
with LOG_PATH.open('w', encoding='utf-8') as fh:
    for e in log_entries:
        fh.write(json.dumps(e, ensure_ascii=False)+'\n')
print(f'Batch cleaning complete. Log written to {LOG_PATH}')