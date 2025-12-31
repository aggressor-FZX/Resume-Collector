#!/usr/bin/env python3
"""Prepare an HF upload bundle by chunking generated JSONL files into .hf_upload_tmp and creating an upload manifest.

This script does NOT upload to Hugging Face. It prepares the chunked files and a manifest for review.
"""
import os
import json
from pathlib import Path
from datetime import datetime

# Try to import chunking helper from upload_to_hf if available
try:
    from upload_to_hf import chunk_jsonl_files, TMP_DIR
except Exception:
    # Fallback: simple chunker implementation
    TMP_DIR = Path('.hf_upload_tmp')
    def chunk_jsonl_files(input_dir, tmp_dir, lines_per_chunk=20000, only_file=None):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        files = [Path(input_dir) / only_file] if only_file else sorted(Path(input_dir).glob('*.jsonl'))
        for src in files:
            if not src.exists():
                continue
            with src.open('r', encoding='utf-8') as f:
                chunk_idx = 0
                out = None
                for i, line in enumerate(f, start=1):
                    if (i - 1) % lines_per_chunk == 0:
                        if out:
                            out.close()
                        chunk_idx += 1
                        chunk_path = tmp_dir / f"{src.stem}-part-{chunk_idx:05d}.jsonl"
                        out = chunk_path.open('w', encoding='utf-8')
                        chunk_paths.append(chunk_path)
                    out.write(line)
                if out:
                    out.close()
        return chunk_paths


def main():
    input_dir = os.environ.get('HF_INPUT', 'training-data/imaginator_generated')
    chunk_lines = int(os.environ.get('HF_CHUNK_LINES', '20000'))
    tmp_dir = Path('.hf_upload_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunking JSONL files from {input_dir} into {tmp_dir} (chunk lines={chunk_lines})...")
    chunk_paths = chunk_jsonl_files(input_dir, tmp_dir, lines_per_chunk=chunk_lines)

    manifest = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'source_dir': str(input_dir),
        'chunks': [],
    }

    for p in chunk_paths:
        record_count = 0
        try:
            with p.open('r', encoding='utf-8') as f:
                for _ in f:
                    record_count += 1
        except Exception:
            record_count = None
        manifest['chunks'].append({'path': str(p), 'records': record_count})

    manifest_path = Path('training-data/upload_manifest.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, indent=2)

    # Log a short progress line
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / 'progress.log').open('a', encoding='utf-8') as lf:
        lf.write(f"{datetime.utcnow().isoformat()}Z | prepare_upload_bundle | completed | {len(chunk_paths)} chunks prepared, manifest={manifest_path}\n")

    print(f"Prepared {len(chunk_paths)} chunks and wrote manifest to {manifest_path}")

if __name__ == '__main__':
    main()
