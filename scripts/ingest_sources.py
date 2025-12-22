#!/usr/bin/env python3
"""Ingest external datasets (Hugging Face + Kaggle) and normalize to training-data/formatted JSONL.

Creates files: training-data/formatted/<source>.jsonl
"""
import os
import json
import glob
import hashlib
import shutil
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME') or 'cogitometric'
KAGGLE_KEY = os.getenv('KAGGLE_API_KEY')
# Fallback: if not loaded, try to parse .env manually
if not KAGGLE_KEY:
    env_path = Path('.env')
    if env_path.exists():
        with env_path.open('r', encoding='utf-8') as f:
            for ln in f:
                if ln.startswith('KAGGLE_API_KEY'):
                    _, v = ln.strip().split('=', 1)
                    KAGGLE_KEY = v.strip()
                    break

HF_DATASETS = [
    'datasetmaster/resumes',
    'MikePfunk28/resume-training-dataset'
]

KAGGLE_DATASETS = [
    'isaacoresanya/freelancer',
    'asaniczka/upwork-job-postings-dataset-2024-50k-records',
    'muhammadadiltalay/fiverr-data-gigs',
    'rayyankauchali0/resume-dataset',
    'snehaanbhawal/resume-dataset',
    'gauravduttakiit/resume-dataset',
    #'mikepfunk28/resume-training-dataset'  # primarily HF
]

OUT_DIR = Path('training-data/formatted')
RAW_DIR = Path('data/raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Simple anonymizer: remove emails and URLs
import re
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r'https?://\S+')


def anonymize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = EMAIL_RE.sub('[REDACTED_EMAIL]', text)
    t = URL_RE.sub('[REDACTED_URL]', t)
    # remove long numeric sequences
    t = re.sub(r'\b\d{6,}\b', '[REDACTED_NUM]', t)
    return t.strip()


def write_jsonl(records, outpath: Path):
    with outpath.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def make_messages(content: str, context: str = 'General tech role'):
    messages = [
        {'role': 'system', 'content': f'You are a world-class tech resume writer. Context: {context}'},
        {'role': 'user', 'content': f'Improve this resume bullet:\n"{content}"'},
        {'role': 'assistant', 'content': ''}
    ]
    return {'messages': messages}


def hf_ingest():
    all_records = []
    seen = set()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    for ds in HF_DATASETS:
        records = []
        print('Inspecting HF dataset', ds)
        try:
            info = api.dataset_info(ds)
        except Exception as e:
            print('  Failed to fetch dataset info for', ds, e)
            continue
        # prefer any .jsonl files in the dataset
        files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.jsonl') or f.rfilename.endswith('.json')]
        if not files:
            print('  No json/jsonl files found for', ds, 'skipping')
            continue
        # pick largest-ish file by name heuristics
        chosen = None
        for name in files:
            if 'master' in name or 'train' in name or name.endswith('.jsonl'):
                chosen = name
                break
        if not chosen:
            chosen = files[0]
        print('  Downloading file', chosen)
        try:
            local_path = hf_hub_download(repo_id=ds, filename=chosen, repo_type='dataset', cache_dir=str(RAW_DIR))
        except Exception as e:
            print('  Failed to download', chosen, e)
            continue
        print('  parsing', local_path)
        try:
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # try to get text fields
                    text_candidates = []
                    for key in ['content', 'resume_text', 'text', 'description', 'summary', 'original', 'raw']:
                        if key in obj and obj.get(key):
                            text_candidates.append(obj.get(key))
                    if not text_candidates:
                        s = ' '.join([str(v) for v in obj.values() if isinstance(v, str)])
                        if s:
                            text_candidates.append(s)
                    for t in text_candidates:
                        t2 = anonymize_text(t)
                        h = hashlib.sha1(t2.encode('utf-8')).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)
                        records.append(make_messages(t2, context=ds))
        except Exception as e:
            print('  Error parsing', local_path, e)

        outpath = OUT_DIR / (ds.replace('/', '__') + '.jsonl')
        write_jsonl(records, outpath)
        print(f'Wrote {len(records)} records to {outpath}')
        all_records.extend(records)
    return all_records


def ensure_kaggle_auth():
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    if not kaggle_json.exists():
        if not KAGGLE_KEY:
            raise RuntimeError('KAGGLE_API_KEY not set in env; cannot authenticate')
        kaggle_json.write_text(json.dumps({'username': KAGGLE_USERNAME, 'key': KAGGLE_KEY}))
        kaggle_json.chmod(0o600)
        print('Wrote ~/.kaggle/kaggle.json')
    else:
        print('Existing ~/.kaggle/kaggle.json found')


def kaggle_ingest():
    try:
        from kaggle import api as kaggle_api
    except Exception as e:
        print('kaggle api not available:', e)
        return []

    all_records = []
    seen = set()

    for slug in KAGGLE_DATASETS:
        print('Downloading Kaggle dataset', slug)
        try:
            target = RAW_DIR / slug.replace('/', '__')
            if target.exists():
                print('  already downloaded at', target)
            else:
                target.mkdir(parents=True, exist_ok=True)
                kaggle_api.dataset_download_files(slug, path=str(target), unzip=True, quiet=False)
        except Exception as e:
            print('  failed to download', slug, e)
            continue
        # find CSV/JSON files
        files = list(target.glob('**/*'))
        records = []
        for f in files:
            if f.suffix.lower() in ('.csv', '.json', '.jsonl'):
                try:
                    if f.suffix.lower() == '.csv':
                        df = pd.read_csv(f)
                        for _, row in df.iterrows():
                            # heuristics for content
                            content = None
                            for col in ['content', 'resume', 'text', 'description', 'summary', 'job_title', 'job_description']:
                                if col in row and pd.notnull(row[col]):
                                    content = str(row[col])
                                    break
                            if not content:
                                # try concat
                                content = ' '.join([str(x) for x in row.values if isinstance(x, (str,)) and x.strip()])
                            content = anonymize_text(content)
                            h = hashlib.sha1(content.encode('utf-8')).hexdigest()
                            if h in seen or not content.strip():
                                continue
                            seen.add(h)
                            records.append(make_messages(content, context=slug))
                    else:
                        with f.open('r', encoding='utf-8', errors='ignore') as fh:
                            for line in fh:
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    obj = None
                                if obj:
                                    # try keys
                                    content = obj.get('content') or obj.get('text') or obj.get('resume') or ''
                                    if not content:
                                        content = ' '.join([str(v) for v in obj.values() if isinstance(v, str)])
                                    content = anonymize_text(content)
                                    h = hashlib.sha1(content.encode('utf-8')).hexdigest()
                                    if h in seen or not content.strip():
                                        continue
                                    seen.add(h)
                                    records.append(make_messages(content, context=slug))
                except Exception as e:
                    print('   failed to read', f, e)
        outpath = OUT_DIR / (slug.replace('/', '__') + '.jsonl')
        write_jsonl(records, outpath)
        print(f'Wrote {len(records)} records to {outpath}')
        all_records.extend(records)
    return all_records


if __name__ == '__main__':
    print('Starting ingestion')
    # HF first
    hf_records = hf_ingest()

    # Kaggle
    try:
        ensure_kaggle_auth()
        kg_records = kaggle_ingest()
    except Exception as e:
        print('Skipping Kaggle ingestion due to:', e)
        kg_records = []

    total = len(hf_records) + len(kg_records)
    print(f'Ingestion complete: {total} records normalized (HF={len(hf_records)}, Kaggle={len(kg_records)})')
    print('Run `python scripts/upload_to_hf.py` to push formatted data to Hugging Face')
