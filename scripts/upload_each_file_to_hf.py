#!/usr/bin/env python3
"""Upload each formatted JSONL file separately to the Hugging Face dataset repo.
This creates smaller, shorter-running upload jobs per file to reduce chance of long-run termination.
"""
import os
import json
import math
from pathlib import Path
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
HF_TOKEN = os.getenv('HUGGING_FACE_API_KEY')
REPO_ID = 'jeff-calderon/ResumeData'
CHUNK_SIZE = 1000


def read_jsonl(path: Path):
    arr = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # normalize to single final_text
                msgs = obj.get('messages') or []
                content = ''
                for m in msgs:
                    if m.get('role') == 'user':
                        content = m.get('content','')
                        break
                if not content:
                    # fallback to any str fields
                    content = ' '.join([str(v) for v in obj.values() if isinstance(v,str)])
                arr.append({'content': content})
            except Exception:
                continue
    return arr


def chunked_upload(dataset, repo_id, chunk_size=1000):
    total_records = len(dataset)
    num_chunks = math.ceil(total_records / chunk_size)
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = dataset.select(range(start, min(end, total_records)))
        print(f'  Uploading chunk {i+1}/{num_chunks} ({len(chunk)})')
        chunk.push_to_hub(repo_id, private=False, commit_message=f'Upload file chunk {i+1}/{num_chunks}')


if __name__ == '__main__':
    if not HF_TOKEN:
        print('HUGGING_FACE_API_KEY not set; aborting')
        raise SystemExit(1)

    login(token=HF_TOKEN)
    files = sorted(Path('training-data/formatted').glob('*.jsonl'))
    print('Found files:', files)
    for p in files:
        print('Processing', p)
        recs = read_jsonl(p)
        if not recs:
            print('  no records, skipping')
            continue
        df = pd.DataFrame(recs)
        ds = Dataset.from_pandas(df)
        print(f'  Pushing {len(ds)} records from {p.name}')
        chunked_upload(ds, REPO_ID, chunk_size=CHUNK_SIZE)
    print('All files processed')
