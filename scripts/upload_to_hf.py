#!/usr/bin/env python3
import os
import json
from datasets import Dataset
from huggingface_hub import login, HfApi
import pandas as pd
from dotenv import load_dotenv
import math

load_dotenv()

HF_TOKEN = os.getenv('HUGGING_FACE_API_KEY')
DATA_PATHS = [
    'training-data/formatted/latest.jsonl',
    'training-data/formatted/resumes-$(date +%Y%m%d).jsonl',
    'training-data/formatted/sample_run.jsonl',
    'training-data/formatted/sample.jsonl'
]
OUT_JSON = 'data/anonymized_combined_resume_dataset.json'
OUT_PARQUET = 'data/anonymized_combined_resume_dataset_hf.parquet'
REPO_ID = 'jeff-calderon/ResumeData'


def load_jsonl(path):
    arr = []
    if not os.path.exists(path):
        return arr
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                arr.append(obj)
            except Exception:
                continue
    return arr


def normalize_example(ex):
    # ex is expected to be {"messages": [{role, content}, ...]}
    msgs = ex.get('messages') or []
    content = ''
    orig = ''
    improved = ''
    # find user and assistant messages
    for m in msgs:
        r = m.get('role')
        c = m.get('content','')
        if r == 'system':
            # include system context as prefix
            content += f"[CONTEXT] {c}\n"
        elif r == 'user':
            orig = orig + '\n' + c
        elif r == 'assistant':
            improved = improved + '\n' + c
    # final content field: prefer improved if present, otherwise user
    final_text = (improved.strip() or orig.strip() or content.strip())
    return {
        'name': None,
        'title': None,
        'content': final_text,
        'resume_text': final_text,
        'skills': [],
        'experience': [],
        'total_experience_years': None,
        'location': None,
        'source': 'Resume-Collector-sample',
        'anonymized': True
    }


import argparse


def chunked_upload(dataset, repo_id, chunk_size=1000, resume=False, checkpoint_path='data/upload_checkpoint.json'):
    """Upload dataset to Hugging Face in chunks with resumable checkpointing."""
    total_records = len(dataset)
    num_chunks = math.ceil(total_records / chunk_size)

    start_chunk = 0
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as fh:
                cp = json.load(fh)
                start_chunk = cp.get('next_chunk', 0)
            print(f'🔁 Resuming upload from chunk {start_chunk + 1}/{num_chunks}')
        except Exception:
            start_chunk = 0

    for i in range(start_chunk, num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = dataset.select(range(start, min(end, total_records)))
        print(f'🔄 Uploading chunk {i + 1}/{num_chunks} ({len(chunk)} records)')
        try:
            chunk.push_to_hub(repo_id, private=False, commit_message=f'Upload chunk {i + 1}/{num_chunks}')
            print(f'✅ Successfully uploaded chunk {i + 1}/{num_chunks}')
            # write checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as fh:
                json.dump({'next_chunk': i + 1}, fh)
        except Exception as e:
            print(f'❌ Failed uploading chunk {i + 1}/{num_chunks}: {e}')
            raise

    # cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# allow CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--chunk-size', type=int, default=1000)
parser.add_argument('--resume', action='store_true')
args, _ = parser.parse_known_args()

def main():
    if not HF_TOKEN:
        print('❌ HUGGING_FACE_API_KEY not set in .env')
        return 1

    # Verify Hugging Face destination
    try:
        print('🔐 Verifying Hugging Face destination...')
        api = HfApi()
        api.whoami()  # Ensure authentication works
        try:
            # Check dataset repo specifically
            repo_info = api.repo_info(REPO_ID, repo_type='dataset')
            print(f'✅ Destination repository {REPO_ID} is reachable and ready.')
        except Exception:
            # Try to create the dataset repo if it does not exist
            print(f'⚠️ Dataset repo {REPO_ID} not found; attempting to create it...')
            api.create_repo(repo_id=REPO_ID, repo_type='dataset', private=False, exist_ok=True)
            print(f'✅ Created dataset repo {REPO_ID}.')
    except Exception as e:
        print(f'❌ Failed to verify or create Hugging Face destination: {e}')
        return 1

    # find all available jsonl files in training-data/formatted and load them
    examples = []
    import glob
    files = sorted(glob.glob('training-data/formatted/*.jsonl'))
    for p in files:
        arr = load_jsonl(p)
        if arr:
            print(f'📂 Loaded {len(arr)} examples from {p}')
            examples.extend(arr)

    if not examples:
        print('⚠️ No formatted JSONL found in training-data/formatted. Ensure you have run the formatter.')
        return 1

    # normalize each example to dataset schema
    records = [normalize_example(e) for e in examples]

    # ensure data directory
    os.makedirs('data', exist_ok=True)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'✅ Wrote {len(records)} records to {OUT_JSON}')

    # create parquet
    df = pd.DataFrame(records)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f'✅ Wrote parquet {OUT_PARQUET}')

    try:
        print('🔐 Logging into HuggingFace...')
        login(token=HF_TOKEN)
        ds = Dataset.from_pandas(df)
        print('🔄 Pushing dataset to hub as', REPO_ID)
        chunked_upload(ds, REPO_ID, chunk_size=args.chunk_size, resume=args.resume)
        print('✅ Successfully pushed dataset to https://huggingface.co/datasets/' + REPO_ID)

        # write simple dataset card
        card = '# Tech Resumes\n\nSample dataset uploaded from Resume-Collector. See repository for full pipeline.'
        with open('dataset_card.md', 'w') as f:
            f.write(card)
        print('📝 Wrote dataset_card.md')

        return 0
    except Exception as e:
        print('❌ Error uploading dataset:', e)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
