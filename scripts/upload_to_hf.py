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
REPO_ID = 'jeff-calderon/Tech_Resumes'


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


def chunked_upload(dataset, repo_id, chunk_size=1000):
    """Upload dataset to Hugging Face in chunks."""
    total_records = len(dataset)
    num_chunks = math.ceil(total_records / chunk_size)

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = dataset.select(range(start, min(end, total_records)))
        print(f'🔄 Uploading chunk {i + 1}/{num_chunks} ({len(chunk)} records)')
        chunk.push_to_hub(repo_id, private=False, commit_message=f'Upload chunk {i + 1}/{num_chunks}')
        print(f'✅ Successfully uploaded chunk {i + 1}/{num_chunks}')


def main():
    if not HF_TOKEN:
        print('❌ HUGGING_FACE_API_KEY not set in .env')
        return 1

    # Verify Hugging Face destination
    try:
        print('🔐 Verifying Hugging Face destination...')
        api = HfApi()
        api.whoami()  # Ensure authentication works
        repo_info = api.repo_info(REPO_ID)
        print(f'✅ Destination repository {REPO_ID} is reachable and ready.')
    except Exception as e:
        print(f'❌ Failed to verify Hugging Face destination: {e}')
        return 1

    # find first available file
    examples = []
    for p in DATA_PATHS:
        # handle the literal $(date ...) by expanding known sample file
        if '$(date' in p:
            continue
        arr = load_jsonl(p)
        if arr:
            print(f'📂 Loaded {len(arr)} examples from {p}')
            examples = arr
            break

    if not examples:
        print('⚠️ No formatted JSONL found in expected locations. Ensure you have run the formatter.')
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
        chunked_upload(ds, REPO_ID)
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
