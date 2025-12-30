#!/usr//bin/env python3
"""
scripts/upload_to_hf.py
Robust chunked uploader for large JSONL datasets to Hugging Face.
Requires: pip install huggingface_hub tqdm backoff psutil
"""


import os
import sys
import signal
import time
import json
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm
import backoff
from dotenv import load_dotenv
import argparse

load_dotenv()

# CLI argument parsing
parser = argparse.ArgumentParser(description="Robust chunked uploader for Hugging Face datasets.")
parser.add_argument('--repo', type=str, help='Hugging Face dataset repo id (e.g. user/dataset)')
parser.add_argument('--input', type=str, help='Input file or directory (JSONL or folder of JSONL)')
parser.add_argument('--chunk-size', type=int, help='Lines per chunk', default=None)
parser.add_argument('--log', type=str, help='Log file path', default=None)
parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
args = parser.parse_args()

HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY") or os.environ.get("HF_TOKEN")
REPO_ID = args.repo or os.environ.get("HF_REPO", "your-username/your-dataset")
INPUT_PATH = args.input or os.environ.get("HF_INPUT", "training-data/formatted")
CHUNK_LINES = args.chunk_size or int(os.environ.get("HF_CHUNK_LINES", "20000"))
TMP_DIR = Path(".hf_upload_tmp")
LOG_PATH = Path(args.log) if args.log else Path("logs/progress.log")

if os.path.isdir(INPUT_PATH):
    INPUT_DIR = Path(INPUT_PATH)
    ONLY_FILE = None
else:
    # If a file is given, treat its parent as input dir and filter for just that file
    INPUT_DIR = Path(INPUT_PATH).parent
    ONLY_FILE = Path(INPUT_PATH).name

# Globals for graceful shutdown
_current_chunk_index = None
_shutdown_requested = False

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{ts} | {msg}\n")
    print(msg, flush=True)

def handle_sigterm(signum, frame):
    global _shutdown_requested
    log(f"SIGTERM received (signal {signum}). Checkpointing and exiting.")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

api = HfApi()

def ensure_repo(repo_id):
    try:
        api.repo_info(repo_id, token=HF_TOKEN)
        log(f"Verified HF repo {repo_id}.")
    except Exception:
        log(f"Repo {repo_id} not found. Creating...")
        api.create_repo(repo_id=repo_id, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
        log(f"Created repo {repo_id}.")

def chunk_jsonl_files(input_dir, tmp_dir, lines_per_chunk=20000, only_file=None):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    files = [Path(input_dir) / only_file] if only_file else sorted(input_dir.glob("*.jsonl"))
    for src in files:
        with src.open("r", encoding="utf-8") as f:
            chunk_idx = 0
            out = None
            for i, line in enumerate(f, start=1):
                if (i - 1) % lines_per_chunk == 0:
                    if out:
                        out.close()
                    chunk_idx += 1
                    chunk_path = tmp_dir / f"{src.stem}-part-{chunk_idx:05d}.jsonl"
                    out = chunk_path.open("w", encoding="utf-8")
                    chunk_paths.append(chunk_path)
                out.write(line)
                if _shutdown_requested:
                    if out:
                        out.close()
                    log("Shutdown requested during chunking; returning partial chunk list.")
                    return chunk_paths
            if out:
                out.close()
    return chunk_paths

@backoff.on_exception(backoff.expo, Exception, max_time=300)
def upload_file_with_retry(repo_id, local_path, dest_path):
    # huggingface_hub HfApi.upload_file supports path_or_fileobj
    log(f"Uploading {local_path} -> {repo_id}:{dest_path}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=dest_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    log(f"Uploaded {dest_path}")

def upload_readme(repo_id, readme_path="dataset_card.md"):
    """Uploads the dataset card (README.md) to the repo root."""
    readme = Path(readme_path)
    if not readme.exists():
        log(f"README file not found at {readme_path}. Skipping README upload.")
        return
    try:
        log(f"Uploading {readme_path} as README.md to {repo_id}")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md", # Upload as README.md
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="feat: upload dataset card"
        )
        log("Successfully uploaded dataset card.")
    except Exception as e:
        log(f"Failed to upload README.md: {e}")

def main():
    if not HF_TOKEN:
        log("HUGGING_FACE_API_KEY not set. Aborting.")
        sys.exit(1)

    ensure_repo(REPO_ID)
    upload_readme(REPO_ID) # Upload the README first

    log(f"Starting chunking of JSONL files from {INPUT_DIR} (only_file={ONLY_FILE if 'ONLY_FILE' in globals() else None})...")
    chunk_paths = chunk_jsonl_files(INPUT_DIR, TMP_DIR, CHUNK_LINES, only_file=globals().get('ONLY_FILE', None))
    if not chunk_paths:
        log("No chunks found. Nothing to upload.")
        return

    # Resume support: check which files already exist in repo
    existing_files = set()
    try:
        repo_info = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
        existing_files = set(repo_info)
    except Exception:
        log("Could not list remote files; proceeding without resume check.")

    for idx, chunk in enumerate(tqdm(chunk_paths, desc="chunks")):
        if _shutdown_requested:
            log("Shutdown requested; stopping upload loop.")
            break
        dest_name = f"data/{chunk.name}"
        if dest_name in existing_files:
            log(f"Skipping already uploaded chunk {dest_name}")
            continue
        try:
            upload_file_with_retry(REPO_ID, chunk, dest_name)
        except Exception as e:
            log(f"Failed to upload {chunk}: {e}")
            # If backoff exhausted, re-raise to allow orchestrator to see failure
            raise

    log("Upload run complete. Cleaning up temporary files.")
    # Optionally remove tmp files
    # for p in TMP_DIR.glob("*"): p.unlink()
    log("Done.")

if __name__ == "__main__":
    main()
