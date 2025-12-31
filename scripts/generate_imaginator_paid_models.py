#!/usr/bin/env python3
"""
Generate imaginator data using paid models only (thedrummer/skyfall-36b-v2 and x-ai/grok-4.1-fast).

- Fetches GitHub profiles (200 by default) using the repository's GitHub query logic and applies an
  additional filter to ensure profiles contain explicit skills/accomplishments indicators.
- Tests model availability and performs a small test call to each model before mass generation.
- Generates `per_model` examples per model (default 150), using `workers` concurrency (default 2 per model).
- Writes outputs to two separate JSONL files: `skyfall_imagin.jsonl` and `grok_imagin.jsonl`.

Usage (basic):
    python scripts/generate_imaginator_paid_models.py --num_seeds 200 --per_model 150 --workers 2 --test-only

"""

import os
import json
import time
import random
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import ijson
from dotenv import load_dotenv

load_dotenv()

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GITHUB_API_BASE = "https://api.github.com"
OUTPUT_DIR = Path('training-data/imaginator_generated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to use (paid OpenRouter-style model ids)
SKYFALL = "thedrummer/skyfall-36b-v2"
GROK = "x-ai/grok-4.1-fast"

DEFAULT_TIMEOUT = 200  # seconds for chat completions (per your request)
RETRY_DELAY = 2  # seconds between tries
MAX_RETRIES = 3

# Filters to ensure the profile contains 'skills' or 'accomplishments'
SKILL_KEYWORDS = ["skills", "experience", "accomplish", "achiev", "projects", "contributed", "worked on"]
MIN_BIO_LEN = 30
MIN_README_LEN = 80


def fetch_github_profiles(num_profiles: int):
    """Fetch candidate GitHub profiles using the repo's simplified query and apply extra filters."""
    headers = {"Authorization": f"token {GITHUB_API_KEY}"} if GITHUB_API_KEY else {}
    query = "language:python language:go language:typescript followers:>500"
    params = {'q': query, 'per_page': num_profiles + 50}

    try:
        resp = requests.get(f"{GITHUB_API_BASE}/search/users", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        users = resp.json().get('items', [])
    except Exception as e:
        logging.warning(f"GitHub search failed: {e}. Falling back to local seeds if available.")
        return []

    profiles = []
    for user in users:
        if len(profiles) >= num_profiles:
            break
        try:
            user_url = user['url']
            uresp = requests.get(user_url, headers=headers, timeout=30)
            if uresp.status_code != 200:
                continue
            u = uresp.json()
            bio = u.get('bio') or ''
            if len(bio) < MIN_BIO_LEN:
                continue

            repos_resp = requests.get(u['repos_url'], headers=headers, timeout=30)
            if repos_resp.status_code != 200:
                continue
            repos = repos_resp.json()
            top_repos = sorted([r for r in repos if not r.get('fork')], key=lambda x: x.get('stargazers_count', 0), reverse=True)[:5]
            if not top_repos:
                continue

            # Fetch README for at least one top repo and do quick heuristic checks
            readme_texts = []
            for repo in top_repos[:3]:
                try:
                    readme_resp = requests.get(f"{GITHUB_API_BASE}/repos/{repo['full_name']}/readme", headers=headers, timeout=20)
                    if readme_resp.status_code == 200:
                        rd = readme_resp.json()
                        download_url = rd.get('download_url')
                        if download_url:
                            content_resp = requests.get(download_url, timeout=20)
                            if content_resp.status_code == 200 and len(content_resp.text) > 0:
                                readme_texts.append(content_resp.text)
                except Exception:
                    continue

            combined = bio + "\n" + "\n".join(readme_texts)
            # Heuristic: require at least one skill/achievement keyword in combined bio/readme or non-empty repo descriptions
            repo_descs = " ".join([r.get('description') or "" for r in top_repos])
            has_keyword = any(k in combined.lower() for k in SKILL_KEYWORDS) or any(k in repo_descs.lower() for k in SKILL_KEYWORDS)
            has_readme = any(len(r) >= MIN_README_LEN for r in readme_texts)
            has_repo_desc = any(r.get('description') for r in top_repos)

            if not (has_keyword and (has_readme or has_repo_desc)):
                logging.info(f"Skipping {u.get('login')}: missing skills/accomplishments indicators")
                continue

            # Construct seed prompt text similar to the existing generator format
            profile_text = f"**Login:** {u.get('login')}\n**Bio:**\n{bio}\n\n**Top Repos:**\n"
            for repo in top_repos:
                profile_text += f"- {repo['name']} ({repo.get('language')}) - {repo.get('stargazers_count', 0)} stars\n  - {repo.get('description') or ''}\n"

            profiles.append(profile_text)
            time.sleep(0.6)  # small pacing to be gentle on API
        except Exception as e:
            logging.debug(f"Skipping user {user.get('login') if user else 'unknown'} due to error: {e}")
            continue

    logging.info(f"Collected {len(profiles)} qualified GitHub seeds (requested {num_profiles})")
    return profiles


def load_local_seeds(path: str, num: int):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local seed file not found: {p}")
    arr = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # reuse seed detection from project (content|text|resume_text) or 'messages'
                text = None
                for k in ('content', 'text', 'resume_text'):
                    if isinstance(obj.get(k), str) and len(obj.get(k)) > 80:
                        text = obj[k]
                        break
                if not text and isinstance(obj.get('messages'), list):
                    msgs = [m.get('content') for m in obj['messages'] if isinstance(m.get('content'), str)]
                    if msgs:
                        text = "\n\n".join(msgs)
                if text:
                    arr.append(text)
            except Exception:
                continue
    logging.info(f"Loaded {len(arr)} local seeds from {p}")
    return arr[:num]


def load_main_dataset_sample(path: str, num: int):
    """Sample `num` resume texts from a large JSON array file using reservoir sampling.

    This avoids loading the entire file into memory by streaming through it with ijson.
    We extract candidate text fields similarly to the rest of the repo: 'content', 'text', 'resume_text'.
    Only candidates longer than ~120 chars are accepted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Main dataset not found: {p}")

    reservoir = []
    seen = 0
    MIN_LEN = 120

    with p.open('rb') as f:
        for obj in ijson.items(f, 'item'):
            seen += 1
            text = None
            for k in ('content', 'text', 'resume_text'):
                val = obj.get(k)
                if isinstance(val, str) and len(val.strip()) >= MIN_LEN:
                    text = val.strip()
                    break
            if not text:
                # fallback: try to join string fields of the object
                parts = [str(v).strip() for v in obj.values() if isinstance(v, str) and len(v.strip()) >= MIN_LEN]
                if parts:
                    text = "\n\n".join(parts)
            if not text:
                continue

            if len(reservoir) < num:
                reservoir.append(text)
            else:
                # reservoir sampling: replace with decreasing probability
                r = random.randint(0, seen - 1)
                if r < num:
                    reservoir[r] = text

    logging.info(f"Reservoir sampled {len(reservoir)} seeds from {seen} scanned records in {p}")
    return reservoir


def load_main_dataset_sample(path: str, sample_size: int):
    """Stream-sample `sample_size` seeds from a potentially large JSON or JSONL main dataset using reservoir sampling.
    Each line may be a JSON object or the file may be a single JSON array; we handle both streaming line JSONL and fallback to array parse if needed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Main dataset not found: {p}")

    reservoir = []
    total = 0

    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # Attempt to parse a JSON object from the line
                try:
                    obj = json.loads(line)
                except Exception:
                    # If parsing fails for a line, skip it
                    continue

                # Extract candidate text similar to other loaders
                text = None
                if isinstance(obj, dict):
                    for k in ('content', 'text', 'resume_text', 'description'):
                        val = obj.get(k)
                        if isinstance(val, str) and len(val) > 80:
                            text = val.strip()
                            break
                    if not text and isinstance(obj.get('messages'), list):
                        msgs = [m.get('content') for m in obj['messages'] if isinstance(m.get('content'), str)]
                        if msgs:
                            text = "\n\n".join(msgs)
                    if not text:
                        # As a last resort, concatenate string fields
                        concat = " ".join([str(v) for v in obj.values() if isinstance(v, str)])
                        if len(concat) > 80:
                            text = concat
                elif isinstance(obj, str) and len(obj) > 80:
                    text = obj

                if not text:
                    continue

                total += 1
                if len(reservoir) < sample_size:
                    reservoir.append(text)
                else:
                    # Replace with decreasing probability
                    import random as _random
                    s = _random.randint(0, total - 1)
                    if s < sample_size:
                        reservoir[s] = text

        if reservoir:
            logging.info(f"Sampled {len(reservoir)} seeds from main dataset ({path}). Processed {total} candidate lines.")
            return reservoir
        # If we didn't collect via streaming, try loading as a single JSON array (fallback)
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            arr = []
            for obj in data:
                text = None
                if isinstance(obj, dict):
                    for k in ('content', 'text', 'resume_text', 'description'):
                        val = obj.get(k)
                        if isinstance(val, str) and len(val) > 80:
                            text = val.strip()
                            break
                    if not text and isinstance(obj.get('messages'), list):
                        msgs = [m.get('content') for m in obj['messages'] if isinstance(m.get('content'), str)]
                        if msgs:
                            text = "\n\n".join(msgs)
                    if not text:
                        concat = " ".join([str(v) for v in obj.values() if isinstance(v, str)])
                        if len(concat) > 80:
                            text = concat
                elif isinstance(obj, str) and len(obj) > 80:
                    text = obj
                if text:
                    arr.append(text)
            if not arr:
                raise RuntimeError(f"No usable text entries found in main dataset: {p}")
            import random as _rnd
            sampled = _rnd.sample(arr, min(sample_size, len(arr)))
            logging.info(f"Loaded main dataset as array and sampled {len(sampled)} seeds")
            return sampled

    except Exception as e:
        raise RuntimeError(f"Failed to sample main dataset: {e}")


def test_paid_model(model_id: str):
    """Make a quick test call to the paid OpenRouter model to ensure it is reachable and returns expected structure."""
    logging.info(f"Testing paid model: {model_id}")
    if not OPENROUTER_API_KEY:
        return False, "OPENROUTER_API_KEY not set"
    prompt = [
        {"role": "system", "content": "You are a world-class tech resume writer."},
        {"role": "user", "content": "Summarize this short developer profile into a one-sentence headline: 'Bio: I build distributed systems; Top project: high-perf router (2000 stars); Skills: Python, Rust.'"}
    ]
    try:
        resp = requests.post(
            url=f"{OPENROUTER_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": model_id, "messages": prompt},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Basic sanity checks
            if isinstance(data, dict) and 'choices' in data and data['choices']:
                content = data['choices'][0]['message'].get('content')
                if isinstance(content, str) and len(content) > 5:
                    return True, content
                return False, "Model returned no usable content"
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, str(e)


def call_model_once(model_id: str, prompt_messages, timeout=DEFAULT_TIMEOUT):
    """Single attempt to call a paid OpenRouter model with configurable timeout."""
    try:
        resp = requests.post(
            url=f"{OPENROUTER_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": model_id, "messages": prompt_messages},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message'].get('content')
        return content, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def generate_for_model(model_id: str, seeds: list, per_model: int, out_file: Path, workers: int = 2):
    """Generate `per_model` outputs for `model_id` using `workers` parallel workers and write to `out_file` (JSONL)."""

    logging.info(f"Starting generation for {model_id}: target={per_model}, workers={workers}")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required to call paid models")

    # If we don't have enough seeds, allow sampling with replacement
    if not seeds:
        raise RuntimeError("No seeds provided for generation")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    futures = []
    written = 0

    def worker_task(seed_text, idx):
        nonlocal written
        prompt = [
            {"role": "system", "content": "You are a world-class tech resume writer. Rewrite the candidate's bio into a concise, achievement-focused resume summary and 3 strong bullet achievements based on their top projects and skills."},
            {"role": "user", "content": seed_text}
        ]

        for attempt in range(1, MAX_RETRIES + 1):
            content, err = call_model_once(model_id, prompt, timeout=DEFAULT_TIMEOUT)
            if content:
                obj = {
                    "model": model_id,
                    "seed": seed_text,
                    "generation": content,
                    "attempt": attempt,
                    "timestamp": int(time.time())
                }
                line = json.dumps(obj, ensure_ascii=False)
                # Write atomically by appending
                with out_file.open('a', encoding='utf-8') as f:
                    f.write(line + '\n')
                logging.info(f"[{model_id}] Generated item {idx}")
                return True, None
            else:
                logging.warning(f"[{model_id}] Attempt {attempt} failed: {err}")
                time.sleep(RETRY_DELAY)
        return False, f"All {MAX_RETRIES} attempts failed for seed idx {idx}"

    # Use ThreadPoolExecutor and submit tasks — pick seeds by sampling (with replacement) if needed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i in range(per_model):
            seed = random.choice(seeds)
            futures.append(ex.submit(worker_task, seed, i + 1))

        for fut in as_completed(futures):
            success, msg = fut.result()
            if not success:
                logging.error(msg)

    logging.info(f"Completed generation for {model_id}. Output file: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=200)
    parser.add_argument('--per_model', type=int, default=150)
    parser.add_argument('--workers', type=int, default=2, help='Workers per model (2 means total 4)')
    parser.add_argument('--seed_source', choices=['github', 'local', 'main'], default='github')
    parser.add_argument('--seed_local_file', type=str, default='training-data/formatted/high_quality_seeds_1000.jsonl')
    parser.add_argument('--main_dataset_path', type=str, default='data/anonymized_combined_resume_dataset.json')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # 1) Quick test of models (do this before fetching seeds when running --test-only)
    ok_sky, reply_sky = test_paid_model(SKYFALL)
    ok_grok, reply_grok = test_paid_model(GROK)

    logging.info(f"Skyfall test: ok={ok_sky}, sample_reply={str(reply_sky)[:200]}")
    logging.info(f"Grok test: ok={ok_grok}, sample_reply={str(reply_grok)[:200]}")

    if args.test_only:
        logging.info("Test-only mode; exiting after model tests.")
        return

    if not ok_sky or not ok_grok:
        raise RuntimeError("One or more paid models failed the test. Check OPENROUTER_API_KEY and model availability before running.")

    # 2) Gather seeds (only after tests pass)
    seeds = []
    if args.seed_source == 'github':
        seeds = fetch_github_profiles(args.num_seeds)
        if len(seeds) < args.num_seeds:
            logging.warning(f"Only fetched {len(seeds)} GitHub seeds; attempting to load local seeds to fill up to {args.num_seeds}.")
            local = load_local_seeds(args.seed_local_file, args.num_seeds - len(seeds))
            seeds.extend(local)
    elif args.seed_source == 'local':
        seeds = load_local_seeds(args.seed_local_file, args.num_seeds)
    else:
        # 'main' dataset sampling
        logging.info(f"Sampling {args.num_seeds} seeds from main dataset: {args.main_dataset_path}")
        seeds = load_main_dataset_sample(args.main_dataset_path, args.num_seeds)

    if not seeds:
        raise RuntimeError("No seeds available — aborting")

    # 3) Generate for each model separately using two workers each (total 4 concurrent workers across both models if run in parallel)
    sky_out = OUTPUT_DIR / 'skyfall_imagin.jsonl'
    grok_out = OUTPUT_DIR / 'grok_imagin.jsonl'

    # Remove files if present to start fresh
    for f in (sky_out, grok_out):
        if f.exists():
            logging.info(f"Removing existing output file: {f}")
            f.unlink()

    # We'll run both generators in parallel (two ThreadPoolExecutors), but each generator will itself use workers=2
    with ThreadPoolExecutor(max_workers=2) as top_ex:
        future_sky = top_ex.submit(generate_for_model, SKYFALL, seeds, args.per_model, sky_out, args.workers)
        future_grok = top_ex.submit(generate_for_model, GROK, seeds, args.per_model, grok_out, args.workers)

        # Wait for completion
        for fut in as_completed([future_sky, future_grok]):
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Generation exception: {e}")

    logging.info("All generation jobs completed.")


if __name__ == '__main__':
    main()
