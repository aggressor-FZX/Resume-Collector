#!/usr/bin/env python3
"""
Generates the "Imaginator-Specific" fine-tuning dataset using a cascade of free models.

This script implements a cost-effective strategy to generate high-quality synthetic data.
It starts with a list of top-tier free models and cycles through them. If a model
fails (e.g., due to rate limiting), it automatically switches to the next one.

If all free models are exhausted before the target number of examples is reached,
it reports the remaining count, allowing for completion with a paid model.
"""

import os
import json
import random
import time
import requests
from pathlib import Path
import argparse
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import threading
import signal
from datasets import Dataset
from huggingface_hub import HfApi, login, InferenceClient
from collections import defaultdict
import queue
import subprocess
# Optional OpenAI library for SambaNova/Groq integration; make import non-fatal for dev environments
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None
    logging.warning("Optional 'openai' library not available; SambaNova/Groq calls will be disabled or use fallbacks.")

# --- Logging Setup ---
log_filename = f"logs/imaginator_generation_{int(time.time())}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

load_dotenv()

# --- Configuration ---
RAW_DATA_DIR = Path('data/raw')
OUTPUT_DIR = Path('training-data/imaginator_generated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
# Use .env Hugging Face key for uploads and API
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY") or os.getenv("HF_TOKEN")
SAMBA_API_KEY = os.getenv("SAMBA_API_KEY")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
HF_API_BASE = "https://api-inference.huggingface.co/v1"
SAMBANOVA_API_BASE = "https://api.sambanova.ai/v1"
GITHUB_API_BASE = "https://api.github.com"
YOUR_SITE_URL = "https://github.com/aggressor-FZX/Resume-Collector"
YOUR_APP_NAME = "ResumeCollectorImaginator"
# Optional Hugging Face repo id for upload (owner/repo)
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# --- Model Cascade ---
"""
Expanded model cascade to include top Hugging Face open-weight models (as of late 2025):
  - deepseek-ai/DeepSeek-R1 (671B MoE)
  - meta-llama/Llama-3.3-70B-Instruct (70B)
  - Qwen/Qwen2.5-72B-Instruct (72B)
  - mistralai/Mistral-Large-Instruct-2407 (123B)
  - CohereForAI/c4ai-command-r-plus (104B)
  - Qwen/Qwen2.5-Coder-32B-Instruct (32B)
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (32B)
  - Qwen/Qwen2.5-VL-72B-Instruct (72B, multimodal)
  - google/gemma-2-27b-it (27B)
  - mistralai/Mistral-Small-Instruct-2407 (24B)
  - tiiuae/falcon-11b (11B)
  - plus legacy OpenRouter free models for fallback.
"""
MODEL_CASCADE = [
    # SambaNova Production Models (>17B)
    "DeepSeek-R1-0528",
    "DeepSeek-V3.1",
    "DeepSeek-R1-Distill-Llama-70B",
    "Meta-Llama-3.3-70B-Instruct",
    # SambaNova Preview Models (>17B)
    "Llama-4-Maverick-17B-128E-Instruct",
    "gpt-oss-120b",
    "Qwen3-32B",
    "Llama-3.3-Swallow-70B-Instruct-v0.4",
    # Hugging Face Inference API models (open-weight, >10B), prioritized for free tier
    "deepseek/deepseek-r1-distill-qwen-32b", # Best balance of logic/speed on free tier
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    # Legacy OpenRouter free models (for fallback)

    "nex-agi/deepseek-v3.1-nex-n1:free",

    "google/gemma-3-27b-it:free",
    # Paid OpenRouter models (for dedicated pool of 3 workers)
    "nousresearch/hermes-4-70b",
    "thedrummer/rocinante-12b",
    "deepseek/deepseek-v3.2",
    "thedrummer/skyfall-36b-v2",
    "deepseek/deepseek-v3.2-speciale",
]

HF_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
}

model_cycler = cycle(MODEL_CASCADE)
model_lock = threading.Lock()
# Track temporary cooldowns (timestamp) for models that hit rate limits
model_cooldowns = {}
# Models that are disabled for the duration of this run (e.g., 410 Gone)
banned_models = set()
# Rate limit for OpenRouter free tier is ~300 requests per hour (~1 req/sec). SambaNova/HF also have limits.
# Make OpenRouter free-tier concurrency configurable via env var OPENROUTER_CONCURRENCY (default 9).
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "9"))
api_call_semaphore = threading.Semaphore(OPENROUTER_CONCURRENCY) # Allows concurrent OpenRouter free-tier calls

# Spacing controls for paid OpenRouter calls to tune RPS without changing worker counts.
# - OPENROUTER_PAID_SPACING: seconds to sleep *before* each paid call (default 0.5)
# - OPENROUTER_AFTER_SUCCESS_WAIT: seconds to sleep *after* a successful scenario generation (default 1.1)
OPENROUTER_PAID_SPACING = float(os.getenv("OPENROUTER_PAID_SPACING", "0.5"))
OPENROUTER_AFTER_SUCCESS_WAIT = float(os.getenv("OPENROUTER_AFTER_SUCCESS_WAIT", "2"))
# Limit concurrent SambaNova requests to 3
samba_semaphore = threading.Semaphore(3)
# Limit concurrent paid OpenRouter requests to 3
paid_openrouter_semaphore = threading.Semaphore(3)
# If we detect an authentication/entitlement failure for OpenRouter paid models, set this flag to avoid repeated 401s
openrouter_auth_failed = False
# Support repeating the same model after success
last_success_model = None
repeat_on_success = False
# CLI-populated runtime blacklist (models to avoid this run)
runtime_blacklist = set()
# Global append mode flag (set from args)
append_mode = False

# Toggle used to alternate preference between Samba and paid providers (thread-safe by using model_lock)
prefer_paid_toggle = False


def is_samba_model(model: str) -> bool:
    """Detect SambaNova model ids (they are bare ids without owner slash or OpenRouter ':free' suffix)."""
    return (":" not in model) and ("/" not in model)


def is_paid_openrouter_model(model: str) -> bool:
    """Detects if a model is a paid OpenRouter model based on its ID structure.
    Paid OpenRouter models do not have ':free' in their ID but do typically
    have an owner/model_name structure.
    """
    # A rough heuristic: if it has a slash and is NOT a SambaNova model, and is NOT a free OpenRouter model
    return ("/" in model) and (":free" not in model) and (not is_samba_model(model))

# --- Paid OpenRouter rotation ---
PAID_OPENROUTER_MODELS = [m for m in MODEL_CASCADE if is_paid_openrouter_model(m)]
paid_model_cycler = cycle(PAID_OPENROUTER_MODELS) if PAID_OPENROUTER_MODELS else None

# Validate paid OpenRouter model IDs at startup and suggest canonical replacements if necessary.
# This performs a lightweight /models lookup and maps configured IDs to available canonical slugs.
# If models are missing, a log entry is emitted and the missing models are removed from the rotation.
# Callers may still see runtime 401/404 errors for entitlement or privacy settings; those are handled
# in make_paid_openrouter_api_call with clearer diagnostics.

def validate_paid_openrouter_models():
    global PAID_OPENROUTER_MODELS, paid_model_cycler
    if not OPENROUTER_API_KEY:
        logging.debug("OPENROUTER_API_KEY not set; skipping paid OpenRouter model validation.")
        return
    try:
        resp = requests.get(f"{OPENROUTER_API_BASE}/models", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=30)
        if resp.status_code != 200:
            logging.warning(f"Failed to fetch OpenRouter /models for validation: {resp.status_code} {resp.text[:200]}")
            return
        data = resp.json()
        models_list = None
        if isinstance(data, dict) and 'data' in data:
            if isinstance(data['data'], list):
                models_list = data['data']
            elif isinstance(data['data'], dict) and 'models' in data['data']:
                models_list = data['data']['models']
        elif isinstance(data, dict) and 'models' in data:
            models_list = data['models']
        elif isinstance(data, list):
            models_list = data
        if not models_list:
            logging.warning("Unexpected /models structure during validation; skipping paid model mapping.")
            return
        available_ids = [m.get('id') or m.get('canonical_slug') or m.get('name') for m in models_list]
        available_ids = [a for a in available_ids if a]
        # Map configured PAID_OPENROUTER_MODELS to available canonical slugs, using exact, substring, or fuzzy matches.
        import difflib
        replacements = {}
        new_list = []
        for m in PAID_OPENROUTER_MODELS:
            target = m
            # exact
            match = next((c for c in available_ids if c.lower() == target.lower()), None)
            if match:
                new_list.append(match)
                continue
            # substring
            match = next((c for c in available_ids if target.lower() in c.lower() or c.lower() in target.lower()), None)
            if match:
                replacements[m] = match
                new_list.append(match)
                continue
            # fuzzy
            close = difflib.get_close_matches(m, available_ids, n=1, cutoff=0.6)
            if close:
                replacements[m] = close[0]
                new_list.append(close[0])
                continue
            # missing
            logging.warning(f"Configured paid OpenRouter model is missing from /models listing: {m}")
        if replacements:
            logging.info("Paid OpenRouter model mapping suggestions:")
            for k, v in replacements.items():
                logging.info(f"  {k} -> {v}")
        # Replace only if any canonical matches were found; otherwise keep original list except missing ones removed
        if new_list:
            PAID_OPENROUTER_MODELS = new_list
            paid_model_cycler = cycle(PAID_OPENROUTER_MODELS) if PAID_OPENROUTER_MODELS else None
            logging.info(f"Using {len(PAID_OPENROUTER_MODELS)} validated paid OpenRouter models for rotation.")
    except Exception as e:
        logging.warning(f"Exception while validating paid OpenRouter models: {e}")

# Run validation at startup so subsequent runtime calls are more likely to succeed.
validate_paid_openrouter_models()


def get_next_model():
    """Return the next available model, skipping banned models and those on cooldown.
    If all models are on cooldown, wait until the earliest cooldown expires.
    """
    with model_lock:
        # If repeat-on-success is enabled and we previously had a working model, prefer it first
        global last_success_model, repeat_on_success, prefer_paid_toggle
        if repeat_on_success and last_success_model:
            m = last_success_model
            if m not in banned_models:
                cd = model_cooldowns.get(m)
                if not cd or time.time() >= cd:
                    logging.debug(f"Repeating last successful model: {m}")
                    return m

        start_model = next(model_cycler)
        current_model = start_model

        # Alternate preference between Samba and paid models to distribute usage and ensure paid models rotate even when Samba is available.
        # Use the `prefer_paid_toggle` to alternate each time this function is called (thread-safe via model_lock).
        prefer_paid = prefer_paid_toggle
        prefer_paid_toggle = not prefer_paid_toggle

        def try_prefer_samba_first():
            try:
                if samba_semaphore.acquire(blocking=False):
                    # Immediately release - we only check availability here
                    samba_semaphore.release()
                    for m in MODEL_CASCADE:
                        if is_samba_model(m) and m not in banned_models:
                            cooldown_until = model_cooldowns.get(m)
                            if not cooldown_until or time.time() >= cooldown_until:
                                logging.debug(f"Preferring Samba model: {m}")
                                return m
            except Exception:
                pass
            return None

        def try_prefer_paid_first():
            try:
                # If we've detected an account-level OpenRouter auth failure, skip paid models entirely
                if openrouter_auth_failed:
                    logging.debug("OpenRouter paid auth previously failed; skipping paid models.")
                    return None

                if paid_openrouter_semaphore.acquire(blocking=False):
                    paid_openrouter_semaphore.release()
                    # Use a dedicated cycler for paid models to rotate through them fairly and avoid constant retries
                    if paid_model_cycler:
                        for _ in range(len(PAID_OPENROUTER_MODELS)):
                            m = next(paid_model_cycler)
                            if m in banned_models:
                                logging.debug(f"Skipping banned paid model: {m}")
                                continue
                            cooldown_until = model_cooldowns.get(m)
                            if cooldown_until and time.time() < cooldown_until:
                                logging.debug(f"Paid model {m} on cooldown until {cooldown_until}. Skipping.")
                                continue
                            logging.debug(f"Preferring paid OpenRouter model (rotated): {m}")
                            return m
                    else:
                        # Fallback: scan the model cascade if no paid cycler is configured
                        for m in MODEL_CASCADE:
                            if is_paid_openrouter_model(m) and m not in banned_models:
                                cooldown_until = model_cooldowns.get(m)
                                if not cooldown_until or time.time() >= cooldown_until:
                                    logging.debug(f"Preferring paid OpenRouter model: {m}")
                                    return m
            except Exception:
                pass
            return None

        # Try in the selected order
        if prefer_paid:
            paid_choice = try_prefer_paid_first()
            if paid_choice:
                return paid_choice
            samba_choice = try_prefer_samba_first()
            if samba_choice:
                return samba_choice
        else:
            samba_choice = try_prefer_samba_first()
            if samba_choice:
                return samba_choice
            paid_choice = try_prefer_paid_first()
            if paid_choice:
                return paid_choice

        while True:
            # Skip if explicitly banned for this run
            if current_model in banned_models:
                logging.debug(f"Skipping banned model: {current_model}")
                current_model = next(model_cycler)
                if current_model == start_model:
                    logging.error("All models are banned for this run. Waiting briefly before retrying.")
                    time.sleep(5)
                    continue
                continue

            # Skip models currently on cooldown
            cooldown_until = model_cooldowns.get(current_model)
            if cooldown_until and time.time() < cooldown_until:
                logging.debug(f"Model {current_model} on cooldown until {cooldown_until}. Skipping.")
                current_model = next(model_cycler)
                if current_model == start_model:
                    # All models are on cooldown — wait until the earliest cooldown ends
                    earliest = min(model_cooldowns.values())
                    sleep_time = max(0, earliest - time.time())
                    logging.info(f"All models on cooldown. Sleeping for {sleep_time + 1:.1f}s until a model is available.")
                    time.sleep(sleep_time + 1)
                continue

            # Model is available
            # Remove stale cooldowns
            if current_model in model_cooldowns and (not model_cooldowns[current_model] or time.time() >= model_cooldowns[current_model]):
                del model_cooldowns[current_model]
            return current_model

def fetch_github_profiles(num_profiles: int):
    """
    Fetches high-quality developer profiles from GitHub to use as seed data.
    """
    print("--- Fetching seed data from GitHub ---")
    profiles = []
    headers = {"Authorization": f"token {GITHUB_API_KEY}"}
    # A query to find active, influential developers
    query = "language:python language:go language:typescript followers:>500"
    
    try:
        response = requests.get(
            f"{GITHUB_API_BASE}/search/users",
            params={'q': query, 'per_page': num_profiles + 20}, # Fetch extra to filter
            headers=headers
        )
        response.raise_for_status()
        users = response.json().get('items', [])

        for user in users:
            if len(profiles) >= num_profiles:
                break
            
            user_url = user['url']
            user_data_res = requests.get(user_url, headers=headers)
            if user_data_res.status_code != 200: continue
            user_data = user_data_res.json()

            if not user_data.get('bio'): continue # Skip profiles without a bio

            repos_res = requests.get(user_data['repos_url'], headers=headers)
            if repos_res.status_code != 200: continue
            repos = repos_res.json()
            
            top_repos = sorted([r for r in repos if not r.get('fork')], key=lambda x: x.get('stargazers_count', 0), reverse=True)[:3]

            if not top_repos: continue

            profile_text = f"**Bio:**\n{user_data['bio']}\n\n**Top Repositories:**\n"
            for repo in top_repos:
                profile_text += f"- **Project:** `{repo['name']}` ({repo.get('language')})\n"
                profile_text += f"  - **Impact:** {repo.get('stargazers_count', 0)} GitHub stars\n"
                profile_text += f"  - **Description:** {repo.get('description')}\n"

            profiles.append(profile_text)
        
        print(f"Successfully fetched {len(profiles)} high-quality profiles from GitHub.")
        return profiles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GitHub: {e}. Falling back to local raw data.")
        return get_raw_resumes(num_profiles) # Fallback to original method

def get_raw_resumes(sample_size: int):
    """
    Scans the raw data directory for resume files and samples them.
    Handles nested directories and different file formats (.json, .jsonl).
    """
    all_files = list(RAW_DATA_DIR.glob('**/*.json')) + list(RAW_DATA_DIR.glob('**/*.jsonl'))
    if not all_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {RAW_DATA_DIR}")

    resumes = []
    for file_path in all_files:
        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Heuristically find resume text
                        text_candidates = []
                        if isinstance(data, dict):
                            for key in ['resume_text', 'text', 'content', 'description', 'resume']:
                                if isinstance(data.get(key), str):
                                    text_candidates.append(data[key])
                            if not text_candidates:
                                 # Fallback: combine all string values
                                text_candidates.append(" ".join([str(v) for v in data.values() if isinstance(v, str)]))
                        elif isinstance(data, str):
                            text_candidates.append(data)
                        
                        for text in text_candidates:
                            if text and len(text) > 200: # Filter for reasonably long texts
                                resumes.append(text)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except FileNotFoundError:
            # File disappeared between glob and open; skip it
            continue
    
    if len(resumes) < sample_size:
        print(f"Warning: Found only {len(resumes)} suitable resume texts, less than the requested sample size of {sample_size}.")
        return resumes
    
    return random.sample(resumes, sample_size)


def load_local_github_seeds(local_path: str, sample_size: int):
    """Load seed resume texts from a local GitHub-prepared JSONL file.

    Supports two formats:
    - legacy flat objects with keys like `content`, `text`, or `resume_text`.
    - OpenAI-style `messages` objects (as produced by `prepare_for_llm.py`).

    Provides detailed diagnostics when no usable entries are found.
    """
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Local seed file not found: {p}")

    SEED_MIN_LENGTH = int(os.getenv("LOCAL_SEED_MIN_LENGTH", 100))

    arr = []
    decoder = json.JSONDecoder()

    total_lines_read = 0
    total_json_objects = 0
    total_with_text_candidate = 0

    with p.open('r', encoding='utf-8') as f:
        for raw_line in f:
            total_lines_read += 1
            line = raw_line.strip()
            if not line:
                continue
            idx = 0
            # It is possible a single physical line contains multiple JSON objects; iterate until consumed
            while idx < len(line):
                try:
                    obj, end = decoder.raw_decode(line[idx:])
                    total_json_objects += 1
                    idx += end

                    # 1) Prefer explicit top-level text fields
                    text = None
                    for key in ('content', 'text', 'resume_text'):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            text = val.strip()
                            break

                    # 2) Support the `messages` format (list of role/content dicts)
                    if not text and isinstance(obj.get('messages'), list):
                        msgs = obj['messages']
                        # Prefer user messages (these contain the candidate bio + projects)
                        user_msgs = [m for m in msgs if m.get('role') == 'user' and isinstance(m.get('content'), str)]
                        if user_msgs:
                            text = "\n\n".join([m['content'].strip() for m in user_msgs])
                        else:
                            # Fallback: concatenate any available message contents
                            contents = [m.get('content') for m in msgs if isinstance(m.get('content'), str)]
                            if contents:
                                text = "\n\n".join([c.strip() for c in contents])

                    # 3) Fallback: if the whole object is a string
                    if not text and isinstance(obj, str):
                        text = obj

                    if isinstance(text, str):
                        total_with_text_candidate += 1
                        # Accept prepared prompts above the configured threshold
                        if len(text) > SEED_MIN_LENGTH:
                            arr.append(text)

                    # Skip any separating whitespace or commas, and also handle literal escaped newlines (\n) inserted accidentally
                    while idx < len(line):
                        if line[idx] in ',\n\r\t ':
                            idx += 1
                            continue
                        if line.startswith('\\n', idx):
                            idx += 2
                            continue
                        break
                except json.JSONDecodeError:
                    # If we can't decode, break out to avoid infinite loop and move to next physical line
                    break

    logging.info(f"Local seed diagnostics: lines={total_lines_read}, json_objects={total_json_objects}, with_text_candidate={total_with_text_candidate}, accepted={len(arr)}, min_len={SEED_MIN_LENGTH}")

    if not arr:
        raise FileNotFoundError(
            f"No usable entries found in local seed file: {p} (lines={total_lines_read}, json_objects={total_json_objects}, with_text_candidate={total_with_text_candidate})"
        )
    return random.sample(arr, min(sample_size, len(arr)))

def make_api_call(prompt_messages, model):
    """
    Dispatcher: OpenRouter if model contains ':free', Hugging Face if it contains '/', otherwise SambaNova (bare id).
    """
    if not model:
        return None, "No more models available in the cascade."

    if ":free" in model:
        return make_openrouter_api_call(prompt_messages, model)
    elif model in HF_MODELS:
        return make_hf_api_call(prompt_messages, model)
    elif is_paid_openrouter_model(model):
        return make_paid_openrouter_api_call(prompt_messages, model)
    else:
        return make_sambanova_api_call(prompt_messages, model)

def make_openrouter_api_call(prompt_messages, model):
    """Makes a call to the OpenRouter API with a given model and prompt."""
    logging.info(f"  Attempting generation with OpenRouter model: {model}...")
    try:
        with api_call_semaphore:
            response = requests.post(
                url=f"{OPENROUTER_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": YOUR_SITE_URL,
                    "X-Title": YOUR_APP_NAME,
                },
                json={
                    "model": model,
                    "messages": prompt_messages,
                    "response_format": {"type": "json_object"},
                },
                timeout=180
            )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], None
    except requests.exceptions.RequestException as e:
        error_message = f"API call failed for model {model}: {e}"
        logging.error(f"    - {error_message}")
        # Check for 402 Payment Required to detect exhausted monthly credits
        if e.response and e.response.status_code == 402:
            logging.warning("OpenRouter free credits may be exhausted (402 Payment Required).")
        elif e.response and e.response.status_code == 400: # Add 400 to banned models
            banned_models.add(model)
            error_message += ". Model returned 400 Bad Request; banning for the remainder of this run."
        return None, error_message

def make_paid_openrouter_api_call(prompt_messages, model):
    """Makes a call to the OpenRouter API with a paid model and prompt.
    Uses a semaphore to limit concurrency.
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY not set for paid OpenRouter model call."

    logging.info(f"  Attempting generation with Paid OpenRouter model: {model}...")
    acquired = False
    try:
        acquired = paid_openrouter_semaphore.acquire(blocking=False)
        if not acquired:
            return None, "Paid OpenRouter concurrency limit reached; try next model."

        # Respect configurable spacing to control paid OpenRouter request rate.
        time.sleep(OPENROUTER_PAID_SPACING) # Add a small delay to further space out requests

        response = requests.post(
            url=f"{OPENROUTER_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_APP_NAME,
            },
            json={
                "model": model,
                "messages": prompt_messages,
                "response_format": {"type": "json_object"},
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], None
    except requests.exceptions.RequestException as e:
        error_message = f"API call failed for paid OpenRouter model {model}: {e}"
        logging.error(f"    - {error_message}")
        if e.response and e.response.status_code == 402:
            logging.warning("OpenRouter paid credits may be exhausted (402 Payment Required).")
        elif e.response and e.response.status_code == 401:
            # Account-level authentication/entitlement failure for OpenRouter paid models.
            # Disable paid models for the remainder of this run and set a flag to avoid repeated 401s.
            global openrouter_auth_failed
            openrouter_auth_failed = True
            for pm in PAID_OPENROUTER_MODELS:
                banned_models.add(pm)
            error_message += ". Received 401 Unauthorized for paid OpenRouter models — disabling paid OpenRouter models for this run. Please verify OPENROUTER_API_KEY and account entitlements."
        elif e.response and e.response.status_code in [400, 404]:
            # Distinguish between generic 400/404 and OpenRouter-specific "No endpoints found matching your data policy" case
            try:
                body = e.response.json()
                msg = None
                if isinstance(body, dict):
                    msg = body.get('error', {}).get('message') or body.get('message') or str(body)
                else:
                    msg = str(body)
            except Exception:
                msg = e.response.text[:500]

            if isinstance(msg, str) and 'No endpoints found matching your data policy' in msg:
                logging.error(
                    f"Paid OpenRouter model {model} exists but is blocked by your account data/privacy settings: {msg}."
                )
                error_message += ". Model requires OpenRouter account data/privacy configuration (see https://openrouter.ai/settings/privacy); banning for this run."
            else:
                error_message += ". Model returned 400/404; banning for the remainder of this run."
            banned_models.add(model)
        elif e.response and e.response.status_code in [429, 503]:
            cooldown_seconds = 600  # 10 minutes
            model_cooldowns[model] = time.time() + cooldown_seconds
            error_message += f". Model is rate-limited or unavailable. Placing on cooldown for {cooldown_seconds} seconds."
        return None, error_message
    finally:
        if acquired:
            try:
                paid_openrouter_semaphore.release()
            except Exception:
                pass

def make_hf_api_call(prompt_messages, model):
    """Makes a call to the Hugging Face Inference API with a given model and prompt."""
    if not HF_TOKEN:
        return None, "HUGGING_FACE_API_KEY not set for HF model call."

    logging.info(f"  Attempting generation with HF model: {model}...")
    try:
        with api_call_semaphore:
            client = InferenceClient(model=model, token=HF_TOKEN)

            # Use the chat interface - it handles the prompt template for you
            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages, # Directly pass the messages array
                max_tokens=2048, # Increased token limit for generation
                stream=False,
                temperature=0.7,
                top_p=0.9,
                seed=random.randint(0, 1000000), # Use a random seed for diverse outputs
            )

        return response.choices[0].message.content, None
    except Exception as e:
        err_str = str(e)
        error_message = f"API call failed for HF model {model}: {err_str}"
        # Treat model-not-found or similar as a ban for this run
        if "model not found" in err_str.lower() or "does not exist" in err_str.lower() or "model_not_found" in err_str.lower():
            banned_models.add(model)
            error_message += ". Model not found for HF inference; banning for the remainder of this run."
            logging.error(f"    - {error_message}")
            return None, error_message
        # InferenceClient raises InferenceTimeoutError for timeouts or model loading (similar to 503)
        if "InferenceTimeoutError" in err_str:
            error_message += ". Model may be loading or timed out; will be retried later in the cascade."
            # No explicit cooldown for InferenceTimeoutError here, rely on client's internal retries
        elif "410 Client Error: Gone" in err_str:
            banned_models.add(model)
            error_message += ". Model returned 410 Gone; disabling for the remainder of this run."
        elif "429 Client Error" in err_str or "503 Service Unavailable" in err_str or "Too Many Requests" in err_str:
            cooldown_seconds = 600  # 10 minutes (increased due to persistent rate limiting)
            model_cooldowns[model] = time.time() + cooldown_seconds
            error_message += f". Model is rate-limited or unavailable. Placing on cooldown for {cooldown_seconds} seconds."
        logging.error(f"    - {error_message}")
        return None, error_message
    except requests.exceptions.RequestException as e:
        error_message = f"API call failed for HF model {model}: {e}"
        # Treat 410 Gone as a sign the model is not available for this run; ban it.
        if hasattr(e, 'response') and e.response and e.response.status_code == 410:
            banned_models.add(model)
            error_message += ". Model returned 410 Gone; disabling for the remainder of this run."
        # Check for rate-limiting or service unavailable errors to trigger a cooldown
        elif hasattr(e, 'response') and e.response and e.response.status_code in [429, 503]:
            cooldown_seconds = 600  # 10 minutes (increased due to persistent rate limiting)
            model_cooldowns[model] = time.time() + cooldown_seconds
            error_message += f". Model is rate-limited or unavailable. Placing on cooldown for {cooldown_seconds} seconds."
        logging.error(f"    - {error_message}")
        return None, error_message

def make_sambanova_api_call(prompt_messages, model):
    """Makes a call to the SambaNova API with a given model and prompt.
    Uses a semaphore to limit concurrency and bans models that do not exist on SambaNova.
    """
    if not SAMBA_API_KEY:
        return None, "SAMBANOVA_API_KEY not set for SambaNova model call."

    logging.info(f"  Attempting generation with SambaNova model: {model}...")
    acquired = False
    try:
        # Try to acquire a Samba slot without blocking; if none available, let caller try next model
        acquired = samba_semaphore.acquire(blocking=False)
        if not acquired:
            return None, "Samba concurrency limit reached; try next model."

        # Add a small delay to further space out requests, even with the semaphore
        time.sleep(0.5)

        client = OpenAI(
            base_url=SAMBANOVA_API_BASE,
            api_key=SAMBA_API_KEY,
            max_retries=0, # Exponential backoff retries for rate limits or transient errors
            timeout=180, # Global timeout for the request
        )
        
        # SambaNova uses OpenAI-compatible chat completions
        response = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        error_message = f"API call failed for SambaNova model {model}: {e}"
        # If Samba returns a 400/404 indicating model not found, ban it for this run
        try:
            if hasattr(e, 'response') and e.response and getattr(e.response, 'status_code', None) in [400, 404]:
                banned_models.add(model)
                error_message += ". Model not found on SambaNova; banning for the remainder of this run."
            elif hasattr(e, 'response') and e.response and getattr(e.response, 'status_code', None) == 429:
                cooldown_seconds = 600  # 10 minutes
                model_cooldowns[model] = time.time() + cooldown_seconds
                error_message += f". Model is rate-limited. Placing on cooldown for {cooldown_seconds} seconds."
        except Exception:
            pass
        logging.error(f"    - {error_message}")
        return None, error_message
    finally:
        if acquired:
            try:
                samba_semaphore.release()
            except Exception:
                pass

def parse_json_from_response(response_text: str):
    """Robustly extract a JSON object from model responses.

    Strategies (in order):
    1. Try json.loads on the whole response (handles pure JSON or JSON arrays).
    2. If it's a list, scan entries for a dict with expected keys or 'parameters' containing a nested JSON string.
    3. Strip markdown fences and attempt to find the first balanced JSON object by scanning for matching braces.
    4. Search for nested JSON strings inside keys like "json" and attempt to unescape and parse.

    Returns (obj, None) on success or (None, error_message) on failure.
    """
    text = response_text or ""

    # 1) Try direct parse
    try:
        parsed = json.loads(text)
        # If parsed is list, try to find useful dict inside
        if isinstance(parsed, dict):
            return parsed, None
        if isinstance(parsed, list) and parsed:
            # Try common patterns: list of dicts where dict contains our keys
            for entry in parsed:
                if isinstance(entry, dict):
                    # If entry itself looks like the scenario, return it
                    if any(k in entry for k in ("job_ad_text", "extracted_skills_json", "domain_insights_json")):
                        return entry, None
                    # Handle function-call wrapper: {'type':..., 'parameters': {'json': '...'}}
                    params = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else None
                    if params and 'json' in params:
                        try:
                            inner = json.loads(params['json'])
                            if isinstance(inner, dict):
                                return inner, None
                        except Exception:
                            pass
            # Fall through to other strategies if nothing suitable
    except Exception:
        pass

    # 2) Clean common markdown fences and code blocks
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):].lstrip()
    if stripped.startswith("```"):
        stripped = stripped[3:].lstrip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()

    # 3) Attempt to find the first balanced JSON object by scanning for braces
    def find_balanced_json(s: str):
        for i, ch in enumerate(s):
            if ch != '{':
                continue
            depth = 0
            for j in range(i, len(s)):
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = s[i:j+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
        return None

    result = find_balanced_json(stripped)
    if result is not None:
        return result, None

    # 4) Search for nested JSON string like "json": "{...}"
    try:
        import re
        m = re.search(r'"json"\s*:\s*"(\{.*?\})"', text, flags=re.S)
        if m:
            inner_escaped = m.group(1)
            # Unescape common escaped quotes
            inner_unescaped = inner_escaped.encode('utf-8').decode('unicode_escape')
            try:
                inner = json.loads(inner_unescaped)
                return inner, None
            except Exception:
                pass
    except Exception:
        pass

    # Give a helpful error message quoting a snippet for debugging
    snippet = (text[:400] + '...') if len(text) > 400 else text
    return None, f"No JSON object found in the response. Snippet: {snippet!r}"

def generate_scenario_and_solution(resume_text):
    """
    Performs the two-step generation process for a single resume.
    """
    current_model = get_next_model()
    
    # Decide if this is a negative example (bad fit)
    is_negative = random.random() < 0.1  # 10% chance for negative examples
    
    # Step 1: Generate the scenario
    if is_negative:
        user_content = f"Here is the candidate's resume text:\n---\n{resume_text}\n---\n\nBased on this, generate a JSON object with keys: `job_ad_text`, `extracted_skills_json`, and `domain_insights_json`. The job ad should be a significant 'reach' for the candidate to create a large skill gap. The insights must identify this gap and suggest an 'inferred_skill' that is NOT transferable from the candidate's background. Provide a brief explanation of why this skill is not transferable."
    else:
        user_content = f"Here is the candidate's resume text:\n---\n{resume_text}\n---\n\nBased on this, generate a JSON object with keys: `job_ad_text`, `extracted_skills_json`, and `domain_insights_json`. The job ad should be a slight 'reach' for the candidate to create a skill gap. The insights must identify this gap and suggest an 'inferred_skill' based on transferable skills from the candidate's background. Provide a brief explanation of why this skill is transferable."
    
    prompt_step1 = [
        {"role": "system", "content": "You are an expert career advisor. Your task is to create a realistic job application scenario based on a candidate's resume. IMPORTANT: Your final output MUST be a single, valid JSON object and nothing else. Do not include any text before or after the JSON object."},
        {"role": "user", "content": user_content}
    ]
    
    # --- Retry logic for robust scenario generation ---
    max_retries = len(MODEL_CASCADE)
    retries_per_model = 3
    scenario_succeeded = False
    for i in range(max_retries // retries_per_model + 1):
        for retry in range(retries_per_model):
            scenario_response_str, error = make_api_call(prompt_step1, current_model)
            if not error:
                scenario_succeeded = True
                # Wait a configurable amount after a successful call to tune throughput and avoid transient rate limitations
                time.sleep(OPENROUTER_AFTER_SUCCESS_WAIT) # Wait after a successful call
                break
            # Wait even on failure to space out calls
            time.sleep(OPENROUTER_AFTER_SUCCESS_WAIT)
        if scenario_succeeded:
            break
        logging.warning(f"  Worker failed with {current_model} after {retries_per_model} retries. Switching to next model...")
        current_model = get_next_model()
    
    if not scenario_succeeded:
        return None, f"Failed to generate scenario after {max_retries} retries.", current_model

    scenario, parse_error = parse_json_from_response(scenario_response_str)
    if parse_error:
        return None, f"Failed to parse scenario: {parse_error}", current_model

    # Step 2: Generate the solution
    # --- Start of Robust Schema Validation ---
    try:
        # Defensively handle cases where models return lists or strings instead of dicts
        extracted_skills_data = scenario.get('extracted_skills_json', {})
        if isinstance(extracted_skills_data, list) and extracted_skills_data:
            extracted_skills_data = extracted_skills_data[0]
        if not isinstance(extracted_skills_data, dict):
            extracted_skills_data = {}

        domain_insights_data = scenario.get('domain_insights_json', {})
        if isinstance(domain_insights_data, list) and domain_insights_data:
            domain_insights_data = domain_insights_data[0]
        if not isinstance(domain_insights_data, dict):
            domain_insights_data = {}

        # Safely extract all fields, providing sensible defaults
        candidate_role = extracted_skills_data.get('title', 'N/A')
        
        skills_list = extracted_skills_data.get('skills', [])
        if not isinstance(skills_list, list):
            skills_list = []
            
        candidate_skills = ', '.join([s.get('skill') for s in skills_list[:3] if isinstance(s, dict)])

        job_ad_text = scenario.get('job_ad_text', 'N/A')
        if not isinstance(job_ad_text, str):
            job_ad_text = 'N/A'
            
        job_target_role = job_ad_text.splitlines()[0]

        gap = domain_insights_data.get('skill_gap_priority', 'N/A')
        
        insights = domain_insights_data.get('insights', ['N/A'])
        if not isinstance(insights, list):
            insights = ['N/A']
            
        inferred_skill = insights[0]

    except Exception as e:
        return None, f"Critical error during schema validation: {e}. Skipping record.", current_model
    # --- End of Robust Schema Validation ---

    # First, we need a concise summary of the generated scenario for the user prompt
    task_instruction = "Rewrite the Experience section to target this new role, using the inferred skill to bridge the gap."
    if is_negative:
        task_instruction = "Attempt to rewrite the Experience section to target this new role, but note that the inferred skill does not adequately bridge the gap and include a disclaimer that the candidate may not be a good fit."
    
    user_prompt_context = f"""CANDIDATE DATA:
- Role: {candidate_role}
- Skills: {candidate_skills}

JOB TARGET:
- Role: {job_target_role}

INSIGHTS:
- Gap: {gap}
- Inferred Skill: {inferred_skill}

Task: {task_instruction}"""

    system_content = "You are the Imaginator. First, provide a <reasoning> block explaining how you will bridge the gaps using inferred skills and why the synthesis is logical. Then, synthesize the input data into a tailored resume section."
    if is_negative:
        system_content = "You are the Imaginator. First, provide a <reasoning> block explaining the attempt to bridge the gaps, noting if the inferred skills are insufficient. Then, synthesize the input data into a tailored resume section, including a disclaimer if it's a bad fit."
    
    prompt_step2 = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt_context}
    ]

    solution_succeeded = False
    for i in range(max_retries // retries_per_model + 1):
        for retry in range(retries_per_model):
            solution_text, error = make_api_call(prompt_step2, current_model)
            if not error:
                solution_succeeded = True
                # Wait a configurable amount after a successful call to tune throughput and avoid transient rate limitations
                time.sleep(OPENROUTER_AFTER_SUCCESS_WAIT) # Wait after a successful call
                break
            # Wait even on failure to space out calls
            time.sleep(OPENROUTER_AFTER_SUCCESS_WAIT)
        if solution_succeeded:
            break
        logging.warning(f"  Worker failed (solution) with {current_model} after {retries_per_model} retries. Switching to next model...")
        current_model = get_next_model()

    if not solution_succeeded:
        return None, f"Failed to generate solution after {max_retries} retries.", current_model

    # Assemble the final record
    final_record = {
        "resume_text": resume_text,
        "job_ad_text": scenario.get("job_ad_text"),
        "extracted_skills_json": scenario.get("extracted_skills_json"),
        "domain_insights_json": scenario.get("domain_insights_json"),
        "messages": prompt_step2 + [{"role": "assistant", "content": solution_text}],
        "model_used": current_model, # Store the model that successfully generated the record
        "is_negative": is_negative
    }

    return final_record, None, current_model


def process_profile(profile, thread_name):
    """
    Main worker function to process a single profile.
    Each thread will call this function.
    """
    logging.info(f"Starting processing for a profile.")
    current_model = get_next_model()
    
    # Add a 1.1-second delay to be respectful of the APIs
    time.sleep(1.1)

    if not current_model:
        logging.warning("All free models have been tried and failed. Stopping generation for this profile.")
        return None

    success = False
    while not success and current_model:
        record, error, model_used = generate_scenario_and_solution(profile)
        if record:
            logging.info(f"Successfully generated record with {model_used}.")
            return record
        else:
            logging.error(f"Failed to generate record with {model_used}. Error: {error}")
            current_model = get_next_model() # Move to the next model
            if current_model:
                logging.info(f"Switching to next model: {current_model}")
                time.sleep(1.1)  # Wait 1.1 second between retries to be respectful
            else:
                logging.warning("All models exhausted for this record.")
                return None
    return None

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)
# Also handle SIGTERM and SIGINT to allow graceful shutdown and final summary
try:
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.signal(signal.SIGINT, timeout_handler)
except Exception:
    # Signal may not be available on some platforms (Windows) - ignore in that case
    pass

def upload_to_huggingface(data_records, repo_id, split="train"):
    """
    Uploads a list of data records to a Hugging Face Dataset repository.
    """
    if not HF_TOKEN:
        logging.warning("HF_TOKEN not set. Skipping Hugging Face upload.")
        return

    if not data_records:
        return

    try:
        # Convert records to Hugging Face Dataset format
        dataset = Dataset.from_list(data_records)
        dataset.push_to_hub(repo_id, split=split, private=False)
        logging.info(f"Successfully uploaded {len(data_records)} records to Hugging Face Hub: {repo_id}/{split}")
    except Exception as e:
        logging.error(f"Failed to upload to Hugging Face Hub: {e}")

def finalize_run(generated_records, num_initial_records, output_path, model_success_counts, start_time):
    """Write generated records to file and log a concise summary (used in normal and graceful exits)."""
    end_time = time.perf_counter()
    total_generated_this_run = len(generated_records)

    # Respect explicit append_mode to avoid accidental overwrites when requested
    global append_mode
    if append_mode:
        mode = 'a'
    else:
        mode = 'a' if output_path.exists() and output_path.stat().st_size > 0 else 'w'

    try:
        with output_path.open(mode, encoding='utf-8') as f:
            for record in generated_records:
                f.write(json.dumps(record) + '\n')
        logging.info(f"Saved generated data to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save generated data: {e}")

    logging.info(f"\n--- Generation Summary ---")
    logging.info(f"Successfully generated {total_generated_this_run} records this run.")
    logging.info(f"Total records in file: {total_generated_this_run + num_initial_records}")
    logging.info(f"Total time taken this run: {end_time - start_time:.2f} seconds")

    total_successful_records = sum(model_success_counts.values())
    if total_successful_records > 0:
        logging.info("\n--- Model Contribution Percentages ---")
        for model, count in model_success_counts.items():
            percentage = (count / total_successful_records) * 100
            logging.info(f"- {model}: {percentage:.2f}% ({count} records)")
    else:
        logging.info("\nNo records successfully generated to calculate percentages.")


def main():

    # Default alarm is 5 hours; allow user to override via --timeout (seconds) for shorter runs
    # (e.g., --timeout 7200 for 2 hours)
    signal.alarm(5 * 3600)  # Set a 5-hour alarm by default
    try:
        parser = argparse.ArgumentParser(description="Generate Imaginator dataset using a cascade of models.")
        parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to generate.")
        parser.add_argument("--output_file", type=str, default="imaginator_complex_free_TEST_50.jsonl", help="Output JSONL file name.")
        parser.add_argument("--seed_source", type=str, choices=['github','local'], default='local', help="Seed source: 'github' to fetch profiles from GitHub, 'local' to use pre-fetched local file.")
        parser.add_argument("--seed_local_file", type=str, default='training-data/formatted/github_profiles_prepared.jsonl', help="Local seed file path used when seed_source=local")
        parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers.")
        parser.add_argument("--append", action="store_true", help="Append to the output file if it exists.")
        parser.add_argument("--blacklist", type=str, default="", help="Comma-separated model IDs to blacklist for this run.")
        parser.add_argument("--repeat_on_success", action="store_true", help="If set, repeat the same model after a successful generation until it fails.")
        # New flags to support paid-only runs and custom timeout
        parser.add_argument("--only_paid", action="store_true", help="Use only the configured paid OpenRouter models for generation (paid models must be validated).")
        parser.add_argument("--timeout", type=int, default=None, help="Maximum runtime in seconds (overrides default alarm).")
        args = parser.parse_args()

        # Apply --timeout override if provided
        if args.timeout and args.timeout > 0:
            signal.alarm(args.timeout)
            logging.info(f"Overriding default alarm: setting timeout to {args.timeout} seconds.")
        
        # Apply runtime blacklist if provided
        if args.blacklist:
            bl = [m.strip() for m in args.blacklist.split(',') if m.strip()]
            for m in bl:
                banned_models.add(m)
            if bl:
                logging.info(f"Runtime blacklist applied: {bl}")

        # Set repeat-on-success behavior if requested
        global repeat_on_success, append_mode
        if args.repeat_on_success:
            repeat_on_success = True
            logging.info("Repeat-on-success mode enabled: successful models will be retried until they fail.")
        if args.append:
            append_mode = True
            logging.info("Append mode: output will be appended to rather than overwritten.")
        # If requested, limit the model cascade to only paid OpenRouter models (validated at startup)
        if args.only_paid:
            if not PAID_OPENROUTER_MODELS:
                logging.error("Requested paid-only run but no paid OpenRouter models are configured/validated. Aborting.")
                return
            logging.info(f"Only-paid mode requested. Using {len(PAID_OPENROUTER_MODELS)} paid OpenRouter models for generation.")
            # Replace the dynamic model cascade for this run with the paid models
            global MODEL_CASCADE, model_cycler
            MODEL_CASCADE = list(PAID_OPENROUTER_MODELS)
            model_cycler = cycle(MODEL_CASCADE)

        logging.info("--- Starting Imaginator Data Generation (Free Tier) ---")
        
        # Seed selection
        if args.seed_source == 'github':
            try:
                seed_profiles = fetch_github_profiles(args.num_examples)
            except Exception as e:
                logging.error(f"Error fetching from GitHub: {e}. Falling back to local seeds.")
                try:
                    seed_profiles = load_local_github_seeds(args.seed_local_file, args.num_examples)
                    logging.info(f"Loaded {len(seed_profiles)} local seeds from {args.seed_local_file}.")
                except Exception as e2:
                    logging.error(f"Error loading local seeds: {e2}")
                    seed_profiles = get_raw_resumes(args.num_examples)
        else:
            try:
                seed_profiles = load_local_github_seeds(args.seed_local_file, args.num_examples)
                logging.info(f"Loaded {len(seed_profiles)} local seeds from {args.seed_local_file}.")
            except Exception as e:
                logging.error(f"Error loading local seeds: {e}.")

                # Attempt to regenerate the prepared seeds automatically if the raw harvest is available
                raw_input_path = Path('training-data/raw/github_profiles_raw.json')
                if raw_input_path.exists():
                    logging.info("Attempting to regenerate prepared seeds from raw harvest file and retrying load...")
                    try:
                        subprocess.run(['python', 'scripts/prepare_for_llm.py', '-i', str(raw_input_path), '-o', args.seed_local_file], check=True)
                        seed_profiles = load_local_github_seeds(args.seed_local_file, args.num_examples)
                        logging.info(f"Loaded {len(seed_profiles)} local seeds from {args.seed_local_file} after regenerating.")
                    except Exception as e2:
                        logging.error(f"Regeneration or reload failed: {e2}. Falling back to raw resumes.")
                        seed_profiles = get_raw_resumes(args.num_examples)
                else:
                    logging.info("No raw GitHub harvest file found. Falling back to raw resumes.")
                    seed_profiles = get_raw_resumes(args.num_examples)

        num_to_generate = min(args.num_examples, len(seed_profiles))
        logging.info(f"Using {num_to_generate} seed profiles. Will generate {num_to_generate} examples with {args.workers} workers.")
        
        num_initial_records = 0
        output_path = OUTPUT_DIR / args.output_file
        if args.append and output_path.exists():
            with output_path.open('r', encoding='utf-8') as f:
                num_initial_records = sum(1 for _ in f)
            logging.info(f"Append mode active: {num_initial_records} existing records in {output_path}.")
        elif output_path.exists() and not args.append:
            logging.info(f"Output file {output_path} exists and will be overwritten unless --append is used.")

        generated_records = []
        model_success_counts = defaultdict(int)
        model_fail_counts = defaultdict(int)
        total_generated_this_run = 0

        start_time = time.perf_counter()
        last_record_count_print_time = start_time
        last_total_time_print_time = start_time

        denied_request_cooldown_active = False
        cooldown_end_time = 0

        # Create a queue to hold profiles that need to be processed
        profile_queue = queue.Queue()
        for profile in seed_profiles:
            profile_queue.put(profile)

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures_to_profile = {}
            while total_generated_this_run < num_to_generate and (not profile_queue.empty() or futures_to_profile):
                if denied_request_cooldown_active and time.perf_counter() < cooldown_end_time:
                    remaining_cooldown = int(cooldown_end_time - time.perf_counter())
                    logging.info(f"Cooldown active. Pausing for {remaining_cooldown} seconds...")
                    time.sleep(min(remaining_cooldown, 30)) # Sleep in chunks during cooldown
                    continue

                # Submit new tasks if workers are free and there are profiles to process
                while len(futures_to_profile) < args.workers and not profile_queue.empty():
                    profile = profile_queue.get()
                    future = executor.submit(generate_scenario_and_solution, profile)
                    futures_to_profile[future] = profile

                # Process completed futures
                done_futures = [f for f in futures_to_profile if f.done()]
                for future in done_futures:
                    profile = futures_to_profile.pop(future) # Get the original profile back
                    record, error, model_used = future.result()
                    if record:
                        generated_records.append(record)
                        total_generated_this_run += 1
                        model_success_counts[model_used] += 1
                        
                        # Determine API provider
                        if ":" in model_used:
                            provider = "OpenRouter"
                        elif "/" in model_used:
                            provider = "HuggingFace"
                        else:
                            provider = "SambaNova"
                        
                        logging.info(f"Successfully generated record with {model_used} ({provider}). Total: {len(generated_records) + num_initial_records}")

                    # Update repeat-on-success preference to keep using this model if requested
                    try:
                        if args.repeat_on_success:
                            with model_lock:
                                last_success_model = model_used
                    except Exception:
                        pass

                    # Log live model contribution percentages
                    total_successful = sum(model_success_counts.values())
                    if total_successful > 0:
                        # Build top contributors line
                        top = sorted(model_success_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_str = ", ".join([f"{m}={c}({c/total_successful*100:.1f}%)" for m,c in top])
                        logging.info(f"Contribs: {top_str} | Total successful: {total_successful}")

                    # Upload to Hugging Face every 10 records (if repo configured)
                    if total_generated_this_run % 10 == 0:
                        if HF_REPO_ID:
                            upload_to_huggingface(generated_records[-10:], HF_REPO_ID)
                        else:
                            logging.debug("HF_REPO_ID not set; skipping HF upload.")
                    else:
                        model_fail_counts[model_used] += 1
                        logging.error(f"Failed to generate record. Error: {error}. Model: {model_used}")

                        # If this model was the last successful model and it failed now, clear the repeat preference
                        try:
                            with model_lock:
                                if last_success_model == model_used:
                                    last_success_model = None
                        except Exception:
                            pass

                        # Check for repeated denied requests to trigger cooldown
                        try:
                            err_str = str(error) if error is not None else ""
                            if "429 Client Error" in err_str or "Too Many Requests" in err_str:
                                logging.warning("Repeated denied requests detected. Activating 5-minute cooldown.")
                                denied_request_cooldown_active = True
                                cooldown_end_time = time.perf_counter() + (5 * 60) # 5 minutes
                        except Exception:
                            # Defensive fallback: ignore malformed error values
                            pass
                        
                        # If a profile failed, put it back in the queue to be retried by another model/worker
                        # but only if it hasn't exhausted all models already
                        try:
                            err_text = str(error) if error is not None else ""
                            if "Failed to generate scenario after" not in err_text and "Failed to generate solution after" not in err_text:
                                profile_queue.put(profile) # Re-queue the original profile
                        except Exception:
                            # Defensive: if error is malformed, requeue the profile so it can be retried
                            profile_queue.put(profile) # Re-queue the original profile

                # Periodic printouts
                current_time = time.perf_counter()
                if current_time - last_record_count_print_time >= 60: # Every 1 minute
                    logging.info(f"Records generated so far: {len(generated_records) + num_initial_records}")
                    last_record_count_print_time = current_time
                
                if current_time - last_total_time_print_time >= 300: # Every 5 minutes
                    elapsed_time = current_time - start_time
                    logging.info(f"Total time elapsed: {elapsed_time:.2f} seconds.")
                    last_total_time_print_time = current_time

                # Prevent busy-waiting if no futures are ready yet
                if not done_futures and futures_to_profile:
                    time.sleep(0.5)
            
            # Ensure all remaining futures are completed before exiting the executor
            for future in as_completed(futures_to_profile):
                profile = futures_to_profile.pop(future)
                record, error, model_used = future.result()
                if record:
                    generated_records.append(record)
                    total_generated_this_run += 1
                    model_success_counts[model_used] += 1
                    
                    # Determine API provider
                    if ":" in model_used:
                        provider = "OpenRouter"
                    elif "/" in model_used:
                        provider = "HuggingFace"
                    else:
                        provider = "SambaNova"
                    
                    logging.info(f"Successfully generated record with {model_used} ({provider}). Total: {len(generated_records) + num_initial_records}")
                    if total_generated_this_run % 10 == 0:
                        if HF_REPO_ID:
                            upload_to_huggingface(generated_records[-10:], HF_REPO_ID)
                        else:
                            logging.debug("HF_REPO_ID not set; skipping HF upload.")
                else:
                    model_fail_counts[model_used] += 1
                    logging.error(f"Failed to generate record. Error: {error}. Model: {model_used}")
                    # No re-queuing here, as the loop has already finished its main iteration

        end_time = time.perf_counter()
        logging.info(f"\n--- Generation Complete ---")
        logging.info(f"Successfully generated {len(generated_records)} out of {num_to_generate} targeted examples this run.")
        logging.info(f"Total records in file: {len(generated_records) + num_initial_records}")
        logging.info(f"Total time taken this run: {end_time - start_time:.2f} seconds")
        
        output_path = OUTPUT_DIR / args.output_file
        mode = 'a'
        with output_path.open(mode, encoding='utf-8') as f:
            for record in generated_records:
                f.write(json.dumps(record) + '\n')
        
        logging.info(f"Saved generated data to {output_path}")

        # Calculate and print source percentages
        total_successful_records = sum(model_success_counts.values())
        if total_successful_records > 0:
            logging.info("\n--- Model Contribution Percentages ---")
            for model, count in model_success_counts.items():
                percentage = (count / total_successful_records) * 100
                logging.info(f"- {model}: {percentage:.2f}% ({count} records)")
        else:
            logging.info("\nNo records successfully generated to calculate percentages.")

        if total_generated_this_run < num_to_generate:
            remaining = num_to_generate - total_generated_this_run
            logging.warning(f"--- ACTION REQUIRED ---")
            logging.warning(f"{remaining} records could not be generated with the free tier.")
    except TimeoutException:
        logging.warning("Generation timed out (signal or timeout reached). Performing graceful shutdown and summary.")
        try:
            finalize_run(generated_records, num_initial_records, output_path, model_success_counts, start_time)
        except Exception as e:
            logging.error(f"Error during graceful finalize: {e}")
    finally:
        signal.alarm(0)  # Disable the alarm

if __name__ == "__main__":
    main()
