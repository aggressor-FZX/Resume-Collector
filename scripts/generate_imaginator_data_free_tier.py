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
from huggingface_hub import HfApi, login
from collections import defaultdict
import queue
import subprocess

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

load_dotenv()

# --- Configuration ---
RAW_DATA_DIR = Path('data/raw')
OUTPUT_DIR = Path('training-data/imaginator_generated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "aggressor-FZX/Imaginator-Generated-Data"

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GITHUB_API_BASE = "https://api.github.com"
YOUR_SITE_URL = "https://github.com/aggressor-FZX/Resume-Collector"
YOUR_APP_NAME = "ResumeCollectorImaginator"

# --- Model Cascade ---
MODEL_CASCADE = [
    "z-ai/glm-4.5-air:free",
    "tngtech/deepseek-r1t-chimera:free",
    "nex-agi/deepseek-v3.1-nex-n1:free",
    "google/gemma-3-4b-it:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/devstral-2512:free",
    "xiaomi/mimo-v2-flash:free",
    "kwaipilot/kat-coder-pro:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-coder:free",
    "moonshotai/kimi-k2:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
]

model_cycler = cycle(MODEL_CASCADE)
model_lock = threading.Lock()
api_call_semaphore = threading.Semaphore(1)

def get_next_model():
    with model_lock:
        return next(model_cycler)

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
    """Makes a call to the OpenRouter API with a given model and prompt."""
    if not model:
        return None, "No more models available in the free tier list."

    logging.info(f"  Attempting generation with model: {model}...")
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
        return None, error_message

def parse_json_from_response(response_text: str):
    """Extracts a JSON object from a model's text response, even with surrounding text."""
    # Find the start and end of the JSON object
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    
    if start_brace == -1 or end_brace == -1:
        return None, "No JSON object found in the response."

    json_str = response_text[start_brace:end_brace+1]
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError:
        return None, "Failed to parse the extracted JSON object."

def generate_scenario_and_solution(resume_text):
    """
    Performs the two-step generation process for a single resume.
    """
    current_model = get_next_model()
    
    # Step 1: Generate the scenario
    prompt_step1 = [
        {"role": "system", "content": "You are an expert career advisor. Your task is to create a realistic job application scenario based on a candidate's resume. IMPORTANT: Your final output MUST be a single, valid JSON object and nothing else. Do not include any text before or after the JSON object."},
        {"role": "user", "content": f"Here is the candidate's resume text:\n---\n{resume_text}\n---\n\nBased on this, generate a JSON object with keys: `job_ad_text`, `extracted_skills_json`, and `domain_insights_json`. The job ad should be a slight 'reach' for the candidate to create a skill gap. The insights must identify this gap and suggest an 'inferred_skill'."}
    ]
    
    # --- Retry logic for robust scenario generation ---
    max_retries = len(MODEL_CASCADE)
    scenario_succeeded = False
    for i in range(max_retries):
        scenario_response_str, error = make_api_call(prompt_step1, current_model)
        if not error:
            scenario_succeeded = True
            time.sleep(1) # Wait 1 second after a successful call
            break
        logging.warning(f"  Worker failed with {current_model}. Retrying with next model...")
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
    user_prompt_context = f"""CANDIDATE DATA:
- Role: {candidate_role}
- Skills: {candidate_skills}

JOB TARGET:
- Role: {job_target_role}

INSIGHTS:
- Gap: {gap}
- Inferred Skill: {inferred_skill}

Task: Rewrite the Experience section to target this new role, using the inferred skill to bridge the gap."""

    prompt_step2 = [
        {"role": "system", "content": "You are the Imaginator. Synthesize the input data into a tailored resume section. Use the 'Inferred Skills' to bridge any gaps between the candidate and the Job Ad."},
        {"role": "user", "content": user_prompt_context}
    ]

    solution_succeeded = False
    for i in range(max_retries):
        solution_text, error = make_api_call(prompt_step2, current_model)
        if not error:
            solution_succeeded = True
            time.sleep(1) # Wait 1 second after a successful call
            break
        logging.warning(f"  Worker failed (solution) with {current_model}. Retrying with next model...")
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
        "model_used": current_model # Store the model that successfully generated the record
    }

    return final_record, None, current_model


def process_profile(profile, thread_name):
    """
    Main worker function to process a single profile.
    Each thread will call this function.
    """
    logging.info(f"Starting processing for a profile.")
    current_model = get_next_model()
    
    # Add a 1-second delay to be respectful of the APIs
    time.sleep(1)

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

def main():
    signal.alarm(5 * 3600)  # Set a 5-hour alarm
    try:
        parser = argparse.ArgumentParser(description="Generate Imaginator dataset using a cascade of free models.")
        parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to generate.")
        parser.add_argument("--output_file", type=str, default="imaginator_complex_free_TEST_50.jsonl", help="Output JSONL file name.")
        parser.add_argument("--seed_source", type=str, choices=['github','local'], default='local', help="Seed source: 'github' to fetch profiles from GitHub, 'local' to use pre-fetched local file.")
        parser.add_argument("--seed_local_file", type=str, default='training-data/formatted/github_profiles_prepared.jsonl', help="Local seed file path used when seed_source=local")
        parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers.")
        parser.add_argument("--append", action="store_true", help="Append to the output file if it exists.")
        args = parser.parse_args()

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

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
                    time.sleep(0.1) # Stagger worker submissions to prevent calls at exactly the same time

                # Process completed futures
                done_futures = [f for f in futures_to_profile if f.done()]
                for future in done_futures:
                    profile = futures_to_profile.pop(future) # Get the original profile back
                    record, error, model_used = future.result()
                    if record:
                        generated_records.append(record)
                        total_generated_this_run += 1
                        model_success_counts[model_used] += 1
                        logging.info(f"Successfully generated record with {model_used}. Total: {len(generated_records) + num_initial_records}")

                        # Upload to Hugging Face every 10 records
                        if total_generated_this_run % 10 == 0:
                            upload_to_huggingface(generated_records[-10:], HF_REPO_ID)
                    else:
                        model_fail_counts[model_used] += 1
                        logging.error(f"Failed to generate record. Error: {error}. Model: {model_used}")
                        
                        # Check for repeated denied requests to trigger cooldown
                        if "429 Client Error" in error or "Too Many Requests" in error:
                            logging.warning("Repeated denied requests detected. Activating 5-minute cooldown.")
                            denied_request_cooldown_active = True
                            cooldown_end_time = time.perf_counter() + (5 * 60) # 5 minutes
                        
                        # If a profile failed, put it back in the queue to be retried by another model/worker
                        # but only if it hasn't exhausted all models already
                        if "Failed to generate scenario after" not in error and "Failed to generate solution after" not in error:
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
                    logging.info(f"Successfully generated record with {model_used}. Total: {len(generated_records) + num_initial_records}")
                    if total_generated_this_run % 10 == 0:
                        upload_to_huggingface(generated_records[-10:], HF_REPO_ID)
                else:
                    model_fail_counts[model_used] += 1
                    logging.error(f"Failed to generate record. Error: {error}. Model: {model_used}")
                    # No re-queuing here, as the loop has already finished its main iteration

        end_time = time.perf_counter()
        logging.info(f"\n--- Generation Complete ---")
        logging.info(f"Successfully generated {len(generated_records)} out of {num_to_generate} targeted examples this run.")
        logging.info(f"Total records in file (including appended): {len(generated_records) + num_initial_records}")
        logging.info(f"Total time taken this run: {end_time - start_time:.2f} seconds")
        
        output_path = OUTPUT_DIR / args.output_file
        mode = 'a' if args.append else 'w'
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
        print("Generation timed out after 25 minutes.")
    finally:
        signal.alarm(0)  # Disable the alarm

if __name__ == "__main__":
    main()
