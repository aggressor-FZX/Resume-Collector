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

load_dotenv()

# --- Configuration ---
RAW_DATA_DIR = Path('data/raw')
OUTPUT_DIR = Path('training-data/imaginator_generated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GITHUB_API_BASE = "https://api.github.com"
YOUR_SITE_URL = "https://github.com/aggressor-FZX/Resume-Collector"
YOUR_APP_NAME = "ResumeCollectorImaginator"

# --- Model Cascade ---
MODEL_CASCADE = [
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/devstral-2512:free",
    "allenai/olmo-3.1-32b-think:free",
    "google/gemma-3-27b-it:free",
]

model_iterator = iter(MODEL_CASCADE)

def get_next_model():
    """Cycles through the model list, returning the next available model."""
    global model_iterator
    try:
        return next(model_iterator)
    except StopIteration:
        # Reset iterator for the next full cycle if needed, though script will likely exit
        model_iterator = iter(MODEL_CASCADE)
        return None

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
    
    if len(resumes) < sample_size:
        print(f"Warning: Found only {len(resumes)} suitable resume texts, less than the requested sample size of {sample_size}.")
        return resumes
    
    return random.sample(resumes, sample_size)

def make_api_call(prompt_messages, model):
    """Makes a call to the OpenRouter API with a given model and prompt."""
    if not model:
        return None, "No more models available in the free tier list."

    print(f"  Attempting generation with model: {model}...")
    try:
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
        print(f"    - {error_message}")
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

def generate_scenario_and_solution(resume_text, model):
    """
    Performs the two-step generation process for a single resume.
    """
    # Step 1: Generate the scenario
    prompt_step1 = [
        {"role": "system", "content": "You are an expert career advisor. Your task is to create a realistic job application scenario based on a candidate's resume. IMPORTANT: Your final output MUST be a single, valid JSON object and nothing else. Do not include any text before or after the JSON object."},
        {"role": "user", "content": f"Here is the candidate's resume text:\n---\n{resume_text}\n---\n\nBased on this, generate a JSON object with keys: `job_ad_text`, `extracted_skills_json`, and `domain_insights_json`. The job ad should be a slight 'reach' for the candidate to create a skill gap. The insights must identify this gap and suggest an 'inferred_skill'."}
    ]
    
    scenario_response_str, error = make_api_call(prompt_step1, model)
    if error:
        return None, error, model

    scenario, parse_error = parse_json_from_response(scenario_response_str)
    if parse_error:
        return None, f"Failed to parse scenario: {parse_error}", model

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
        return None, f"Critical error during schema validation: {e}. Skipping record.", model
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

    solution_text, error = make_api_call(prompt_step2, model)
    if error:
        # We don't advance the model here, as the first call succeeded.
        return None, error, model

    # Assemble the final record
    final_record = {
        "resume_text": resume_text,
        "job_ad_text": scenario.get("job_ad_text"),
        "extracted_skills_json": scenario.get("extracted_skills_json"),
        "domain_insights_json": scenario.get("domain_insights_json"),
        "messages": prompt_step2 + [{"role": "assistant", "content": solution_text}]
    }

    return final_record, None, model


def main():
    parser = argparse.ArgumentParser(description="Generate Imaginator dataset using a cascade of free models.")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to generate.")
    parser.add_argument("--output_file", type=str, default="imaginator_complex_free_TEST_50.jsonl", help="Output JSONL file name.")
    args = parser.parse_args()

    print("--- Starting Imaginator Data Generation (Free Tier) ---")
    
    try:
        # We now prioritize fetching high-quality profiles directly from GitHub
        seed_profiles = fetch_github_profiles(args.num_examples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    num_to_generate = min(args.num_examples, len(seed_profiles))
    print(f"Using {len(seed_profiles)} seed profiles. Will generate {num_to_generate} examples.")
    
    generated_records = []
    current_model = get_next_model()

    for i, profile in enumerate(seed_profiles):
        print(f"\n--- Generating record {i+1}/{num_to_generate} ---")
        
        # Add a 1-second delay to be respectful of the APIs
        time.sleep(1)

        if not current_model:
            print("All free models have been tried and failed. Stopping generation.")
            break

        success = False
        while not success and current_model:
            # Pass the GitHub profile as the resume_text
            record, error, model_used = generate_scenario_and_solution(profile, current_model)
            if record:
                generated_records.append(record)
                success = True
                print(f"  Successfully generated record with {model_used}.")
            else:
                print(f"  Failed to generate record with {model_used}. Error: {error}")
                current_model = get_next_model() # Move to the next model
                if current_model:
                    print(f"  Switching to next model: {current_model}")
                else:
                    print("  All models exhausted for this record.")

    print(f"\n--- Generation Complete ---")
    print(f"Successfully generated {len(generated_records)} out of {num_to_generate} targeted examples.")

    output_path = OUTPUT_DIR / args.output_file
    with output_path.open('w', encoding='utf-8') as f:
        for record in generated_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nSaved generated data to {output_path}")

    if len(generated_records) < num_to_generate:
        remaining = num_to_generate - len(generated_records)
        print(f"\n--- ACTION REQUIRED ---")
        print(f"{remaining} records could not be generated with the free tier.")
        print("You can now run the paid generation script to complete the dataset.")

if __name__ == "__main__":
    main()
