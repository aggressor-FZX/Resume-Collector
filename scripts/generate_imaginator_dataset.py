#!/usr/bin/env python3
"""
Generates a specialized "Imaginator" dataset for fine-tuning and uploads it.

This script performs a multi-step process:
1. Creates a new Hugging Face dataset repository if it doesn't exist.
2. Scans for existing high-quality training data (`__rewritten.jsonl` files).
3. Samples from raw resume text files to create new, complex training examples.
4. For each sample, it uses a powerful LLM (via a Hugging Face Space) to:
   a. Simulate a target job ad and domain insights.
   b. Construct a detailed "Imaginator" user prompt.
   c. Generate a high-quality, tailored resume section as the assistant's response.
5. Saves the newly generated data to `imaginator_complex.jsonl`.
6. Merges the existing data with the new Imaginator data into `dataset_final.jsonl`.
7. Uploads the final merged dataset to the specified Hugging Face repository.

Usage:
    python scripts/generate_imaginator_dataset.py \\
        --repo-id "your-hf-username/your-new-dataset-repo" \\
        --num-examples 500 \\
        --llm-space "HuggingFaceH4/zephyr-141b-beta"
"""

import os
import json
import random
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
import backoff
import requests

# --- Configuration ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
RAW_DATA_DIR = Path('data/raw')
FORMATTED_DATA_DIR = Path('training-data/formatted')
OUTPUT_DIR = Path('training-data/generated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Hugging Face API Client ---
class HfSpaceClient:
    """A client to interact with a Hugging Face Space for LLM inference."""
    def __init__(self, space_id):
        self.space_id = space_id
        self.api_url = f"https://huggingfaceh4-zephyr-141b-beta.hf.space/chat"
        self.headers = {"Content-Type": "application/json"}

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
    def query(self, messages, max_new_tokens=1024):
        payload = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
            },
            "stream": False
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('generated_text', '')

def create_hf_repo(api: HfApi, repo_id: str):
    """Creates a Hugging Face dataset repository if it doesn't exist."""
    try:
        api.repo_info(repo_id=repo_id, repo_type='dataset')
        print(f"HF repo '{repo_id}' already exists.")
    except Exception:
        print(f"HF repo '{repo_id}' not found. Creating...")
        api.create_repo(repo_id=repo_id, repo_type='dataset', private=False)
        print(f"Successfully created HF repo '{repo_id}'.")

def find_base_training_files(data_dir: Path):
    """Finds the best existing .jsonl files for training, preferring rewritten versions."""
    all_files = list(data_dir.glob("*.jsonl"))
    # A simplified heuristic to find primary training files
    chosen_files = [f for f in all_files if '__rewritten' in f.name or ('__' not in f.name and 'pairs' not in f.name)]
    print(f"Found {len(chosen_files)} base training files to merge.")
    return chosen_files

def get_raw_resumes(raw_dir: Path, max_files=50):
    """Loads content from raw resume files, searching recursively."""
    print(f"Searching for raw resume files in {raw_dir}...")
    resume_files = list(raw_dir.glob("**/*.json")) + list(raw_dir.glob("**/*.jsonl"))
    random.shuffle(resume_files)
    resumes = []
    
    # Let's limit the number of files we try to open to avoid excessive processing
    files_to_process = resume_files[:max_files*5] # Read more files to find enough valid ones

    for f in files_to_process:
        if len(resumes) >= max_files:
            break
        try:
            with f.open('r', encoding='utf-8') as rf:
                # Handle both JSON and JSONL
                if f.suffix == '.jsonl':
                    for line in rf:
                        data = json.loads(line)
                        content = " ".join([str(v) for v in data.values() if isinstance(v, str)])
                        if len(content) > 150:
                            resumes.append(content)
                            if len(resumes) >= max_files: break
                else: # .json
                    data = json.load(rf)
                    # Heuristic to find the main text content
                    if isinstance(data, list): # a list of records
                         for item in data:
                             content = " ".join([str(v) for v in item.values() if isinstance(v, str)])
                             if len(content) > 150:
                                 resumes.append(content)
                                 if len(resumes) >= max_files: break
                    elif isinstance(data, dict):
                        content = " ".join([str(v) for v in data.values() if isinstance(v, str)])
                        if len(content) > 150:
                            resumes.append(content)

        except (json.JSONDecodeError, IOError):
            # Ignore files that are not valid JSON or can't be read
            continue
            
    print(f"Loaded {len(resumes)} raw resume texts for generation.")
    return resumes[:max_files]


def generate_imaginator_example(resume_text: str, llm_client: HfSpaceClient):
    """Generates a single, complex 'Imaginator' training example."""
    print("  Generating Imaginator example...")
    # 1. Simulate job ad and insights with an LLM
    simulation_prompt = [
        {"role": "system", "content": "You are a creative assistant. Based on the provided resume text, invent a plausible target job role, a brief job ad for it, a list of skills the candidate likely has, and some market insights. Output ONLY a valid JSON object with keys: 'target_role', 'job_ad_text', 'extracted_skills', 'domain_insights'."},
        {"role": "user", "content": f"Resume Text:\n---\n{resume_text[:1500]}\n---"}
    ]
    
    try:
        simulated_data_str = llm_client.query(simulation_prompt)
        simulated_data = json.loads(simulated_data_str)
    except Exception as e:
        print(f"    Failed to simulate data: {e}")
        return None

    # 2. Construct the Imaginator user prompt
    user_prompt_content = f"""CANDIDATE DATA:
- Role: Inferred from resume
- Skills: {simulated_data.get('extracted_skills', {}).get('skills', ['N/A'])}

JOB TARGET:
- Role: {simulated_data.get('target_role', 'N/A')}
- Requirement: Inferred from Job Ad

INSIGHTS:
- {simulated_data.get('domain_insights', {}).get('insights', ['N/A'])}

Task: Rewrite the Experience section of the candidate's resume to specifically target this new role, using the insights to bridge any gaps.
"""

    # 3. Generate the final assistant response
    assistant_generation_prompt = [
        {"role": "system", "content": "You are the Imaginator. Synthesize the input data into a tailored resume. Use the 'Inferred Skills' to bridge any gaps between the candidate and the Job Ad."},
        {"role": "user", "content": user_prompt_content}
    ]
    
    try:
        assistant_response = llm_client.query(assistant_generation_prompt, max_new_tokens=1500)
    except Exception as e:
        print(f"    Failed to generate assistant response: {e}")
        return None

    # 4. Assemble the final record
    final_record = {
        "source_resume_text": resume_text,
        "generated_job_ad_text": simulated_data.get('job_ad_text'),
        "generated_extracted_skills_json": simulated_data.get('extracted_skills'),
        "generated_domain_insights_json": simulated_data.get('domain_insights'),
        "target_role": simulated_data.get('target_role'),
        "messages": [
            {"role": "system", "content": "You are the Imaginator. Synthesize the input data into a tailored resume. Use the 'Inferred Skills' to bridge any gaps between the candidate and the Job Ad."},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    print(f"  Successfully generated example for target role: {simulated_data.get('target_role')}")
    return final_record


def main():
    parser = argparse.ArgumentParser(description="Generate and upload an Imaginator fine-tuning dataset.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., 'user/repo').")
    parser.add_argument("--num_examples", type=int, default=500, help="Number of Imaginator examples to generate.")
    parser.add_argument("--llm_space", type=str, default="HuggingFaceH4/zephyr-141b-beta", help="HF Space to use for generation.")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("HUGGING_FACE_API_KEY not found in environment. Please set it.")
        return

    api = HfApi()
    llm_client = HfSpaceClient(args.llm_space)

    # Step 1: Create HF Repo
    create_hf_repo(api, args.repo_id)

    # Step 2: Generate Imaginator Data
    raw_resumes = get_raw_resumes(RAW_DATA_DIR)
    if not raw_resumes:
        print("Could not find any raw resumes to process. Exiting.")
        return
        
    generated_records = []
    imaginator_output_file = OUTPUT_DIR / 'imaginator_complex.jsonl'

    for i in range(args.num_examples):
        print(f"--- Generating record {i+1}/{args.num_examples} ---")
        resume_text = random.choice(raw_resumes)
        record = generate_imaginator_example(resume_text, llm_client)
        if record:
            generated_records.append(record)
            # Write progress periodically
            with imaginator_output_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
    
    print(f"\nGenerated a total of {len(generated_records)} Imaginator records.")

    # Step 3: Merge and Upload
    print("\n--- Merging all datasets ---")
    final_dataset_path = OUTPUT_DIR / 'dataset_final.jsonl'
    base_files = find_base_training_files(FORMATTED_DATA_DIR)
    
    with final_dataset_path.open('w', encoding='utf-8') as final_out:
        # Add base files
        for f in base_files:
            with f.open('r', encoding='utf-8') as fin:
                for line in fin:
                    final_out.write(line)
        # Add new imaginator files
        with imaginator_output_file.open('r', encoding='utf-8') as fin:
            for line in fin:
                final_out.write(line)
    
    print(f"Final merged dataset created at '{final_dataset_path}'")

    # Step 4: Upload to Hugging Face
    print(f"\n--- Uploading to {args.repo_id} ---")
    try:
        api.upload_file(
            path_or_fileobj=str(final_dataset_path),
            path_in_repo="dataset_final.jsonl",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("Successfully uploaded 'dataset_final.jsonl' to Hugging Face.")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")


if __name__ == "__main__":
    main()
