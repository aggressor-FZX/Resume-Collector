#!/usr/bin/env python3
"""
Prepares raw, harvested GitHub profile data for an LLM resume writer.

This script acts as a "bridge" between the raw data collector and the LLM.
It takes the JSON output from `harvest_github_profiles.py`, filters for high-impact
repositories, and formats the user's bio and key projects into a clean,
"evidence-based" text context suitable for prompting a large language model.

Usage:
    python scripts/prepare_for_llm.py --input training-data/raw/github_profiles_raw.json
"""

import json
import argparse
from pathlib import Path

def format_repo_for_resume(repo):
    """
    Converts raw GitHub repo JSON into a clean string for a resume context.
    Filters out forks and low-starred repos.
    """
    # 1. Filter out weak repos
    if repo.get('fork') or repo.get('stargazers_count', 0) < 5:
        return None
    
    # 2. Extract "Impact" metrics
    stars = repo.get('stargazers_count', 0)
    language = repo.get('language') or "Code"
    description = repo.get('description') or "No description provided."
    
    # 3. Create a high-quality summary string
    return (
        f"- Project: {repo.get('name', 'N/A')} ({language})\n"
        f"  - Impact: {stars} GitHub stars (demonstrating community adoption)\n"
        f"  - Description: {description}\n"
        f"  - URL: {repo.get('html_url', '#')}"
    )

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Prepare GitHub data for LLM context.")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file from the harvesting script.")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file for prepared data.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    if not profiles:
        print("No profiles found in the input file.")
        return

    prepared_data = []
    for user in profiles:
        bio = user.get('profile', {}).get('bio', 'No bio provided.')
        repositories = user.get('repositories', [])

        # Get top 5 repos sorted by stars (Quality over Quantity)
        sorted_repos = sorted(repositories, key=lambda x: x.get('stargazers_count', 0), reverse=True)[:5]
        
        formatted_repos = [
            formatted for r in sorted_repos 
            if (formatted := format_repo_for_resume(r)) is not None
        ]

        if not formatted_repos:
            continue

        # FINAL PROMPT CONTEXT
        llm_context = {
            "messages": [
                {"role": "system", "content": "You are a professional resume writer. Rewrite the following developer bio and project list into a compelling, evidence-based resume."},
                {"role": "user", "content": f"CANDIDATE BIO:\\n{bio}\\n\\nKEY OPEN SOURCE PROJECTS:\\n{chr(10).join(formatted_repos)}"}
            ]
        }
        prepared_data.append(llm_context)

    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        for item in prepared_data:
            # Use print to reliably emit a platform-correct newline and avoid accidental escaping of the literal "\\n"
            print(json.dumps(item), file=f)
            f.flush()

    print(f"--- Successfully prepared {len(prepared_data)} profiles. ---")
    print(f"Prepared data saved to: {output_path}")

if __name__ == "__main__":
    main()
