#!/usr/bin/env python3
"""
Harvests rich GitHub user profile data based on search criteria.

This script performs a multi-stage process to collect comprehensive data for each user:
1. Searches for users on GitHub based on specified criteria.
2. For each user, it fetches:
   - Detailed user profile (bio, company, name).
   - A list of their public repositories.
   - The content of their profile README.md, if it exists.
3. It saves this aggregated raw data into a single JSON file.

Usage:
    python scripts/harvest_github_profiles.py --query "language:python location:newyork" --output training-data/raw/github_profiles_raw.json --limit 50
"""

import os
import requests
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_BASE_URL = "https://api.github.com"
TOKEN = os.getenv("GH_TOKEN") or os.getenv("GITHUB_API_KEY")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# --- Constants for Filtering ---
MIN_REPO_COUNT = 5
MIN_STAR_COUNT = 50
TARGET_LANGUAGES = {"python", "c++", "go", "rust", "typescript", "javascript"}
MIN_BIO_LENGTH = 10

# --- Functions ---

def get_repo_readme_content(repo_full_name):
    """Fetches the README content for a specific repository."""
    readme_url = f"{API_BASE_URL}/repos/{repo_full_name}/readme"
    readme_response = requests.get(readme_url, headers=HEADERS)
    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        download_url = readme_data.get("download_url")
        if download_url:
            content_response = requests.get(download_url)
            if content_response.status_code == 200:
                return content_response.text
    return None

def is_high_quality_profile(user_details):
    """
    Checks if a user profile meets the high-quality criteria based on the "Three C's".
    """
    profile = user_details.get("profile", {})
    repos = user_details.get("repositories", [])

    # 1. Complexity & Clout: Repo and Star Count
    if not repos or len(repos) < MIN_REPO_COUNT:
        print(f"[{user_details['login']}] Skipping: Fewer than {MIN_REPO_COUNT} repos.")
        return False

    has_highly_starred_repo = any(repo.get("stargazers_count", 0) > MIN_STAR_COUNT for repo in repos)
    if not has_highly_starred_repo:
        print(f"[{user_details['login']}] Skipping: No repo with >{MIN_STAR_COUNT} stars.")
        return False

    # 2. Complexity: Technical Languages
    repo_languages = {repo.get("language", "").lower() for repo in repos if repo.get("language")}
    if not any(lang in TARGET_LANGUAGES for lang in repo_languages):
        print(f"[{user_details['login']}] Skipping: No repos with target languages.")
        return False

    # 3. Context: Bio and READMEs
    if not profile.get("bio") or len(profile.get("bio", "")) < MIN_BIO_LENGTH:
        print(f"[{user_details['login']}] Skipping: Bio is too short.")
        return False
        
    # Check if at least one of the top repos has a README
    for repo in repos[:MIN_REPO_COUNT]: # Check top N repos
        if get_repo_readme_content(repo['full_name']):
            return True

    print(f"[{user_details['login']}] Skipping: No README found in top repos.")
    return False

def search_users(query, per_page=100):
    """Searches for users on GitHub."""
    print(f"Searching for users with query: '{query}'")
    url = f"{API_BASE_URL}/search/users"
    params = {"q": query, "per_page": per_page}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("items", [])

def get_user_details(user_login):
    """Fetches detailed profile, repos, and README for a user."""
    print(f"Fetching details for user: {user_login}")
    
    # 1. Get user profile
    profile_url = f"{API_BASE_URL}/users/{user_login}"
    profile_response = requests.get(profile_url, headers=HEADERS)
    profile_response.raise_for_status()
    profile = profile_response.json()

    # 2. Get user repositories (up to 100)
    repos_url = f'{profile["repos_url"]}?per_page=100'
    repos_response = requests.get(repos_url, headers=HEADERS)
    repos_response.raise_for_status()
    repos = repos_response.json()

    # 3. Get profile README
    readme_url = f"{API_BASE_URL}/repos/{user_login}/{user_login}/readme"
    readme_response = requests.get(readme_url, headers=HEADERS)
    readme_content = ""
    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        readme_download_url = readme_data.get("download_url")
        if readme_download_url:
            readme_content_response = requests.get(readme_download_url)
            if readme_content_response.status_code == 200:
                readme_content = readme_content_response.text

    return {
        "login": user_login,
        "profile": profile,
        "repositories": repos,
        "profile_readme": readme_content
    }

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Harvest rich GitHub profiles.")
    parser.add_argument("-q", "--query", required=True, help="GitHub user search query.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path for raw data.")
    parser.add_argument("-l", "--limit", type=int, default=50, help="Maximum number of profiles to fetch.")
    args = parser.parse_args()

    if not TOKEN:
        print("Error: GITHUB_API_KEY or GH_TOKEN not found.")
        exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    users = search_users(args.query, per_page=args.limit * 2) # Fetch more to filter down
    
    all_profiles_data = []
    qualified_profiles_count = 0
    
    for user in users:
        if qualified_profiles_count >= args.limit:
            print(f"Reached limit of {args.limit} qualified profiles.")
            break
        try:
            user_data = get_user_details(user["login"])
            
            if is_high_quality_profile(user_data):
                print(f"✅ [{user['login']}] QUALIFIED: Profile meets high-quality criteria.")
                all_profiles_data.append(user_data)
                qualified_profiles_count += 1
            else:
                print(f"❌ [{user['login']}] SKIPPED: Profile did not meet criteria.")

            # Respect rate limits
            time.sleep(1) 
            
        except requests.HTTPError as e:
            print(f"Error fetching data for {user['login']}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {user['login']}: {e}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles_data, f, indent=4)

    print(f"\nSuccessfully harvested {qualified_profiles_count} high-quality profiles.")
    print(f"Raw data saved to: {output_path}")

if __name__ == "__main__":
    main()
