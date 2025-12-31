import json
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Replace bullet points and special chars
    text = text.replace('\u00e2\u0080\u00a2', '-')
    text = text.replace('\u00c3\u00af', 'i')  # example, add more if needed
    # Add more replacements as needed
    return text

def standardize_extracted_skills(skills):
    if isinstance(skills, list):
        return {"skills": skills}
    elif isinstance(skills, dict):
        if "skills" in skills:
            return skills
        else:
            # Assume it's a dict of categories, flatten to list
            all_skills = []
            for key, value in skills.items():
                if isinstance(value, list):
                    all_skills.extend(value)
                elif isinstance(value, str):
                    all_skills.append(value)
            return {"skills": all_skills}
    else:
        return {"skills": []}

def standardize_domain_insights(insights):
    if not isinstance(insights, dict):
        return {"skill_gap": "N/A", "inferred_skill": "N/A"}
    skill_gap = insights.get("skill_gap", "N/A")
    inferred_skill = insights.get("inferred_skill", "N/A")
    # Flatten skill_gap
    if isinstance(skill_gap, list):
        gap_parts = []
        for item in skill_gap:
            if isinstance(item, dict):
                gap_parts.append("; ".join([f"{k}: {v}" for k, v in item.items()]))
            else:
                gap_parts.append(str(item))
        skill_gap = "; ".join(gap_parts)
    elif isinstance(skill_gap, dict):
        skill_gap = "; ".join([f"{k}: {v}" for k, v in skill_gap.items()])
    if isinstance(inferred_skill, dict):
        inferred_skill = inferred_skill.get("name", str(inferred_skill))
    return {"skill_gap": skill_gap, "inferred_skill": inferred_skill}

def clean_entry(entry):
    # Clean text fields if they exist
    if 'resume_text' in entry:
        entry['resume_text'] = clean_text(entry['resume_text'])
    if 'job_ad_text' in entry:
        entry['job_ad_text'] = clean_text(entry['job_ad_text'])
    # Clean messages
    if 'messages' in entry:
        for msg in entry['messages']:
            if 'content' in msg:
                msg['content'] = clean_text(msg['content'])
    # Standardize extracted_skills_json if exists
    if 'extracted_skills_json' in entry:
        entry['extracted_skills_json'] = standardize_extracted_skills(entry['extracted_skills_json'])
    # Standardize domain_insights_json if exists
    if 'domain_insights_json' in entry:
        entry['domain_insights_json'] = standardize_domain_insights(entry['domain_insights_json'])
    return entry

def process_file(filepath):
    cleaned_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cleaned_entry = clean_entry(entry)
                cleaned_lines.append(json.dumps(cleaned_entry, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {filepath}: {e}")
                continue
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    directory = "/workspaces/Resume-Collector/training-data/imaginator_generated"
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}")
            process_file(filepath)
    print("Cleaning complete.")