---
license: mit
language:
  - en
tags:
  - text-generation
  - summarization
---

# Dataset Card for GitHub Profiles for Resume Generation

## Dataset Summary

This dataset contains curated GitHub user profiles designed to fine-tune language models for the task of professional resume writing. It is composed of profiles that meet a "high-quality" standard based on technical complexity, rich documentation, and community engagement (the "Three C's"). The goal is to train models that can generate compelling, evidence-based resume content from a developer's raw bio and project history.

## Supported Tasks & Leaderboards

- **`text-generation`**: This dataset is primarily intended for fine-tuning models for text generation, specifically for creating resume summaries and project descriptions.
- **`summarization`**: The data can also be used for summarization tasks, where the model learns to condense a developer's career and projects into a concise, impactful narrative.

## Languages

The dataset primarily consists of English-language content from GitHub profiles. The code repositories associated with these profiles are predominantly in the following languages:
- Python
- C++
- Go
- Rust
- TypeScript
- JavaScript

## Dataset Structure

The dataset is provided in a single JSONL file (`github_profiles_prepared.jsonl`). Each line in the file is a JSON object representing one training example. The structure is formatted for conversational fine-tuning (e.g., with OpenAI models):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional resume writer. Rewrite the following developer bio and project list into a compelling, evidence-based resume."
    },
    {
      "role": "user",
      "content": "CANDIDATE BIO:\\n{bio}\\n\\nKEY OPEN SOURCE PROJECTS:\\n{formatted project list}"
    }
  ]
}
```

### Data Fields

- `messages`: An array of message objects.
    - `role`: The role of the speaker, either `system` or `user`.
    - `content`: The text of the message.

### Data Splits

Currently, the dataset consists of a single split: `train`. No validation or test splits are provided at this time.

## Data Collection Process

### Source Data

The data was collected from public user profiles on GitHub.

### Data Collection

The data was collected in a two-stage process:

1.  **Harvesting**: A Python script (`scripts/harvest_github_profiles.py`) was used to search for GitHub users with a high follower count and repositories in target languages.
2.  **Filtering & Preparation**: The harvested profiles were then filtered based on the "Three C's":
    - **Complexity**: At least 5 public repos in "hard" engineering languages.
    - **Context**: A bio of at least 10 characters and the presence of READMEs in their top repositories.
    - **Clout**: At least one repository with more than 50 stars.

A second script (`scripts/prepare_for_llm.py`) then formatted the qualifying profiles into the final JSONL structure.

## Additional Information

### Licensing Information

The dataset itself is released under the [MIT License](https://opensource.org/licenses/MIT). However, the underlying content (user bios, repository information, and READMEs) is subject to the terms of service of GitHub and the licenses of the individual repositories.

### Citation Information

```
@dataset{
  author={aggressor-FZX},
  title={GitHub Profiles for Resume Generation},
  year={2025},
  url={https://huggingface.co/datasets/jeff-calderon/ResumeData}
}
```

### Contributions

Contributions are welcome! If you have suggestions for improving the dataset or the collection process, please open an issue or pull request in the [GitHub repository](https://github.com/aggressor-FZX/Resume-Collector).