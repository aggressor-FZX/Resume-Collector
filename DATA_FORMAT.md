# Resume-Collector Dataset Documentation

## Overview

The Resume-Collector project provides a pipeline for collecting, formatting, validating, and preparing resume-related data for fine-tuning large language models (LLMs). The data is sourced from various platforms (e.g., GitHub, Kaggle, Stack Exchange) and processed into a standardized format suitable for instruction-tuning tasks.

## Data Structure

### Directory Layout
- `training-data/raw/` — Contains raw API responses and backups in JSON format. These files are the initial outputs from data collection scripts and may vary in structure depending on the source.
- `training-data/formatted/` — Contains processed JSONL files, each with one JSON object per line. These files are ready for fine-tuning and follow a consistent schema.

### File Naming Conventions
- Files are named according to their source, e.g., `datasetmaster__resumes.jsonl`, `MikePfunk28__resume-training-dataset.jsonl`.
- Files ending with `__rewritten.jsonl` (e.g., `asaniczka__upwork-job-postings-dataset-2024-50k-records__rewritten.jsonl`) contain data that has been enhanced or rewritten, typically by an LLM or post-processing step. These are preferred for fine-tuning.
- Other suffixes (e.g., `__pairs`, `__clean_report`) indicate intermediate or diagnostic files and are not used for training.

## Data Format

Each line in a formatted `.jsonl` file is a JSON object with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a world-class tech resume writer. Context: ..."},
    {"role": "user", "content": "Improve this resume bullet:\n\"...original bullet...\""},
    {"role": "assistant", "content": "...improved bullet..."}
  ]
}
```

- The `messages` array follows the instruction-tuning format used by models like OpenAI GPT, Anthropic Claude, and Llama.
- The `system` message sets the context for the assistant (e.g., resume writing for a specific role).
- The `user` message provides the original resume bullet or prompt to be improved.
- The `assistant` message contains the improved or rewritten bullet, which is the target output for fine-tuning.

## Data Quality and Processing

- **Anonymization:** All personal information (emails, phone numbers, names) is replaced with tokens like `[EMAIL]`, `[PHONE]`, and `[NAME]`.
- **Deduplication:** Duplicate entries are removed to ensure a diverse and high-quality training set.
- **Validation:** Data is checked for schema consistency, minimum content length, and encoding issues. Records with empty assistant responses or encoding errors are filtered out.
- **Chunked Uploads:** Large datasets are split into smaller chunks for reliable uploading to Hugging Face.

## Usage Notes

- The formatted data is ready for use in LLM fine-tuning pipelines that support instruction-tuning formats.
- For best results, use the `__rewritten.jsonl` files when available, as these contain the highest-quality examples.
- Before training, further split the data into train, validation, and test sets to enable robust evaluation.

## Example Workflow

1. Collect raw data using scrapers (e.g., `src/scrapers/stackexchange_scraper.py`).
2. Format raw data into JSONL using the provided scripts (e.g., `scripts/format-for-llm.js`).
3. Validate and anonymize the formatted data.
4. Upload the processed data to Hugging Face using `scripts/upload_to_hf.py`.
5. Use the formatted JSONL files for fine-tuning and evaluation.

## References
- See the project README for setup, usage, and pipeline details.
- For more on the instruction-tuning format, refer to OpenAI, Anthropic, or Llama documentation.
