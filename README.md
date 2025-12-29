# Resume-Collector
Collects and ingest data to transform for LLM finetuning

## Fine-tuning Data Pipeline (resume writer)

This repo includes a minimal pipeline to fetch raw resumes, format them into instruction-tuning JSONL, validate the output, and commit formatted training data back into the repo.

Folders created:

- `training-data/raw/` — API responses and backups (JSON)
- `training-data/formatted/` — JSONL ready for fine-tuning (one JSON per line)
- `scripts/format-for-llm.js` — Node script to transform `raw` ➜ `formatted`
- `.github/workflows/sync-training-data.yml` — GitHub Actions workflow to keep training data synced

Usage

1. Add repo secrets and variables:
   - `RESUME_API_KEY` (secret)
   - `RESUME_API_URL` (repository variable)

2. Test locally:

```bash
node scripts/format-for-llm.js --input training-data/raw/sample.json --output training-data/formatted/sample.jsonl --format openai
```

3. Validate the JSONL:

```bash
python -c "import sys,json; [json.loads(line) for line in open('training-data/formatted/sample.jsonl','r')]"
```

4. To run on GitHub Actions, either trigger manually from Actions > Sync Resume Training Data, or use:

```bash
# trigger a dispatch event
gh api repos/<owner>/Resume-Collector/dispatches -f event_type=resume-api-update
```

Notes

- The formatter expects input JSON with a top-level `resumes` array. Example entries: `{ "original_bullet": "...", "improved_bullet": "...", "context": "..." }`.
- You can set `--format anthropic|llama` to change message structure if needed.

Privacy & data handling

- **Anonymization**: the formatter runs lightweight anonymization to replace emails, phone numbers, and some name patterns with tokens like `[EMAIL]`, `[PHONE]`, and `[NAME]`.
- **Deduplication**: duplicate bullets are deduplicated case-insensitively to reduce noisy examples in training data.
- **No auto fine-tuning**: This pipeline only collects and formats training data. It does **not** trigger or upload to any fine-tuning/training provider automatically — that step is intentionally manual to allow review and governance.

Testing & CI

- A `Validate Formatter` workflow runs on PRs and checks that the TypeScript formatter and unit tests pass against sample data.

## Development setup 🔧

Create a virtual environment and install Python dependencies (example using venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommendations:

- Copy `.env.example` to `.env` and fill in provider API keys (e.g., `PERPLEXITY_API_KEY`, `GITHUB_API_KEY`).
- Use `python -m pytest` to run the test suite.
- Add new processors and schemas in `src/processors/` and `src/schemas/` respectively.



## Uploading to Hugging Face

The `scripts/upload_to_hf.py` script is used to upload formatted datasets to Hugging Face. It supports chunked uploads for large datasets and verifies the destination repository before starting the upload.

### Prerequisites

1. Ensure you have a Hugging Face account and API token.
2. Add the token to your `.env` file:

```bash
HUGGING_FACE_API_KEY=<your_huggingface_token>
```

### Running the Script

To upload the dataset:

```bash
python scripts/upload_to_hf.py
```

### Features

- **Chunked Uploads**: The script splits large datasets into smaller chunks for reliable uploads.
- **Destination Verification**: Ensures the Hugging Face repository is reachable and properly configured before uploading.
- **Dataset Card**: Automatically generates a simple dataset card (`dataset_card.md`).

### Notes

- The script looks for formatted JSONL files in `training-data/formatted/`.
- Ensure the dataset is properly formatted and anonymized before uploading.
- Check the Hugging Face repository after upload: [https://huggingface.co/datasets/](https://huggingface.co/datasets/).

## Running the Full Pipeline

Follow these steps to run the entire pipeline from data collection to uploading the dataset to Hugging Face:

### 1. Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/aggressor-FZX/Resume-Collector.git
   cd Resume-Collector
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and configure API keys:
   ```bash
   cp .env.example .env
   # Fill in the required keys, e.g., HUGGING_FACE_API_KEY
   ```

### 2. Data Collection

Run the scrapers to collect raw resume data. For example, to run the Stack Exchange scraper:
```bash
python src/scrapers/stackexchange_scraper.py
```

### 3. Data Formatting

Format the raw data into JSONL files suitable for fine-tuning:
```bash
node scripts/format-for-llm.js --input training-data/raw/sample.json --output training-data/formatted/sample.jsonl --format openai
```

### 4. Validation

Validate the formatted JSONL files:
```bash
python -c "import sys,json; [json.loads(line) for line in open('training-data/formatted/sample.jsonl','r')]"
```

### 5. Upload to Hugging Face

Upload the formatted dataset to Hugging Face:
```bash
python scripts/upload_to_hf.py
```

This script supports chunked uploads for large datasets and verifies the Hugging Face repository before starting the upload.

### Notes

- Ensure all API keys and environment variables are correctly set in the `.env` file.
- Check the Hugging Face repository after upload: [https://huggingface.co/datasets/](https://huggingface.co/datasets/).
- For large datasets, the upload script will automatically split the data into smaller chunks to ensure reliability.

## Progress

2025-12-28T22:23:56Z | upload_manifest created with 9 datasets, total_records=36344
