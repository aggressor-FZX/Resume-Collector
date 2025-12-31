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

2025-12-31T02:54:57Z | imaginator_generation | completed | Generated 180 imaginator records in 300Run_imaginator_fuel.jsonl, uploaded to HF repo jeff-calderon/imaginator-dataset. Model contributions: nex-agi/deepseek-v3.1-nex-n1:free 22.22% (40), meta-llama/llama-3.3-70b-instruct 8.89% (16), deepseek/deepseek-v3.2-speciale 8.33% (15), qwen/qwen-2.5-72b-instruct 8.33% (15), nousresearch/hermes-4-70b 9.44% (17), deepseek/deepseek-v3.2 7.78% (14), thedrummer/rocinante-12b 7.78% (14), deepseek/deepseek-r1-distill-qwen-32b 6.11% (11), thedrummer/skyfall-36b-v2 6.67% (12), DeepSeek-V3.1 3.89% (7), Meta-Llama-3.3-70B-Instruct 4.44% (8), DeepSeek-R1-0528 3.33% (6), gpt-oss-120b 2.78% (5)
2025-12-30T21:32:00Z | SambaNova Qwen3-235B integrated into model cascade, HF free models added, successful 3-minute test with 4 records generated, data uploaded to jeff-calderon/Tech_Resumes
2025-12-28T22:23:56Z | upload_manifest created with 9 datasets, total_records=36344

---

## Imaginator Training Plan — Hybrid Instruction Tuning 🧠💡

**Overview:** This plan describes a production-ready approach to fine-tuning a large base model so it can do more than "text polishing" — it will perform strategic reasoning, taking inputs from your pipeline (Hermes, FastSVM, Job Ads) and produce resumes that *bridge skill gaps*.

### 1) The Core Philosophy — **Hybrid Instruction Tuning**

- We fine-tune a large "Teacher" model (e.g., `meta-llama/llama-3.3-70b-instruct` or a Magnum 72B variant) using a hybrid dataset composed of:
  - **Dataset A — The Stylist (80%)**: Real Upwork/collected rows (~26k). Teaches tone, active phrasing, and metric-driven bullets.
  - **Dataset B — The Strategist (20%)**: Synthetic "Imaginator Scenarios" (1k–2k). Trains strategic reasoning and gap-bridging logic.

### 2) Step-by-Step Training Workflow

**Phase 1 — Synthesizing the "Strategist" Data**
- Problem: We lack examples of `Hermes Inputs -> Perfect Resumes`.
- Solution: Use a stronger Teacher model (e.g., Hermes 3 405B or GPT-4o) to generate ideal responses from simulated scenarios.
- Generator script behavior:
  - Input: A raw resume from your dataset.
  - Simulation: Randomly assign a Target Job and simulate Hermes outputs (e.g., "Flagged Gap: Kubernetes").
  - Teacher prompt (example):

```text
"You are an expert Resume Strategist. Given a candidate profile, a Target Job, and identified Gaps, craft a resume segment that uses inferred bridges (e.g., Docker -> Kubernetes) to minimize the gap. Output must be interview-ready and metric-driven."
```

- Save the paired (scenario input, Teacher Output) as a JSONL training record.

**Phase 2 — Constructing the Training Dataset**
- Format records into the conversational JSONL used by Unsloth/Hugging Face, e.g.:

```json
{
  "messages": [
    {"role": "system", "content": "You are the Imaginator. Synthesize the input into a tailored resume. Use 'Inferred Skills' to bridge gaps."},
    {"role": "user", "content": "CONTEXT:\n- Candidate: Senior Python Dev\n- Target Job: Cloud Architect\n\nLOGIC LAYER:\n- Critical Gap: AWS Security\n- Inferred Bridge: 'Identity Management' (from Django Auth)\n- Market Trend: Serverless\n\nTASK: Rewrite Project Experience for 'E-Commerce Backend' role."},
    {"role": "assistant", "content": "Project: E-Commerce Backend Refactor\n- Architected a serverless backend using Python and AWS Lambda, implementing Identity Management (OAuth2) to secure user data..."}
  ]
}
```

**Phase 3 — Fine-Tuning Environment**
- Use **Unsloth** on A100 GPUs to run **QLoRA** (quantized LoRA adapters).
- Base model: `meta-llama/llama-3.3-70b-instruct` quantized (4-bit preferred).
- Why QLoRA: Freeze base weights and train small adapter layers (~1–2% params) so the model learns domain-specific reasoning without catastrophic forgetting.
- Target modules to adapt: all linear modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This focuses training on reasoning pathways, not surface-level vocabulary.

**Phase 4 — The Training Run**
- Load hybrid dataset (80% Stylist + 20% Strategist).
- Train for ~1–2 epochs with a low LR (e.g., 2e-4) and early stopping.
- Save only LoRA adapters (compact artifacts, ~200MB) for production deployment.
- Validation: run held-out job-ad alignment tests and manual spot-checks to detect catastrophic degradation.

### 3) Production Architecture & Inference Flow

- Frontend: user uploads resume + target job URL.
- Preprocessing: Document Reader extracts text; Hermes & FastSVM extract skills & signals.
- Logic Module: computes adjacency and gap scores (e.g., Python=90%, Azure=0% — Azure adjacent to AWS).
- Prompt Engineering (the Bridge): Structured prompt communicates Profile, Gap, and Strategy to `Imaginator`.
- Inference: a quantized `Llama-3.3-70B` + LoRA adapter produces a role-targeted, metric-driven resume output.

**Summary Checklist ✅**
- **Model**: Llama 3.3 70B Instruct (quantized)
- **Technique**: QLoRA via Unsloth
- **Data Strategy**: 80% Stylist (real Upwork), 20% Strategist (synthetic Imaginator scenarios)
- **Hardware**: A100 (80GB) or A100 (40GB) ×1–2
- **Outcome**: Model that bridges skill gaps and writes interview-focused, metric-driven resume content.

---

If you'd like, I can also add supporting scripts and example training notebooks to `scripts/` and `notebooks/` to automate Phase 1+2 (scenario generation, teacher calls, JSONL formatting). Let me know which you'd prefer next.
