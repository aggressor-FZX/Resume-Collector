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

If you want me to adjust anonymization rules or add stronger PII scrubbing (e.g., named-entity recognition based), I can add that as an optional step.