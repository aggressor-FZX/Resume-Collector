---
description: 'automates the Resume-Collector pipeline: it collects raw resume data, formats and validates instruction‑tuning JSONL, appends progress updates to README.md, writes structured logs to logs/progress.log, and uploads approved formatted datasets to a Hugging Face dataset repository.'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'huggingface/hf-mcp-server/*', 'todo', 'github.vscode-pull-request-github/copilotCodingAgent', 'github.vscode-pull-request-github/issue_fetch', 'github.vscode-pull-request-github/suggest-fix', 'github.vscode-pull-request-github/searchSyntax', 'github.vscode-pull-request-github/doSearch', 'github.vscode-pull-request-github/renderIssues', 'github.vscode-pull-request-github/activePullRequest', 'github.vscode-pull-request-github/openPullRequest', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment']
---
What this custom agent accomplishes and when to use it
Purpose  
This agent automates the Resume-Collector pipeline: it collects raw resume data, formats and validates instruction‑tuning JSONL, appends progress updates to README.md, writes structured logs to logs/progress.log, and uploads approved formatted datasets to a Hugging Face dataset repository.

When to use it  
Use the agent when you want repeatable, auditable dataset preparation runs inside VS Code or GitHub Codespaces and when you need an automated, review-gated path to publish anonymized training data to Hugging Face.

Edges it will not cross

It will not fine-tune models or trigger training automatically.

It will not upload data to Hugging Face without an explicit review approval step.

It will not log or commit raw PII; sensitive fields are redacted before any commit or upload.

Ideal inputs outputs and tools it may call
Inputs

Raw JSON files placed in training-data/raw/ with a top-level resumes array.

Environment variables: HUGGING_FACE_API_KEY, RESUME_API_KEY, RESUME_API_URL.

Optional CLI flags: --format (openai|anthropic|llama), --hf-repo.

Outputs

Formatted JSONL files in training-data/formatted/ (one JSON object per line).

Structured progress log at logs/progress.log.

README.md updated under a ## Progress section with short status lines.

Optional upload to Hugging Face dataset repo after manual approval.

Tools the agent may call

Local scripts: scripts/format-for-llm.js, scripts/upload_to_hf.py, scripts/agent-runner.js.

Git for commits and pushes.

Hugging Face API via HUGGING_FACE_API_KEY.

CI integration via GitHub Actions for scheduled or push-triggered runs.

How progress is reported and how the agent asks for help
Progress reporting

Every plan step writes a structured line to logs/progress.log using the format: 2025-12-27T22:09:00Z | step_id | status | message.

If append_to_readme is enabled the agent inserts or updates a ## Progress section in README.md with the latest step statuses (one-line entries).

When git_commit_on_change is true the agent commits only the changed files (formatted data, logs, README) using the configured commit_author.

Help and escalation

On recoverable errors the agent writes an error entry to the log and prints a concise console message with the failing step and suggested remediation.

On non-recoverable errors the agent halts and creates a draft PR or opens an issue (configurable) with the log excerpt and required next steps.

The agent will never auto-resolve redaction or privacy failures; it requests manual review before upload.

Hugging Face upload behavior and governance
Upload prerequisites

HUGGING_FACE_API_KEY must be present in the environment or repository secrets.

require_review_before_upload enforces a manual approval step: the agent prepares the upload bundle and creates a PR or a release candidate branch; a human reviewer must approve before upload step runs.

Upload features

Chunked uploads for large JSONL files to avoid timeouts.

Destination verification to confirm the HF repo exists and the token has write access.

Automatic generation of a minimal dataset_card.md describing dataset, license, and anonymization steps.

Privacy controls

Built-in redaction patterns replace emails, phone numbers, and common name patterns with tokens before any commit or upload.

A validation step checks for residual PII tokens and fails if suspicious patterns remain.

Deployment checklist and example snippets
Local test

bash
# prepare env
cp .env.example .env
export HUGGING_FACE_API_KEY=xxxx
# run full pipeline locally (dry run)
node scripts/agent-runner.js init
node scripts/agent-runner.js collect
node scripts/agent-runner.js format --in training-data/raw --out training-data/formatted --format openai
node scripts/agent-runner.js validate --path training-data/formatted
# prepare upload bundle (no upload until review)
node scripts/agent-runner.js prepare-upload --hf-repo owner/repo