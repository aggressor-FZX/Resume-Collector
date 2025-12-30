# All Resumes (Aggregated)

Short Description
-----------------
An aggregated collection of cleaned and formatted resume bullet points and rewritten resume content derived from multiple public and collected resume datasets. Each record contains a short resume bullet or improved resume text suited for LLM fine-tuning of resume-writing assistants.

Dataset Details
---------------
- Source: Aggregated from multiple cleaned datasets included in the repository (see `data/cleaned_test/`).
- Records: 13,852 (post-deduplication)
- Format: JSON Lines (one JSON object per line) and Parquet
- License: MIT (please verify and confirm licensing of source datasets before public distribution)
- Languages: English (predominantly)

Fields
------
- `content` (string): The cleaned/improved resume text used as the main training target.
- `original` (string): Original user-provided content (source bullet) where available.
- `source` (string): Source dataset identifier (e.g., `gauravduttakiit__resume-dataset`).
- Additional fields may exist depending on source transform.

Provenance
----------
This dataset was produced by combining formatted outputs (`data/formatted/*__openai.jsonl`), extracting the assistant/user content, running deduplication (exact and fuzzy) and producing quality metrics (see `data/qc_report.json` and `data/qc_summary.txt`).

How to Use
----------
- Load JSONL directly or use the provided Parquet file for efficient analytics.
- Use `content` as the training target for LLM fine-tuning.

Citation
--------
Please attribute the dataset to `jeff-calderon` and include a link to the repository and the Hugging Face dataset once published.

Contact
-------
`jeff-calderon` on Hugging Face.
