# Hugging Face Resume-Collector Dataset Format

## Data Format in Hugging Face

The data already uploaded to Hugging Face from Resume-Collector is in **JSONL (JSON Lines)** format, specifically designed for instruction-tuning large language models.

### File Structure
- Each file contains one JSON object per line.
- Files are named by their source, e.g., `datasetmaster__resumes__rewritten.jsonl`, `MikePfunk28__resume-training-dataset__rewritten.jsonl`.
- The `__rewritten.jsonl` files are the preferred, high-quality data for fine-tuning.

### Example Record
Each line in a `.jsonl` file follows this schema:

```json
{
  "messages": [
    {"role": "system", "content": "You are a world-class tech resume writer. Context: ..."},
    {"role": "user", "content": "Improve this resume bullet:\n\"...original bullet...\""},
    {"role": "assistant", "content": "...improved bullet..."}
  ]
}
```

- **messages**: An array of message objects, each with:
    - **role**: One of `system`, `user`, or `assistant`.
    - **content**: The text for that role.
- The `system` message sets the context for the assistant.
- The `user` message provides the original resume bullet to be improved.
- The `assistant` message contains the improved bullet, which is the target output for fine-tuning.

### Key Points
- All records are anonymized and deduplicated before upload.
- The format matches standard instruction-tuning datasets for LLMs (OpenAI, Anthropic, Llama, etc.).
- Only the `__rewritten.jsonl` files (and similar formatted files) in Hugging Face are intended for training; diagnostic or intermediate files are not used.

## Summary

**The Hugging Face dataset consists of JSONL files, each line containing a `messages` array with `system`, `user`, and `assistant` roles, ready for direct use in LLM fine-tuning workflows.**
