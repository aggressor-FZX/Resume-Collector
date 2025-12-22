#!/usr/bin/env python3
"""
Resume-Collector Main Entry Point

This script provides the main interface for running the resume data collection pipeline.
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

def run_command(cmd: List[str], cwd: Optional[str] = None) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(f"✓ Command succeeded: {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False

def run_data_collection(sources: List[str], output_dir: str) -> bool:
    """Run data collection from specified sources."""
    print(f"📊 Starting data collection from sources: {sources}")

    success = True
    for source in sources:
        if source == "stackexchange":
            print("Running Stack Exchange scraper...")
            # Import and run the scraper
            try:
                from scrapers.stackexchange_scraper import StackExchangeScraper
                # This would need proper implementation
                print("Stack Exchange scraper not fully implemented yet")
            except ImportError:
                print("Stack Exchange scraper not available")
                success = False
        else:
            print(f"Source '{source}' not implemented yet")
            success = False

    return success

def run_formatting(input_file: str, output_file: str, format_type: str = "openai") -> bool:
    """Run the LLM formatting script."""
    print(f"🔧 Running formatter on {input_file} -> {output_file}")

    # Use the TypeScript formatter
    cmd = [
        "npx", "ts-node",
        "scripts/format-for-llm.ts",
        "--input", input_file,
        "--output", output_file,
        "--format", format_type
    ]

    return run_command(cmd)

def run_validation(jsonl_file: str) -> bool:
    """Validate the formatted JSONL file."""
    print(f"✅ Validating JSONL file: {jsonl_file}")

    if not os.path.exists(jsonl_file):
        print(f"✗ File not found: {jsonl_file}")
        return False

    try:
        import json
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"✗ Invalid JSON at line {i}: {e}")
                    return False
        print(f"✓ JSONL validation passed for {jsonl_file}")
        return True
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False

def run_upload(dataset_path: str, repo_id: str) -> bool:
    """Upload dataset to Hugging Face."""
    print(f"📤 Uploading to Hugging Face: {repo_id}")

    cmd = ["python", "scripts/upload_to_hf.py"]
    return run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="Resume-Collector: Collect, format, validate, and upload resume data")
    parser.add_argument('--sources', nargs='+', default=[], help='Data sources (stackexchange)')
    parser.add_argument('--input', help='Raw input JSON file for formatting')
    parser.add_argument('--output', help='Output JSONL file')
    parser.add_argument('--format', action='store_true', help='Run formatting step')
    parser.add_argument('--validate', action='store_true', help='Validate JSONL')
    parser.add_argument('--upload', action='store_true', help='Upload to HF')
    parser.add_argument('--full', action='store_true', help='Run full pipeline: collect -> format -> validate -> upload')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (print commands only)')

    args = parser.parse_args()

    print("🚀 Resume-Collector Pipeline")
    print(f"Sources: {args.sources}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    success = True

    if args.full or args.format:
        if not args.input or not args.output:
            print("✗ --input and --output required for formatting")
            success = False
        else:
            if not args.dry_run:
                success = success and run_formatting(args.input, args.output)
            else:
                print("DRY: npx ts-node scripts/format-for-llm.ts ...")

    if args.full or args.validate:
        if args.output and not args.dry_run:
            success = success and run_validation(args.output)
        elif args.dry_run:
            print("DRY: python validate_jsonl.py ...")

    if args.full or args.upload:
        if not args.dry_run:
            success = success and run_upload(args.output, "jeff-calderon/ResumeData")
        else:
            print("DRY: python scripts/upload_to_hf.py")

    if args.sources and not args.dry_run:
        success = success and run_data_collection(args.sources, "training-data/raw")

    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
