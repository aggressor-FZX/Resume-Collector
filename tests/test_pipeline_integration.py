"""
Integration tests for the main pipeline.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path

def test_formatter_validation_integration():
    """Test the complete formatter + validation pipeline."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample input data
        sample_data = {
            "resumes": [
                {
                    "original_bullet": "Developed web applications using Python and Django",
                    "improved_bullet": "Engineered scalable web applications using Python and Django framework, resulting in 40% performance improvement",
                    "context": "Software Engineer"
                },
                {
                    "original_bullet": "Managed team of 5 developers",
                    "improved_bullet": "Led cross-functional team of 5 developers in agile environment, delivering projects 20% ahead of schedule",
                    "context": "Engineering Manager"
                }
            ]
        }

        input_file = Path(temp_dir) / "input.json"
        output_file = Path(temp_dir) / "output.jsonl"

        # Write sample data
        with open(input_file, 'w') as f:
            json.dump(sample_data, f)

        # Run the formatter using subprocess
        import subprocess
        import sys

        cmd = [
            sys.executable, "src/main.py",
            "--format-only",
            "--input", str(input_file),
            "--output-file", str(output_file),
            "--format-type", "openai"
        ]

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

        # Check that the command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check that output file was created
        assert output_file.exists(), "Output file was not created"

        # Validate the JSONL content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

            for line in lines:
                # Parse JSON
                data = json.loads(line.strip())
                assert "messages" in data, "Missing messages field"
                messages = data["messages"]
                assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"

                # Check message structure
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert messages[2]["role"] == "assistant"

                # Check content
                assert "Context:" in messages[0]["content"]
                assert "Improve this resume bullet" in messages[1]["content"]


def test_validation_only():
    """Test the validation-only functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid JSONL file
        jsonl_file = Path(temp_dir) / "test.jsonl"
        test_data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "system", "content": "You are an AI"}]}
        ]

        with open(jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Run validation
        import subprocess
        import sys

        cmd = [
            sys.executable, "src/main.py",
            "--validate-only", str(jsonl_file)
        ]

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

        # Should succeed
        assert result.returncode == 0
        assert "JSONL validation passed" in result.stdout


def test_invalid_jsonl_validation():
    """Test validation fails on invalid JSONL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create invalid JSONL file
        jsonl_file = Path(temp_dir) / "invalid.jsonl"

        with open(jsonl_file, 'w') as f:
            f.write('{"invalid": json}\n')
            f.write('{"valid": "json"}\n')

        # Run validation
        import subprocess
        import sys

        cmd = [
            sys.executable, "src/main.py",
            "--validate-only", str(jsonl_file)
        ]

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

        # Should fail
        assert result.returncode == 1
        assert "Invalid JSON" in result.stdout or "Invalid JSON" in result.stderr