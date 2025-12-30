import pytest
import os
import subprocess
import json
from pathlib import Path

@pytest.fixture
def sample_raw_data(tmp_path):
    """Create sample raw input for formatter."""
    raw_path = tmp_path / "sample.json"
    sample_data = {
        "resumes": [
            {
                "original_bullet": "Wrote Python code",
                "improved_bullet": "Developed scalable Python backend serving 10k+ users",
                "context": "Software Engineer at TechCorp"
            }
        ]
    }
    with open(raw_path, "w") as f:
        json.dump(sample_data, f)
    return raw_path

def test_formatter_integration(sample_raw_data, tmp_path):
    """Test TS formatter + JSONL validation end-to-end."""
    input_file = sample_raw_data
    output_file = tmp_path / "formatted.jsonl"
    
    os.chdir("/workspaces/Resume-Collector")
    
    # Run formatter
    cmd = [
        "npx", "ts-node", "scripts/format-for-llm.ts",
        "--input", str(input_file),
        "--output", str(output_file),
        "--format", "openai"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Formatter failed: {result.stderr}"
    assert output_file.exists()
    
    # Validate JSONL
    with open(output_file, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                json.loads(line)

def test_main_pipeline(sample_raw_data, tmp_path):
    """Test main.py orchestration."""
    input_file = sample_raw_data
    output_file = tmp_path / "test_formatted.jsonl"
    
    os.chdir("/workspaces/Resume-Collector")
    
    cmd = [
        "python", "src/main.py", "format",
        "--input", str(input_file),
        "--output", str(output_file)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert output_file.exists()
    assert "✓ JSONL validation passed" in result.stdout or "Formatter succeeded" in result.stdout
