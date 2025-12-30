"""
Tests for Pydantic v2 migration - ensuring no deprecation warnings.
"""

import warnings
import pytest
from src.config import Config, APIKeys, RateLimits, LoggingConfig, ProjectConfig
from src.schemas.resume_data import ResumeData, ExperienceEntry


def test_config_no_deprecation_warnings():
    """Test that Config classes don't emit deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test APIKeys
        api_keys = APIKeys()
        assert isinstance(api_keys.github_api_key, (str, type(None)))

        # Test RateLimits
        rate_limits = RateLimits()
        assert isinstance(rate_limits.github_rate_limit, int)

        # Test LoggingConfig
        logging_config = LoggingConfig()
        assert isinstance(logging_config.level, str)

        # Test ProjectConfig
        project_config = ProjectConfig()
        assert isinstance(project_config.name, str)

        # Test main Config
        config = Config()
        assert isinstance(config.project.name, str)

        # Check for any Pydantic deprecation warnings
        pydantic_warnings = [warning for warning in w if "Pydantic" in str(warning.message)]
        assert len(pydantic_warnings) == 0, f"Found Pydantic deprecation warnings: {[str(w.message) for w in pydantic_warnings]}"


def test_resume_data_no_deprecation_warnings():
    """Test that ResumeData doesn't emit deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test ExperienceEntry
        exp = ExperienceEntry(
            title="Software Engineer",
            company="Tech Corp",
            description="Developed web applications"
        )
        assert exp.title == "Software Engineer"

        # Test ResumeData creation
        resume = ResumeData(
            name="John Doe",
            title="Software Engineer",
            content="Experienced software engineer...",
            resume_text="Experienced software engineer...",
            total_experience_years=5.0,
            source="github"
        )
        assert resume.name == "John Doe"

        # Test serialization
        data = resume.to_dict()
        assert data["name"] == "John Doe"

        # Test deserialization
        resume2 = ResumeData.from_dict(data)
        assert resume2.name == "John Doe"

        # Check for any Pydantic deprecation warnings
        pydantic_warnings = [warning for warning in w if "Pydantic" in str(warning.message)]
        assert len(pydantic_warnings) == 0, f"Found Pydantic deprecation warnings: {[str(w.message) for w in pydantic_warnings]}"


def test_resume_data_datetime_serialization():
    """Test that datetime serialization works without warnings."""
    import datetime

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create resume with datetime
        collected_at = datetime.datetime(2023, 1, 1, 12, 0, 0)
        resume = ResumeData(
            name="Test",
            title="Engineer",
            content="Content",
            resume_text="Text",
            total_experience_years=1.0,
            source="test",
            collected_at=collected_at
        )

        # Test serialization
        data = resume.to_dict()
        assert "collected_at" in data
        assert isinstance(data["collected_at"], str)

        # Check for any deprecation warnings
        pydantic_warnings = [warning for warning in w if "Pydantic" in str(warning.message)]
        assert len(pydantic_warnings) == 0, f"Found Pydantic deprecation warnings: {[str(w.message) for w in pydantic_warnings]}"