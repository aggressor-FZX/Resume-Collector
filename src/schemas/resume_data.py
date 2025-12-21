"""
ResumeData schema definition using Pydantic for data validation and type safety.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_core import PydanticCustomError
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json


class SchemaVersion(str, Enum):
    """Schema version enumeration for ResumeData migrations"""
    V1_0 = "1.0"  # Initial version
    V1_1 = "1.1"  # Added experience validation
    V1_2 = "1.2"  # Enhanced quality scoring
    V2_0 = "2.0"  # Major restructuring (future)


class SchemaMigration:
    """Handles schema migrations between versions"""

    @staticmethod
    def migrate(data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate data from one schema version to another"""
        if from_version == to_version:
            return data

        # Parse versions for comparison
        from_parts = [int(x) for x in from_version.split('.')]
        to_parts = [int(x) for x in to_version.split('.')]

        # Simple migration path for now (could be more complex)
        if from_parts < [1, 1] and to_parts >= [1, 1]:
            data = SchemaMigration._migrate_v1_0_to_v1_1(data)
        if from_parts < [1, 2] and to_parts >= [1, 2]:
            data = SchemaMigration._migrate_v1_1_to_v1_2(data)
        if from_parts < [2, 0] and to_parts >= [2, 0]:
            data = SchemaMigration._migrate_v1_2_to_v2_0(data)

        data['schema_version'] = to_version
        return data

    @staticmethod
    def _migrate_v1_0_to_v1_1(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migration from v1.0 to v1.1: Add experience validation"""
        # Ensure experience entries have required fields
        if 'experience' in data and data['experience']:
            validated_experience = []
            for exp in data['experience']:
                if isinstance(exp, dict):
                    # Ensure required fields exist
                    if 'title' not in exp or not exp['title']:
                        continue
                    if 'company' not in exp or not exp['company']:
                        continue
                    validated_experience.append(exp)
            data['experience'] = validated_experience
        return data

    @staticmethod
    def _migrate_v1_1_to_v1_2(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migration from v1.1 to v1.2: Enhanced quality scoring"""
        # Add any new fields or transformations needed for v1.2
        # For now, this is a no-op but structure is ready for future enhancements
        return data

    @staticmethod
    def _migrate_v1_2_to_v2_0(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migration from v1.2 to v2.0: Major restructuring"""
        # Placeholder for major version changes
        # This would handle breaking changes in v2.0
        return data


class SchemaRegistry:
    """Registry for schema versions and their specifications"""

    CURRENT_VERSION = SchemaVersion.V1_2

    @staticmethod
    def get_supported_versions() -> List[str]:
        """Get list of supported schema versions"""
        return [v.value for v in SchemaVersion]

    @staticmethod
    def is_version_supported(version: str) -> bool:
        """Check if a schema version is supported"""
        return version in SchemaRegistry.get_supported_versions()

    @staticmethod
    def validate_version_compatibility(data_version: str, current_version: str = CURRENT_VERSION.value) -> bool:
        """Check if data version is compatible with current version"""
        # For now, allow any supported version
        return SchemaRegistry.is_version_supported(data_version)


@dataclass
class QualityScore:
    """Quality assessment for resume data"""
    overall_score: float  # 0-100
    completeness_percentage: float  # 0-100
    field_scores: Dict[str, float]  # Individual field scores
    issues: List[str]  # List of quality issues found

    def __str__(self) -> str:
        return f"QualityScore(overall={self.overall_score:.1f}, complete={self.completeness_percentage:.1f}%)"


class ExperienceEntry(BaseModel):
    """Individual work experience entry"""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM or YYYY)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM or YYYY, or 'Present')")
    description: Optional[str] = Field(None, description="Job description")
    location: Optional[str] = Field(None, description="Job location")


class ResumeData(BaseModel):
    """Standardized resume data model for the Resume-Collector project"""

    # Core identity fields
    name: str = Field(..., min_length=1, description="Full name of the person")
    title: str = Field(..., min_length=1, description="Current or most recent job title")

    # Content fields
    content: str = Field(..., min_length=200, description="Raw resume content text")
    resume_text: str = Field(..., min_length=1, description="Cleaned and processed resume text")

    # Skills and experience
    skills: List[str] = Field(default_factory=list, min_length=5, max_length=50, description="List of technical and soft skills")
    experience: List[ExperienceEntry] = Field(default_factory=list, description="Work experience history")
    total_experience_years: float = Field(..., ge=1.0, le=20.0, description="Total years of professional experience")

    # Metadata
    location: Optional[str] = Field(None, description="Current location")
    source: str = Field(..., description="Data source (e.g., 'github', 'stackoverflow', 'kaggle')")
    anonymized: bool = Field(default=False, description="Whether PII has been removed")

    # Optional metadata
    source_url: Optional[str] = Field(None, description="Original source URL")
    collected_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="When data was collected")
    schema_version: str = Field(default=SchemaRegistry.CURRENT_VERSION.value, description="Schema version for migrations")

    @field_validator('name', 'title', 'resume_text')
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure required string fields are not just whitespace"""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty or whitespace only')
        return v.strip()

    @field_validator('skills')
    @classmethod
    def validate_skills_content(cls, v: List[str]) -> List[str]:
        """Ensure skills are meaningful (not empty, not just numbers)"""
        if not v:
            return v
        validated_skills = []
        for skill in v:
            skill = skill.strip()
            if not skill:
                continue  # Skip empty skills
            if len(skill) < 2:
                continue  # Skip very short skills
            if skill.isdigit():
                continue  # Skip pure numbers
            validated_skills.append(skill)
        return validated_skills

    @field_validator('content')
    @classmethod
    def validate_content_quality(cls, v: str) -> str:
        """Ensure content has reasonable quality indicators"""
        content = v.strip()
        if len(content) < 200:
            raise ValueError('Content must be at least 200 characters')
        # Check for minimum word count (roughly 30 words for 200 chars)
        words = content.split()
        if len(words) < 20:
            raise ValueError('Content must contain at least 20 words')
        return content

    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source is from allowed values"""
        allowed_sources = {'github', 'stackoverflow', 'kaggle', 'huggingface', 'semantic_scholar', 'openalex', 'manual'}
        source = v.lower().strip()
        if source not in allowed_sources:
            # Allow custom sources but log warning
            pass  # Could add logging here
        return source

    @field_validator('schema_version')
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Validate schema version is supported"""
        if not SchemaRegistry.is_version_supported(v):
            supported = SchemaRegistry.get_supported_versions()
            raise ValueError(f"Unsupported schema version '{v}'. Supported versions: {supported}")
        return v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"ResumeData(name='{self.name}', title='{self.title}', source='{self.source}', skills={len(self.skills)})"

    def quality_score(self) -> QualityScore:
        """Calculate quality score based on completeness and data richness"""
        issues = []
        field_scores = {}

        # Core fields (high weight)
        core_fields = ['name', 'title', 'content', 'resume_text', 'skills', 'total_experience_years']
        for field in core_fields:
            value = getattr(self, field)
            if field in ['content', 'resume_text']:
                # Content quality based on length
                score = min(100, len(value) / 10)  # 1000 chars = 100 score
                field_scores[field] = score
                if len(value) < 500:
                    issues.append(f"Content too short ({len(value)} chars)")
            elif field == 'skills':
                # Skills quality based on count and diversity
                skill_count = len(value)
                score = min(100, skill_count * 5)  # 20 skills = 100 score
                field_scores[field] = score
                if skill_count < 5:
                    issues.append(f"Too few skills ({skill_count})")
                elif skill_count > 30:
                    issues.append(f"Too many skills ({skill_count})")
            elif field == 'total_experience_years':
                # Experience score based on years
                score = min(100, value * 5)  # 20 years = 100 score
                field_scores[field] = score
                if value < 1:
                    issues.append("Very little experience")
            else:
                # Basic presence check
                score = 100 if value else 0
                field_scores[field] = score
                if not value:
                    issues.append(f"Missing {field}")

        # Optional fields (medium weight)
        optional_fields = ['location', 'experience']
        for field in optional_fields:
            value = getattr(self, field)
            if field == 'experience':
                exp_count = len(value) if value else 0
                score = min(100, exp_count * 25)  # 4 experiences = 100 score
                field_scores[field] = score
                if exp_count == 0:
                    issues.append("No work experience listed")
            else:
                score = 100 if value else 0
                field_scores[field] = score

        # Calculate completeness percentage
        all_scores = list(field_scores.values())
        completeness_percentage = sum(all_scores) / len(all_scores) if all_scores else 0

        # Overall score with weighted factors
        core_weight = 0.7
        optional_weight = 0.3

        core_avg = sum(field_scores[f] for f in core_fields) / len(core_fields)
        optional_avg = sum(field_scores[f] for f in optional_fields) / len(optional_fields)

        overall_score = (core_avg * core_weight) + (optional_avg * optional_weight)

        # Bonus for high completeness
        if completeness_percentage > 90:
            overall_score = min(100, overall_score * 1.1)

        return QualityScore(
            overall_score=round(overall_score, 1),
            completeness_percentage=round(completeness_percentage, 1),
            field_scores=field_scores,
            issues=issues
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeData":
        """Create from dictionary with automatic migration"""
        # Check if schema version is supported before migration
        data_version = data.get('schema_version', '1.0')  # Default to 1.0 for legacy data
        if not SchemaRegistry.is_version_supported(data_version):
            supported = SchemaRegistry.get_supported_versions()
            raise ValueError(f"Unsupported schema version '{data_version}'. Supported versions: {supported}")

        # Check if migration is needed
        current_version = SchemaRegistry.CURRENT_VERSION.value
        if data_version != current_version:
            data = SchemaMigration.migrate(data, data_version, current_version)

        return cls.model_validate(data)

    def migrate_to_version(self, target_version: str) -> "ResumeData":
        """Migrate this instance to a target schema version"""
        if self.schema_version == target_version:
            return self

        # Convert to dict, migrate, then back to model
        data = self.model_dump()
        migrated_data = SchemaMigration.migrate(data, self.schema_version, target_version)

        return ResumeData.model_validate(migrated_data)

    def to_hf_dict(self) -> Dict[str, Any]:
        """Convert to Hugging Face dataset format dictionary"""
        data = self.to_dict()

        # Experience is already a list of dicts from model_dump(), no need to change
        # Convert datetime to string
        if data.get('collected_at'):
            data['collected_at'] = data['collected_at'].isoformat()

        # Ensure all values are HF-compatible (strings, numbers, lists)
        return data

    @classmethod
    def from_hf_dict(cls, data: Dict[str, Any]) -> "ResumeData":
        """Create from Hugging Face dataset format dictionary"""
        # Deep copy to avoid modifying original
        data = data.copy()

        # Parse experience JSON strings back to objects
        if data.get('experience'):
            experience_list = []
            for exp in data['experience']:
                if isinstance(exp, str):
                    # If stored as JSON string, parse it
                    import json
                    exp = json.loads(exp)
                experience_list.append(exp)
            data['experience'] = experience_list

        # Parse datetime string back
        if data.get('collected_at') and isinstance(data['collected_at'], str):
            data['collected_at'] = datetime.fromisoformat(data['collected_at'])

        return cls.model_validate(data)

    @staticmethod
    def to_hf_dataset(resumes: List["ResumeData"]) -> Dict[str, List[Any]]:
        """Convert list of ResumeData to HF dataset format"""
        if not resumes:
            return {}

        # Get all unique keys
        all_keys = set()
        for resume in resumes:
            all_keys.update(resume.to_hf_dict().keys())

        # Create columnar data
        dataset_dict = {}
        for key in all_keys:
            dataset_dict[key] = []

        for resume in resumes:
            hf_dict = resume.to_hf_dict()
            for key in all_keys:
                dataset_dict[key].append(hf_dict.get(key))

        return dataset_dict

    @staticmethod
    def from_hf_dataset(dataset_dict: Dict[str, List[Any]]) -> List["ResumeData"]:
        """Convert HF dataset format back to list of ResumeData"""
        if not dataset_dict:
            return []

        # Get number of examples
        first_key = next(iter(dataset_dict.keys()))
        num_examples = len(dataset_dict[first_key])

        resumes = []
        for i in range(num_examples):
            example = {}
            for key, values in dataset_dict.items():
                if i < len(values):
                    example[key] = values[i]
            try:
                resumes.append(ResumeData.from_hf_dict(example))
            except Exception as e:
                print(f"Warning: Failed to parse example {i}: {e}")
                continue

        return resumes


if __name__ == "__main__":
    # Test the model with valid data
    sample_data = {
        "name": "John Doe",
        "title": "Software Engineer",
        "content": "Experienced software engineer with 5 years of professional development experience. Skilled in Python, JavaScript, and React development. Previously worked at Tech Corp developing web applications and APIs. Holds a Bachelor's degree in Computer Science.",
        "resume_text": "Experienced software engineer with 5 years...",
        "skills": ["Python", "JavaScript", "React", "Django", "PostgreSQL", "Git"],
        "experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "start_date": "2020-01",
                "end_date": "Present",
                "description": "Developed web applications"
            }
        ],
        "total_experience_years": 5.0,
        "location": "San Francisco, CA",
        "source": "github",
        "anonymized": False
    }

    try:
        resume = ResumeData.from_dict(sample_data)
        print("✅ ResumeData created successfully:")
        print(resume)
        print(f"Skills: {resume.skills}")
        print(f"Experience entries: {len(resume.experience)}")

        # Test quality scoring
        quality = resume.quality_score()
        print(f"\n📊 Quality Score: {quality}")
        print(f"Field scores: {quality.field_scores}")
        if quality.issues:
            print(f"Issues: {quality.issues}")

        # Test HF serialization
        hf_dict = resume.to_hf_dict()
        print(f"\n🤗 HF dict keys: {list(hf_dict.keys())}")

        # Test round-trip
        resume2 = ResumeData.from_hf_dict(hf_dict)
        print(f"Round-trip successful: {resume.name == resume2.name}")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    # Test schema versioning and migration
    print("\n--- Testing schema versioning ---")
    legacy_data = {
        "name": "Jane Smith",
        "title": "Data Scientist",
        "content": "Experienced data scientist with over 4 years of expertise in machine learning and statistical analysis. Proficient in Python, R, SQL, and various data science tools. Strong background in predictive modeling, data visualization, and statistical analysis. Previously worked at Data Corp developing machine learning models for customer behavior prediction and revenue optimization. Holds a Master's degree in Data Science from a top university. Has published several papers on machine learning applications in industry. Skilled in deploying ML models to production environments and building scalable data pipelines. Experience with cloud platforms like AWS and GCP for data processing and model training.",
        "resume_text": "Experienced data scientist...",
        "skills": ["Python", "R", "SQL", "Machine Learning", "Statistics", "Tableau"],
        "experience": [
            {
                "title": "Data Scientist",
                "company": "Data Corp",
                "start_date": "2019-06",
                "end_date": "Present",
                "description": "Built ML models"
            }
        ],
        "total_experience_years": 4.5,
        "location": "New York, NY",
        "source": "kaggle",
        "anonymized": True,
        "schema_version": "1.0"  # Legacy version
    }

    try:
        legacy_resume = ResumeData.from_dict(legacy_data)
        print("✅ Legacy data migrated successfully:")
        print(f"Original version: 1.0 -> Migrated version: {legacy_resume.schema_version}")
        print(f"Resume: {legacy_resume}")

        # Test migration method
        migrated = legacy_resume.migrate_to_version("1.1")
        print(f"Manual migration to 1.1: {migrated.schema_version}")

    except Exception as e:
        print(f"❌ Migration error: {e}")

    # Test invalid schema version
    print("\n--- Testing invalid schema version ---")
    invalid_version_data = {
        "name": "Test User",
        "title": "Test Title",
        "content": "This is a test content with enough words to pass validation. It needs at least twenty words for the content validation to pass. This should be sufficient for testing purposes. Adding more words to ensure we meet the minimum character requirement for the content field validation in our ResumeData schema. The validation requires at least 200 characters and 20 words minimum.",
        "resume_text": "Test resume text",
        "skills": ["Python", "JavaScript", "React", "Django", "SQL"],
        "experience": [],
        "total_experience_years": 5.0,
        "location": "Test City",
        "source": "github",
        "anonymized": False,
        "schema_version": "99.9"  # Invalid version that won't be migrated
    }

    try:
        invalid_resume = ResumeData.from_dict(invalid_version_data)
        print("❌ Should have failed with invalid version")
    except (ValidationError, ValueError) as e:
        print("✅ Invalid version correctly rejected:")
        print(f"  - {e}")

    # Test invalid data
    print("\n--- Testing validation ---")
    invalid_data = {
        "name": "",  # Empty name
        "title": "   ",  # Whitespace only
        "content": "Too short",  # Too short content
        "resume_text": "Valid text",
        "skills": ["Python"],  # Too few skills
        "experience": [],
        "total_experience_years": 25.0,  # Too many years
        "source": "invalid_source"
    }

    try:
        invalid_resume = ResumeData.from_dict(invalid_data)
        print("❌ Should have failed validation")
    except ValidationError as e:
        print("✅ Validation correctly failed:")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")
