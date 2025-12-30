"""
ResumeData schema definition using Pydantic for data validation and type safety.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from pydantic import ConfigDict


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core identity fields
    name: str = Field(..., description="Full name of the person")
    title: str = Field(..., description="Current or most recent job title")

    # Content fields
    content: str = Field(..., description="Raw resume content text")
    resume_text: str = Field(..., description="Cleaned and processed resume text")

    # Skills and experience
    skills: List[str] = Field(default_factory=list, description="List of technical and soft skills")
    experience: List[ExperienceEntry] = Field(default_factory=list, description="Work experience history")
    total_experience_years: float = Field(..., description="Total years of professional experience")

    # Metadata
    location: Optional[str] = Field(None, description="Current location")
    source: str = Field(..., description="Data source (e.g., 'github', 'stackoverflow', 'kaggle')")
    anonymized: bool = Field(default=False, description="Whether PII has been removed")

    # Optional metadata
    source_url: Optional[str] = Field(None, description="Original source URL")
    collected_at: Optional[datetime] = Field(default_factory=datetime.now, description="When data was collected")
    schema_version: str = Field(default="1.0", description="Schema version for migrations")

    @field_serializer('collected_at')
    def serialize_collected_at(self, value: Optional[datetime]) -> Optional[str]:
        if value is not None:
            return value.isoformat()
        return None

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"ResumeData(name='{self.name}', title='{self.title}', source='{self.source}', skills={len(self.skills)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (Pydantic v2 compatible)"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeData":
        """Create from dictionary (Pydantic v2 compatible)"""
        return cls.model_validate(data)


if __name__ == "__main__":
    # Test the model
    sample_data = {
        "name": "John Doe",
        "title": "Software Engineer",
        "content": "Experienced software engineer with 5 years...",
        "resume_text": "Experienced software engineer with 5 years...",
        "skills": ["Python", "JavaScript", "React"],
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

    resume = ResumeData.from_dict(sample_data)
    print("ResumeData created successfully:")
    print(resume)
    print(f"Skills: {resume.skills}")
    print(f"Experience entries: {len(resume.experience)}")
