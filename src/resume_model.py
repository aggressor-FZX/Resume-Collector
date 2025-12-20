from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ResumeData:
    name: Optional[str]
    title: Optional[str]
    content: Optional[str]
    resume_text: Optional[str]
    skills: List[str]
    experience: List[str]
    total_experience_years: Optional[int]
    location: Optional[str]
    source: Optional[str]
    anonymized: bool = True
