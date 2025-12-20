import asyncio
import pytest
from src.scrapers.stackexchange_scraper import StackExchangeScraper
from src.resume_model import ResumeData

@pytest.mark.asyncio
async def test_determine_title_and_skills():
    s = StackExchangeScraper()
    tags = [
        {'tag_name': 'python', 'post_score': 100, 'answer_count': 200},
        {'tag_name': 'django', 'post_score': 50, 'answer_count': 30},
    ]
    title = s._determine_title_from_tags(tags)
    assert 'Python' in title
    skills = s._extract_skills_from_tags(tags)
    assert 'Python' in skills and 'Django' in skills

@pytest.mark.asyncio
async def test_create_resume_from_user():
    s = StackExchangeScraper()

    # patch network calls
    async def fake_get_user_tags(user_id):
        return [{'tag_name': 'python', 'post_score': 10, 'answer_count': 5}]

    async def fake_get_user_top_answers(user_id):
        return [{'score': 42, 'is_accepted': True}, {'score': 10, 'is_accepted': False}]

    s._get_user_tags = fake_get_user_tags
    s._get_user_top_answers = fake_get_user_top_answers

    user_data = {
        'user_id': 123,
        'display_name': 'Test Dev',
        'reputation': 1500,
        'badge_counts': {'gold': 2, 'silver': 5, 'bronze': 10},
        'creation_date': 1577836800,  # Jan 1, 2020
        'about_me': '<p>Experienced developer</p>',
        'answer_count': 50
    }

    resume = await s._create_resume_from_user(user_data)
    assert isinstance(resume, ResumeData)
    assert resume.name == 'Test Dev'
    assert 'Reputation' in (resume.content or '')
    assert resume.total_experience_years is not None
    assert 'python' in ','.join([sk.lower() for sk in resume.skills])

@pytest.mark.asyncio
async def test_estimate_experience():
    s = StackExchangeScraper()
    user_data = {'creation_date': 1609459200, 'reputation': 6000}  # 2021-01-01
    years = s._estimate_experience_from_so(user_data)
    assert isinstance(years, int)
    assert years >= 1 and years <= 20
