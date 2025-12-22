from typing import List, Dict, Any, Optional
import re
import logging

from src.base_scraper import BaseScraper, RateLimiter
import src.config as cfg
from src.schemas.resume_data import ResumeData, ExperienceEntry

logger = logging.getLogger(__name__)

# Use the config module values
STACK_BASE = getattr(cfg, 'STACK_EXCHANGE_API_BASE', 'https://api.stackexchange.com/2.3')
STACK_KEY = getattr(cfg, 'STACK_EXCHANGE_API_KEY', None)
RATE_LIMIT = getattr(cfg, 'STACK_EXCHANGE_RATE_LIMIT', 1000)


class StackExchangeScraper(BaseScraper):
    """Scraper for Stack Exchange API to extract developer profile data."""

    def __init__(self):
        super().__init__("StackExchange")
        self.rate_limiter = RateLimiter(RATE_LIMIT, 86400)  # Daily limit
        self.base_url = STACK_BASE
        self.headers = {'User-Agent': 'Resume-Collector/1.0', 'Accept': 'application/json'}

    def get_source_name(self) -> str:
        return "StackExchange"

    async def scrape(self, *args, **kwargs) -> List[ResumeData]:
        """
        Main scraping method - delegate to scrape_resumes.
        """
        query = kwargs.get('query', '')
        limit = kwargs.get('limit', 100)
        return await self.scrape_resumes(query, limit)

    async def scrape_resumes(self, query: str, limit: int = 100) -> List[ResumeData]:
        resumes: List[ResumeData] = []
        page = 1
        page_size = min(100, limit)

        while len(resumes) < limit:
            search_url = f"{self.base_url}/users"
            params = {
                'site': 'stackoverflow',
                'sort': 'reputation',
                'order': 'desc',
                'page': page,
                'pagesize': page_size,
                'filter': 'default'
            }

            if STACK_KEY:
                params['key'] = STACK_KEY

            logger.info(f"Searching Stack Overflow users: page {page}")
            search_results = await self.make_request('GET', search_url, headers=self.headers, params=params)

            if not search_results or 'items' not in search_results:
                logger.warning("No more Stack Overflow users found")
                break

            users = search_results['items']
            if not users:
                break

            for user_data in users:
                if len(resumes) >= limit:
                    break
                if user_data.get('reputation', 0) < 100:
                    continue
                resume = await self._create_resume_from_user(user_data)
                if resume:
                    resumes.append(resume)

            page += 1
            if not search_results.get('has_more', False):
                break

        logger.info(f"Scraped {len(resumes)} Stack Overflow profiles")
        return resumes

    async def _create_resume_from_user(self, user_data: Dict[str, Any]) -> Optional[ResumeData]:
        user_id = user_data.get('user_id')
        if not user_id:
            return None

        tags_data = await self._get_user_tags(user_id)
        answers_data = await self._get_user_top_answers(user_id)

        display_name = user_data.get('display_name', f"Developer_{user_id}")
        location = user_data.get('location')
        about_me = user_data.get('about_me', '')

        title = self._determine_title_from_tags(tags_data)
        skills = self._extract_skills_from_tags(tags_data)
        content = self._create_resume_content(user_data, tags_data, answers_data, about_me)
        experience_years = self._estimate_experience_from_so(user_data)
        experience = self._create_experience_list(user_data, tags_data)

        source_url = user_data.get('link', None)
        # standardize to ResumeData dataclass
        return ResumeData(
            name=display_name,
            title=title,
            content=content,
            resume_text=content,
            total_experience_years=float(experience_years) if experience_years is not None else 0.0,
            experience=experience,
            skills=skills,
            location=location,
            source='StackExchange',
            anonymized=True,
            source_url=source_url
        )

    async def _get_user_tags(self, user_id: int) -> List[Dict]:
        url = f"{self.base_url}/users/{user_id}/top-tags"
        params = {'site': 'stackoverflow', 'pagesize': 20}
        if STACK_KEY:
            params['key'] = STACK_KEY
        result = await self.make_request('GET', url, headers=self.headers, params=params)
        return result.get('items', []) if result else []

    async def _get_user_top_answers(self, user_id: int) -> List[Dict]:
        url = f"{self.base_url}/users/{user_id}/answers"
        params = {'site': 'stackoverflow', 'sort': 'votes', 'order': 'desc', 'pagesize': 5, 'filter': 'withbody'}
        if STACK_KEY:
            params['key'] = STACK_KEY
        result = await self.make_request('GET', url, headers=self.headers, params=params)
        return result.get('items', []) if result else []

    def _determine_title_from_tags(self, tags: List[Dict]) -> str:
        if not tags:
            return "Software Developer"
        tag_names = [tag.get('tag_name', '').lower() for tag in tags[:5]]
        title_mappings = {
            'python': 'Python Developer',
            'javascript': 'JavaScript Developer',
            'java': 'Java Developer',
            'c#': 'C# Developer',
            'php': 'PHP Developer',
            'c++': 'C++ Developer',
            'react': 'React Developer',
            'angular': 'Angular Developer',
            'node.js': 'Node.js Developer',
            'machine-learning': 'Machine Learning Engineer',
            'data-science': 'Data Scientist',
            'android': 'Android Developer',
            'ios': 'iOS Developer',
            'devops': 'DevOps Engineer',
            'docker': 'DevOps Engineer',
            'kubernetes': 'DevOps Engineer',
            'aws': 'Cloud Engineer',
            'azure': 'Cloud Engineer',
            'sql': 'Database Developer',
            'mysql': 'Database Developer',
            'postgresql': 'Database Developer'
        }
        for tag in tag_names:
            if tag in title_mappings:
                return title_mappings[tag]
        for tag in tag_names:
            for key, title in title_mappings.items():
                if key in tag or tag in key:
                    return title
        if tag_names:
            return f"{tag_names[0].title()} Developer"
        return "Software Developer"

    def _extract_skills_from_tags(self, tags: List[Dict]) -> List[str]:
        skills = []
        for tag in tags:
            tag_name = tag.get('tag_name', '')
            if tag_name:
                skill = tag_name.replace('-', ' ').title()
                if skill not in skills:
                    skills.append(skill)
        return skills

    def _create_resume_content(self, user: Dict, tags: List[Dict], answers: List[Dict], about_me: str) -> str:
        content_parts = []
        if about_me:
            clean_about = re.sub(r'<[^>]+>', '', about_me)
            content_parts.append(f"PROFESSIONAL SUMMARY:\n{clean_about}\n")
        reputation = user.get('reputation', 0)
        badge_counts = user.get('badge_counts', {})
        gold_badges = badge_counts.get('gold', 0)
        silver_badges = badge_counts.get('silver', 0)
        bronze_badges = badge_counts.get('bronze', 0)
        stats = [f"Reputation: {reputation:,}", f"Badges: {gold_badges} gold, {silver_badges} silver, {bronze_badges} bronze"]
        content_parts.append(f"STACK OVERFLOW PROFILE:\n{' | '.join(stats)}\n")
        if tags:
            content_parts.append("TECHNICAL EXPERTISE:")
            for tag in tags[:10]:
                tag_name = tag.get('tag_name', '')
                post_score = tag.get('post_score', 0)
                answer_count = tag.get('answer_count', 0)
                content_parts.append(f"• {tag_name}: {answer_count} answers, {post_score} score")
            content_parts.append("")
        if answers:
            content_parts.append("NOTABLE CONTRIBUTIONS:")
            for answer in answers[:3]:
                score = answer.get('score', 0)
                is_accepted = answer.get('is_accepted', False)
                accepted_text = " (Accepted)" if is_accepted else ""
                content_parts.append(f"• High-quality answer with {score} upvotes{accepted_text}")
            content_parts.append("")
        if tags:
            all_skills = [tag.get('tag_name', '') for tag in tags]
            content_parts.append(f"CORE TECHNOLOGIES:\n{', '.join(all_skills[:15])}")
        return '\n'.join(content_parts)

    def _estimate_experience_from_so(self, user: Dict) -> Optional[int]:
        from datetime import datetime
        creation_date = user.get('creation_date')
        if not creation_date:
            return None
        try:
            account_created = datetime.fromtimestamp(creation_date)
            account_age_years = (datetime.now() - account_created).days / 365.25
            reputation = user.get('reputation', 0)
            experience = max(1, int(account_age_years))
            if reputation > 1000:
                experience += 1
            if reputation > 5000:
                experience += 2
            if reputation > 10000:
                experience += 3
            if reputation > 25000:
                experience += 5
            return min(experience, 20)
        except Exception:
            return None

    def _create_experience_list(self, user: Dict, tags: List[Dict]) -> List[ExperienceEntry]:
        experience = []
        reputation = user.get('reputation', 0)
        if reputation > 1000:
            experience.append(ExperienceEntry(
                title="Stack Overflow Contributor",
                company="Stack Overflow",
                description=f"Earned {reputation:,} reputation points demonstrating technical expertise",
                start_date=None,
                end_date=None,
                location=None
            ))
        badge_counts = user.get('badge_counts', {})
        gold_badges = badge_counts.get('gold', 0)
        if gold_badges > 0:
            experience.append(ExperienceEntry(
                title="Recognized Expert",
                company="Stack Overflow",
                description=f"Earned {gold_badges} gold badges for exceptional contributions to the developer community",
                start_date=None,
                end_date=None,
                location=None
            ))
        if tags:
            top_tags = [tag.get('tag_name', '') for tag in tags[:3]]
            experience.append(ExperienceEntry(
                title="Technical Expert",
                company="Self-employed/Open Source",
                description=f"Demonstrated expertise in: {', '.join(top_tags)}",
                start_date=None,
                end_date=None,
                location=None
            ))
        answer_count = user.get('answer_count', 0)
        if answer_count and answer_count > 10:
            experience.append(ExperienceEntry(
                title="Community Helper",
                company="Stack Overflow",
                description=f"Provided {answer_count} answers helping fellow developers solve technical challenges",
                start_date=None,
                end_date=None,
                location=None
            ))
        return experience
