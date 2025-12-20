import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, capacity: int, period_seconds: int = 86400):
        self.capacity = capacity
        self.period_seconds = period_seconds


class BaseScraper:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if not self.session:
            try:
                import aiohttp
                self.session = aiohttp.ClientSession()
            except Exception:
                # aiohttp not available in this environment; session remains None
                logger.warning('aiohttp not available; network requests will fail if attempted')
                self.session = None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def make_request(self, url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self._ensure_session()
        if not self.session:
            logger.warning('No HTTP session available to make requests')
            return None
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=30) as resp:
                if resp.status != 200:
                    logger.warning(f"Request to {url} returned status {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger.exception(f"HTTP request failed: {e}")
            return None
