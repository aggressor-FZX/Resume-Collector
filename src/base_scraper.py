"""
Base scraper framework for the Resume-Collector project.
Provides abstract base class with async HTTP capabilities and common scraping utilities.
"""

import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    diskcache = None
    DISKCACHE_AVAILABLE = False

from src import config as cfg
from src.schemas.resume_data import ResumeData

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascade failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.state != CircuitBreakerState.OPEN:
            return False
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    def _record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

    async def call(self, func, *args, **kwargs):
        """
        Call a function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            raise e

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class RetryConfig:
    """
    Configuration for retry logic with exponential backoff.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


async def retry_with_backoff(
    func: Callable[[], Awaitable[Any]],
    config: RetryConfig
) -> Any:
    """
    Execute function with exponential backoff retry logic.

    Args:
        func: Async function to retry
        config: Retry configuration

    Returns:
        Function result

    Raises:
        Last exception from function
    """
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                # Last attempt failed
                raise e

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.backoff_factor ** attempt),
                config.max_delay
            )

            # Add jitter to prevent thundering herd
            if config.jitter:
                delay = delay * (0.5 + 0.5 * asyncio.get_event_loop().time() % 1)

            logger.debug(f"Retry attempt {attempt + 1}/{config.max_attempts} after {delay:.2f}s delay")
            await asyncio.sleep(delay)

    # This should never be reached, but just in case
    raise last_exception


class ErrorClassifier:
    """
    Classifies errors as retryable or non-retryable.
    """

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception to classify

        Returns:
            True if error is retryable
        """
        # Network-related errors are usually retryable
        if isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError, OSError)):
            return True

        # Rate limit errors are retryable
        if isinstance(error, RateLimitError):
            return True

        # HTTP 5xx errors are retryable
        if isinstance(error, HTTPError):
            # Extract status code from error message if possible
            error_str = str(error).lower()
            if '500' in error_str or '502' in error_str or '503' in error_str or '504' in error_str:
                return True

        # Default to non-retryable
        return False


class ScraperError(Exception):
    """Base exception for scraper errors"""
    pass


class HTTPError(ScraperError):
    """HTTP request related errors"""
    pass


class RateLimitError(ScraperError):
    """Rate limiting related errors"""
    pass


class ProgressTracker:
    """
    Tracks progress of long-running operations with completion percentages and ETA.
    """

    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.completed_items = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a progress callback function.

        Args:
            callback: Function that receives progress info dict
        """
        self.callbacks.append(callback)

    def update(self, completed: int = None, increment: int = 1) -> None:
        """
        Update progress.

        Args:
            completed: Absolute number of completed items (optional)
            increment: Number of items to increment (default: 1)
        """
        if completed is not None:
            self.completed_items = completed
        else:
            self.completed_items += increment

        self.completed_items = min(self.completed_items, self.total_items)
        self.last_update_time = time.time()

        # Notify callbacks
        progress_info = self.get_progress_info()
        for callback in self.callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information.

        Returns:
            Dictionary with progress details
        """
        elapsed = time.time() - self.start_time
        progress = self.completed_items / self.total_items if self.total_items > 0 else 0

        # Calculate ETA
        eta_seconds = None
        if self.completed_items > 0 and progress > 0:
            avg_time_per_item = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta_seconds = avg_time_per_item * remaining_items

        return {
            'description': self.description,
            'completed': self.completed_items,
            'total': self.total_items,
            'progress': progress,
            'percentage': progress * 100,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta_seconds,
            'rate_per_second': self.completed_items / elapsed if elapsed > 0 else 0
        }

    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.completed_items >= self.total_items


class StatePersistence:
    """
    Handles state persistence for resumable operations using checkpoint files.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize state persistence.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, operation_id: str, state: Dict[str, Any]) -> None:
        """
        Save operation state to checkpoint file.

        Args:
            operation_id: Unique identifier for the operation
            state: State dictionary to save
        """
        checkpoint_file = self.checkpoint_dir / f"{operation_id}.json"

        checkpoint_data = {
            'operation_id': operation_id,
            'timestamp': time.time(),
            'state': state
        }

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.debug(f"Saved checkpoint for operation {operation_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {operation_id}: {e}")

    def load_checkpoint(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load operation state from checkpoint file.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            State dictionary or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{operation_id}.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            logger.debug(f"Loaded checkpoint for operation {operation_id}")
            return checkpoint_data.get('state', {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {operation_id}: {e}")
            return None

    def delete_checkpoint(self, operation_id: str) -> None:
        """
        Delete checkpoint file.

        Args:
            operation_id: Unique identifier for the operation
        """
        checkpoint_file = self.checkpoint_dir / f"{operation_id}.json"

        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            logger.debug(f"Deleted checkpoint for operation {operation_id}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint for {operation_id}: {e}")

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoint operation IDs.

        Returns:
            List of operation IDs
        """
        try:
            return [f.stem for f in self.checkpoint_dir.glob("*.json")]
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []


class BatchProcessor:
    """
    Handles batch processing with configurable chunk sizes and progress tracking.
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 3,
        progress_tracker: Optional[ProgressTracker] = None,
        state_persistence: Optional[StatePersistence] = None
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batches
            progress_tracker: Optional progress tracker
            state_persistence: Optional state persistence
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.progress_tracker = progress_tracker
        self.state_persistence = state_persistence
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_batches(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Awaitable[List[Any]]],
        operation_id: Optional[str] = None
    ) -> List[Any]:
        """
        Process items in batches.

        Args:
            items: List of items to process
            process_func: Async function to process a batch
            operation_id: Optional operation ID for resumable processing

        Returns:
            List of processed results
        """
        if not items:
            return []

        # Load checkpoint if available
        start_index = 0
        results = []

        if operation_id and self.state_persistence:
            checkpoint_state = self.state_persistence.load_checkpoint(operation_id)
            if checkpoint_state:
                start_index = checkpoint_state.get('processed_count', 0)
                results = checkpoint_state.get('results', [])
                logger.info(f"Resuming from checkpoint: {start_index}/{len(items)} items processed")

        # Create batches
        batches = []
        for i in range(start_index, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append((i, batch))

        # Process batches concurrently
        tasks = []
        for batch_index, batch in batches:
            task = asyncio.create_task(
                self._process_batch_with_semaphore(batch_index, batch, process_func, operation_id)
            )
            tasks.append(task)

        # Collect results
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            results.extend(batch_result)

        # Clean up checkpoint on completion
        if operation_id and self.state_persistence:
            self.state_persistence.delete_checkpoint(operation_id)

        return results

    async def _process_batch_with_semaphore(
        self,
        batch_index: int,
        batch: List[Any],
        process_func: Callable[[List[Any]], Awaitable[List[Any]]],
        operation_id: Optional[str] = None
    ) -> List[Any]:
        """
        Process a single batch with semaphore control.
        """
        async with self.semaphore:
            try:
                batch_results = await process_func(batch)

                # Update progress
                if self.progress_tracker:
                    self.progress_tracker.update(increment=len(batch))

                # Save checkpoint
                if operation_id and self.state_persistence:
                    checkpoint_state = {
                        'processed_count': batch_index + len(batch),
                        'results': [],  # Results are accumulated separately
                        'last_batch_index': batch_index
                    }
                    self.state_persistence.save_checkpoint(operation_id, checkpoint_state)

                return batch_results
            except Exception as e:
                logger.error(f"Batch {batch_index} failed: {e}")
                raise


class Cache:
    """
    Disk-based cache with TTL support for HTTP responses and scraper data.
    """

    def __init__(self, cache_dir: str = "./cache", default_ttl: int = 3600):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds
        """
        if DISKCACHE_AVAILABLE:
            self.cache = diskcache.Cache(cache_dir, size_limit=10**9)  # 1GB limit
            self.default_ttl = default_ttl
        else:
            self.cache = None
            self.default_ttl = default_ttl
            logger.warning("diskcache not available, caching disabled")

    def _make_key(self, method: str, url: str, params: Optional[Dict] = None,
                  data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> str:
        """
        Create a cache key from request parameters.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Form data
            json_data: JSON data

        Returns:
            Cache key string
        """
        key_parts = [method.upper(), url]
        if params:
            key_parts.append(str(sorted(params.items())))
        if data:
            key_parts.append(str(sorted(data.items())))
        if json_data:
            key_parts.append(str(sorted(json_data.items())))
        return "|".join(key_parts)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.cache:
            return None

        try:
            return self.cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        if not self.cache:
            return

        try:
            expire = time.time() + (ttl or self.default_ttl)
            self.cache.set(key, value, expire=expire)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def clear(self) -> None:
        """Clear all cache entries."""
        if self.cache:
            try:
                self.cache.clear()
            except Exception as e:
                logger.warning(f"Cache clear failed: {e}")


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    """

    def __init__(self, capacity: int, period_seconds: int = 86400, burst_limit: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            capacity: Maximum tokens per period
            period_seconds: Refill period in seconds
            burst_limit: Maximum burst capacity
        """
        self.capacity = capacity
        self.period_seconds = period_seconds
        self.burst_limit = burst_limit or capacity
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = int(elapsed / self.period_seconds * self.capacity)

        if refill_amount > 0:
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self.lock:
            await self._refill_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    @asynccontextmanager
    async def limit(self, tokens: int = 1):
        """
        Context manager for rate limiting.

        Args:
            tokens: Number of tokens required

        Raises:
            RateLimitError: If rate limit exceeded
        """
        if not await self.acquire(tokens):
            raise RateLimitError(f"Rate limit exceeded: {tokens} tokens required, {self.tokens} available")
        yield


class MultiDomainRateLimiter:
    """
    Rate limiter that supports different limits for different domains.
    """

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.default_limiter: Optional[RateLimiter] = None

    def add_domain(self, domain: str, capacity: int, period_seconds: int = 86400, burst_limit: Optional[int] = None) -> None:
        """
        Add rate limiter for a specific domain.

        Args:
            domain: Domain name (e.g., 'api.github.com')
            capacity: Token capacity
            period_seconds: Refill period
            burst_limit: Burst limit
        """
        self.limiters[domain] = RateLimiter(capacity, period_seconds, burst_limit)

    def set_default(self, capacity: int, period_seconds: int = 86400, burst_limit: Optional[int] = None) -> None:
        """
        Set default rate limiter for domains without specific limits.

        Args:
            capacity: Token capacity
            period_seconds: Refill period
            burst_limit: Burst limit
        """
        self.default_limiter = RateLimiter(capacity, period_seconds, burst_limit)

    def get_limiter(self, domain: str) -> RateLimiter:
        """
        Get rate limiter for a domain.

        Args:
            domain: Domain name

        Returns:
            RateLimiter instance
        """
        return self.limiters.get(domain, self.default_limiter)

    @asynccontextmanager
    async def limit(self, domain: str, tokens: int = 1):
        """
        Context manager for domain-specific rate limiting.

        Args:
            domain: Domain name
            tokens: Number of tokens to consume
        """
        limiter = self.get_limiter(domain)
        if limiter:
            async with limiter.limit(tokens):
                yield
        else:
            # No limiter configured, proceed without limiting
            yield


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers providing common functionality.
    """

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers: Dict[str, str] = {}
        self._is_initialized = False

        # Rate limiting
        self.rate_limiter: Optional[RateLimiter] = None
        self.domain_limiters = MultiDomainRateLimiter()

        # Caching
        self.cache_enabled = True
        self.cache: Optional[Cache] = None

        # Circuit breaker
        self.enable_circuit_breaker = False
        self.circuit_breaker: Optional[CircuitBreaker] = None

        # Retry configuration
        self.retry_config = RetryConfig()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def scrape(self, *args, **kwargs) -> List[ResumeData]:
        """
        Main scraping method to be implemented by subclasses.

        Returns:
            List of scraped ResumeData objects
        """
        pass

    async def initialize(self) -> None:
        """Initialize the scraper with session and configuration."""
        if self._is_initialized:
            return

        # Create HTTP session
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
        else:
            logger.warning("aiohttp not available, HTTP requests will fail")

        # Initialize cache
        if self.cache_enabled and DISKCACHE_AVAILABLE:
            self.cache = Cache()

        self._is_initialized = True
        logger.info(f"{self.name}: Initialized scraper")

    async def close(self) -> None:
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.cache:
            await self.cache.clear()

        self._is_initialized = False
        logger.info(f"{self.name}: Closed scraper")

    def configure_rate_limiting(self, capacity: int, period_seconds: int = 86400,
                               burst_limit: Optional[int] = None) -> None:
        """
        Configure global rate limiting.

        Args:
            capacity: Maximum requests per period
            period_seconds: Period in seconds
            burst_limit: Burst capacity
        """
        self.rate_limiter = RateLimiter(capacity, period_seconds, burst_limit)
        logger.info(f"{self.name}: Configured rate limiter with capacity {capacity}")

    def configure_domain_rate_limiting(self, domain: str, capacity: int,
                                      period_seconds: int = 86400,
                                      burst_limit: Optional[int] = None) -> None:
        """
        Configure rate limiting for a specific domain.

        Args:
            domain: Domain name
            capacity: Maximum requests per period
            period_seconds: Period in seconds
            burst_limit: Burst capacity
        """
        self.domain_limiters.add_domain(domain, capacity, period_seconds, burst_limit)
        logger.info(f"{self.name}: Configured domain rate limiter for {domain}")

    def enable_caching(self, cache_dir: str = "./cache", default_ttl: int = 3600) -> None:
        """
        Enable response caching.

        Args:
            cache_dir: Cache directory
            default_ttl: Default TTL in seconds
        """
        if DISKCACHE_AVAILABLE:
            self.cache = Cache(cache_dir, default_ttl)
            self.cache_enabled = True
            logger.info(f"{self.name}: Enabled caching with TTL {default_ttl}s")
        else:
            logger.warning(f"{self.name}: Cannot enable caching - diskcache not available")

    def disable_caching(self) -> None:
        """Disable response caching."""
        self.cache_enabled = False
        if self.cache:
            asyncio.create_task(self.cache.clear())
        self.cache = None
        logger.info(f"{self.name}: Disabled caching")

    def configure_retry(self, max_attempts: int = 3, base_delay: float = 1.0,
                       max_delay: float = 60.0, backoff_factor: float = 2.0,
                       jitter: bool = True) -> None:
        """
        Configure retry behavior.

        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay
            backoff_factor: Exponential backoff factor
            jitter: Whether to add jitter to delays
        """
        self.retry_config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter
        )
        logger.info(f"{self.name}: Configured retry with {max_attempts} max attempts")

    def enable_circuit_breaker_protection(self, failure_threshold: int = 5,
                                         recovery_timeout: float = 60.0) -> None:
        """
        Enable circuit breaker protection.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
        """
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=ScraperError
        )
        self.enable_circuit_breaker = True
        logger.info(f"{self.name}: Enabled circuit breaker with threshold={failure_threshold}")

    def disable_circuit_breaker(self) -> None:
        """Disable circuit breaker protection"""
        self.enable_circuit_breaker = False
        logger.info(f"{self.name}: Disabled circuit breaker")

    def get_circuit_breaker_state(self) -> Optional[CircuitBreakerState]:
        """
        Get current circuit breaker state.

        Returns:
            CircuitBreakerState or None if disabled
        """
        return self.circuit_breaker.get_state() if self.circuit_breaker else None

    async def make_request(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request using the scraper's session with rate limiting.

        Args:
            url: Request URL
            method: HTTP method
            headers: Additional headers
            params: Query parameters
            data: Form data
            json_data: JSON data
            **kwargs: Additional aiohttp arguments

        Returns:
            JSON response data or None on failure
        """
        if not self._is_initialized:
            await self.initialize()

        if not self.session:
            logger.error(f"{self.name}: No session available for request")
            return None

        # Apply rate limiting
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        tokens = 1  # Default token cost
        async with self.domain_limiters.limit(domain, tokens):
            if self.rate_limiter:
                async with self.rate_limiter.limit(tokens):
                    return await self._make_cached_request(
                        url, method, headers, params, data, json_data, **kwargs
                    )
            else:
                return await self._make_cached_request(
                    url, method, headers, params, data, json_data, **kwargs
                )

    async def _make_cached_request(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]],
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make a cached HTTP request with retry and circuit breaker logic.

        Args:
            url: Request URL
            method: HTTP method
            headers: Additional headers
            params: Query parameters
            data: Form data
            json_data: JSON data
            cache_ttl: Cache TTL override
            **kwargs: Additional arguments

        Returns:
            Response data
        """
        # Check if circuit breaker is open
        if self.enable_circuit_breaker and self.circuit_breaker:
            try:
                return await self.circuit_breaker.call(
                    self._make_request_with_retry_and_cache,
                    url, method, headers, params, data, json_data, cache_ttl, **kwargs
                )
            except CircuitBreakerOpen:
                logger.warning(f"{self.name}: Circuit breaker open, skipping request to {url}")
                return None
        else:
            return await self._make_request_with_retry_and_cache(
                url, method, headers, params, data, json_data, cache_ttl, **kwargs
            )

    async def _make_request_with_retry_and_cache(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]],
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make request with caching and retry logic.
        """
        # Only cache GET requests
        if self.cache_enabled and self.cache and method.upper() == 'GET':
            cache_key = self.cache._make_key(method, url, params, data, json_data)

            # Try to get from cache first
            cached_response = await self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"{self.name}: Cache hit for {url}")
                return cached_response

            # Not in cache, make request and cache it
            logger.debug(f"{self.name}: Cache miss for {url}")

        # Make request with retry logic
        response = await self._execute_request_with_retry(
            url, method, headers, params, data, json_data, **kwargs
        )

        # Cache successful responses
        if (self.cache_enabled and self.cache and method.upper() == 'GET' and
            response is not None and 'cache_key' in locals()):
            ttl = cache_ttl or (self.cache.default_ttl if self.cache else 3600)
            await self.cache.set(cache_key, response, ttl)

        return response

    async def _execute_request_with_retry(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Execute request with retry logic.
        """
        async def _single_attempt():
            return await self._execute_request(url, method, headers, params, data, json_data, **kwargs)

        try:
            return await retry_with_backoff(_single_attempt, self.retry_config)
        except Exception as e:
            if ErrorClassifier.is_retryable(e):
                logger.error(f"{self.name}: Request to {url} failed after retries: {e}")
            else:
                logger.error(f"{self.name}: Request to {url} failed (non-retryable): {e}")
            raise e

    async def _execute_request(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the actual HTTP request.
        """
        # Merge headers
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)

        try:
            logger.debug(f"{self.name}: Making {method} request to {url}")

            async with self.session.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                data=data,
                json=json_data,
                **kwargs
            ) as response:
                if response.status == 429:
                    raise RateLimitError(f"Rate limited: {url}")
                elif response.status >= 400:
                    raise HTTPError(f"HTTP {response.status}: {url}")

                return await response.json()

        except Exception as e:
            logger.debug(f"{self.name}: Request failed: {e}")
            raise e

    def create_progress_tracker(self, total_items: int, description: str = None) -> ProgressTracker:
        """
        Create a progress tracker for long-running operations.

        Args:
            total_items: Total number of items to process
            description: Description of the operation

        Returns:
            ProgressTracker instance
        """
        if description is None:
            description = f"{self.name} scraping"

        tracker = ProgressTracker(total_items, description)

        # Add default logging callback
        def log_progress(info: Dict[str, Any]):
            percentage = info['percentage']
            completed = info['completed']
            total = info['total']
            eta = info.get('eta_seconds')
            eta_str = f" ETA: {eta:.0f}s" if eta else ""
            logger.info(f"{self.name}: {percentage:.1f}% complete ({completed}/{total}){eta_str}")

        tracker.add_callback(log_progress)
        return tracker

    def create_batch_processor(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 3,
        progress_tracker: Optional[ProgressTracker] = None
    ) -> BatchProcessor:
        """
        Create a batch processor for efficient data processing.

        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batches
            progress_tracker: Optional progress tracker

        Returns:
            BatchProcessor instance
        """
        state_persistence = StatePersistence() if self.cache_enabled else None
        return BatchProcessor(
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent_batches,
            progress_tracker=progress_tracker,
            state_persistence=state_persistence
        )

    async def process_with_progress_and_resume(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Awaitable[List[Any]]],
        operation_id: str,
        batch_size: int = 100,
        description: str = None
    ) -> List[Any]:
        """
        Process items with progress tracking and resumable operations.

        Args:
            items: List of items to process
            process_func: Async function to process batches
            operation_id: Unique operation identifier for checkpoints
            batch_size: Number of items per batch
            description: Progress description

        Returns:
            List of processed results
        """
        if not items:
            return []

        # Create progress tracker
        tracker = self.create_progress_tracker(len(items), description)

        # Create batch processor
        processor = self.create_batch_processor(
            batch_size=batch_size,
            progress_tracker=tracker
        )

        logger.info(f"{self.name}: Starting batch processing of {len(items)} items (batch_size={batch_size})")

        try:
            results = await processor.process_batches(items, process_func, operation_id)
            logger.info(f"{self.name}: Completed processing {len(results)} items")
            return results
        except Exception as e:
            logger.error(f"{self.name}: Batch processing failed: {e}")
            raise

    def validate_resume_data(self, data: ResumeData) -> bool:
        """
        Validate ResumeData object.

        Args:
            data: ResumeData to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check required fields
            if not data.name or not data.title or not data.content:
                return False

            # Check quality score
            quality = data.quality_score()
            return quality.overall_score >= 30  # Minimum quality threshold

        except Exception as e:
            logger.warning(f"{self.name}: Validation error: {e}")
            return False

    def log_scraping_progress(self, current: int, total: int, source: str) -> None:
        """
        Log scraping progress.

        Args:
            current: Current number of items processed
            total: Total number expected
            source: Source identifier
        """
        if total > 0:
            percentage = (current / total) * 100
            logger.info(f"{self.name}: Processed {current}/{total} items from {source} ({percentage:.1f}%)")
        else:
            logger.info(f"{self.name}: Processed {current} items from {source}")
