"""
Configuration management for Resume-Collector

Uses Pydantic for validation and environment variable loading.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Configuration management for Resume-Collector

Uses Pydantic for validation and environment variable loading.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIKeys(BaseModel):
    """API keys for various data sources"""
    github_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_API_KEY"))
    stack_exchange_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("STACK_EXCHANGE_API_KEY"))
    semantic_scholar_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    hugging_face_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("HUGGING_FACE_API_KEY"))
    perplexity_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    deepseek_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))

class RateLimits(BaseModel):
    """Rate limiting configuration"""
    github_rate_limit: int = Field(default_factory=lambda: int(os.getenv("GITHUB_RATE_LIMIT", "5000")))
    stack_exchange_rate_limit: int = Field(default_factory=lambda: int(os.getenv("STACK_EXCHANGE_RATE_LIMIT", "1000")))
    semantic_scholar_rate_limit: int = Field(default_factory=lambda: int(os.getenv("SEMANTIC_SCHOLAR_RATE_LIMIT", "1000")))
    openalex_rate_limit: int = Field(default_factory=lambda: int(os.getenv("OPENALEX_RATE_LIMIT", "10")))

class HTTPConfig(BaseModel):
    """HTTP client configuration"""
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    backoff_factor: float = Field(default=0.3, description="Backoff factor for retries")
    user_agent: str = Field(default="Resume-Collector/1.0", description="User agent string")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, description="Log file path")

class ProjectConfig(BaseModel):
    """Project-wide configuration"""
    name: str = Field(default_factory=lambda: os.getenv("PROJECT_NAME", "scrap_tool"))
    data_output_dir: str = Field(default_factory=lambda: os.getenv("DATA_OUTPUT_DIR", "./data"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

class Config(BaseModel):
    """Main configuration class"""
    api_keys: APIKeys = Field(default_factory=APIKeys)
    rate_limits: RateLimits = Field(default_factory=RateLimits)
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls()

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from a JSON/YAML file (future enhancement)"""
        # For now, just load from env
        return cls.from_env()

# Global config instance
config = Config.from_env()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def get_headers(provider: str) -> dict:
    """Legacy function for backward compatibility"""
    return {
        'User-Agent': config.http.user_agent,
        'Accept': 'application/json'
    }

# Legacy constants for backward compatibility
STACK_EXCHANGE_API_BASE = 'https://api.stackexchange.com/2.3'
STACK_EXCHANGE_API_KEY = config.api_keys.stack_exchange_api_key
STACK_EXCHANGE_RATE_LIMIT = config.rate_limits.stack_exchange_rate_limit

if __name__ == "__main__":
    # Test configuration loading
    cfg = get_config()
    print("Configuration loaded successfully")
    print(f"Project name: {cfg.project.name}")
    print(f"GitHub API key configured: {cfg.api_keys.github_api_key is not None}")
    print(f"Log level: {cfg.logging.level}")
