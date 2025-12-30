"""
Structured logging utilities for Resume-Collector

Provides JSON-formatted logging with different levels and performance metrics.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from config import get_config

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

class Logger:
    """Centralized logger configuration"""

    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._setup_logger(name)
        return cls._loggers[name]

    @classmethod
    def _setup_logger(cls, name: str) -> logging.Logger:
        """Set up a logger with appropriate handlers and formatters"""
        config = get_config()
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)

        # File handler with rotation if file path is specified
        if config.logging.file_path:
            file_path = Path(config.logging.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)

        return logger

def log_performance(logger: logging.Logger, operation: str, duration: float,
                   records_processed: Optional[int] = None, **kwargs):
    """Log performance metrics for operations"""
    extra_fields = {
        "operation": operation,
        "duration_seconds": duration,
        "records_processed": records_processed,
        **kwargs
    }

    logger.info(f"Performance: {operation} completed in {duration:.2f}s",
               extra={"extra_fields": extra_fields})

def log_scraping_activity(logger: logging.Logger, source: str, status: str,
                         records_collected: Optional[int] = None,
                         error_message: Optional[str] = None, **kwargs):
    """Log scraping activity"""
    extra_fields = {
        "source": source,
        "status": status,
        "records_collected": records_collected,
        **kwargs
    }

    if error_message:
        extra_fields["error_message"] = error_message
        logger.error(f"Scraping {source}: {status}", extra={"extra_fields": extra_fields})
    else:
        logger.info(f"Scraping {source}: {status}", extra={"extra_fields": extra_fields})

def setup_global_logging():
    """Set up global logging configuration"""
    # Prevent duplicate log messages
    logging.getLogger().setLevel(logging.WARNING)

    # Set up root logger to prevent warnings from propagating
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

# Convenience functions
def get_logger(name: str = "resume_collector") -> logging.Logger:
    """Get the default logger"""
    return Logger.get_logger(name)

if __name__ == "__main__":
    # Test logging
    setup_global_logging()
    logger = get_logger("test")

    logger.info("Test message")
    logger.warning("Test warning", extra={"extra_fields": {"test_field": "value"}})

    # Test performance logging
    log_performance(logger, "test_operation", 1.23, records_processed=100)

    # Test scraping logging
    log_scraping_activity(logger, "github", "completed", records_collected=50)
