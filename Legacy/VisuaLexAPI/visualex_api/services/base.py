"""
Compatibility module for visualex.scrapers.base
Re-exports BaseScraper and related classes from their actual locations
"""
from ..tools.sys_op import BaseScraper
from ..tools.exceptions import (
    NetworkError,
    DocumentNotFoundError,
    ParsingError,
)

# ScraperConfig - create a simple dataclass if needed
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScraperConfig:
    """Configuration for scrapers."""
    timeout: Optional[int] = 30
    retries: int = 3
    user_agent: Optional[str] = None

# ScraperError - base exception for all scraper errors
class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass

# Make NetworkError, DocumentNotFoundError, ParsingError inherit from ScraperError
# They already exist in exceptions.py, so we just re-export them

__all__ = [
    "BaseScraper",
    "ScraperConfig",
    "ScraperError",
    "NetworkError",
    "DocumentNotFoundError",
    "ParsingError",
]
