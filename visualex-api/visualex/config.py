import os
from pathlib import Path
from typing import Any, List, Optional

# Use APP_BASE_PATH env var in Docker, otherwise calculate from file location
BASE_PATH = Path(os.getenv("APP_BASE_PATH", Path(__file__).resolve().parents[2]))

SCRAPING_THROUGHPUT_PROFILES = {
    "safe": {
        "http_max_concurrency": 3,
        "fetch_queue_workers": 2,
    },
    "balanced": {
        "http_max_concurrency": 5,
        "fetch_queue_workers": 4,
    },
    "aggressive": {
        "http_max_concurrency": 8,
        "fetch_queue_workers": 6,
    },
}
DEFAULT_SCRAPING_THROUGHPUT_PROFILE = "balanced"
MAX_HTTP_CONCURRENCY = 8


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_scraping_profile() -> str:
    profile = os.getenv(
        "SCRAPING_THROUGHPUT_PROFILE",
        DEFAULT_SCRAPING_THROUGHPUT_PROFILE,
    ).strip().lower()
    if profile not in SCRAPING_THROUGHPUT_PROFILES:
        return DEFAULT_SCRAPING_THROUGHPUT_PROFILE
    return profile

MAX_CACHE_SIZE = 10000
HISTORY_LIMIT = 50
HISTORY_FILE = BASE_PATH / "data" / "history.json"
DOSSIER_FILE = BASE_PATH / "data" / "dossiers.json"
DOSSIER_LIMIT = 100  # Max number of dossiers
RATE_LIMIT = 1000  # Limit to 100 requests per minute
RATE_LIMIT_WINDOW = 600  # Window size in seconds

PERSISTENT_CACHE_DIR = BASE_PATH / "download" / "cache"
PERSISTENT_CACHE_TTL = _get_env_int("PERSISTENT_CACHE_TTL", 86400)
QUERY_STATS_MAX_BODY_BYTES = _get_env_int("QUERY_STATS_MAX_BODY_BYTES", 16384)

SCRAPING_THROUGHPUT_PROFILE = _get_scraping_profile()
_scraping_profile_defaults = SCRAPING_THROUGHPUT_PROFILES[SCRAPING_THROUGHPUT_PROFILE]

HTTP_MAX_CONCURRENCY = _clamp(
    _get_env_int(
        "HTTP_MAX_CONCURRENCY",
        _scraping_profile_defaults["http_max_concurrency"],
    ),
    1,
    MAX_HTTP_CONCURRENCY,
)
HTTP_MIN_INTERVAL = _get_env_float("HTTP_MIN_INTERVAL", 0.5)
HTTP_MAX_RETRIES = _get_env_int("HTTP_MAX_RETRIES", 4)
HTTP_BACKOFF_FACTOR = _get_env_float("HTTP_BACKOFF_FACTOR", 2.0)
HTTP_INITIAL_BACKOFF = _get_env_float("HTTP_INITIAL_BACKOFF", 0.5)
HTTP_JITTER = _get_env_float("HTTP_JITTER", 0.3)
HTTP_TIMEOUT = _get_env_int("HTTP_TIMEOUT", 30)

FETCH_QUEUE_WORKERS = _clamp(
    _get_env_int(
        "FETCH_QUEUE_WORKERS",
        _scraping_profile_defaults["fetch_queue_workers"],
    ),
    1,
    HTTP_MAX_CONCURRENCY,
)
FETCH_QUEUE_DELAY = _get_env_float("FETCH_QUEUE_DELAY", 0.3)


class Settings:
    """
    Configuration settings manager.
    
    Provides access to configuration values via environment variables
    with sensible defaults.
    """
    
    def __init__(self) -> None:
        self._cache: dict = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key (will be uppercased for env var lookup)
            default: Default value if not found
            
        Returns:
            The configuration value
        """
        if key in self._cache:
            return self._cache[key]
        
        env_key = key.upper().replace(".", "_")
        value = os.getenv(env_key, default)
        self._cache[key] = value
        return value
    
    def get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """
        Get a configuration value as a list.
        
        Args:
            key: The configuration key
            default: Default list if not found
            
        Returns:
            List of configuration values
        """
        if default is None:
            default = []
        
        value = self.get(key)
        if value is None:
            return default
        
        if isinstance(value, list):
            return value
        
        # Parse comma-separated string
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        
        return default
