import importlib.util
from pathlib import Path
import sys


def load_config(monkeypatch, **env_overrides):
    """Reload config with temporary environment overrides."""
    for key in (
        "SCRAPING_THROUGHPUT_PROFILE",
        "HTTP_MAX_CONCURRENCY",
        "FETCH_QUEUE_WORKERS",
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in env_overrides.items():
        monkeypatch.setenv(key, str(value))

    module_name = "visualex_config_test_module"
    config_path = Path(__file__).resolve().parents[2] / "visualex" / "config.py"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    assert spec is not None
    assert spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)

    return config_module


def test_balanced_profile_is_default(monkeypatch):
    """Balanced profile should raise throughput with safe defaults."""
    config = load_config(monkeypatch)

    assert config.SCRAPING_THROUGHPUT_PROFILE == "balanced"
    assert config.HTTP_MAX_CONCURRENCY == 5
    assert config.FETCH_QUEUE_WORKERS == 4


def test_safe_profile_preserves_conservative_limits(monkeypatch):
    """Safe profile should keep previous low-risk throughput settings."""
    config = load_config(monkeypatch, SCRAPING_THROUGHPUT_PROFILE="safe")

    assert config.HTTP_MAX_CONCURRENCY == 3
    assert config.FETCH_QUEUE_WORKERS == 2


def test_aggressive_profile_is_clamped(monkeypatch):
    """Worker defaults should never exceed the effective HTTP concurrency cap."""
    config = load_config(monkeypatch, SCRAPING_THROUGHPUT_PROFILE="aggressive")

    assert config.HTTP_MAX_CONCURRENCY == 8
    assert config.FETCH_QUEUE_WORKERS == 6


def test_invalid_profile_falls_back_to_balanced(monkeypatch):
    """Unknown profiles should not silently pick an unsafe configuration."""
    config = load_config(monkeypatch, SCRAPING_THROUGHPUT_PROFILE="unknown")

    assert config.SCRAPING_THROUGHPUT_PROFILE == "balanced"
    assert config.HTTP_MAX_CONCURRENCY == 5
    assert config.FETCH_QUEUE_WORKERS == 4


def test_explicit_env_overrides_are_clamped(monkeypatch):
    """Manual overrides still work but remain within controlled limits."""
    config = load_config(
        monkeypatch,
        SCRAPING_THROUGHPUT_PROFILE="safe",
        HTTP_MAX_CONCURRENCY=12,
        FETCH_QUEUE_WORKERS=20,
    )

    assert config.HTTP_MAX_CONCURRENCY == 8
    assert config.FETCH_QUEUE_WORKERS == 8
