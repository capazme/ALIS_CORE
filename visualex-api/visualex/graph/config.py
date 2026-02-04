"""
FalkorDB Configuration
======================

Configuration for FalkorDB graph database connection.

Supports environment variables for flexible deployment.

Environment Variables:
    FALKORDB_HOST: Server host (default: localhost)
    FALKORDB_PORT: Server port (default: 6379)
    FALKORDB_GRAPH_NAME: Graph name (default: visualex_dev)
    FALKORDB_PASSWORD: Password (default: empty)
    FALKORDB_MAX_CONNECTIONS: Max pool connections (default: 10)
    FALKORDB_TIMEOUT_MS: Operation timeout in ms (default: 5000)

Naming Convention:
    - visualex_dev: Development environment
    - visualex_test: Test environment
    - visualex_prod: Production environment
"""

import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["FalkorDBConfig"]


def _get_env_str(key: str, default: str) -> str:
    """Read environment variable as string."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Read environment variable as integer."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class FalkorDBConfig:
    """
    FalkorDB connection configuration.

    All fields support override via environment variables.

    Attributes:
        host: FalkorDB server host
        port: Server port (6379 for FalkorDB default)
        graph_name: Graph name (use _dev/_test/_prod for environments)
        max_connections: Maximum pool connections
        timeout_ms: Operation timeout in milliseconds
        password: Authentication password (optional)
    """

    host: str = field(
        default_factory=lambda: _get_env_str("FALKORDB_HOST", "localhost")
    )
    port: int = field(
        default_factory=lambda: _get_env_int("FALKORDB_PORT", 6379)
    )
    graph_name: str = field(
        default_factory=lambda: _get_env_str("FALKORDB_GRAPH_NAME", "visualex_dev")
    )
    # Reserved for future connection pool implementation
    max_connections: int = field(
        default_factory=lambda: _get_env_int("FALKORDB_MAX_CONNECTIONS", 10)
    )
    # Reserved for future timeout implementation
    timeout_ms: int = field(
        default_factory=lambda: _get_env_int("FALKORDB_TIMEOUT_MS", 5000)
    )
    password: Optional[str] = field(
        default_factory=lambda: _get_env_str("FALKORDB_PASSWORD", "") or None
    )

    def __repr__(self) -> str:
        return (
            f"FalkorDBConfig(host={self.host!r}, port={self.port}, "
            f"graph_name={self.graph_name!r})"
        )
