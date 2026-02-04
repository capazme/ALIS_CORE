"""
Tests for Graph Config Module
=============================

Unit tests for FalkorDB configuration.
"""

import os
import pytest
from visualex.graph.config import FalkorDBConfig


class TestFalkorDBConfigDefaults:
    """Tests for default configuration values."""

    def test_default_host(self):
        """Default host is localhost."""
        config = FalkorDBConfig()
        assert config.host == "localhost"

    def test_default_port(self):
        """Default port is 6379."""
        config = FalkorDBConfig()
        assert config.port == 6379

    def test_default_graph_name(self):
        """Default graph name is visualex_dev."""
        config = FalkorDBConfig()
        assert config.graph_name == "visualex_dev"

    def test_default_max_connections(self):
        """Default max connections is 10."""
        config = FalkorDBConfig()
        assert config.max_connections == 10

    def test_default_timeout_ms(self):
        """Default timeout is 5000ms."""
        config = FalkorDBConfig()
        assert config.timeout_ms == 5000

    def test_default_password_is_none(self):
        """Default password is None."""
        config = FalkorDBConfig()
        assert config.password is None


class TestFalkorDBConfigEnvVars:
    """Tests for environment variable configuration."""

    def test_host_from_env(self, monkeypatch):
        """Host can be set via FALKORDB_HOST."""
        monkeypatch.setenv("FALKORDB_HOST", "graph.example.com")
        config = FalkorDBConfig()
        assert config.host == "graph.example.com"

    def test_port_from_env(self, monkeypatch):
        """Port can be set via FALKORDB_PORT."""
        monkeypatch.setenv("FALKORDB_PORT", "6380")
        config = FalkorDBConfig()
        assert config.port == 6380

    def test_graph_name_from_env(self, monkeypatch):
        """Graph name can be set via FALKORDB_GRAPH_NAME."""
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "visualex_prod")
        config = FalkorDBConfig()
        assert config.graph_name == "visualex_prod"

    def test_password_from_env(self, monkeypatch):
        """Password can be set via FALKORDB_PASSWORD."""
        monkeypatch.setenv("FALKORDB_PASSWORD", "secret123")
        config = FalkorDBConfig()
        assert config.password == "secret123"

    def test_max_connections_from_env(self, monkeypatch):
        """Max connections can be set via FALKORDB_MAX_CONNECTIONS."""
        monkeypatch.setenv("FALKORDB_MAX_CONNECTIONS", "50")
        config = FalkorDBConfig()
        assert config.max_connections == 50

    def test_timeout_from_env(self, monkeypatch):
        """Timeout can be set via FALKORDB_TIMEOUT_MS."""
        monkeypatch.setenv("FALKORDB_TIMEOUT_MS", "10000")
        config = FalkorDBConfig()
        assert config.timeout_ms == 10000

    def test_empty_password_becomes_none(self, monkeypatch):
        """Empty password string becomes None."""
        monkeypatch.setenv("FALKORDB_PASSWORD", "")
        config = FalkorDBConfig()
        assert config.password is None


class TestFalkorDBConfigOverride:
    """Tests for explicit configuration override."""

    def test_explicit_host_overrides_env(self, monkeypatch):
        """Explicit host in constructor takes precedence."""
        monkeypatch.setenv("FALKORDB_HOST", "from-env.com")
        config = FalkorDBConfig(host="explicit.com")
        assert config.host == "explicit.com"

    def test_explicit_port_overrides_env(self, monkeypatch):
        """Explicit port in constructor takes precedence."""
        monkeypatch.setenv("FALKORDB_PORT", "1234")
        config = FalkorDBConfig(port=5678)
        assert config.port == 5678

    def test_explicit_graph_name_overrides_env(self, monkeypatch):
        """Explicit graph name takes precedence."""
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "from_env")
        config = FalkorDBConfig(graph_name="explicit_graph")
        assert config.graph_name == "explicit_graph"


class TestFalkorDBConfigInvalidEnv:
    """Tests for invalid environment variable handling."""

    def test_invalid_port_uses_default(self, monkeypatch):
        """Invalid port falls back to default."""
        monkeypatch.setenv("FALKORDB_PORT", "not_a_number")
        config = FalkorDBConfig()
        assert config.port == 6379

    def test_invalid_max_connections_uses_default(self, monkeypatch):
        """Invalid max_connections falls back to default."""
        monkeypatch.setenv("FALKORDB_MAX_CONNECTIONS", "abc")
        config = FalkorDBConfig()
        assert config.max_connections == 10


class TestFalkorDBConfigRepr:
    """Tests for configuration repr."""

    def test_repr_shows_key_info(self):
        """Repr shows host, port, and graph name."""
        config = FalkorDBConfig(
            host="myhost",
            port=6380,
            graph_name="mygraph",
        )
        repr_str = repr(config)
        assert "myhost" in repr_str
        assert "6380" in repr_str
        assert "mygraph" in repr_str

    def test_repr_does_not_show_password(self):
        """Repr does not expose password."""
        config = FalkorDBConfig(password="secret")
        repr_str = repr(config)
        assert "secret" not in repr_str


class TestFalkorDBConfigEnvironments:
    """Tests for environment naming convention."""

    def test_dev_environment(self, monkeypatch):
        """Development environment uses _dev suffix."""
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "visualex_dev")
        config = FalkorDBConfig()
        assert config.graph_name.endswith("_dev")

    def test_prod_environment(self, monkeypatch):
        """Production environment uses _prod suffix."""
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "visualex_prod")
        config = FalkorDBConfig()
        assert config.graph_name.endswith("_prod")

    def test_test_environment(self, monkeypatch):
        """Test environment uses _test suffix."""
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "visualex_test")
        config = FalkorDBConfig()
        assert config.graph_name.endswith("_test")
