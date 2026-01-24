"""
Pytest configuration for merlt-models tests.
"""
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory with test data."""
    test_dir = tmp_path_factory.mktemp("test_data")

    # Create test config files
    config_dir = test_dir / "config"
    config_dir.mkdir()

    test_config = config_dir / "test_config.yaml"
    test_config.write_text("""
version: "1.0"
retrieval:
  alpha: 0.7
  over_retrieve_factor: 3
expert_traversal:
  literal:
    defines: 0.8
    modifies: 0.6
    """)

    # Create test weights directory
    weights_dir = test_dir / "weights"
    weights_dir.mkdir()

    yield test_dir


@pytest.fixture
def temp_weights_dir(tmp_path):
    """Create a temporary directory for weight files."""
    weights = tmp_path / "weights"
    weights.mkdir()
    return weights


@pytest.fixture
def sample_weight_config():
    """Sample weight configuration dictionary."""
    return {
        "version": "2.0",
        "schema_version": "1.0",
        "retrieval": {
            "alpha": {
                "default": 0.7,
                "bounds": [0.3, 0.9],
                "learnable": True,
                "learning_rate": 0.01,
            },
            "over_retrieve_factor": 3,
            "max_graph_hops": 3,
            "default_graph_score": 0.5,
        },
        "rlcf": {
            "baseline_credentials": {
                "default": 0.4,
                "bounds": [0.1, 0.6],
                "learnable": True,
            },
            "track_record": {
                "default": 0.4,
                "bounds": [0.2, 0.7],
                "learnable": True,
            },
            "recent_performance": {
                "default": 0.2,
                "bounds": [0.1, 0.4],
                "learnable": True,
            },
        },
    }
