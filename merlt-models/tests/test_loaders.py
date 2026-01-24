"""
Tests for model and config loading functions.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_dict(self, test_data_dir, monkeypatch):
        """Test that load_config returns a dictionary."""
        from merlt_models import load_config, CONFIG_DIR

        # Patch CONFIG_DIR to use test data
        monkeypatch.setattr("merlt_models.CONFIG_DIR", test_data_dir / "config")

        config = load_config("test_config")

        assert isinstance(config, dict)
        assert "version" in config

    def test_load_config_file_not_found(self, monkeypatch, tmp_path):
        """Test that load_config raises error for missing config."""
        from merlt_models import load_config

        monkeypatch.setattr("merlt_models.CONFIG_DIR", tmp_path)

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")


class TestLoadModel:
    """Tests for load_model function."""

    @pytest.mark.skipif(
        True,  # Skip by default as it requires PyTorch
        reason="Requires PyTorch and model files"
    )
    def test_load_model_returns_state_dict(self, temp_weights_dir, monkeypatch):
        """Test that load_model returns model weights."""
        import torch
        from merlt_models import load_model

        # Create a dummy model file
        dummy_weights = {"layer1.weight": torch.zeros(10, 10)}
        model_path = temp_weights_dir / "test_model.pt"
        torch.save(dummy_weights, model_path)

        monkeypatch.setattr("merlt_models.WEIGHTS_DIR", temp_weights_dir)

        result = load_model("test_model")

        assert isinstance(result, dict)
        assert "layer1.weight" in result

    def test_load_model_file_not_found(self, monkeypatch, tmp_path):
        """Test that load_model raises error for missing model."""
        from merlt_models import load_model

        monkeypatch.setattr("merlt_models.WEIGHTS_DIR", tmp_path)
        monkeypatch.setattr("merlt_models.CHECKPOINTS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model")


class TestListFunctions:
    """Tests for list_models and list_configs functions."""

    def test_list_configs_returns_list(self, test_data_dir, monkeypatch):
        """Test that list_configs returns a list."""
        from merlt_models import list_configs

        monkeypatch.setattr("merlt_models.CONFIG_DIR", test_data_dir / "config")

        configs = list_configs()

        assert isinstance(configs, list)

    def test_list_models_returns_list(self, temp_weights_dir, monkeypatch):
        """Test that list_models returns a list."""
        from merlt_models import list_models

        monkeypatch.setattr("merlt_models.WEIGHTS_DIR", temp_weights_dir)
        monkeypatch.setattr("merlt_models.CHECKPOINTS_DIR", temp_weights_dir)

        models = list_models()

        assert isinstance(models, list)

    def test_list_models_includes_checkpoints(self, monkeypatch, tmp_path):
        """Test that list_models includes checkpoint directory."""
        from merlt_models import list_models

        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()

        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Create a checkpoint file
        (checkpoints_dir / "model_v1.pt").write_bytes(b"dummy")

        monkeypatch.setattr("merlt_models.WEIGHTS_DIR", weights_dir)
        monkeypatch.setattr("merlt_models.CHECKPOINTS_DIR", checkpoints_dir)

        models = list_models()

        assert "checkpoint/model_v1" in models
