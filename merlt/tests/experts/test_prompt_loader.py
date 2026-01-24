"""
Test PromptLoader
==================

Test per il caricamento e gestione dei prompt da YAML.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml

from merlt.experts.prompt_loader import (
    PromptLoader,
    PromptVersion,
    PromptUsage,
    get_prompt_loader,
)


class TestPromptVersion:
    """Test per PromptVersion dataclass."""

    def test_creation(self):
        """Test creazione PromptVersion."""
        version = PromptVersion(
            version="1.0.0",
            content="Test prompt content",
            created="2025-12-29",
            status="active",
        )

        assert version.version == "1.0.0"
        assert version.content == "Test prompt content"
        assert version.status == "active"
        assert version.metadata == {}
        assert version.performance_metrics == {}


class TestPromptUsage:
    """Test per PromptUsage dataclass."""

    def test_creation(self):
        """Test creazione PromptUsage."""
        usage = PromptUsage(
            expert_type="literal",
            prompt_name="system_prompt",
            version="1.0.0",
            timestamp="2025-12-29T10:00:00",
            trace_id="abc123",
            query_type="definitional",
        )

        assert usage.expert_type == "literal"
        assert usage.prompt_name == "system_prompt"
        assert usage.version == "1.0.0"
        assert usage.trace_id == "abc123"


class TestPromptLoader:
    """Test per PromptLoader."""

    @pytest.fixture
    def temp_prompts_dir(self):
        """Crea directory temporanea con prompts.yaml di test."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            prompts_yaml = config_dir / "prompts.yaml"
            test_prompts = {
                "version": "1.0.0",
                "experts": {
                    "literal": {
                        "system_prompt": "Sei un esperto di interpretazione letterale.",
                        "metadata": {
                            "description": "Interpretazione letterale",
                            "created": "2025-12-29",
                        },
                    },
                    "systemic": {
                        "system_prompt": "Sei un esperto di interpretazione sistematica.",
                        "metadata": {
                            "description": "Interpretazione sistematica",
                        },
                    },
                    "principles": {
                        "system_prompt": "Sei un esperto di interpretazione teleologica.",
                    },
                    "precedent": {
                        "system_prompt": "Sei un esperto di giurisprudenza.",
                    },
                },
                "synthesizer": {
                    "convergent": {
                        "convergent": "Sintetizza in modo convergente.",
                        "instructions": "Istruzioni convergent.",
                    },
                    "divergent": {
                        "divergent": "Presenta le alternative.",
                    },
                },
            }

            with open(prompts_yaml, "w", encoding="utf-8") as f:
                yaml.dump(test_prompts, f, allow_unicode=True)

            yield prompts_yaml

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton prima di ogni test."""
        PromptLoader._instance = None
        yield
        PromptLoader._instance = None

    def test_initialization(self, temp_prompts_dir):
        """Test inizializzazione con file esistente."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        assert loader._initialized is True
        assert loader.version == "1.0.0"
        assert "literal" in loader.available_experts

    def test_singleton_pattern(self, temp_prompts_dir):
        """Test che PromptLoader sia singleton."""
        loader1 = PromptLoader(config_path=temp_prompts_dir)
        loader2 = PromptLoader()

        assert loader1 is loader2

    def test_get_prompt_basic(self, temp_prompts_dir):
        """Test caricamento prompt base."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        prompt = loader.get_prompt("literal")

        assert "interpretazione letterale" in prompt.lower()

    def test_get_prompt_fallback(self, temp_prompts_dir):
        """Test fallback per expert non trovato."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        prompt = loader.get_prompt("unknown_expert")

        assert "esperto giuridico" in prompt.lower()

    def test_get_prompt_synthesizer(self, temp_prompts_dir):
        """Test prompt synthesizer."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        prompt = loader.get_prompt("synthesizer", "convergent")

        assert "convergente" in prompt.lower() or "Sintetizza" in prompt

    def test_get_prompt_with_metadata(self, temp_prompts_dir):
        """Test get_prompt_with_metadata."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        result = loader.get_prompt_with_metadata("literal")

        assert "prompt" in result
        assert "version" in result
        assert result["version"] == "1.0.0"
        assert result["expert_type"] == "literal"

    def test_get_metadata(self, temp_prompts_dir):
        """Test get_metadata."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        metadata = loader.get_metadata("literal")

        assert "description" in metadata
        assert "Interpretazione letterale" in metadata["description"]

    def test_track_usage(self, temp_prompts_dir):
        """Test tracking utilizzo."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        loader.track_usage(
            expert_type="literal",
            prompt_name="system_prompt",
            trace_id="test123",
            query_type="definitional",
        )

        stats = loader.get_usage_stats()

        assert stats["total_calls"] == 1
        assert "literal/system_prompt" in stats["by_prompt"]

    def test_track_usage_limit(self, temp_prompts_dir):
        """Test che tracking limiti a 1000 record."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        # Aggiungi 1100 record
        for i in range(1100):
            loader.track_usage("literal", trace_id=str(i))

        stats = loader.get_usage_stats()

        # Deve limitare a 1000
        assert stats["total_calls"] == 1000

    def test_list_prompts(self, temp_prompts_dir):
        """Test list_prompts."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        prompts = loader.list_prompts()

        assert "literal/system_prompt" in prompts
        assert "systemic/system_prompt" in prompts
        assert any("synthesizer" in p for p in prompts)

    def test_list_prompts_filtered(self, temp_prompts_dir):
        """Test list_prompts con filtro expert."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        prompts = loader.list_prompts(expert_type="literal")

        assert "literal/system_prompt" in prompts
        assert "systemic/system_prompt" not in prompts

    def test_reload(self, temp_prompts_dir):
        """Test reload dei prompts."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        original_version = loader.version

        # Modifica file
        with open(temp_prompts_dir, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["version"] = "2.0.0"
        with open(temp_prompts_dir, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        # Reload
        loader.reload()

        assert loader.version == "2.0.0"
        assert loader.version != original_version

    def test_available_experts(self, temp_prompts_dir):
        """Test available_experts property."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        experts = loader.available_experts

        assert "literal" in experts
        assert "systemic" in experts
        assert "principles" in experts
        assert "precedent" in experts

    def test_missing_config_file(self):
        """Test comportamento con file mancante."""
        loader = PromptLoader(config_path=Path("/nonexistent/path.yaml"))

        # Deve funzionare con fallback
        prompt = loader.get_prompt("literal")

        assert len(prompt) > 0


class TestGetPromptLoader:
    """Test per factory function."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global state."""
        import merlt.experts.prompt_loader as module
        module._loader = None
        PromptLoader._instance = None
        yield
        module._loader = None
        PromptLoader._instance = None

    def test_get_prompt_loader_singleton(self):
        """Test che get_prompt_loader restituisca singleton."""
        loader1 = get_prompt_loader()
        loader2 = get_prompt_loader()

        assert loader1 is loader2


class TestPromptLoaderCaching:
    """Test per caching dei prompt."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton prima di ogni test."""
        PromptLoader._instance = None
        yield
        PromptLoader._instance = None

    @pytest.fixture
    def temp_prompts_dir(self):
        """Crea directory temporanea."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            prompts_yaml = config_dir / "prompts.yaml"
            test_prompts = {
                "version": "1.0.0",
                "experts": {
                    "literal": {
                        "system_prompt": "Test prompt",
                    },
                },
            }

            with open(prompts_yaml, "w", encoding="utf-8") as f:
                yaml.dump(test_prompts, f)

            yield prompts_yaml

    def test_cache_hit(self, temp_prompts_dir):
        """Test che cache funzioni."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        # Prima chiamata
        prompt1 = loader.get_prompt("literal")

        # Seconda chiamata (dovrebbe essere cached)
        prompt2 = loader.get_prompt("literal")

        assert prompt1 == prompt2

        # Verifica cache info
        cache_info = loader.get_prompt.cache_info()
        assert cache_info.hits >= 1

    def test_cache_clear(self, temp_prompts_dir):
        """Test clear della cache."""
        loader = PromptLoader(config_path=temp_prompts_dir)

        # Popola cache
        loader.get_prompt("literal")
        loader.get_prompt("literal")

        # Clear
        loader._clear_cache()

        # Cache dovrebbe essere vuota
        cache_info = loader.get_prompt.cache_info()
        assert cache_info.currsize == 0
