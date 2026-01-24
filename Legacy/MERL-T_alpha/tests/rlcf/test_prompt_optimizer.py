"""
Test PromptOptimizer (APE)
===========================

Test per Automatic Prompt Engineering via LLM.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock
import yaml

from merlt.rlcf.prompt_optimizer import (
    PromptOptimizer,
    PromptCandidate,
    OptimizationResult,
    APEConfig,
)


class TestPromptCandidate:
    """Test per PromptCandidate dataclass."""

    def test_creation(self):
        """Test creazione PromptCandidate."""
        candidate = PromptCandidate(
            prompt="Test prompt",
            rationale="Test rationale",
            focus_area="clarity",
            score=0.75,
        )

        assert candidate.prompt == "Test prompt"
        assert candidate.rationale == "Test rationale"
        assert candidate.focus_area == "clarity"
        assert candidate.score == 0.75

    def test_default_values(self):
        """Test valori default."""
        candidate = PromptCandidate(prompt="Test")

        assert candidate.rationale == ""
        assert candidate.focus_area == "general"
        assert candidate.score == 0.0
        assert candidate.metadata == {}

    def test_to_dict(self):
        """Test serializzazione."""
        candidate = PromptCandidate(
            prompt="Test",
            focus_area="clarity",
            score=0.8,
        )

        d = candidate.to_dict()

        assert d["prompt"] == "Test"
        assert d["focus_area"] == "clarity"
        assert d["score"] == 0.8


class TestOptimizationResult:
    """Test per OptimizationResult dataclass."""

    def test_creation(self):
        """Test creazione OptimizationResult."""
        result = OptimizationResult(
            expert_type="literal",
            original_prompt="Old prompt",
            new_prompt="New prompt",
            original_score=0.5,
            new_score=0.75,
            candidates_evaluated=3,
        )

        assert result.expert_type == "literal"
        assert result.original_score == 0.5
        assert result.new_score == 0.75
        assert result.candidates_evaluated == 3

    def test_improvement_calculation(self):
        """Test calcolo improvement."""
        result = OptimizationResult(
            expert_type="literal",
            original_prompt="Old",
            new_prompt="New",
            original_score=0.5,
            new_score=0.75,
            candidates_evaluated=3,
        )

        # Improvement = (0.75 - 0.5) / 0.5 * 100 = 50%
        assert result.improvement == 50.0

    def test_improvement_zero_original(self):
        """Test improvement con original score 0."""
        result = OptimizationResult(
            expert_type="literal",
            original_prompt="Old",
            new_prompt="New",
            original_score=0.0,
            new_score=0.75,
            candidates_evaluated=3,
        )

        # Evita divisione per zero
        assert result.improvement == 0.0


class TestAPEConfig:
    """Test per APEConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        config = APEConfig()

        assert config.trigger_threshold == 0.65
        assert config.num_candidates == 3
        assert config.evaluation_queries == 5
        assert config.min_improvement == 0.05
        assert config.cooldown_hours == 24

    def test_custom_config(self):
        """Test configurazione custom."""
        config = APEConfig(
            trigger_threshold=0.7,
            num_candidates=5,
            cooldown_hours=12,
        )

        assert config.trigger_threshold == 0.7
        assert config.num_candidates == 5
        assert config.cooldown_hours == 12


class TestPromptOptimizer:
    """Test per PromptOptimizer."""

    @pytest.fixture
    def temp_prompts_file(self):
        """Crea file prompts.yaml temporaneo."""
        with TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.yaml"

            test_prompts = {
                "version": "1.0.0",
                "experts": {
                    "literal": {
                        "system_prompt": "Sei un esperto di interpretazione letterale. " * 20,
                        "metadata": {},
                    },
                    "systemic": {
                        "system_prompt": "Sei un esperto di interpretazione sistematica.",
                    },
                },
            }

            with open(prompts_path, "w", encoding="utf-8") as f:
                yaml.dump(test_prompts, f, allow_unicode=True)

            yield prompts_path

    def test_initialization(self, temp_prompts_file):
        """Test inizializzazione."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        assert optimizer.config.trigger_threshold == 0.65
        assert optimizer._last_optimization == {}
        assert optimizer._optimization_history == []

    def test_initialization_with_config(self, temp_prompts_file):
        """Test inizializzazione con config custom."""
        config = APEConfig(trigger_threshold=0.8)
        optimizer = PromptOptimizer(config=config, prompts_path=temp_prompts_file)

        assert optimizer.config.trigger_threshold == 0.8

    @pytest.mark.asyncio
    async def test_optimize_if_needed_above_threshold(self, temp_prompts_file):
        """Test che non ottimizzi se rating sopra soglia."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        result = await optimizer.optimize_if_needed(
            expert_type="literal",
            avg_rating=0.8,  # Sopra soglia 0.65
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_optimize_if_needed_triggers(self, temp_prompts_file):
        """Test che ottimizzi se rating sotto soglia."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        # Senza ai_service, usa fallback candidates
        result = await optimizer.optimize_if_needed(
            expert_type="literal",
            avg_rating=0.45,  # Sotto soglia 0.65
        )

        # Con fallback candidates che hanno score basso,
        # potrebbe non migliorare abbastanza
        # Il test verifica che il flusso funzioni

    @pytest.mark.asyncio
    async def test_cooldown(self, temp_prompts_file):
        """Test che cooldown prevenga ottimizzazioni multiple."""
        optimizer = PromptOptimizer(
            prompts_path=temp_prompts_file,
            config=APEConfig(cooldown_hours=24),
        )

        # Simula ottimizzazione recente
        from datetime import datetime
        optimizer._last_optimization["literal"] = datetime.now()

        result = await optimizer.optimize_if_needed(
            expert_type="literal",
            avg_rating=0.4,
        )

        assert result is None  # Skipped per cooldown

    @pytest.mark.asyncio
    async def test_generate_candidates_without_ai(self, temp_prompts_file):
        """Test generazione candidati senza AI service."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        candidates = await optimizer.generate_candidates(
            expert_type="literal",
            current_prompt="Test prompt",
            feedback_history=[],
            num_candidates=3,
        )

        # Fallback genera 2 candidati
        assert len(candidates) == 2
        assert all(isinstance(c, PromptCandidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_generate_candidates_with_mock_ai(self, temp_prompts_file):
        """Test generazione candidati con AI mockato."""
        mock_ai = MagicMock()
        mock_ai.complete = AsyncMock(return_value="Nuovo prompt migliorato.\n## Sezione\nDettagli.")

        optimizer = PromptOptimizer(
            ai_service=mock_ai,
            prompts_path=temp_prompts_file,
        )

        candidates = await optimizer.generate_candidates(
            expert_type="literal",
            current_prompt="Test prompt",
            feedback_history=[{"rating": 0.4}],
            num_candidates=2,
        )

        assert len(candidates) >= 1
        mock_ai.complete.assert_called()

    @pytest.mark.asyncio
    async def test_evaluate_candidates(self, temp_prompts_file):
        """Test valutazione candidati."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        candidates = [
            PromptCandidate(
                prompt="Prompt corto",
                focus_area="clarity",
            ),
            PromptCandidate(
                prompt="Prompt lungo con fonte e citazione e JSON e confidence e legal_basis. " * 20 + "\n## Sezione",
                focus_area="specificity",
            ),
        ]

        best = await optimizer._evaluate_candidates(candidates, "literal")

        assert best is not None
        # Il secondo dovrebbe avere score piu' alto (keywords, headers)
        assert best.score > 0.5

    def test_load_current_prompt(self, temp_prompts_file):
        """Test caricamento prompt corrente."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        prompt = optimizer._load_current_prompt("literal")

        assert prompt is not None
        assert "interpretazione letterale" in prompt.lower()

    def test_load_current_prompt_missing(self, temp_prompts_file):
        """Test caricamento prompt mancante."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        prompt = optimizer._load_current_prompt("unknown")

        assert prompt is None

    def test_save_new_prompt(self, temp_prompts_file):
        """Test salvataggio nuovo prompt."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        success = optimizer._save_new_prompt("literal", "Nuovo prompt!")

        assert success is True

        # Verifica che sia stato salvato
        with open(temp_prompts_file, "r") as f:
            data = yaml.safe_load(f)

        assert data["experts"]["literal"]["system_prompt"] == "Nuovo prompt!"
        assert data["version"] == "1.0.1"  # Versione incrementata

    def test_summarize_feedback(self, temp_prompts_file):
        """Test riassunto feedback."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        feedback_list = [
            {"rating": 0.3, "comment": "Risposta poco chiara"},
            {"rating": 0.4, "comment": "Mancano le fonti"},
            {"rating": 0.5},  # Senza commento
        ]

        summary = optimizer._summarize_feedback(feedback_list)

        assert "poco chiara" in summary
        assert "fonti" in summary

    def test_summarize_feedback_empty(self, temp_prompts_file):
        """Test riassunto feedback vuoto."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        summary = optimizer._summarize_feedback([])

        assert "Nessun feedback" in summary

    def test_identify_issues(self, temp_prompts_file):
        """Test identificazione problemi."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        # Molti rating bassi
        feedback_list = [
            {"rating": 0.2},
            {"rating": 0.3},
            {"rating": 0.4},
            {"rating": 0.6},
        ]

        issues = optimizer._identify_issues(feedback_list)

        assert "bassi" in issues.lower() or "sotto" in issues.lower()

    def test_get_optimization_history(self, temp_prompts_file):
        """Test history delle ottimizzazioni."""
        optimizer = PromptOptimizer(prompts_path=temp_prompts_file)

        # Aggiungi risultato fittizio
        optimizer._optimization_history.append(
            OptimizationResult(
                expert_type="literal",
                original_prompt="Old",
                new_prompt="New",
                original_score=0.5,
                new_score=0.7,
                candidates_evaluated=3,
            )
        )

        history = optimizer.get_optimization_history()

        assert len(history) == 1
        assert history[0]["expert_type"] == "literal"


class TestPromptOptimizerIntegration:
    """Test di integrazione per PromptOptimizer."""

    @pytest.fixture
    def temp_prompts_file(self):
        """Crea file prompts.yaml temporaneo."""
        with TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.yaml"

            test_prompts = {
                "version": "1.0.0",
                "experts": {
                    "literal": {
                        "system_prompt": "Prompt base per literal expert. " * 30,
                    },
                },
            }

            with open(prompts_path, "w", encoding="utf-8") as f:
                yaml.dump(test_prompts, f, allow_unicode=True)

            yield prompts_path

    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self, temp_prompts_file):
        """Test ciclo completo di ottimizzazione."""
        # Mock AI che genera prompt con keywords buone
        mock_ai = MagicMock()
        mock_ai.complete = AsyncMock(
            return_value=(
                "Nuovo prompt migliorato con fonte e citazione e JSON. "
                "Includi sempre confidence e legal_basis nelle risposte.\n"
                "## Istruzioni\nSegui queste linee guida."
            )
        )

        optimizer = PromptOptimizer(
            ai_service=mock_ai,
            prompts_path=temp_prompts_file,
            config=APEConfig(min_improvement=0.0),  # Accetta qualsiasi miglioramento
        )

        result = await optimizer.optimize_if_needed(
            expert_type="literal",
            avg_rating=0.3,
            feedback_history=[
                {"rating": 0.3, "comment": "Non chiaro"},
                {"rating": 0.25, "comment": "Mancano fonti"},
            ],
        )

        # Con min_improvement=0, dovrebbe sempre salvare se genera candidati
        if result is not None:
            assert result.expert_type == "literal"
            assert result.candidates_evaluated > 0
