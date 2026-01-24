"""
Test Disagreement Explainer
============================

Test per modulo explainability con Integrated Gradients.
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip se torch non disponibile
torch_available = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


from merlt.disagreement.types import (
    DisagreementAnalysis,
    DisagreementType,
    DisagreementLevel,
    ExpertPairConflict,
)


class TestIntegratedGradients:
    """Test per IntegratedGradients."""

    @pytest.fixture
    def mock_model(self):
        """Crea mock del modello."""
        from merlt.disagreement.heads import HeadsOutput

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 2)

            def forward(self, x):
                # x: [batch, num_experts, hidden_size]
                batch_size = x.shape[0]

                # Aggregate over experts
                agg = x.mean(dim=1)  # [batch, hidden_size]
                binary_logits = self.linear(agg)

                return HeadsOutput(
                    binary_logits=binary_logits,
                    binary_probs=torch.softmax(binary_logits, dim=-1),
                    type_logits=torch.randn(batch_size, 6),
                    type_probs=torch.softmax(torch.randn(batch_size, 6), dim=-1),
                    level_logits=torch.randn(batch_size, 4),
                    level_probs=torch.softmax(torch.randn(batch_size, 4), dim=-1),
                    intensity=torch.rand(batch_size, 1),
                    resolvability=torch.rand(batch_size, 1),
                    pairwise_matrix=torch.rand(batch_size, 4, 4),
                    confidence=torch.rand(batch_size, 1),
                )

        return MockModel()

    def test_compute_shape(self, mock_model):
        """Test shape delle attributions."""
        from merlt.disagreement.explainer import IntegratedGradients

        ig = IntegratedGradients(mock_model, n_steps=10)
        inputs = torch.randn(2, 4, 768)

        attributions = ig.compute(inputs, target_class=1, task="binary")

        assert attributions.shape == inputs.shape

    def test_compute_with_zero_baseline(self, mock_model):
        """Test con baseline zero."""
        from merlt.disagreement.explainer import IntegratedGradients

        ig = IntegratedGradients(mock_model, n_steps=10, baseline_type="zero")
        inputs = torch.randn(1, 4, 768)

        attributions = ig.compute(inputs, target_class=0, task="binary")

        assert not torch.isnan(attributions).any()

    def test_compute_with_mean_baseline(self, mock_model):
        """Test con baseline mean."""
        from merlt.disagreement.explainer import IntegratedGradients

        ig = IntegratedGradients(mock_model, n_steps=10, baseline_type="mean")
        inputs = torch.randn(1, 4, 768)

        attributions = ig.compute(inputs, target_class=1, task="binary")

        assert not torch.isnan(attributions).any()

    def test_compute_token_attributions(self, mock_model):
        """Test attributions per token."""
        from merlt.disagreement.explainer import IntegratedGradients

        ig = IntegratedGradients(mock_model, n_steps=10)
        inputs = torch.randn(1, 4, 768)

        token_attrs = ig.compute_token_attributions(
            inputs,
            target_class=1,
            task="binary",
        )

        # Una lista per batch
        assert len(token_attrs) == 1

        # 4 expert attributions
        assert len(token_attrs[0]) == 4

        # Verifica struttura TokenAttribution
        attr = token_attrs[0][0]
        assert hasattr(attr, "token")
        assert hasattr(attr, "score")
        assert hasattr(attr, "expert_source")


class TestAttentionAnalyzer:
    """Test per AttentionAnalyzer."""

    @pytest.fixture
    def mock_model_with_attention(self):
        """Crea mock con attention."""
        from merlt.disagreement.heads import HeadsOutput

        class MockCrossAttention(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                return {
                    "attended": x,
                    "attention_weights": torch.rand(batch_size, 4, 4, 4),
                    "contrast_features": torch.rand(batch_size, 4, 4, 768),
                    "aggregate": torch.rand(batch_size, 768),
                }

        class MockHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_attention = MockCrossAttention()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = MockHeads()

            def forward(self, x):
                batch_size = x.shape[0]
                return HeadsOutput(
                    binary_logits=torch.randn(batch_size, 2),
                    binary_probs=torch.softmax(torch.randn(batch_size, 2), dim=-1),
                    type_logits=torch.randn(batch_size, 6),
                    type_probs=torch.softmax(torch.randn(batch_size, 6), dim=-1),
                    level_logits=torch.randn(batch_size, 4),
                    level_probs=torch.softmax(torch.randn(batch_size, 4), dim=-1),
                    intensity=torch.rand(batch_size, 1),
                    resolvability=torch.rand(batch_size, 1),
                    pairwise_matrix=torch.rand(batch_size, 4, 4),
                    confidence=torch.rand(batch_size, 1),
                )

        return MockModel()

    def test_extract_attention(self, mock_model_with_attention):
        """Test estrazione attention weights."""
        from merlt.disagreement.explainer import AttentionAnalyzer

        analyzer = AttentionAnalyzer(mock_model_with_attention)
        inputs = torch.randn(2, 4, 768)

        result = analyzer.extract_attention(inputs)

        assert "attention_weights" in result
        assert result["attention_weights"] is not None

    def test_find_top_pairs(self, mock_model_with_attention):
        """Test ricerca coppie con alta attenzione."""
        from merlt.disagreement.explainer import AttentionAnalyzer

        analyzer = AttentionAnalyzer(mock_model_with_attention)

        # Crea matrice con coppia (0,1) ad alta attenzione
        attention = torch.zeros(1, 4, 4)
        attention[0, 0, 1] = 0.9
        attention[0, 1, 0] = 0.9

        pairs = analyzer._find_top_pairs(attention, top_k=3)

        assert len(pairs) == 1  # Un batch
        assert len(pairs[0]) >= 1  # Almeno una coppia
        assert pairs[0][0][:2] == (0, 1)  # Coppia (0,1)


class TestExplanationGenerator:
    """Test per ExplanationGenerator."""

    def test_generate_with_disagreement(self):
        """Test generazione con disagreement."""
        from merlt.disagreement.explainer import ExplanationGenerator

        generator = ExplanationGenerator()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
            confidence=0.85,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="literal",
                    expert_b="principles",
                    conflict_score=0.8,
                    contention_point="Interpretazione della buona fede",
                )
            ],
        )

        explanation = generator.generate(analysis)

        assert "divergenza" in explanation.natural_explanation.lower()
        assert "metodologic" in explanation.natural_explanation.lower()
        assert "85%" in explanation.natural_explanation

    def test_generate_without_disagreement(self):
        """Test generazione senza disagreement."""
        from merlt.disagreement.explainer import ExplanationGenerator

        generator = ExplanationGenerator()

        analysis = DisagreementAnalysis(
            has_disagreement=False,
            confidence=0.9,
        )

        explanation = generator.generate(analysis)

        assert "non" in explanation.natural_explanation.lower()
        assert "convergono" in explanation.natural_explanation.lower()

    def test_type_descriptions(self):
        """Test descrizioni tipi."""
        from merlt.disagreement.explainer import ExplanationGenerator

        generator = ExplanationGenerator()

        # Verifica che tutti i tipi abbiano descrizione
        for dtype in DisagreementType:
            assert dtype in generator.TYPE_DESCRIPTIONS

    def test_level_descriptions(self):
        """Test descrizioni livelli."""
        from merlt.disagreement.explainer import ExplanationGenerator

        generator = ExplanationGenerator()

        for level in DisagreementLevel:
            assert level in generator.LEVEL_DESCRIPTIONS

    def test_resolution_suggestions(self):
        """Test suggerimenti risoluzione."""
        from merlt.disagreement.explainer import ExplanationGenerator

        generator = ExplanationGenerator()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.HIERARCHICAL,
            confidence=0.8,
        )

        explanation = generator.generate(analysis)

        # Il tipo HIERARCHICAL ha criteri di risoluzione
        assert len(explanation.resolution_suggestions) > 0


class TestExplainabilityModule:
    """Test per ExplainabilityModule principale."""

    @pytest.fixture
    def simple_model(self):
        """Crea modello semplice."""
        from merlt.disagreement.heads import HeadsOutput

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 2)

            def forward(self, x):
                batch_size = x.shape[0]
                agg = x.mean(dim=1)
                binary_logits = self.linear(agg)

                return HeadsOutput(
                    binary_logits=binary_logits,
                    binary_probs=torch.softmax(binary_logits, dim=-1),
                    type_logits=torch.randn(batch_size, 6),
                    type_probs=torch.softmax(torch.randn(batch_size, 6), dim=-1),
                    level_logits=torch.randn(batch_size, 4),
                    level_probs=torch.softmax(torch.randn(batch_size, 4), dim=-1),
                    intensity=torch.rand(batch_size, 1),
                    resolvability=torch.rand(batch_size, 1),
                    pairwise_matrix=torch.rand(batch_size, 4, 4),
                    confidence=torch.rand(batch_size, 1),
                )

        return SimpleModel()

    def test_initialization(self, simple_model):
        """Test inizializzazione."""
        from merlt.disagreement.explainer import ExplainabilityModule

        explainer = ExplainabilityModule(
            model=simple_model,
            n_integration_steps=10,
        )

        assert explainer.ig is not None
        assert explainer.attention_analyzer is not None
        assert explainer.explanation_generator is not None

    @pytest.mark.asyncio
    async def test_explain_basic(self, simple_model):
        """Test spiegazione base."""
        from merlt.disagreement.explainer import ExplainabilityModule

        explainer = ExplainabilityModule(
            model=simple_model,
            n_integration_steps=5,
        )

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.ANTINOMY,
            confidence=0.8,
        )

        inputs = torch.randn(1, 4, 768)

        explanation = await explainer.explain(
            analysis=analysis,
            inputs=inputs,
            compute_attributions=True,
            compute_attention=False,
        )

        assert explanation.natural_explanation is not None
        assert len(explanation.natural_explanation) > 0

    @pytest.mark.asyncio
    async def test_explain_without_inputs(self, simple_model):
        """Test spiegazione senza input tensors."""
        from merlt.disagreement.explainer import ExplainabilityModule

        explainer = ExplainabilityModule(model=simple_model)

        analysis = DisagreementAnalysis(
            has_disagreement=False,
            confidence=0.9,
        )

        explanation = await explainer.explain(
            analysis=analysis,
            inputs=None,
            compute_attributions=False,
        )

        assert explanation.natural_explanation is not None

    def test_compute_feature_importance(self, simple_model):
        """Test calcolo feature importance."""
        from merlt.disagreement.explainer import ExplainabilityModule

        explainer = ExplainabilityModule(
            model=simple_model,
            n_integration_steps=5,
        )

        inputs = torch.randn(2, 4, 768)
        importance = explainer.compute_feature_importance(inputs, task="binary")

        assert "literal" in importance
        assert "systemic" in importance
        assert "principles" in importance
        assert "precedent" in importance

        # Valori dovrebbero essere float
        for name, value in importance.items():
            assert isinstance(value, float)


class TestExplanationIntegration:
    """Test di integrazione per explainability."""

    def test_full_explanation_pipeline(self):
        """Test pipeline completa di spiegazione."""
        from merlt.disagreement.explainer import ExplanationGenerator
        from merlt.disagreement.types import TokenAttribution

        generator = ExplanationGenerator()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.INTERPRETIVE_GAP,
            disagreement_level=DisagreementLevel.SEMANTIC,
            intensity=0.6,
            resolvability=0.7,
            confidence=0.82,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="literal",
                    expert_b="systemic",
                    conflict_score=0.65,
                )
            ],
        )

        attributions = [
            TokenAttribution(token="literal", score=0.8, expert_source="literal", position=0),
            TokenAttribution(token="systemic", score=0.6, expert_source="systemic", position=1),
        ]

        explanation = generator.generate(
            analysis=analysis,
            attributions=attributions,
        )

        # Verifica struttura completa
        assert explanation.natural_explanation is not None
        assert len(explanation.key_tokens) > 0
        assert len(explanation.expert_pair_scores) > 0
