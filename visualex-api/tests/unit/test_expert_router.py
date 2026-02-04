"""
Tests for Expert Router.

Tests cover:
- AC1: Query type classification (DEFINITION, INTERPRETATION, etc.)
- AC2: Expert activation order (Art. 12 Preleggi)
- AC3: Expert relevance scores
- AC4: Routing rationale for F2 feedback
- AC5: Entity-based weight adjustment
"""

import pytest
from unittest.mock import MagicMock

from visualex.experts import (
    ExpertRouter,
    RouterConfig,
    RoutingDecision,
    ExpertWeight,
    QueryType,
    ExpertType,
)
from visualex.ner import ExtractionResult, ExtractedEntity, EntityType


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestExpertWeight:
    """Tests for ExpertWeight dataclass."""

    def test_creation(self):
        """Test basic creation."""
        weight = ExpertWeight(
            expert=ExpertType.LITERAL,
            weight=0.6,
            is_primary=True,
        )

        assert weight.expert == ExpertType.LITERAL
        assert weight.weight == 0.6
        assert weight.is_primary is True
        assert weight.skip_reason is None

    def test_to_dict(self):
        """Test serialization."""
        weight = ExpertWeight(
            expert=ExpertType.SYSTEMIC,
            weight=0.25,
            is_primary=False,
            skip_reason="Low relevance",
        )

        d = weight.to_dict()

        assert d["expert"] == "systemic"
        assert d["weight"] == 0.25
        assert d["is_primary"] is False
        assert d["skip_reason"] == "Low relevance"


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_creation(self):
        """Test basic creation."""
        decision = RoutingDecision(
            query_type=QueryType.DEFINITION,
            expert_weights=[
                ExpertWeight(ExpertType.LITERAL, 0.6, is_primary=True),
                ExpertWeight(ExpertType.SYSTEMIC, 0.2),
            ],
            confidence=0.85,
            rationale="Test rationale",
        )

        assert decision.query_type == QueryType.DEFINITION
        assert len(decision.expert_weights) == 2
        assert decision.confidence == 0.85

    def test_to_dict(self):
        """Test serialization."""
        decision = RoutingDecision(
            query_type=QueryType.INTERPRETATION,
            expert_weights=[
                ExpertWeight(ExpertType.LITERAL, 0.35),
            ],
            confidence=0.7,
            rationale="Test",
        )

        d = decision.to_dict()

        assert d["query_type"] == "INTERPRETATION"
        assert d["confidence"] == 0.7
        assert len(d["expert_weights"]) == 1
        assert "activation_order" in d

    def test_get_activation_order(self):
        """Test Expert activation order."""
        decision = RoutingDecision(
            query_type=QueryType.DEFINITION,
            expert_weights=[
                ExpertWeight(ExpertType.LITERAL, 0.6),
                ExpertWeight(ExpertType.SYSTEMIC, 0.2),
                ExpertWeight(ExpertType.PRINCIPLES, 0.1),
                ExpertWeight(ExpertType.PRECEDENT, 0.05),  # Below threshold
            ],
            confidence=0.8,
            rationale="",
        )

        order = decision.get_activation_order(threshold=0.1)

        assert order[0] == "literal"
        assert order[1] == "systemic"
        assert order[2] == "principles"
        assert "precedent" not in order  # Below threshold

    def test_get_primary_expert(self):
        """Test getting primary Expert."""
        decision = RoutingDecision(
            query_type=QueryType.CASE_ANALYSIS,
            expert_weights=[
                ExpertWeight(ExpertType.LITERAL, 0.2),
                ExpertWeight(ExpertType.PRECEDENT, 0.5),
            ],
            confidence=0.75,
            rationale="",
        )

        primary = decision.get_primary_expert()

        assert primary == ExpertType.PRECEDENT


# =============================================================================
# Router Configuration Tests
# =============================================================================


class TestRouterConfig:
    """Tests for RouterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RouterConfig()

        assert config.weight_threshold == 0.1
        assert config.confidence_threshold == 0.5
        assert config.boost_factor == 1.2
        assert config.record_rationale is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RouterConfig(
            weight_threshold=0.2,
            boost_factor=1.5,
        )

        assert config.weight_threshold == 0.2
        assert config.boost_factor == 1.5


# =============================================================================
# Expert Router Tests
# =============================================================================


class TestExpertRouter:
    """Tests for ExpertRouter."""

    def setup_method(self):
        """Create router for tests."""
        self.router = ExpertRouter()

    @pytest.mark.asyncio
    async def test_route_definition_query(self):
        """Test routing definitional query (AC1)."""
        decision = await self.router.route("Cos'è la risoluzione del contratto?")

        assert decision.query_type == QueryType.DEFINITION
        assert decision.get_primary_expert() == ExpertType.LITERAL

    @pytest.mark.asyncio
    async def test_route_interpretation_query(self):
        """Test routing interpretive query (AC1)."""
        decision = await self.router.route(
            "Come si interpreta l'art. 1453 c.c.?"
        )

        assert decision.query_type == QueryType.INTERPRETATION

    @pytest.mark.asyncio
    async def test_route_comparison_query(self):
        """Test routing comparison query (AC1)."""
        decision = await self.router.route(
            "Qual è la differenza tra risoluzione e rescissione?"
        )

        assert decision.query_type == QueryType.COMPARISON
        # Systemic should have high weight for comparisons
        systemic_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.SYSTEMIC
        )
        assert systemic_weight.weight > 0.2

    @pytest.mark.asyncio
    async def test_route_case_analysis_query(self):
        """Test routing case analysis query (AC1)."""
        decision = await self.router.route(
            "Nel caso in cui il debitore non paghi, posso risolvere il contratto?"
        )

        assert decision.query_type == QueryType.CASE_ANALYSIS
        # Precedent should have high weight for case analysis
        precedent_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.PRECEDENT
        )
        assert precedent_weight.weight >= 0.2

    @pytest.mark.asyncio
    async def test_art12_activation_order(self):
        """Test Art. 12 Preleggi activation order (AC2)."""
        decision = await self.router.route("Test query")

        # Verify all 4 experts are present
        expert_types = [w.expert for w in decision.expert_weights]
        assert ExpertType.LITERAL in expert_types
        assert ExpertType.SYSTEMIC in expert_types
        assert ExpertType.PRINCIPLES in expert_types
        assert ExpertType.PRECEDENT in expert_types

    @pytest.mark.asyncio
    async def test_expert_relevance_scores(self):
        """Test Expert relevance scores (AC3)."""
        decision = await self.router.route("Cos'è l'inadempimento?")

        # All weights should be between 0 and 1
        for weight in decision.expert_weights:
            assert 0.0 <= weight.weight <= 1.0

        # Weights should sum to approximately 1
        total_weight = sum(w.weight for w in decision.expert_weights)
        assert 0.99 <= total_weight <= 1.01

    @pytest.mark.asyncio
    async def test_routing_rationale_recorded(self):
        """Test routing rationale for F2 feedback (AC4)."""
        decision = await self.router.route("Cos'è la risoluzione?")

        assert decision.rationale != ""
        assert "DEFINITION" in decision.rationale
        assert "Expert primario" in decision.rationale

    @pytest.mark.asyncio
    async def test_confidence_score(self):
        """Test confidence score is provided."""
        decision = await self.router.route("Cos'è la legittima difesa?")

        assert 0.0 <= decision.confidence <= 1.0


class TestExpertRouterWithNER:
    """Tests for Expert Router with NER integration (AC5)."""

    def setup_method(self):
        """Create router for tests."""
        self.router = ExpertRouter()

    @pytest.mark.asyncio
    async def test_boost_literal_with_norm_refs(self):
        """Test Literal boost with norm references."""
        ner_result = ExtractionResult(
            text="art. 1453 c.c.",
            entities=[
                ExtractedEntity(
                    text="art. 1453",
                    entity_type=EntityType.ARTICLE_REF,
                    start=0,
                    end=9,
                ),
            ],
        )

        decision = await self.router.route("Cosa dice l'art. 1453?", ner_result)

        # Literal should be boosted
        literal_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.LITERAL
        )
        assert literal_weight.weight > 0.3

    @pytest.mark.asyncio
    async def test_boost_principles_with_concepts(self):
        """Test Principles boost with principle-related concepts."""
        ner_result = ExtractionResult(
            text="principio di libertà",
            entities=[
                ExtractedEntity(
                    text="libertà",
                    entity_type=EntityType.LEGAL_CONCEPT,
                    start=13,
                    end=20,
                ),
            ],
        )

        decision = await self.router.route(
            "Qual è il principio di libertà contrattuale?",
            ner_result,
        )

        # Principles should be boosted
        principles_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.PRINCIPLES
        )
        assert principles_weight.weight > 0.1

    @pytest.mark.asyncio
    async def test_metadata_includes_entity_count(self):
        """Test metadata includes NER entity count."""
        ner_result = ExtractionResult(
            text="test",
            entities=[
                ExtractedEntity("art. 1", EntityType.ARTICLE_REF, 0, 6),
                ExtractedEntity("risoluzione", EntityType.LEGAL_CONCEPT, 10, 21),
            ],
        )

        decision = await self.router.route("Test query", ner_result)

        assert decision.metadata["ner_entity_count"] == 2


class TestExpertRouterKeywords:
    """Tests for keyword-based weight adjustments."""

    def setup_method(self):
        """Create router for tests."""
        self.router = ExpertRouter()

    @pytest.mark.asyncio
    async def test_boost_systemic_with_evolution(self):
        """Test Systemic boost with evolution keywords."""
        decision = await self.router.route(
            "Qual è l'evoluzione storica dell'articolo?"
        )

        systemic_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.SYSTEMIC
        )
        assert systemic_weight.weight > 0.2

    @pytest.mark.asyncio
    async def test_boost_principles_with_ratio(self):
        """Test Principles boost with ratio keywords."""
        decision = await self.router.route(
            "Qual è la ratio legis dell'art. 1453?"
        )

        principles_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.PRINCIPLES
        )
        assert principles_weight.weight > 0.2

    @pytest.mark.asyncio
    async def test_boost_precedent_with_giurisprudenza(self):
        """Test Precedent boost with giurisprudenza keywords."""
        decision = await self.router.route(
            "Cosa dice la giurisprudenza della Cassazione?"
        )

        precedent_weight = next(
            w for w in decision.expert_weights if w.expert == ExpertType.PRECEDENT
        )
        # Precedent should be boosted above baseline (~0.20)
        assert precedent_weight.weight > 0.2


class TestExpertRouterEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Create router for tests."""
        self.router = ExpertRouter()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test handling empty query."""
        decision = await self.router.route("")

        # Should return valid decision with default type
        assert decision.query_type is not None
        assert len(decision.expert_weights) == 4

    @pytest.mark.asyncio
    async def test_general_query_defaults(self):
        """Test general query uses INTERPRETATION as default."""
        decision = await self.router.route("Una domanda generica qualsiasi")

        # Should default to INTERPRETATION
        assert decision.query_type == QueryType.INTERPRETATION

    @pytest.mark.asyncio
    async def test_multiple_query_patterns(self):
        """Test that multiple patterns can match."""
        # Query with both DEFINITION and INTERPRETATION patterns
        decision = await self.router.route(
            "Cos'è e come si interpreta la risoluzione?"
        )

        # Should match one of the types (priority based on pattern matching)
        assert decision.query_type in [QueryType.DEFINITION, QueryType.INTERPRETATION]

    @pytest.mark.asyncio
    async def test_custom_config(self):
        """Test router with custom config."""
        config = RouterConfig(weight_threshold=0.3, record_rationale=False)
        router = ExpertRouter(config)

        decision = await router.route("Test query")

        # Rationale should be empty with record_rationale=False
        assert decision.rationale == ""

        # Higher threshold should skip more experts
        activation = decision.get_activation_order(threshold=0.3)
        assert len(activation) < 4  # Some experts should be skipped
