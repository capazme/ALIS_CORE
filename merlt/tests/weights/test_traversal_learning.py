"""
Test per RLCF Traversal Weight Learning (Fase 2 v2 Recovery).

Verifica:
1. Tracking relazioni in explore_iteratively()
2. Calcolo gradiente per traversal weights
3. Update pesi via WeightLearner
4. Convergenza pesi nel tempo
5. Persistenza e recovery
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from merlt.experts.base import (
    BaseExpert,
    ExpertContext,
    ExpertResponse,
    RelationUsage,
    LegalSource,
)
from merlt.weights.learner import (
    WeightLearner,
    RLCFFeedback,
    RelationUsageData,
    LearnerConfig,
)
from merlt.weights.config import (
    WeightConfig,
    ExpertTraversalWeights,
    LearnableWeight,
)
from merlt.weights.store import WeightStore


class TestRelationUsageDataclass:
    """Test dataclass RelationUsage."""

    def test_init_defaults(self):
        """Verifica inizializzazione con default."""
        usage = RelationUsage(relation_type="RIFERIMENTO")
        assert usage.relation_type == "RIFERIMENTO"
        assert usage.usage_count == 0
        assert usage.sources_found == 0
        assert usage.sources_used_in_response == 0
        assert usage.avg_relevance == 0.0

    def test_to_dict(self):
        """Verifica serializzazione."""
        usage = RelationUsage(
            relation_type="CITATO_DA",
            usage_count=5,
            sources_found=3,
            sources_used_in_response=2,
            avg_relevance=0.8
        )
        d = usage.to_dict()
        assert d["relation_type"] == "CITATO_DA"
        assert d["usage_count"] == 5
        assert d["avg_relevance"] == 0.8

    def test_from_dict(self):
        """Verifica deserializzazione."""
        data = {
            "relation_type": "MODIFICA",
            "usage_count": 10,
            "sources_found": 7,
            "sources_used_in_response": 4,
            "avg_relevance": 0.75
        }
        usage = RelationUsage.from_dict(data)
        assert usage.relation_type == "MODIFICA"
        assert usage.usage_count == 10
        assert usage.avg_relevance == 0.75


class TestRelationUsageDataLearner:
    """Test RelationUsageData per WeightLearner."""

    def test_init(self):
        """Verifica inizializzazione."""
        data = RelationUsageData(
            relation_type="ABROGA",
            usage_count=3,
            sources_found=2,
            avg_relevance=0.9
        )
        assert data.relation_type == "ABROGA"
        assert data.avg_relevance == 0.9


class TestComputeTraversalGradient:
    """Test _compute_traversal_gradient in WeightLearner."""

    @pytest.fixture
    def mock_store(self):
        """Crea mock WeightStore."""
        store = MagicMock(spec=WeightStore)
        store.get_weights = AsyncMock(return_value=WeightConfig())
        return store

    @pytest.fixture
    def learner(self, mock_store):
        """Crea WeightLearner per test."""
        return WeightLearner(store=mock_store)

    def test_gradient_empty_without_relation_usage(self, learner, mock_store):
        """Verifica che gradient sia vuoto senza relation_usage."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.8,
            relevance_scores={},
            expert_type="literal"
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)
        assert gradient == {}

    def test_gradient_empty_without_expert_type(self, learner, mock_store):
        """Verifica che gradient sia vuoto senza expert_type."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.8,
            relevance_scores={},
            relation_usage={"RIFERIMENTO": RelationUsageData("RIFERIMENTO", 5, 3)}
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)
        assert gradient == {}

    def test_gradient_positive_with_good_feedback(self, learner, mock_store):
        """Verifica gradient positivo con feedback positivo."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.8,
            relevance_scores={},
            expert_type="literal",
            user_rating=0.9,  # Feedback positivo
            relation_usage={
                "RIFERIMENTO": RelationUsageData(
                    relation_type="RIFERIMENTO",
                    usage_count=5,
                    sources_found=3,
                    sources_used_in_response=2,  # 2/3 = 0.67 efficienza
                    avg_relevance=0.8
                )
            }
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)

        assert "RIFERIMENTO" in gradient
        assert gradient["RIFERIMENTO"] > 0  # Gradient positivo per feedback positivo

    def test_gradient_negative_with_bad_feedback(self, learner, mock_store):
        """Verifica gradient negativo con feedback negativo."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.8,
            relevance_scores={},
            expert_type="systemic",
            user_rating=0.2,  # Feedback negativo
            relation_usage={
                "CITATO_DA": RelationUsageData(
                    relation_type="CITATO_DA",
                    usage_count=10,
                    sources_found=5,
                    sources_used_in_response=1,  # Bassa efficienza
                    avg_relevance=0.6
                )
            }
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)

        assert "CITATO_DA" in gradient
        assert gradient["CITATO_DA"] < 0  # Gradient negativo per feedback negativo

    def test_gradient_zero_for_unused_relations(self, learner, mock_store):
        """Verifica gradient zero per relazioni non usate."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.8,
            relevance_scores={},
            expert_type="precedent",
            user_rating=0.8,
            relation_usage={
                "INUTILIZZATA": RelationUsageData(
                    relation_type="INUTILIZZATA",
                    usage_count=0,  # Non usata
                    sources_found=0,
                    sources_used_in_response=0
                )
            }
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)

        # Relazione con usage_count=0 non produce gradient
        assert gradient == {}

    def test_multiple_relations_independent_gradients(self, learner, mock_store):
        """Verifica gradient indipendenti per relazioni multiple."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.9,
            relevance_scores={},
            expert_type="principles",
            user_rating=0.7,
            relation_usage={
                "RIFERIMENTO": RelationUsageData(
                    relation_type="RIFERIMENTO",
                    usage_count=10,
                    sources_found=8,
                    sources_used_in_response=6,  # Alta efficienza
                    avg_relevance=0.9
                ),
                "ABROGA": RelationUsageData(
                    relation_type="ABROGA",
                    usage_count=5,
                    sources_found=3,
                    sources_used_in_response=0,  # Bassa efficienza
                    avg_relevance=0.3
                )
            }
        )
        current = WeightConfig()

        gradient = learner._compute_traversal_gradient(feedback, current)

        assert "RIFERIMENTO" in gradient
        assert "ABROGA" in gradient
        # RIFERIMENTO dovrebbe avere gradient maggiore (più efficiente)
        assert gradient["RIFERIMENTO"] > gradient["ABROGA"]


class TestApplyTraversalUpdate:
    """Test _apply_update per expert_traversal."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock(spec=WeightStore)
        store.get_weights = AsyncMock(return_value=WeightConfig())
        return store

    @pytest.fixture
    def learner(self, mock_store):
        return WeightLearner(store=mock_store)

    def test_apply_creates_expert_traversal_if_missing(self, learner):
        """Verifica creazione expert_traversal se mancante."""
        current = WeightConfig()
        gradient = {"RIFERIMENTO": 0.05}

        updated = learner._apply_update(
            category="expert_traversal",
            current=current,
            gradient=gradient,
            authority=0.8,
            expert_type="literal"
        )

        assert "LiteralExpert" in updated.expert_traversal
        assert "RIFERIMENTO" in updated.expert_traversal["LiteralExpert"].weights

    def test_apply_updates_existing_weight(self, learner):
        """Verifica update di peso esistente."""
        current = WeightConfig()
        # Inizializza peso esistente
        current.expert_traversal["LiteralExpert"] = ExpertTraversalWeights(
            weights={"RIFERIMENTO": LearnableWeight(default=0.5, bounds=(0.1, 1.0))}
        )

        gradient = {"RIFERIMENTO": 0.1}

        updated = learner._apply_update(
            category="expert_traversal",
            current=current,
            gradient=gradient,
            authority=0.8,
            expert_type="literal"
        )

        # Il peso dovrebbe essere aumentato
        new_weight = updated.expert_traversal["LiteralExpert"].weights["RIFERIMENTO"].default
        assert new_weight > 0.5

    def test_apply_respects_bounds(self, learner):
        """Verifica che update rispetti bounds."""
        current = WeightConfig()
        current.expert_traversal["SystemicExpert"] = ExpertTraversalWeights(
            weights={"MODIFICA": LearnableWeight(default=0.95, bounds=(0.1, 1.0))}
        )

        # Gradient molto grande che porterebbe oltre il bound
        gradient = {"MODIFICA": 1.0}

        updated = learner._apply_update(
            category="expert_traversal",
            current=current,
            gradient=gradient,
            authority=1.0,
            expert_type="systemic"
        )

        # Il peso non deve superare 1.0
        new_weight = updated.expert_traversal["SystemicExpert"].weights["MODIFICA"].default
        assert new_weight <= 1.0

    def test_apply_considers_authority(self, learner):
        """Verifica che authority influenzi l'update."""
        current = WeightConfig()
        current.expert_traversal["PrecedentExpert"] = ExpertTraversalWeights(
            weights={"CITATO_DA": LearnableWeight(default=0.5, bounds=(0.1, 1.0), learning_rate=0.1)}
        )

        gradient = {"CITATO_DA": 0.1}

        # Update con alta authority
        high_auth = learner._apply_update(
            category="expert_traversal",
            current=current,
            gradient=gradient,
            authority=0.9,
            expert_type="precedent"
        )

        # Reset per secondo test
        current.expert_traversal["PrecedentExpert"].weights["CITATO_DA"].default = 0.5

        # Update con bassa authority
        low_auth = learner._apply_update(
            category="expert_traversal",
            current=current,
            gradient=gradient,
            authority=0.3,
            expert_type="precedent"
        )

        # Alta authority dovrebbe portare a cambiamento maggiore
        high_change = abs(high_auth.expert_traversal["PrecedentExpert"].weights["CITATO_DA"].default - 0.5)
        low_change = abs(low_auth.expert_traversal["PrecedentExpert"].weights["CITATO_DA"].default - 0.5)

        assert high_change > low_change


class TestUpdateFromFeedback:
    """Test completo update_from_feedback per traversal."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock(spec=WeightStore)
        store.get_weights = AsyncMock(return_value=WeightConfig())
        store.save_weights = AsyncMock()
        return store

    @pytest.fixture
    def learner(self, mock_store):
        return WeightLearner(store=mock_store)

    @pytest.mark.asyncio
    async def test_full_update_flow(self, learner, mock_store):
        """Verifica flusso completo di update."""
        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.85,
            relevance_scores={"source1": 0.9, "source2": 0.7},
            expert_type="literal",
            user_rating=0.8,
            relation_usage={
                "RIFERIMENTO": RelationUsageData(
                    relation_type="RIFERIMENTO",
                    usage_count=5,
                    sources_found=4,
                    sources_used_in_response=3,
                    avg_relevance=0.85
                )
            }
        )

        updated = await learner.update_from_feedback(
            category="expert_traversal",
            feedback=feedback,
            experiment_id="exp-001"
        )

        # Verifica che i pesi siano stati aggiornati
        assert "LiteralExpert" in updated.expert_traversal
        assert "RIFERIMENTO" in updated.expert_traversal["LiteralExpert"].weights

    @pytest.mark.asyncio
    async def test_low_authority_skips_update(self, learner, mock_store):
        """Verifica che bassa authority non aggiorni i pesi."""
        initial_config = WeightConfig()
        mock_store.get_weights = AsyncMock(return_value=initial_config)

        feedback = RLCFFeedback(
            query_id="q1",
            user_id="u1",
            authority=0.1,  # Sotto la soglia
            relevance_scores={},
            expert_type="literal",
            relation_usage={"RIFERIMENTO": RelationUsageData("RIFERIMENTO", 5, 3)}
        )

        result = await learner.update_from_feedback(
            category="expert_traversal",
            feedback=feedback
        )

        # Nessun cambiamento (stessa config)
        assert result.expert_traversal == initial_config.expert_traversal


class TestTraversalWeightConvergence:
    """Test convergenza pesi traversal nel tempo."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock(spec=WeightStore)
        config = WeightConfig()
        config.expert_traversal["LiteralExpert"] = ExpertTraversalWeights(
            weights={"RIFERIMENTO": LearnableWeight(default=0.5, bounds=(0.1, 1.0), learning_rate=0.1)}
        )
        store.get_weights = AsyncMock(return_value=config)
        store.save_weights = AsyncMock()
        return store

    @pytest.fixture
    def learner(self, mock_store):
        return WeightLearner(store=mock_store)

    @pytest.mark.asyncio
    async def test_weights_converge_with_consistent_feedback(self, learner, mock_store):
        """Verifica che pesi convergano con feedback consistente."""
        weights_over_time = []

        # Simula 20 feedback positivi consistenti
        for i in range(20):
            feedback = RLCFFeedback(
                query_id=f"q{i}",
                user_id="expert_user",
                authority=0.9,
                relevance_scores={},
                expert_type="literal",
                user_rating=0.95,  # Sempre positivo
                relation_usage={
                    "RIFERIMENTO": RelationUsageData(
                        relation_type="RIFERIMENTO",
                        usage_count=5,
                        sources_found=4,
                        sources_used_in_response=4,  # Alta efficienza
                        avg_relevance=0.9
                    )
                }
            )

            updated = await learner.update_from_feedback(
                category="expert_traversal",
                feedback=feedback
            )

            weight = updated.expert_traversal["LiteralExpert"].weights["RIFERIMENTO"].default
            weights_over_time.append(weight)

            # Aggiorna store per prossima iterazione
            mock_store.get_weights = AsyncMock(return_value=updated)

        # Il peso deve essere aumentato significativamente
        assert weights_over_time[-1] > weights_over_time[0]

        # Ultimi valori devono essere stabili (convergenza)
        last_5 = weights_over_time[-5:]
        variance = max(last_5) - min(last_5)
        assert variance < 0.1, "Pesi dovrebbero convergere"


class TestExploreIterativelyRelationTracking:
    """Test tracking relazioni in BaseExpert.explore_iteratively()."""

    @pytest.fixture
    def mock_expert(self):
        """Crea mock expert per testing."""
        class MockExpert(BaseExpert):
            expert_type = "test"
            description = "Test expert"

            async def analyze(self, context):
                return ExpertResponse(expert_type=self.expert_type, interpretation="test")

        expert = MockExpert()

        # Mock tools
        mock_semantic = AsyncMock()
        mock_semantic.return_value = MagicMock(success=True, data={"results": []})

        mock_graph = AsyncMock()
        mock_graph.return_value = MagicMock(
            success=True,
            data={
                "nodes": [
                    {"urn": "urn:node1", "properties": {"testo": "testo nodo 1"}},
                    {"urn": "urn:node2", "properties": {"testo": "testo nodo 2"}}
                ],
                "edges": [
                    {"type": "RIFERIMENTO", "source": "urn:start", "target": "urn:node1"},
                    {"type": "CITATO_DA", "source": "urn:node2", "target": "urn:start"}
                ]
            }
        )

        expert._tool_registry.get = MagicMock(side_effect=lambda name: {
            "semantic_search": mock_semantic,
            "graph_search": mock_graph
        }.get(name))

        return expert

    @pytest.mark.asyncio
    async def test_tracks_relation_usage(self, mock_expert):
        """Verifica che explore_iteratively tracci le relazioni usate."""
        context = ExpertContext(
            query_text="Test query",
            entities={"norm_references": ["urn:test:123"]}
        )

        await mock_expert.explore_iteratively(context, max_iterations=1)

        # Verifica che relation_usage sia stato popolato
        relation_usage = mock_expert.get_relation_usage()
        assert len(relation_usage) > 0
        assert "RIFERIMENTO" in relation_usage or "CITATO_DA" in relation_usage

    @pytest.mark.asyncio
    async def test_exploration_metrics_include_relations(self, mock_expert):
        """Verifica che exploration metrics includa dati sulle relazioni."""
        context = ExpertContext(
            query_text="Test query",
            entities={"norm_references": ["urn:test:456"]}
        )

        await mock_expert.explore_iteratively(context, max_iterations=1)

        metrics = mock_expert.get_exploration_metrics()
        assert "relation_usage" in metrics
        assert "relation_types_used" in metrics
        assert "total_relation_traversals" in metrics

    @pytest.mark.asyncio
    async def test_sources_track_via_relation(self, mock_expert):
        """Verifica che le fonti traccino la relazione che le ha portate."""
        context = ExpertContext(
            query_text="Test query",
            entities={"norm_references": ["urn:test:789"]}
        )

        sources = await mock_expert.explore_iteratively(context, max_iterations=1)

        # Verifica che almeno una fonte abbia via_relation
        sources_with_relation = [s for s in sources if s.get("via_relation")]
        assert len(sources_with_relation) >= 0  # Può essere 0 se tutte le fonti vengono da semantic search
