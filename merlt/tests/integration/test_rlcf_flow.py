"""
RLCF Integration Tests: Feedback → Aggregation → Training Flow
================================================================

Tests the real wiring of the RLCF pipeline against PostgreSQL.
All tests are marked @pytest.mark.integration and excluded by default.

Run with:
    pytest tests/integration/test_rlcf_flow.py -v -m integration

Requirements:
    Docker services running (PostgreSQL on port 5433)
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from merlt.experts.models import QATrace, QAFeedback, AggregatedFeedback
from merlt.rlcf.feedback_aggregation_service import (
    FeedbackAggregationService,
    AggregatedTraceResult,
    AggregationRunResult,
)
from merlt.rlcf.traversal_training_service import TraversalTrainingService
from merlt.storage.bridge.models import BridgeTableEntry


def _make_trace_id() -> str:
    return f"trace_{uuid4().hex[:12]}"


# =============================================================================
# TEST INSERT AND AGGREGATE FEEDBACK
# =============================================================================


@pytest.mark.integration
class TestInsertAndAggregateFeedback:
    """Insert real QATrace + QAFeedback, then aggregate."""

    @pytest.mark.asyncio
    async def test_insert_and_aggregate_feedback(self, rlcf_session: AsyncSession):
        """Insert trace + feedback → aggregate → verify averages."""
        trace_id = _make_trace_id()

        # Insert a trace
        trace = QATrace(
            trace_id=trace_id,
            user_id="test_user_001",
            query="Cos'è la legittima difesa?",
            selected_experts=["literal", "systemic"],
            synthesis_mode="convergent",
            synthesis_text="La legittima difesa è...",
            consent_level="full",
            query_type="definitional",
            confidence=0.85,
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        # Insert feedback
        fb1 = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_001",
            inline_rating=5,
            user_authority=0.9,
        )
        fb2 = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_002",
            inline_rating=3,
            user_authority=0.5,
        )
        rlcf_session.add_all([fb1, fb2])
        await rlcf_session.flush()

        # Aggregate
        svc = FeedbackAggregationService()
        result = await svc.aggregate_trace_feedback(rlcf_session, trace_id)

        assert isinstance(result, AggregatedTraceResult)
        assert result.trace_id == trace_id
        assert result.total_feedback == 2
        assert result.avg_inline_rating == pytest.approx(4.0)
        # Authority-weighted: (5*0.9 + 3*0.5) / (0.9+0.5) = 6.0/1.4 ≈ 4.286
        assert result.authority_weighted_inline == pytest.approx(6.0 / 1.4, rel=1e-3)

    @pytest.mark.asyncio
    async def test_aggregate_empty_trace(self, rlcf_session: AsyncSession):
        """Non-existent trace → empty aggregation."""
        svc = FeedbackAggregationService()
        result = await svc.aggregate_trace_feedback(rlcf_session, "nonexistent_trace")

        assert result.total_feedback == 0
        assert result.avg_inline_rating is None

    @pytest.mark.asyncio
    async def test_detailed_feedback_aggregation(self, rlcf_session: AsyncSession):
        """Insert detailed feedback → verify dimension averages."""
        trace_id = _make_trace_id()

        trace = QATrace(
            trace_id=trace_id,
            user_id="test_user_002",
            query="Interpretazione art. 2043 c.c.",
            consent_level="basic",
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        fb = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_003",
            retrieval_score=0.8,
            reasoning_score=0.9,
            synthesis_score=0.7,
        )
        rlcf_session.add(fb)
        await rlcf_session.flush()

        svc = FeedbackAggregationService()
        result = await svc.aggregate_trace_feedback(rlcf_session, trace_id)

        assert result.total_feedback == 1
        assert result.avg_retrieval_score == pytest.approx(0.8)
        assert result.avg_reasoning_score == pytest.approx(0.9)
        assert result.avg_synthesis_score == pytest.approx(0.7)


# =============================================================================
# TEST COMPONENT AGGREGATION
# =============================================================================


@pytest.mark.integration
class TestComponentAggregation:
    """Aggregate feedback filtered by component type."""

    @pytest.mark.asyncio
    async def test_synthesizer_component(self, rlcf_session: AsyncSession):
        """Feedback with synthesis_score → aggregated under 'synthesizer'."""
        trace_id = _make_trace_id()

        trace = QATrace(
            trace_id=trace_id,
            user_id="user_comp_001",
            query="Test component aggregation",
            consent_level="basic",
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        fb = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_comp",
            synthesis_score=0.85,
            user_authority=0.7,
        )
        rlcf_session.add(fb)
        await rlcf_session.flush()

        svc = FeedbackAggregationService()
        result = await svc.aggregate_component_feedback(
            rlcf_session,
            "synthesizer",
            since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
        )

        assert result.component == "synthesizer"
        assert result.total_feedback == 1
        assert result.avg_rating == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_bridge_component(self, rlcf_session: AsyncSession):
        """Feedback with source_id → aggregated under 'bridge'."""
        trace_id = _make_trace_id()

        trace = QATrace(
            trace_id=trace_id,
            user_id="user_bridge_001",
            query="Test bridge aggregation",
            consent_level="basic",
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        fb = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_bridge",
            source_id="urn:nir:stato:codice.civile:1942;art1453",
            source_relevance=4,
            user_authority=0.8,
        )
        rlcf_session.add(fb)
        await rlcf_session.flush()

        svc = FeedbackAggregationService()
        result = await svc.aggregate_component_feedback(
            rlcf_session,
            "bridge",
            since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
        )

        assert result.component == "bridge"
        assert result.total_feedback == 1
        # source_relevance=4 → normalized = (4-1)/4 = 0.75
        assert result.avg_rating == pytest.approx(0.75)


# =============================================================================
# TEST PERIODIC AGGREGATION
# =============================================================================


@pytest.mark.integration
class TestPeriodicAggregation:
    """Test run_periodic_aggregation across all components."""

    @pytest.mark.asyncio
    async def test_periodic_aggregation_real_db(self, rlcf_session: AsyncSession):
        """Insert mixed feedback → run_periodic_aggregation → verify counts."""
        trace_id = _make_trace_id()

        trace = QATrace(
            trace_id=trace_id,
            user_id="user_periodic",
            query="Test periodic aggregation",
            consent_level="full",
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        # Synthesizer feedback
        fb1 = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_p1",
            synthesis_score=0.9,
            user_authority=0.8,
        )
        # Source/bridge feedback
        fb2 = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_p2",
            source_id="urn:art52cp",
            source_relevance=5,
            user_authority=0.6,
        )
        rlcf_session.add_all([fb1, fb2])
        await rlcf_session.flush()

        svc = FeedbackAggregationService()
        result = await svc.run_periodic_aggregation(
            rlcf_session,
            since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
        )

        assert isinstance(result, AggregationRunResult)
        assert result.components_aggregated >= 2  # at least synthesizer + bridge
        assert result.total_feedback_processed >= 2


# =============================================================================
# TEST AFFINITY UPDATE
# =============================================================================


@pytest.mark.integration
class TestAffinityUpdate:
    """Test AffinityUpdateService with real BridgeTableEntry."""

    @pytest.mark.asyncio
    async def test_affinity_update_real_db(self, rlcf_session: AsyncSession):
        """Insert BridgeTableEntry + source feedback → verify affinity updated."""
        from merlt.rlcf.affinity_service import AffinityUpdateService

        trace_id = _make_trace_id()
        source_urn = "urn:nir:stato:codice.penale:1930;art52"

        # Insert bridge entry
        bridge_entry = BridgeTableEntry(
            chunk_id=uuid4(),
            graph_node_urn=source_urn,
            node_type="Norma",
            relation_type="contained_in",
            confidence=0.9,
            source="test",
        )
        rlcf_session.add(bridge_entry)
        await rlcf_session.flush()

        # Insert trace with expert_results referencing the source
        trace = QATrace(
            trace_id=trace_id,
            user_id="user_affinity",
            query="Legittima difesa art. 52 c.p.",
            selected_experts=["literal", "systemic"],
            consent_level="full",
            full_trace={
                "expert_results": {
                    "literal": {"sources": [{"source_id": source_urn}]},
                }
            },
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        # Insert source feedback (5 stars = very relevant)
        feedback = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_affinity",
            source_id=source_urn,
            source_relevance=5,
            user_authority=0.9,
        )
        rlcf_session.add(feedback)
        await rlcf_session.flush()

        # Run affinity update
        svc = AffinityUpdateService()
        updated = await svc.update_from_source_feedback(
            rlcf_session, trace, feedback
        )

        assert updated is not None
        # literal expert should have increased affinity (from default 0.5)
        # new = 0.5 + 0.3 * (1.0 - 0.5) = 0.65
        assert updated["literal"] == pytest.approx(0.65, rel=1e-2)
        # Other experts should remain at default (0.5)
        assert updated["systemic"] == pytest.approx(0.5)


# =============================================================================
# TEST TRAVERSAL TRAINING DATA
# =============================================================================


@pytest.mark.integration
class TestTraversalTrainingData:
    """Test TraversalTrainingService.prepare_training_data with real DB."""

    @pytest.mark.asyncio
    async def test_prepare_training_data_real_db(self, rlcf_session: AsyncSession):
        """Insert trace + source feedback → prepare_training_data → verify samples."""
        trace_id = _make_trace_id()
        source_urn = "urn:nir:stato:codice.civile:1942;art1453"

        trace = QATrace(
            trace_id=trace_id,
            user_id="user_traversal",
            query="Risoluzione per inadempimento",
            selected_experts=["literal"],
            consent_level="full",
            full_trace={
                "query_embedding": [0.1] * 1024,
                "graph_traversal": {
                    "paths": [
                        {
                            "target_urn": source_urn,
                            "edges": [{"relation_type": "RIFERIMENTO"}],
                        }
                    ]
                },
                "expert_results": {
                    "literal": {"sources": [{"source_id": source_urn}]},
                },
            },
        )
        rlcf_session.add(trace)
        await rlcf_session.flush()

        feedback = QAFeedback(
            trace_id=trace_id,
            user_id="reviewer_traversal",
            source_id=source_urn,
            source_relevance=5,
        )
        rlcf_session.add(feedback)
        await rlcf_session.flush()

        svc = TraversalTrainingService()
        samples = await svc.prepare_training_data(
            rlcf_session,
            since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
        )

        assert len(samples) >= 1
        sample = samples[0]
        assert sample.relation_type == "RIFERIMENTO"
        assert sample.expert_type == "literal"
        assert sample.reward == pytest.approx(1.0)  # (5-1)/4 = 1.0
        # Should use real embedding from trace
        assert len(sample.query_embedding) == 1024
        assert sample.query_embedding[0] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_prepare_training_data_no_feedback(self, rlcf_session: AsyncSession):
        """No source feedback → empty samples."""
        svc = TraversalTrainingService()
        samples = await svc.prepare_training_data(
            rlcf_session,
            since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
        )
        assert samples == []
