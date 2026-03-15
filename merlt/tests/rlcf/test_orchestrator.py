"""Tests for RLCFOrchestrator."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.rlcf.orchestrator import (
    RLCFOrchestrator,
    ExpertFeedbackRecord,
    get_orchestrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global orchestrator singleton between tests."""
    import merlt.rlcf.orchestrator as mod
    mod._orchestrator_instance = None
    yield
    mod._orchestrator_instance = None


def _make_response(
    trace_id="trace_abc123",
    interpretation="risposta di test",
    confidence=0.7,
    legal_basis=None,
):
    resp = MagicMock()
    resp.trace_id = trace_id
    resp.interpretation = interpretation
    resp.confidence = confidence
    resp.legal_basis = legal_basis or ["art.1", "art.2"]
    return resp


def _make_db_row(obj_id=1):
    """Return a mock DB row-like object with .id attribute."""
    obj = MagicMock()
    obj.id = obj_id
    return obj


def _make_orchestrator(min_authority=0.3):
    db = AsyncMock()
    store = MagicMock()
    learner = AsyncMock()
    orch = RLCFOrchestrator(
        db_session=db,
        weight_store=store,
        weight_learner=learner,
        min_authority_for_update=min_authority,
    )
    return orch, db, store, learner


# ---------------------------------------------------------------------------
# ExpertFeedbackRecord
# ---------------------------------------------------------------------------


class TestExpertFeedbackRecord:
    def test_defaults(self):
        rec = ExpertFeedbackRecord(
            trace_id="t1",
            expert_type="literal",
            user_rating=0.8,
        )
        assert rec.feedback_type == "accuracy"
        assert rec.interpretation == ""
        assert rec.sources_cited == 0
        assert rec.confidence == 0.5
        assert rec.feedback_details == {}

    def test_custom_fields(self):
        rec = ExpertFeedbackRecord(
            trace_id="t2",
            expert_type="systemic",
            user_rating=0.5,
            feedback_type="utility",
            sources_cited=3,
            confidence=0.9,
            feedback_details={"extra": "data"},
        )
        assert rec.feedback_type == "utility"
        assert rec.sources_cited == 3
        assert rec.confidence == 0.9
        assert rec.feedback_details == {"extra": "data"}


# ---------------------------------------------------------------------------
# RLCFOrchestrator – init
# ---------------------------------------------------------------------------


class TestRLCFOrchestratorInit:
    def test_attributes_stored(self):
        orch, db, store, learner = _make_orchestrator(min_authority=0.4)
        assert orch.db is db
        assert orch.store is store
        assert orch.learner is learner
        assert orch.min_authority == 0.4


# ---------------------------------------------------------------------------
# RLCFOrchestrator.record_expert_feedback
# ---------------------------------------------------------------------------


class TestRecordExpertFeedback:
    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        orch, db, store, learner = _make_orchestrator()

        task = _make_db_row(10)
        db_response = _make_db_row(20)
        feedback = _make_db_row(30)

        with (
            patch.object(orch, "_get_or_create_task", AsyncMock(return_value=task)),
            patch.object(orch, "_create_response", AsyncMock(return_value=db_response)),
            patch.object(orch, "_create_feedback", AsyncMock(return_value=feedback)),
            patch.object(orch, "_trigger_weight_update", AsyncMock(return_value=True)),
        ):
            resp = _make_response()
            result = await orch.record_expert_feedback(
                expert_type="literal",
                response=resp,
                user_rating=0.8,
            )

        assert result["feedback_id"] == 30
        assert result["response_id"] == 20
        assert result["task_id"] == 10
        assert result["expert_type"] == "literal"
        assert "authority_used" in result
        assert "weights_updated" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_anonymous_feedback_uses_default_authority(self):
        orch, db, store, learner = _make_orchestrator()
        task = _make_db_row(1)
        db_response = _make_db_row(2)
        feedback = _make_db_row(3)

        with (
            patch.object(orch, "_get_or_create_task", AsyncMock(return_value=task)),
            patch.object(orch, "_create_response", AsyncMock(return_value=db_response)),
            patch.object(orch, "_create_feedback", AsyncMock(return_value=feedback)),
            patch.object(orch, "_trigger_weight_update", AsyncMock(return_value=True)),
        ):
            result = await orch.record_expert_feedback(
                expert_type="systemic",
                response=_make_response(),
                user_rating=0.9,
                user_id=None,
            )

        # anonymous user gets default authority 0.5
        assert result["authority_used"] == 0.5

    @pytest.mark.asyncio
    async def test_with_user_id_calls_update_authority(self):
        orch, db, store, learner = _make_orchestrator()
        task = _make_db_row(1)
        db_response = _make_db_row(2)
        feedback = _make_db_row(3)

        with (
            patch.object(orch, "_get_or_create_task", AsyncMock(return_value=task)),
            patch.object(orch, "_create_response", AsyncMock(return_value=db_response)),
            patch.object(orch, "_create_feedback", AsyncMock(return_value=feedback)),
            patch.object(orch, "_update_user_authority", AsyncMock(return_value=0.75)),
            patch.object(orch, "_trigger_weight_update", AsyncMock(return_value=True)),
        ):
            result = await orch.record_expert_feedback(
                expert_type="principles",
                response=_make_response(),
                user_rating=1.0,
                user_id=42,
            )

        assert result["authority_used"] == 0.75

    @pytest.mark.asyncio
    async def test_weights_not_updated_when_authority_below_min(self):
        orch, db, store, learner = _make_orchestrator(min_authority=0.6)
        task = _make_db_row(1)
        db_response = _make_db_row(2)
        feedback = _make_db_row(3)

        mock_trigger = AsyncMock(return_value=True)

        with (
            patch.object(orch, "_get_or_create_task", AsyncMock(return_value=task)),
            patch.object(orch, "_create_response", AsyncMock(return_value=db_response)),
            patch.object(orch, "_create_feedback", AsyncMock(return_value=feedback)),
            # authority 0.4 is below min_authority 0.6
            patch.object(orch, "_update_user_authority", AsyncMock(return_value=0.4)),
            patch.object(orch, "_trigger_weight_update", mock_trigger),
        ):
            result = await orch.record_expert_feedback(
                expert_type="precedent",
                response=_make_response(),
                user_rating=0.5,
                user_id=7,
            )

        mock_trigger.assert_not_awaited()
        assert result["weights_updated"] is False

    @pytest.mark.asyncio
    async def test_response_attributes_extracted_via_getattr(self):
        """Fallback to defaults when response lacks attributes."""
        orch, db, store, learner = _make_orchestrator()
        task = _make_db_row(1)
        db_response = _make_db_row(2)
        feedback = _make_db_row(3)

        bare_response = object()  # no attributes at all

        with (
            patch.object(orch, "_get_or_create_task", AsyncMock(return_value=task)),
            patch.object(orch, "_create_response", AsyncMock(return_value=db_response)),
            patch.object(orch, "_create_feedback", AsyncMock(return_value=feedback)),
            patch.object(orch, "_trigger_weight_update", AsyncMock(return_value=False)),
        ):
            result = await orch.record_expert_feedback(
                expert_type="literal",
                response=bare_response,
                user_rating=0.5,
            )

        assert "feedback_id" in result


# ---------------------------------------------------------------------------
# RLCFOrchestrator._trigger_weight_update
# ---------------------------------------------------------------------------


class TestTriggerWeightUpdate:
    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        orch, db, store, learner = _make_orchestrator()
        orch.learner.update_from_feedback = AsyncMock()

        result = await orch._trigger_weight_update(
            expert_type="literal",
            user_rating=0.9,
            authority=0.8,
            trace_id="trace_xyz",
        )

        assert result is True
        orch.learner.update_from_feedback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        orch, db, store, learner = _make_orchestrator()
        orch.learner.update_from_feedback = AsyncMock(side_effect=RuntimeError("fail"))

        result = await orch._trigger_weight_update(
            expert_type="systemic",
            user_rating=0.5,
            authority=0.7,
            trace_id="trace_err",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_rlcf_feedback_built_correctly(self):
        orch, db, store, learner = _make_orchestrator()
        captured = {}

        async def capture_update(category, feedback, experiment_id):
            captured["category"] = category
            captured["feedback"] = feedback
            captured["experiment_id"] = experiment_id

        orch.learner.update_from_feedback = capture_update

        await orch._trigger_weight_update(
            expert_type="precedent",
            user_rating=0.75,
            authority=0.6,
            trace_id="trace_capture",
        )

        assert captured["category"] == "expert_traversal"
        assert captured["experiment_id"] == "expert_precedent"
        fb = captured["feedback"]
        assert fb.query_id == "trace_capture"
        assert fb.authority == 0.6
        assert fb.relevance_scores == {"precedent": 0.75}


# ---------------------------------------------------------------------------
# RLCFOrchestrator._update_user_authority
# ---------------------------------------------------------------------------


class TestUpdateUserAuthority:
    @pytest.mark.asyncio
    async def test_returns_authority_on_success(self):
        orch, db, store, learner = _make_orchestrator()
        feedback = MagicMock()

        with (
            patch(
                "merlt.rlcf.orchestrator.calculate_quality_score",
                AsyncMock(return_value=0.7),
            ),
            patch("merlt.rlcf.orchestrator.update_track_record", AsyncMock()),
            patch(
                "merlt.rlcf.orchestrator.update_authority_score",
                AsyncMock(return_value=0.65),
            ),
        ):
            result = await orch._update_user_authority(user_id=5, feedback=feedback)

        assert result == 0.65

    @pytest.mark.asyncio
    async def test_returns_default_on_exception(self):
        orch, db, store, learner = _make_orchestrator()
        feedback = MagicMock()

        with patch(
            "merlt.rlcf.orchestrator.calculate_quality_score",
            AsyncMock(side_effect=Exception("db error")),
        ):
            result = await orch._update_user_authority(user_id=99, feedback=feedback)

        assert result == 0.5


# ---------------------------------------------------------------------------
# RLCFOrchestrator.get_expert_feedback_stats
# ---------------------------------------------------------------------------


class TestGetExpertFeedbackStats:
    @pytest.mark.asyncio
    async def test_returns_stats_dict_structure(self):
        orch, db, store, learner = _make_orchestrator()

        row = MagicMock()
        row.total_feedback = 5
        row.avg_accuracy = 3.5
        row.avg_utility = 4.0

        execute_result = MagicMock()
        execute_result.first.return_value = row
        db.execute = AsyncMock(return_value=execute_result)

        # Pass expert_type=None to avoid PostgreSQL JSONB .astext path
        # which requires a real DB connection
        result = await orch.get_expert_feedback_stats(days=7)

        assert result["expert_type"] == "all"
        assert result["period_days"] == 7
        assert result["total_feedback"] == 5
        assert result["avg_accuracy"] == 3.5
        assert result["avg_utility"] == 4.0
        assert "queried_at" in result

    @pytest.mark.asyncio
    async def test_all_experts_when_none_specified(self):
        orch, db, store, learner = _make_orchestrator()

        row = MagicMock()
        row.total_feedback = 0
        row.avg_accuracy = None
        row.avg_utility = None

        execute_result = MagicMock()
        execute_result.first.return_value = row
        db.execute = AsyncMock(return_value=execute_result)

        result = await orch.get_expert_feedback_stats()

        assert result["expert_type"] == "all"
        assert result["avg_accuracy"] == 0.0
        assert result["avg_utility"] == 0.0

    @pytest.mark.asyncio
    async def test_handles_zero_row(self):
        """DB returns a row with None averages (empty aggregation).
        total_feedback=0 guard works; avg fields fall back to 0 via `or 0`."""
        orch, db, store, learner = _make_orchestrator()

        row = MagicMock()
        row.total_feedback = 0
        row.avg_accuracy = None
        row.avg_utility = None

        execute_result = MagicMock()
        execute_result.first.return_value = row
        db.execute = AsyncMock(return_value=execute_result)

        result = await orch.get_expert_feedback_stats()

        assert result["total_feedback"] == 0
        assert result["avg_accuracy"] == 0.0
        assert result["avg_utility"] == 0.0


# ---------------------------------------------------------------------------
# get_orchestrator singleton
# ---------------------------------------------------------------------------


class TestGetOrchestratorSingleton:
    @pytest.mark.asyncio
    async def test_creates_instance_when_none(self):
        db = AsyncMock()
        store = MagicMock()
        learner = AsyncMock()

        instance = await get_orchestrator(db, store, learner)

        assert isinstance(instance, RLCFOrchestrator)

    @pytest.mark.asyncio
    async def test_returns_same_instance_on_second_call(self):
        db = AsyncMock()
        store = MagicMock()
        learner = AsyncMock()

        i1 = await get_orchestrator(db, store, learner)
        i2 = await get_orchestrator(db, store, learner)

        assert i1 is i2

    @pytest.mark.asyncio
    async def test_creates_default_store_and_learner_when_none(self):
        db = AsyncMock()

        with (
            patch("merlt.rlcf.orchestrator.WeightStore") as MockStore,
            patch("merlt.rlcf.orchestrator.WeightLearner") as MockLearner,
        ):
            MockStore.return_value = MagicMock()
            MockLearner.return_value = AsyncMock()

            instance = await get_orchestrator(db)

        assert isinstance(instance, RLCFOrchestrator)
        MockStore.assert_called_once()
        MockLearner.assert_called_once()
