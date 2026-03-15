"""Tests for ReproducibilityService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.rlcf.reproducibility_service import (
    ReproducibilityService,
    ReproducibilityResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    trace_id="trace_abc",
    query="Cos'è la responsabilità contrattuale?",
    selected_experts=None,
    synthesis_mode="convergent",
    confidence=0.8,
    routing_method="neural",
    query_type="definitional",
    sources=None,
    full_trace=None,
):
    t = MagicMock()
    t.trace_id = trace_id
    t.query = query
    t.selected_experts = selected_experts or ["literal", "systemic"]
    t.synthesis_mode = synthesis_mode
    t.confidence = confidence
    t.routing_method = routing_method
    t.query_type = query_type
    t.sources = sources
    t.full_trace = full_trace
    t.user_id = "user_test"
    t.consent_level = "basic"
    return t


# ---------------------------------------------------------------------------
# ReproducibilityResult
# ---------------------------------------------------------------------------


class TestReproducibilityResult:
    def test_to_dict_returns_dict(self):
        r = ReproducibilityResult(
            original_trace_id="t1",
            reproduced_trace_id="t2",
            config_used={"key": "val"},
            diff={"expert_overlap": 1.0},
            reproducibility_score=0.9,
            caveats=["caveat1"],
        )
        d = r.to_dict()
        assert d["original_trace_id"] == "t1"
        assert d["reproduced_trace_id"] == "t2"
        assert d["reproducibility_score"] == 0.9
        assert d["caveats"] == ["caveat1"]


# ---------------------------------------------------------------------------
# ReproducibilityService._extract_config
# ---------------------------------------------------------------------------


class TestExtractConfig:
    def test_extracts_basic_fields(self):
        svc = ReproducibilityService()
        trace = _make_trace(
            selected_experts=["literal"],
            synthesis_mode="divergent",
            confidence=0.75,
            routing_method="llm_fallback",
            query_type="interpretive",
        )
        trace.full_trace = None

        config = svc._extract_config(trace)

        assert config["selected_experts"] == ["literal"]
        assert config["synthesis_mode"] == "divergent"
        assert config["confidence"] == 0.75
        assert config["routing_method"] == "llm_fallback"
        assert config["query_type"] == "interpretive"

    def test_extracts_model_versions_from_full_trace(self):
        svc = ReproducibilityService()
        trace = _make_trace(
            full_trace={
                "model_versions": {"literal": "v1.2"},
                "routing": {"gating_weights": {"literal": 0.4}},
            }
        )
        config = svc._extract_config(trace)

        assert config["model_versions"] == {"literal": "v1.2"}
        assert config["gating_weights"] == {"literal": 0.4}

    def test_no_full_trace_skips_model_versions(self):
        svc = ReproducibilityService()
        trace = _make_trace(full_trace=None)
        config = svc._extract_config(trace)
        assert "model_versions" not in config
        assert "gating_weights" not in config


# ---------------------------------------------------------------------------
# ReproducibilityService._compute_diff
# ---------------------------------------------------------------------------


class TestComputeDiff:
    def test_identical_traces(self):
        svc = ReproducibilityService()
        t = _make_trace(
            selected_experts=["literal", "systemic"],
            synthesis_mode="convergent",
            confidence=0.8,
            sources=[],
        )
        result_obj = MagicMock()
        result_obj.combined_legal_basis = []

        diff = svc._compute_diff(t, t, result_obj)

        assert diff["expert_overlap"] == 1.0
        assert diff["confidence_delta"] == 0.0
        assert diff["mode_match"] is True

    def test_different_experts_overlap(self):
        svc = ReproducibilityService()
        orig = _make_trace(selected_experts=["literal", "systemic"])
        repro = _make_trace(selected_experts=["literal", "precedent"])
        result_obj = MagicMock()
        result_obj.combined_legal_basis = []

        diff = svc._compute_diff(orig, repro, result_obj)

        # intersection={"literal"}, union={"literal","systemic","precedent"} → 1/3
        assert diff["expert_overlap"] == pytest.approx(1 / 3, abs=0.001)

    def test_confidence_delta_calculated(self):
        svc = ReproducibilityService()
        orig = _make_trace(confidence=0.8)
        repro = _make_trace(confidence=0.5)
        result_obj = MagicMock()
        result_obj.combined_legal_basis = []

        diff = svc._compute_diff(orig, repro, result_obj)

        assert diff["confidence_delta"] == pytest.approx(0.3, abs=0.001)
        assert diff["original_confidence"] == 0.8
        assert diff["reproduced_confidence"] == 0.5

    def test_mode_match_false_when_modes_differ(self):
        svc = ReproducibilityService()
        orig = _make_trace(synthesis_mode="convergent")
        repro = _make_trace(synthesis_mode="divergent")
        result_obj = MagicMock()
        result_obj.combined_legal_basis = []

        diff = svc._compute_diff(orig, repro, result_obj)

        assert diff["mode_match"] is False

    def test_source_jaccard_with_matching_sources(self):
        svc = ReproducibilityService()
        orig = _make_trace(
            sources=[
                {"article_urn": "urn:a"},
                {"article_urn": "urn:b"},
            ]
        )
        repro = _make_trace(sources=[])

        src_a = MagicMock()
        src_a.source_id = "urn:a"
        result_obj = MagicMock()
        result_obj.combined_legal_basis = [src_a]

        diff = svc._compute_diff(orig, repro, result_obj)

        # orig_sources = {urn:a, urn:b}, repro_sources = {urn:a}
        # intersection = {urn:a}, union = {urn:a, urn:b} → 0.5
        assert diff["source_jaccard"] == pytest.approx(0.5, abs=0.001)

    def test_empty_experts_gives_overlap_1(self):
        svc = ReproducibilityService()
        orig = _make_trace(selected_experts=[])
        repro = _make_trace(selected_experts=[])
        result_obj = MagicMock()
        result_obj.combined_legal_basis = []

        diff = svc._compute_diff(orig, repro, result_obj)

        assert diff["expert_overlap"] == 1.0


# ---------------------------------------------------------------------------
# ReproducibilityService._compute_reproducibility_score
# ---------------------------------------------------------------------------


class TestComputeReproducibilityScore:
    def test_perfect_score(self):
        svc = ReproducibilityService()
        diff = {
            "expert_overlap": 1.0,
            "confidence_delta": 0.0,
            "source_jaccard": 1.0,
        }
        score = svc._compute_reproducibility_score(diff)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_zero_score_on_error(self):
        svc = ReproducibilityService()
        diff = {"error": "trace not found"}
        score = svc._compute_reproducibility_score(diff)
        assert score == 0.0

    def test_partial_score(self):
        svc = ReproducibilityService()
        diff = {
            "expert_overlap": 0.5,
            "confidence_delta": 0.5,
            "source_jaccard": 0.5,
        }
        # confidence_sim = 1 - 0.5 = 0.5; mean(0.5, 0.5, 0.5) = 0.5
        score = svc._compute_reproducibility_score(diff)
        assert score == pytest.approx(0.5, abs=0.001)

    def test_score_clamped_to_0_1(self):
        svc = ReproducibilityService()
        diff = {
            "expert_overlap": 2.0,  # intentionally bad value
            "confidence_delta": -1.0,
            "source_jaccard": 2.0,
        }
        score = svc._compute_reproducibility_score(diff)
        assert 0.0 <= score <= 1.0

    def test_missing_keys_use_defaults(self):
        svc = ReproducibilityService()
        diff = {}  # all keys missing
        score = svc._compute_reproducibility_score(diff)
        # expert_overlap=0, confidence_sim=1-1=0, source_jaccard=0 → 0
        assert score == 0.0


# ---------------------------------------------------------------------------
# ReproducibilityService.reproduce_query – trace not found
# ---------------------------------------------------------------------------


class TestReproduceQuery:
    @pytest.mark.asyncio
    async def test_returns_error_when_trace_not_found(self):
        svc = ReproducibilityService()
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None)
            )
        )

        result = await svc.reproduce_query(session, "nonexistent_trace")

        assert result.original_trace_id == "nonexistent_trace"
        assert result.reproduced_trace_id is None
        assert result.reproducibility_score == 0.0
        assert "not found" in result.diff.get("error", "")
        assert result.caveats == svc.CAVEATS

    @pytest.mark.asyncio
    async def test_returns_error_when_orchestrator_raises(self):
        svc = ReproducibilityService()
        trace = _make_trace()

        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=trace)
            )
        )

        with patch(
            "merlt.rlcf.reproducibility_service.ReproducibilityService._extract_config",
            return_value={"selected_experts": ["literal"]},
        ):
            with patch.dict(
                "sys.modules",
                {"merlt.api.experts_router": MagicMock(get_orchestrator=MagicMock(side_effect=ImportError("not found")))},
            ):
                result = await svc.reproduce_query(session, "trace_abc")

        assert result.reproducibility_score == 0.0
        assert "error" in result.diff

    @pytest.mark.asyncio
    async def test_caveats_always_present(self):
        svc = ReproducibilityService()
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None)
            )
        )

        result = await svc.reproduce_query(session, "any_trace")

        assert len(result.caveats) == 3
        assert all(isinstance(c, str) for c in result.caveats)
