"""
Tests for Gold Standard Regression Testing module.
"""

import json
import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from visualex.experts.regression import (
    GoldStandardQuery,
    ExpectedResponse,
    RegressionResult,
    QueryResult,
    SimilarityScore,
    RegressionReport,
    GoldStandardSuite,
    SuiteConfig,
    RegressionRunner,
    RunnerConfig,
    ResponseComparator,
    SemanticComparator,
    StructuralComparator,
)
from visualex.experts.regression.models import QueryStatus, ExpertFocus
from visualex.experts.regression.comparator import CombinedComparator, ComparatorConfig


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def expected_response():
    """Create a sample expected response."""
    return ExpectedResponse(
        main_answer="La risoluzione del contratto è lo scioglimento del vincolo contrattuale.",
        expected_experts=["LiteralExpert", "SystemicExpert"],
        expected_citations=["Art. 1453 c.c.", "Art. 1454 c.c."],
        key_concepts=["risoluzione", "scioglimento", "contratto", "inadempimento"],
        confidence_range=(0.7, 0.95),
    )


@pytest.fixture
def gold_standard_query(expected_response):
    """Create a sample gold standard query."""
    return GoldStandardQuery(
        query_id="test-001",
        query_text="Cos'è la risoluzione del contratto?",
        expected=expected_response,
        focus=ExpertFocus.LITERAL,
        topic_tags=["contratti", "obbligazioni"],
        difficulty=3,
        validated_by="legal_expert",
        validated_at=datetime.now(),
        notes="Test query for unit tests",
    )


@pytest.fixture
def suite_config():
    """Create a sample suite config."""
    return SuiteConfig(
        name="test_suite",
        description="Test Suite",
        version="1.0.0",
        pass_threshold=0.7,
        degradation_threshold=0.1,
    )


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    pipeline = AsyncMock()
    pipeline.process.return_value = {
        "response": "La risoluzione del contratto comporta lo scioglimento del vincolo.",
        "metadata": {
            "contributing_experts": ["LiteralExpert", "SystemicExpert"],
            "citations": ["Art. 1453 c.c."],
            "confidence": 0.85,
        },
    }
    return pipeline


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.embed.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Expected
        [0.12, 0.18, 0.32, 0.38, 0.48],  # Actual - similar
    ]
    return service


# ============================================
# Models Tests
# ============================================


class TestExpectedResponse:
    """Tests for ExpectedResponse model."""

    def test_creation(self, expected_response):
        """Test ExpectedResponse creation."""
        assert expected_response.main_answer is not None
        assert len(expected_response.expected_experts) == 2
        assert len(expected_response.expected_citations) == 2
        assert len(expected_response.key_concepts) == 4

    def test_to_dict(self, expected_response):
        """Test conversion to dictionary."""
        data = expected_response.to_dict()
        assert data["main_answer"] == expected_response.main_answer
        assert data["expected_experts"] == expected_response.expected_experts
        assert data["confidence_range"] == list(expected_response.confidence_range)

    def test_from_dict(self, expected_response):
        """Test creation from dictionary."""
        data = expected_response.to_dict()
        restored = ExpectedResponse.from_dict(data)
        assert restored.main_answer == expected_response.main_answer
        assert restored.expected_experts == expected_response.expected_experts


class TestGoldStandardQuery:
    """Tests for GoldStandardQuery model."""

    def test_creation(self, gold_standard_query):
        """Test GoldStandardQuery creation."""
        assert gold_standard_query.query_id == "test-001"
        assert gold_standard_query.focus == ExpertFocus.LITERAL
        assert gold_standard_query.difficulty == 3

    def test_to_dict(self, gold_standard_query):
        """Test conversion to dictionary."""
        data = gold_standard_query.to_dict()
        assert data["query_id"] == "test-001"
        assert data["focus"] == "literal"
        assert data["topic_tags"] == ["contratti", "obbligazioni"]

    def test_from_dict(self, gold_standard_query):
        """Test creation from dictionary."""
        data = gold_standard_query.to_dict()
        restored = GoldStandardQuery.from_dict(data)
        assert restored.query_id == gold_standard_query.query_id
        assert restored.focus == gold_standard_query.focus


class TestSimilarityScore:
    """Tests for SimilarityScore model."""

    def test_creation(self):
        """Test SimilarityScore creation."""
        score = SimilarityScore(
            semantic_score=0.85,
            structural_score=0.75,
            concept_coverage=0.8,
            citation_coverage=0.5,
            expert_match=1.0,
            overall_score=0.8,
        )
        assert score.overall_score == 0.8

    def test_is_passing(self):
        """Test pass threshold check."""
        passing = SimilarityScore(overall_score=0.75)
        failing = SimilarityScore(overall_score=0.65)

        assert passing.is_passing(threshold=0.7)
        assert not failing.is_passing(threshold=0.7)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = SimilarityScore(semantic_score=0.8, overall_score=0.75)
        data = score.to_dict()
        assert data["semantic_score"] == 0.8
        assert data["overall_score"] == 0.75


class TestQueryResult:
    """Tests for QueryResult model."""

    def test_creation(self):
        """Test QueryResult creation."""
        result = QueryResult(
            query_id="test-001",
            status=QueryStatus.PASSED,
            actual_response="Test response",
            similarity=SimilarityScore(overall_score=0.8),
            latency_ms=150.0,
        )
        assert result.status == QueryStatus.PASSED
        assert result.latency_ms == 150.0

    def test_error_result(self):
        """Test error result creation."""
        result = QueryResult(
            query_id="test-001",
            status=QueryStatus.ERROR,
            error_message="Pipeline timeout",
        )
        assert result.status == QueryStatus.ERROR
        assert "timeout" in result.error_message.lower()


class TestRegressionResult:
    """Tests for RegressionResult model."""

    def test_pass_rate(self):
        """Test pass rate calculation."""
        result = RegressionResult(
            total_queries=10,
            passed=7,
            failed=2,
            errors=1,
        )
        assert result.pass_rate == 0.7

    def test_pass_rate_empty(self):
        """Test pass rate with no queries."""
        result = RegressionResult(total_queries=0)
        assert result.pass_rate == 0.0

    def test_is_successful(self):
        """Test success check."""
        successful = RegressionResult(total_queries=10, passed=9, failed=1)
        unsuccessful_rate = RegressionResult(total_queries=10, passed=8, failed=2)
        unsuccessful_degraded = RegressionResult(
            total_queries=10, passed=10, degraded=1
        )

        assert successful.is_successful
        assert not unsuccessful_rate.is_successful  # 80% < 90%
        assert not unsuccessful_degraded.is_successful  # Has degradation


class TestRegressionReport:
    """Tests for RegressionReport model."""

    def test_duration(self):
        """Test duration calculation."""
        started = datetime(2025, 1, 1, 12, 0, 0)
        completed = datetime(2025, 1, 1, 12, 0, 30)

        report = RegressionReport(
            run_id="test-run",
            suite_name="test",
            result=RegressionResult(total_queries=5, passed=5),
            started_at=started,
            completed_at=completed,
        )
        assert report.duration_seconds == 30.0

    def test_summary(self):
        """Test summary generation."""
        report = RegressionReport(
            run_id="test-run",
            suite_name="test_suite",
            result=RegressionResult(
                total_queries=10,
                passed=9,
                failed=1,
            ),
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        summary = report.summary()
        assert "PASS" in summary
        assert "test_suite" in summary
        assert "90.0%" in summary


# ============================================
# Suite Tests
# ============================================


class TestGoldStandardSuite:
    """Tests for GoldStandardSuite."""

    def test_add_query(self, expected_response):
        """Test adding a query."""
        suite = GoldStandardSuite()
        query = suite.add_query(
            query_text="Test query",
            expected=expected_response,
            topic_tags=["test"],
        )
        assert query.query_id is not None
        assert suite.query_count == 1

    def test_add_query_custom_id(self, expected_response):
        """Test adding query with custom ID."""
        suite = GoldStandardSuite()
        query = suite.add_query(
            query_text="Test query",
            expected=expected_response,
            query_id="custom-id",
        )
        assert query.query_id == "custom-id"

    def test_get_query(self, expected_response):
        """Test getting a query by ID."""
        suite = GoldStandardSuite()
        added = suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-id",
        )
        retrieved = suite.get_query("test-id")
        assert retrieved is not None
        assert retrieved.query_id == added.query_id

    def test_get_query_not_found(self):
        """Test getting non-existent query."""
        suite = GoldStandardSuite()
        assert suite.get_query("nonexistent") is None

    def test_update_query(self, expected_response):
        """Test updating a query."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-id",
            notes="Original notes",
        )
        updated = suite.update_query("test-id", notes="Updated notes")
        assert updated is not None
        assert updated.notes == "Updated notes"

    def test_remove_query(self, expected_response):
        """Test removing a query."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-id",
        )
        assert suite.query_count == 1
        assert suite.remove_query("test-id")
        assert suite.query_count == 0

    def test_get_by_focus(self, expected_response):
        """Test filtering by focus."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Literal query",
            expected=expected_response,
            focus=ExpertFocus.LITERAL,
            query_id="lit-1",
        )
        suite.add_query(
            query_text="Systemic query",
            expected=expected_response,
            focus=ExpertFocus.SYSTEMIC,
            query_id="sys-1",
        )

        literal = suite.get_by_focus(ExpertFocus.LITERAL)
        assert len(literal) == 1
        assert literal[0].query_id == "lit-1"

    def test_get_by_tag(self, expected_response):
        """Test filtering by tag."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Contract query",
            expected=expected_response,
            topic_tags=["contracts", "civil"],
            query_id="q1",
        )
        suite.add_query(
            query_text="Criminal query",
            expected=expected_response,
            topic_tags=["criminal"],
            query_id="q2",
        )

        contracts = suite.get_by_tag("contracts")
        assert len(contracts) == 1
        assert contracts[0].query_id == "q1"

    def test_get_by_difficulty(self, expected_response):
        """Test filtering by difficulty."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Easy",
            expected=expected_response,
            difficulty=1,
            query_id="easy",
        )
        suite.add_query(
            query_text="Hard",
            expected=expected_response,
            difficulty=5,
            query_id="hard",
        )

        easy_to_medium = suite.get_by_difficulty(1, 3)
        assert len(easy_to_medium) == 1
        assert easy_to_medium[0].query_id == "easy"

    def test_baseline_scores(self, expected_response):
        """Test baseline score management."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-id",
        )
        suite.set_baseline_score("test-id", 0.85)
        assert suite.get_baseline_score("test-id") == 0.85
        assert suite.get_baseline_score("nonexistent") is None

    def test_save_and_load(self, expected_response, tmp_path):
        """Test saving and loading suite."""
        suite = GoldStandardSuite(
            config=SuiteConfig(name="test", version="1.0.0")
        )
        suite.add_query(
            query_text="Test query",
            expected=expected_response,
            query_id="test-id",
        )
        suite.set_baseline_score("test-id", 0.9)

        path = str(tmp_path / "suite.json")
        suite.save(path)

        loaded = GoldStandardSuite.load(path)
        assert loaded.config.name == "test"
        assert loaded.query_count == 1
        assert loaded.get_baseline_score("test-id") == 0.9

    def test_export_csv(self, expected_response, tmp_path):
        """Test CSV export."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test query",
            expected=expected_response,
            topic_tags=["test"],
            query_id="test-id",
        )

        path = str(tmp_path / "export.csv")
        suite.export(path, format="csv")

        import csv
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["query_id"] == "test-id"


# ============================================
# Comparator Tests
# ============================================


class TestSemanticComparator:
    """Tests for SemanticComparator."""

    @pytest.mark.asyncio
    async def test_word_overlap_fallback(self, expected_response):
        """Test fallback to word overlap without embedding service."""
        comparator = SemanticComparator()
        score = await comparator.compare(
            expected_response,
            "La risoluzione del contratto comporta lo scioglimento.",
        )
        assert 0 < score < 1  # Should have some overlap

    @pytest.mark.asyncio
    async def test_with_embedding_service(self, expected_response, mock_embedding_service):
        """Test with embedding service."""
        comparator = SemanticComparator(embedding_service=mock_embedding_service)
        score = await comparator.compare(
            expected_response,
            "Similar response text",
        )
        assert 0 < score <= 1
        mock_embedding_service.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_service_error_fallback(self, expected_response):
        """Test fallback on embedding service error."""
        failing_service = AsyncMock()
        failing_service.embed.side_effect = Exception("API error")

        comparator = SemanticComparator(embedding_service=failing_service)
        score = await comparator.compare(
            expected_response,
            "La risoluzione del contratto.",
        )
        # Should fall back to word overlap
        assert 0 <= score <= 1

    def test_tokenize(self):
        """Test tokenization."""
        comparator = SemanticComparator()
        tokens = comparator._tokenize("La risoluzione del contratto, è importante!")
        assert "risoluzione" in tokens
        assert "contratto" in tokens
        assert "la" not in tokens  # Short words filtered
        assert "è" not in tokens


class TestStructuralComparator:
    """Tests for StructuralComparator."""

    @pytest.mark.asyncio
    async def test_full_match(self, expected_response):
        """Test with full structural match."""
        comparator = StructuralComparator()
        score = await comparator.compare(
            expected_response,
            "Risoluzione scioglimento contratto inadempimento Art. 1453 c.c. Art. 1454 c.c.",
            {
                "contributing_experts": ["LiteralExpert", "SystemicExpert"],
                "citations": ["Art. 1453 c.c.", "Art. 1454 c.c."],
            },
        )
        assert score > 0.9  # Should be near perfect

    @pytest.mark.asyncio
    async def test_partial_match(self, expected_response):
        """Test with partial match."""
        comparator = StructuralComparator()
        score = await comparator.compare(
            expected_response,
            "La risoluzione del contratto",  # Missing some concepts
            {
                "contributing_experts": ["LiteralExpert"],  # Missing one expert
                "citations": ["Art. 1453 c.c."],  # Missing one citation
            },
        )
        assert 0.3 < score < 0.9

    @pytest.mark.asyncio
    async def test_no_expectations(self):
        """Test with empty expectations."""
        expected = ExpectedResponse(main_answer="Any answer")
        comparator = StructuralComparator()
        score = await comparator.compare(expected, "Some response")
        assert score == 1.0  # All weights return 1.0 when nothing expected

    @pytest.mark.asyncio
    async def test_custom_weights(self, expected_response):
        """Test with custom weights."""
        comparator = StructuralComparator(
            expert_weight=0.5,
            citation_weight=0.3,
            concept_weight=0.2,
        )
        score = await comparator.compare(
            expected_response,
            "Response text",
            {"contributing_experts": ["LiteralExpert", "SystemicExpert"]},
        )
        # Expert weight is high and experts match
        assert score > 0.4


class TestCombinedComparator:
    """Tests for CombinedComparator."""

    @pytest.mark.asyncio
    async def test_combined_score(self, expected_response):
        """Test combined scoring."""
        comparator = CombinedComparator()
        similarity = await comparator.compare(
            expected_response,
            "La risoluzione del contratto comporta lo scioglimento del vincolo. Art. 1453 c.c.",
            {"contributing_experts": ["LiteralExpert"]},
        )

        assert isinstance(similarity, SimilarityScore)
        assert 0 <= similarity.semantic_score <= 1
        assert 0 <= similarity.structural_score <= 1
        assert 0 <= similarity.overall_score <= 1

    @pytest.mark.asyncio
    async def test_config(self, expected_response):
        """Test with custom config."""
        config = ComparatorConfig(
            semantic_weight=0.8,
            structural_weight=0.2,
            pass_threshold=0.6,
        )
        comparator = CombinedComparator(config=config)
        similarity = await comparator.compare(
            expected_response,
            "La risoluzione del contratto è importante.",
        )
        # Semantic weight is higher
        assert similarity.overall_score is not None


# ============================================
# Runner Tests
# ============================================


class TestRegressionRunner:
    """Tests for RegressionRunner."""

    @pytest.mark.asyncio
    async def test_run_empty_suite(self, mock_pipeline):
        """Test running with empty suite."""
        suite = GoldStandardSuite()
        runner = RegressionRunner(suite, mock_pipeline)
        report = await runner.run()

        assert report.result.total_queries == 0
        assert report.result.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_run_single_query(self, expected_response, mock_pipeline):
        """Test running a single query."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test query",
            expected=expected_response,
            query_id="test-1",
        )

        runner = RegressionRunner(suite, mock_pipeline)
        report = await runner.run()

        assert report.result.total_queries == 1
        assert mock_pipeline.process.called

    @pytest.mark.asyncio
    async def test_run_with_callback(self, expected_response, mock_pipeline):
        """Test callback on query completion."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )

        callback_results = []

        def callback(result):
            callback_results.append(result)

        runner = RegressionRunner(
            suite, mock_pipeline, on_query_complete=callback
        )
        await runner.run()

        assert len(callback_results) == 1
        assert callback_results[0].query_id == "test-1"

    @pytest.mark.asyncio
    async def test_run_with_filter_ids(self, expected_response, mock_pipeline):
        """Test filtering by query IDs."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Query 1",
            expected=expected_response,
            query_id="q1",
        )
        suite.add_query(
            query_text="Query 2",
            expected=expected_response,
            query_id="q2",
        )

        runner = RegressionRunner(suite, mock_pipeline)
        report = await runner.run(query_ids=["q1"])

        assert report.result.total_queries == 1
        assert report.result.results[0].query_id == "q1"

    @pytest.mark.asyncio
    async def test_run_with_filter_tags(self, expected_response, mock_pipeline):
        """Test filtering by tags."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Contract query",
            expected=expected_response,
            topic_tags=["contracts"],
            query_id="c1",
        )
        suite.add_query(
            query_text="Criminal query",
            expected=expected_response,
            topic_tags=["criminal"],
            query_id="cr1",
        )

        runner = RegressionRunner(suite, mock_pipeline)
        report = await runner.run(tags=["contracts"])

        assert report.result.total_queries == 1

    @pytest.mark.asyncio
    async def test_pipeline_error(self, expected_response):
        """Test handling pipeline errors."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )

        failing_pipeline = AsyncMock()
        failing_pipeline.process.side_effect = Exception("Pipeline failed")

        runner = RegressionRunner(
            suite,
            failing_pipeline,
            config=RunnerConfig(retry_on_error=0),
        )
        report = await runner.run()

        assert report.result.errors == 1
        assert report.result.results[0].status == QueryStatus.ERROR

    @pytest.mark.asyncio
    async def test_pipeline_timeout(self, expected_response):
        """Test handling pipeline timeout."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )

        async def slow_process(query):
            await asyncio.sleep(10)
            return {"response": "Too late"}

        slow_pipeline = AsyncMock()
        slow_pipeline.process = slow_process

        runner = RegressionRunner(
            suite,
            slow_pipeline,
            config=RunnerConfig(timeout_seconds=0.1, retry_on_error=0),
        )
        report = await runner.run()

        assert report.result.errors == 1
        assert "Timeout" in report.result.results[0].error_message

    @pytest.mark.asyncio
    async def test_degradation_detection(self, expected_response, mock_pipeline):
        """Test degradation detection."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )
        suite.set_baseline_score("test-1", 0.95)  # High baseline

        # Mock a poor response
        mock_pipeline.process.return_value = {
            "response": "Unrelated response",
            "metadata": {},
        }

        runner = RegressionRunner(
            suite,
            mock_pipeline,
            config=RunnerConfig(
                degradation_threshold=0.1,
                pass_threshold=0.3,  # Low threshold to not fail
            ),
        )
        report = await runner.run()

        # Score dropped significantly from baseline
        assert report.result.results[0].status == QueryStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_improvement_detection(self, expected_response, mock_pipeline):
        """Test improvement detection."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )
        suite.set_baseline_score("test-1", 0.3)  # Low baseline

        runner = RegressionRunner(
            suite,
            mock_pipeline,
            config=RunnerConfig(degradation_threshold=0.1),
        )
        report = await runner.run()

        # Score improved significantly from baseline
        assert report.result.results[0].status == QueryStatus.IMPROVED

    @pytest.mark.asyncio
    async def test_run_single(self, expected_response, mock_pipeline):
        """Test running a single query by ID."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
        )

        runner = RegressionRunner(suite, mock_pipeline)
        result = await runner.run_single("test-1")

        assert result is not None
        assert result.query_id == "test-1"

    @pytest.mark.asyncio
    async def test_run_single_not_found(self, mock_pipeline):
        """Test run_single with non-existent ID."""
        suite = GoldStandardSuite()
        runner = RegressionRunner(suite, mock_pipeline)
        result = await runner.run_single("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_baselines(self, expected_response, mock_pipeline):
        """Test baseline update from results."""
        suite = GoldStandardSuite()
        suite.add_query(
            query_text="Test 1",
            expected=expected_response,
            query_id="test-1",
        )
        suite.add_query(
            query_text="Test 2",
            expected=expected_response,
            query_id="test-2",
        )

        runner = RegressionRunner(suite, mock_pipeline)
        report = await runner.run()
        await runner.update_baselines(report.result)

        # Baselines should be set for passing queries
        assert suite.get_baseline_score("test-1") is not None

    @pytest.mark.asyncio
    async def test_parallel_execution(self, expected_response):
        """Test parallel query execution."""
        suite = GoldStandardSuite()
        for i in range(5):
            suite.add_query(
                query_text=f"Query {i}",
                expected=expected_response,
                query_id=f"q{i}",
            )

        execution_order = []

        async def track_process(query):
            execution_order.append(query)
            await asyncio.sleep(0.05)
            return {"response": "Response", "metadata": {}}

        pipeline = AsyncMock()
        pipeline.process = track_process

        runner = RegressionRunner(
            suite,
            pipeline,
            config=RunnerConfig(parallel_queries=3),
        )
        await runner.run()

        # All queries should run
        assert len(execution_order) == 5

    @pytest.mark.asyncio
    async def test_report_metadata(self, expected_response, mock_pipeline):
        """Test report metadata."""
        suite = GoldStandardSuite(
            config=SuiteConfig(name="test_suite", version="1.0.0")
        )
        suite.add_query(
            query_text="Test",
            expected=expected_response,
            query_id="test-1",
            topic_tags=["test"],
        )

        runner = RegressionRunner(
            suite,
            mock_pipeline,
            config=RunnerConfig(
                parallel_queries=3,
                timeout_seconds=30.0,
                pass_threshold=0.7,
            ),
        )
        report = await runner.run(
            query_ids=["test-1"],
            tags=["test"],
            version="2.0.0",
        )

        assert report.suite_name == "test_suite"
        assert report.baseline_version == "1.0.0"
        assert report.current_version == "2.0.0"
        assert report.metadata["config"]["parallel_queries"] == 3
        assert report.metadata["filter"]["query_ids"] == ["test-1"]


class TestRunnerConfig:
    """Tests for RunnerConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = RunnerConfig()
        assert config.parallel_queries == 5
        assert config.timeout_seconds == 60.0
        assert config.pass_threshold == 0.7
        assert config.degradation_threshold == 0.1
        assert config.retry_on_error == 2
        assert config.save_responses is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RunnerConfig(
            parallel_queries=10,
            timeout_seconds=120.0,
            pass_threshold=0.8,
        )
        assert config.parallel_queries == 10
        assert config.timeout_seconds == 120.0
        assert config.pass_threshold == 0.8


class TestRetryBackoff:
    """Tests for retry backoff functionality."""

    def test_retry_delay_calculation(self, mock_pipeline):
        """Test retry delay calculation."""
        suite = GoldStandardSuite()
        runner = RegressionRunner(suite, mock_pipeline)

        # First attempt
        delay0 = runner._retry_delay(0)
        assert 0.5 <= delay0 <= 1.5  # base_delay with jitter

        # Second attempt (exponential)
        delay1 = runner._retry_delay(1)
        assert 1.0 <= delay1 <= 3.0  # 2^1 with jitter

        # Third attempt
        delay2 = runner._retry_delay(2)
        assert 2.0 <= delay2 <= 6.0  # 2^2 with jitter

    def test_retry_delay_max_cap(self, mock_pipeline):
        """Test retry delay is capped at max."""
        suite = GoldStandardSuite()
        runner = RegressionRunner(suite, mock_pipeline)

        # Very high attempt number
        delay = runner._retry_delay(10, max_delay=30.0)
        assert delay <= 45.0  # max_delay with max jitter


class TestWeightValidation:
    """Tests for weight validation in comparators."""

    def test_structural_weight_clamping(self):
        """Test structural comparator clamps weights."""
        comparator = StructuralComparator(
            expert_weight=-0.5,  # Should clamp to 0
            citation_weight=1.5,  # Should clamp to 1
            concept_weight=0.5,
        )
        assert comparator.expert_weight == 0.0
        assert comparator.citation_weight == 1.0
        assert comparator.concept_weight == 0.5

    def test_comparator_config_validation(self):
        """Test ComparatorConfig validates weights."""
        config = ComparatorConfig(
            semantic_weight=-0.1,
            structural_weight=1.5,
            pass_threshold=2.0,
        )
        assert config.semantic_weight == 0.0
        assert config.structural_weight == 1.0
        assert config.pass_threshold == 1.0


class TestDifficultyValidation:
    """Tests for difficulty validation."""

    def test_difficulty_clamped_low(self, expected_response):
        """Test difficulty is clamped to minimum 1."""
        query = GoldStandardQuery(
            query_id="test",
            query_text="Test",
            expected=expected_response,
            difficulty=0,
        )
        assert query.difficulty == 1

    def test_difficulty_clamped_high(self, expected_response):
        """Test difficulty is clamped to maximum 5."""
        query = GoldStandardQuery(
            query_id="test",
            query_text="Test",
            expected=expected_response,
            difficulty=10,
        )
        assert query.difficulty == 5

    def test_difficulty_valid_range(self, expected_response):
        """Test valid difficulty values are unchanged."""
        for diff in [1, 2, 3, 4, 5]:
            query = GoldStandardQuery(
                query_id=f"test-{diff}",
                query_text="Test",
                expected=expected_response,
                difficulty=diff,
            )
            assert query.difficulty == diff


class TestCosineSimliarityEdgeCases:
    """Tests for cosine similarity edge cases."""

    def test_empty_vectors(self):
        """Test with empty vectors."""
        from visualex.experts.regression.comparator import cosine_similarity
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1, 2], []) == 0.0
        assert cosine_similarity([], [1, 2]) == 0.0

    def test_zero_vectors(self):
        """Test with zero vectors."""
        from visualex.experts.regression.comparator import cosine_similarity
        assert cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_identical_vectors(self):
        """Test with identical vectors."""
        from visualex.experts.regression.comparator import cosine_similarity
        result = cosine_similarity([1, 2, 3], [1, 2, 3])
        assert abs(result - 1.0) < 0.001

    def test_different_length_vectors(self):
        """Test with different length vectors.

        Note: zip truncates dot product to shorter length, but norms
        use full vectors. This results in a lower similarity score
        for mismatched lengths, which is reasonable behavior.
        """
        from visualex.experts.regression.comparator import cosine_similarity
        # Different lengths produce a valid but lower score
        result = cosine_similarity([1, 2], [1, 2, 3, 4])
        assert 0 < result < 1  # Valid similarity range
