"""
Test Disagreement Metrics
=========================

Test per le metriche di valutazione (evaluation/metrics.py).
"""

import pytest
import numpy as np

from merlt.disagreement.evaluation.metrics import (
    DisagreementMetrics,
    compute_disagreement_metrics,
    compute_pairwise_metrics,
    _compute_binary_metrics,
    _compute_multiclass_metrics,
    _compute_regression_metrics,
)
from merlt.disagreement.types import (
    DisagreementType,
    DisagreementLevel,
    DisagreementAnalysis,
    DisagreementSample,
    ExpertPairConflict,
)


class TestBinaryMetrics:
    """Test per metriche binarie."""

    def test_perfect_predictions(self):
        """Test con predizioni perfette."""
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 1, 0, 0, 1, 0]

        metrics = _compute_binary_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        """Test con tutte predizioni errate."""
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]

        metrics = _compute_binary_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_mixed_predictions(self):
        """Test con predizioni miste."""
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 1, 0]

        metrics = _compute_binary_metrics(y_true, y_pred)

        # TP=2, FP=1, FN=1, TN=2
        assert metrics["accuracy"] == pytest.approx(4/6)
        assert metrics["precision"] == pytest.approx(2/3)
        assert metrics["recall"] == pytest.approx(2/3)
        assert metrics["f1"] == pytest.approx(2/3)

    def test_empty_input(self):
        """Test con input vuoto."""
        metrics = _compute_binary_metrics([], [])

        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0


class TestMulticlassMetrics:
    """Test per metriche multiclass."""

    def test_perfect_predictions(self):
        """Test con predizioni perfette."""
        y_true = ["ANT", "LAC", "MET", "OVR"]
        y_pred = ["ANT", "LAC", "MET", "OVR"]
        classes = ["ANT", "LAC", "MET", "OVR", "GER", "SPE"]

        metrics = _compute_multiclass_metrics(y_true, y_pred, classes)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_mixed_predictions(self):
        """Test con predizioni miste."""
        y_true = ["ANT", "ANT", "LAC", "MET"]
        y_pred = ["ANT", "LAC", "LAC", "MET"]
        classes = ["ANT", "LAC", "MET"]

        metrics = _compute_multiclass_metrics(y_true, y_pred, classes)

        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["macro_f1"] < 1

        # Per-class metrics
        assert "ANT" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["ANT"]

    def test_empty_input(self):
        """Test con input vuoto."""
        metrics = _compute_multiclass_metrics([], [], ["A", "B"])

        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0


class TestRegressionMetrics:
    """Test per metriche regression."""

    def test_perfect_predictions(self):
        """Test con predizioni perfette."""
        y_true = [0.5, 0.7, 0.3, 0.9]
        y_pred = [0.5, 0.7, 0.3, 0.9]

        metrics = _compute_regression_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0

    def test_small_errors(self):
        """Test con piccoli errori."""
        y_true = [0.5, 0.5, 0.5, 0.5]
        y_pred = [0.6, 0.4, 0.6, 0.4]

        metrics = _compute_regression_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(0.1)
        assert metrics["mse"] == pytest.approx(0.01)

    def test_empty_input(self):
        """Test con input vuoto."""
        metrics = _compute_regression_metrics([], [])

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0


class TestComputeDisagreementMetrics:
    """Test per compute_disagreement_metrics."""

    @pytest.fixture
    def sample_data(self):
        """Crea dati di esempio."""
        predictions = [
            DisagreementAnalysis(
                has_disagreement=True,
                disagreement_type=DisagreementType.METHODOLOGICAL,
                disagreement_level=DisagreementLevel.TELEOLOGICAL,
                intensity=0.7,
                resolvability=0.5,
            ),
            DisagreementAnalysis(
                has_disagreement=False,
            ),
            DisagreementAnalysis(
                has_disagreement=True,
                disagreement_type=DisagreementType.ANTINOMY,
                disagreement_level=DisagreementLevel.SYSTEMIC,
                intensity=0.9,
                resolvability=0.3,
            ),
        ]

        ground_truth = [
            DisagreementSample(
                sample_id="s1",
                query="q1",
                expert_responses={},
                has_disagreement=True,
                disagreement_type=DisagreementType.METHODOLOGICAL,
                disagreement_level=DisagreementLevel.TELEOLOGICAL,
                intensity=0.75,
                resolvability=0.45,
            ),
            DisagreementSample(
                sample_id="s2",
                query="q2",
                expert_responses={},
                has_disagreement=False,
            ),
            DisagreementSample(
                sample_id="s3",
                query="q3",
                expert_responses={},
                has_disagreement=True,
                disagreement_type=DisagreementType.ANTINOMY,
                disagreement_level=DisagreementLevel.SYSTEMIC,
                intensity=0.85,
                resolvability=0.35,
            ),
        ]

        return predictions, ground_truth

    def test_compute_metrics(self, sample_data):
        """Test calcolo metriche completo."""
        predictions, ground_truth = sample_data

        metrics = compute_disagreement_metrics(predictions, ground_truth)

        # Binary metrics - tutte predizioni corrette
        assert metrics.binary_accuracy == 1.0
        assert metrics.binary_f1 == 1.0

        # Type metrics
        assert metrics.type_accuracy == 1.0  # Tutte corrette

        # Level metrics
        assert metrics.level_accuracy == 1.0

        # Regression metrics - piccoli errori
        assert metrics.intensity_mae < 0.1
        assert metrics.resolvability_mae < 0.1

        # Metadata
        assert metrics.n_samples == 3
        assert metrics.n_positive == 2
        assert metrics.n_negative == 1

    def test_mismatched_lengths(self):
        """Test errore con lunghezze diverse."""
        predictions = [DisagreementAnalysis(has_disagreement=True)]
        ground_truth = []

        with pytest.raises(ValueError) as exc_info:
            compute_disagreement_metrics(predictions, ground_truth)

        assert "Mismatch" in str(exc_info.value)

    def test_empty_inputs(self):
        """Test con input vuoti."""
        metrics = compute_disagreement_metrics([], [])

        assert metrics.n_samples == 0
        assert metrics.binary_f1 == 0.0

    def test_to_dict(self, sample_data):
        """Test serializzazione metriche."""
        predictions, ground_truth = sample_data
        metrics = compute_disagreement_metrics(predictions, ground_truth)

        d = metrics.to_dict()

        assert "binary" in d
        assert "type" in d
        assert "level" in d
        assert "regression" in d
        assert "metadata" in d

        assert d["binary"]["accuracy"] == metrics.binary_accuracy
        assert d["metadata"]["n_samples"] == metrics.n_samples

    def test_summary(self, sample_data):
        """Test summary leggibile."""
        predictions, ground_truth = sample_data
        metrics = compute_disagreement_metrics(predictions, ground_truth)

        summary = metrics.summary()

        assert "DisagreementMetrics" in summary
        assert "Binary" in summary
        assert "Type" in summary
        assert "Level" in summary


class TestComputePairwiseMetrics:
    """Test per compute_pairwise_metrics."""

    def test_perfect_pairwise(self):
        """Test con coppie perfette."""
        predictions = [
            DisagreementAnalysis(
                has_disagreement=True,
                conflicting_pairs=[
                    ExpertPairConflict(
                        expert_a="literal",
                        expert_b="principles",
                        conflict_score=0.8,
                    ),
                ],
            ),
        ]

        ground_truth = [
            DisagreementSample(
                sample_id="s1",
                query="q1",
                expert_responses={},
                has_disagreement=True,
                conflicting_pairs=[("literal", "principles")],
            ),
        ]

        metrics = compute_pairwise_metrics(predictions, ground_truth)

        assert metrics["pair_precision"] == 1.0
        assert metrics["pair_recall"] == 1.0
        assert metrics["pair_f1"] == 1.0

    def test_wrong_pairs(self):
        """Test con coppie sbagliate."""
        predictions = [
            DisagreementAnalysis(
                has_disagreement=True,
                conflicting_pairs=[
                    ExpertPairConflict(
                        expert_a="literal",
                        expert_b="systemic",
                        conflict_score=0.8,
                    ),
                ],
            ),
        ]

        ground_truth = [
            DisagreementSample(
                sample_id="s1",
                query="q1",
                expert_responses={},
                has_disagreement=True,
                conflicting_pairs=[("principles", "precedent")],
            ),
        ]

        metrics = compute_pairwise_metrics(predictions, ground_truth)

        assert metrics["pair_precision"] == 0.0
        assert metrics["pair_recall"] == 0.0
        assert metrics["pair_f1"] == 0.0

    def test_empty_pairs(self):
        """Test con coppie vuote."""
        predictions = [
            DisagreementAnalysis(
                has_disagreement=True,
                conflicting_pairs=[],
            ),
        ]

        ground_truth = [
            DisagreementSample(
                sample_id="s1",
                query="q1",
                expert_responses={},
                has_disagreement=True,
                conflicting_pairs=[],
            ),
        ]

        metrics = compute_pairwise_metrics(predictions, ground_truth)

        # Con 0 coppie, metriche sono 0
        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0


class TestDisagreementMetricsDataclass:
    """Test per DisagreementMetrics dataclass."""

    def test_default_values(self):
        """Test valori default."""
        metrics = DisagreementMetrics()

        assert metrics.binary_accuracy == 0.0
        assert metrics.type_macro_f1 == 0.0
        assert metrics.n_samples == 0
        assert metrics.type_per_class == {}

    def test_custom_values(self):
        """Test con valori custom."""
        metrics = DisagreementMetrics(
            binary_accuracy=0.85,
            binary_f1=0.82,
            type_accuracy=0.75,
            type_macro_f1=0.70,
            n_samples=100,
            n_positive=60,
            n_negative=40,
        )

        assert metrics.binary_accuracy == 0.85
        assert metrics.n_positive == 60
