"""
Test per Off-Policy Evaluation
==============================

Test completi per OPEEvaluator e metodi correlati.
"""

import pytest
import math
from unittest.mock import MagicMock, patch
import torch

from merlt.rlcf.off_policy_eval import (
    OPEMethod,
    OPEDataPoint,
    OPEResult,
    OPEConfig,
    OPEEvaluator,
    PolicyComparisonResult,
    compare_two_policies,
    create_ope_evaluator
)
from merlt.rlcf.policy_gradient import GatingPolicy


# =============================================================================
# TEST OPE METHOD
# =============================================================================

class TestOPEMethod:
    """Test per OPEMethod enum."""

    def test_values(self):
        """Test valori enum."""
        assert OPEMethod.IS.value == "importance_sampling"
        assert OPEMethod.WIS.value == "weighted_importance_sampling"
        assert OPEMethod.PDIS.value == "per_decision_is"
        assert OPEMethod.DR.value == "doubly_robust"


# =============================================================================
# TEST OPE DATAPOINT
# =============================================================================

class TestOPEDataPoint:
    """Test per OPEDataPoint."""

    def test_create_datapoint(self):
        """Test creazione datapoint."""
        dp = OPEDataPoint(
            state=[1.0, 2.0],
            action=[0.5, 0.5],
            reward=0.8,
            old_log_prob=-0.5
        )

        assert dp.reward == 0.8
        assert dp.old_log_prob == -0.5
        assert dp.new_log_prob is None

    def test_compute_weight_no_new_prob(self):
        """Test compute weight senza new_log_prob."""
        dp = OPEDataPoint(
            state=[1.0],
            action=[0.5],
            reward=0.5,
            old_log_prob=-0.5
        )

        weight = dp.compute_weight()

        assert weight == 1.0

    def test_compute_weight_same_prob(self):
        """Test weight quando probabilità uguali."""
        dp = OPEDataPoint(
            state=[1.0],
            action=[0.5],
            reward=0.5,
            old_log_prob=-0.5,
            new_log_prob=-0.5
        )

        weight = dp.compute_weight()

        assert abs(weight - 1.0) < 0.001  # exp(0) = 1

    def test_compute_weight_higher_new_prob(self):
        """Test weight quando nuova prob più alta."""
        dp = OPEDataPoint(
            state=[1.0],
            action=[0.5],
            reward=0.5,
            old_log_prob=-1.0,  # Lower prob
            new_log_prob=-0.5   # Higher prob
        )

        weight = dp.compute_weight()

        # exp(-0.5 - (-1.0)) = exp(0.5) ≈ 1.65
        assert weight > 1.0

    def test_compute_weight_lower_new_prob(self):
        """Test weight quando nuova prob più bassa."""
        dp = OPEDataPoint(
            state=[1.0],
            action=[0.5],
            reward=0.5,
            old_log_prob=-0.5,  # Higher prob
            new_log_prob=-1.0   # Lower prob
        )

        weight = dp.compute_weight()

        # exp(-1.0 - (-0.5)) = exp(-0.5) ≈ 0.61
        assert weight < 1.0


# =============================================================================
# TEST OPE RESULT
# =============================================================================

class TestOPEResult:
    """Test per OPEResult."""

    def test_create_result(self):
        """Test creazione result."""
        result = OPEResult(
            method=OPEMethod.WIS,
            estimated_value=0.75,
            ci_lower=0.6,
            ci_upper=0.9,
            effective_sample_size=50.0,
            n_samples=100
        )

        assert result.estimated_value == 0.75
        assert result.effective_sample_size == 50.0

    def test_to_dict(self):
        """Test serializzazione."""
        result = OPEResult(
            method=OPEMethod.IS,
            estimated_value=0.789123,
            ci_lower=0.5,
            ci_upper=1.0,
            effective_sample_size=75.5,
            n_samples=100
        )

        data = result.to_dict()

        assert data["method"] == "importance_sampling"
        assert data["estimated_value"] == 0.7891
        assert data["ess_ratio"] == 0.755


# =============================================================================
# TEST OPE CONFIG
# =============================================================================

class TestOPEConfig:
    """Test per OPEConfig."""

    def test_default_config(self):
        """Test config default."""
        config = OPEConfig()

        assert config.clip_weights is True
        assert config.max_weight == 100.0
        assert config.confidence_level == 0.95

    def test_custom_config(self):
        """Test config custom."""
        config = OPEConfig(
            clip_weights=False,
            max_weight=50.0,
            bootstrap_samples=500
        )

        assert config.clip_weights is False
        assert config.max_weight == 50.0
        assert config.bootstrap_samples == 500

    def test_to_dict(self):
        """Test serializzazione."""
        config = OPEConfig()
        data = config.to_dict()

        assert "clip_weights" in data
        assert "confidence_level" in data


# =============================================================================
# TEST OPE EVALUATOR
# =============================================================================

class TestOPEEvaluator:
    """Test per OPEEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Crea evaluator per test."""
        return OPEEvaluator()

    @pytest.fixture
    def mock_policy(self):
        """Crea mock policy per test."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4, device="cpu")
        return policy

    @pytest.fixture
    def sample_data(self):
        """Crea dati sample per test."""
        return [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.25, 0.25, 0.25, 0.25],
                "reward": 0.7,
                "old_log_prob": -1.0
            },
            {
                "state": torch.randn(64).tolist(),
                "action": [0.4, 0.2, 0.2, 0.2],
                "reward": 0.8,
                "old_log_prob": -0.8
            },
            {
                "state": torch.randn(64).tolist(),
                "action": [0.1, 0.5, 0.2, 0.2],
                "reward": 0.6,
                "old_log_prob": -1.2
            }
        ]

    def test_create_evaluator(self):
        """Test creazione evaluator."""
        evaluator = OPEEvaluator()
        assert evaluator.config is not None

    def test_create_with_config(self):
        """Test con config custom."""
        config = OPEConfig(clip_weights=False)
        evaluator = OPEEvaluator(config)

        assert evaluator.config.clip_weights is False

    def test_compute_importance_weights(self, evaluator, mock_policy, sample_data):
        """Test calcolo importance weights."""
        datapoints = evaluator.compute_importance_weights(
            sample_data, mock_policy, device="cpu"
        )

        assert len(datapoints) == 3
        assert all(dp.importance_weight is not None for dp in datapoints)
        assert all(dp.new_log_prob is not None for dp in datapoints)

    def test_importance_sampling(self, evaluator):
        """Test stima IS."""
        datapoints = [
            OPEDataPoint([1], [0.5], reward=0.8, old_log_prob=-0.5,
                        new_log_prob=-0.5),
            OPEDataPoint([2], [0.5], reward=0.6, old_log_prob=-0.5,
                        new_log_prob=-0.5),
        ]
        for dp in datapoints:
            dp.compute_weight()

        estimated, stats = evaluator.importance_sampling(datapoints)

        # Con pesi = 1, media = (0.8 + 0.6) / 2 = 0.7
        assert abs(estimated - 0.7) < 0.001

    def test_weighted_importance_sampling(self, evaluator):
        """Test stima WIS."""
        datapoints = [
            OPEDataPoint([1], [0.5], reward=0.8, old_log_prob=-0.5,
                        new_log_prob=-0.5),
            OPEDataPoint([2], [0.5], reward=0.6, old_log_prob=-0.5,
                        new_log_prob=-0.5),
        ]
        for dp in datapoints:
            dp.compute_weight()

        estimated, stats = evaluator.weighted_importance_sampling(datapoints)

        # Con pesi = 1, stesso risultato di IS
        assert abs(estimated - 0.7) < 0.001

    def test_wis_with_different_weights(self, evaluator):
        """Test WIS con pesi diversi."""
        datapoints = [
            OPEDataPoint([1], [0.5], reward=1.0, old_log_prob=-1.0,
                        new_log_prob=-0.5),  # Weight > 1
            OPEDataPoint([2], [0.5], reward=0.0, old_log_prob=-0.5,
                        new_log_prob=-1.0),  # Weight < 1
        ]
        for dp in datapoints:
            dp.compute_weight()

        estimated, stats = evaluator.weighted_importance_sampling(datapoints)

        # Sample con reward alto ha peso alto -> stima > 0.5
        # (media semplice sarebbe 0.5)
        assert estimated > 0.5

    def test_effective_sample_size(self, evaluator):
        """Test calcolo ESS."""
        # Pesi uniformi -> ESS = n
        weights_uniform = [1.0, 1.0, 1.0, 1.0]
        ess_uniform = evaluator.compute_effective_sample_size(weights_uniform)
        assert abs(ess_uniform - 4.0) < 0.001

        # Un peso dominante -> ESS basso
        weights_skewed = [10.0, 0.1, 0.1, 0.1]
        ess_skewed = evaluator.compute_effective_sample_size(weights_skewed)
        assert ess_skewed < 4.0

    def test_effective_sample_size_empty(self, evaluator):
        """Test ESS con lista vuota."""
        ess = evaluator.compute_effective_sample_size([])
        assert ess == 0.0

    def test_evaluate(self, evaluator, mock_policy, sample_data):
        """Test valutazione completa."""
        result = evaluator.evaluate(
            mock_policy, sample_data,
            method=OPEMethod.WIS,
            compute_ci=False
        )

        assert isinstance(result, OPEResult)
        assert result.method == OPEMethod.WIS
        assert 0 <= result.estimated_value <= 1
        assert result.n_samples == 3

    def test_evaluate_with_ci(self, evaluator, mock_policy):
        """Test con confidence interval."""
        # Dati più grandi per bootstrap
        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.25, 0.25, 0.25, 0.25],
                "reward": 0.5 + 0.3 * (i % 3) / 3,
                "old_log_prob": -0.8
            }
            for i in range(50)
        ]

        result = evaluator.evaluate(
            mock_policy, data,
            method=OPEMethod.WIS,
            compute_ci=True
        )

        assert result.ci_lower <= result.estimated_value <= result.ci_upper
        assert result.ci_lower < result.ci_upper

    def test_evaluate_empty_data(self, evaluator, mock_policy):
        """Test con dati vuoti."""
        result = evaluator.evaluate(mock_policy, [], method=OPEMethod.IS)

        assert result.estimated_value == 0.0
        assert result.n_samples == 0

    def test_evaluate_is_method(self, evaluator, mock_policy, sample_data):
        """Test metodo IS esplicito."""
        result = evaluator.evaluate(
            mock_policy, sample_data,
            method=OPEMethod.IS
        )

        assert result.method == OPEMethod.IS

    def test_compare_policies(self, evaluator, sample_data):
        """Test confronto multiple policy."""
        policies = [
            GatingPolicy(input_dim=64, hidden_dim=32, device="cpu"),
            GatingPolicy(input_dim=64, hidden_dim=32, device="cpu"),
            GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        ]

        results = evaluator.compare_policies(policies, sample_data)

        assert len(results) == 3
        # Dovrebbero essere ordinate per valore
        assert results[0].estimated_value >= results[1].estimated_value
        assert results[1].estimated_value >= results[2].estimated_value

    def test_weight_clipping(self, mock_policy):
        """Test clipping dei pesi."""
        config = OPEConfig(
            clip_weights=True,
            max_weight=5.0,
            min_weight=0.5
        )
        evaluator = OPEEvaluator(config)

        # Dati che potrebbero avere pesi estremi
        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.9, 0.03, 0.03, 0.04],  # Azione estrema
                "reward": 0.5,
                "old_log_prob": -5.0  # Molto diverso
            }
        ]

        datapoints = evaluator.compute_importance_weights(data, mock_policy)

        # Peso dovrebbe essere clippato
        assert datapoints[0].importance_weight <= 5.0
        assert datapoints[0].importance_weight >= 0.5


# =============================================================================
# TEST COMPARE TWO POLICIES
# =============================================================================

class TestCompareTwoPolicies:
    """Test per compare_two_policies helper."""

    def test_compare_same_policy(self):
        """Test confronto stessa policy (differenza ~0)."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")

        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.25, 0.25, 0.25, 0.25],
                "reward": 0.5,
                "old_log_prob": -1.0
            }
            for _ in range(20)
        ]

        result = compare_two_policies(policy, policy, data)

        # Stessa policy -> differenza piccola
        assert abs(result.difference) < 0.3

    def test_compare_different_policies(self):
        """Test confronto policy diverse."""
        policy_a = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        policy_b = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")

        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.25, 0.25, 0.25, 0.25],
                "reward": 0.5 + 0.2 * (i % 3) / 3,
                "old_log_prob": -0.8
            }
            for i in range(30)
        ]

        result = compare_two_policies(policy_a, policy_b, data)

        assert isinstance(result, PolicyComparisonResult)
        assert result.ci_lower <= result.difference <= result.ci_upper

    def test_comparison_result_to_dict(self):
        """Test serializzazione risultato."""
        result = PolicyComparisonResult(
            policy_a_value=0.75,
            policy_b_value=0.65,
            difference=0.10,
            ci_lower=0.02,
            ci_upper=0.18,
            significant=True,
            preferred="a"
        )

        data = result.to_dict()

        assert data["difference"] == 0.1
        assert data["significant"] is True
        assert data["preferred"] == "a"


# =============================================================================
# TEST FACTORY
# =============================================================================

class TestCreateOPEEvaluator:
    """Test per factory function."""

    def test_create_default(self):
        """Test creazione default."""
        evaluator = create_ope_evaluator()

        assert isinstance(evaluator, OPEEvaluator)
        assert evaluator.config.clip_weights is True

    def test_create_custom(self):
        """Test con parametri custom."""
        evaluator = create_ope_evaluator(
            clip_weights=False,
            max_weight=50.0,
            confidence_level=0.99
        )

        assert evaluator.config.clip_weights is False
        assert evaluator.config.max_weight == 50.0
        assert evaluator.config.confidence_level == 0.99


# =============================================================================
# TEST INTEGRAZIONE
# =============================================================================

class TestOPEIntegration:
    """Test di integrazione OPE."""

    def test_full_evaluation_workflow(self):
        """Test workflow completo di valutazione."""
        # Setup
        policy = GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4, device="cpu")
        evaluator = OPEEvaluator()

        # Genera dati storici simulati
        historical_data = []
        for i in range(100):
            state = torch.randn(64).tolist()

            # Simula azione da vecchia policy (uniform)
            action = [0.25, 0.25, 0.25, 0.25]

            # Simula reward
            reward = 0.5 + 0.3 * (i % 5) / 5

            # Log prob sotto vecchia policy (uniform softmax)
            old_log_prob = math.log(0.25) * 4  # ~-5.5

            historical_data.append({
                "state": state,
                "action": action,
                "reward": reward,
                "old_log_prob": old_log_prob
            })

        # Valuta nuova policy
        result = evaluator.evaluate(policy, historical_data, compute_ci=True)

        # Verifica risultati
        assert 0 <= result.estimated_value <= 1
        assert result.ci_lower <= result.estimated_value <= result.ci_upper
        assert result.effective_sample_size > 0
        assert result.n_samples == 100

        # Verifica diagnostiche
        assert "ess_ratio" in result.diagnostics

    def test_policy_selection_workflow(self):
        """Test workflow selezione policy migliore."""
        # Crea diverse policy
        policies = [
            GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
            for _ in range(3)
        ]

        # Genera dati
        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.25, 0.25, 0.25, 0.25],
                "reward": 0.5 + 0.2 * (i % 4) / 4,
                "old_log_prob": -1.0
            }
            for i in range(50)
        ]

        # Confronta e seleziona
        evaluator = OPEEvaluator()
        results = evaluator.compare_policies(policies, data)

        # Seleziona migliore
        best_idx = results[0].diagnostics["policy_index"]
        best_policy = policies[best_idx]
        best_value = results[0].estimated_value

        # Verifica
        assert isinstance(best_policy, GatingPolicy)
        assert all(r.estimated_value <= best_value for r in results)

    def test_diagnostic_flags(self):
        """Test flag diagnostiche."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")

        # Dati con pesi estremi
        data = [
            {
                "state": torch.randn(64).tolist(),
                "action": [0.9, 0.03, 0.03, 0.04],  # Molto skewed
                "reward": 0.5,
                "old_log_prob": -5.0  # Molto basso
            }
            for _ in range(20)
        ]

        evaluator = OPEEvaluator()
        result = evaluator.evaluate(policy, data)

        # Dovrebbe avere flag di warning
        assert "extreme_weights" in result.diagnostics
        assert "ess_warning" in result.diagnostics
