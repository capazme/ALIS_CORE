"""
Tests for weight configuration Pydantic models.
"""
import pytest
from pydantic import ValidationError


class TestLearnableWeight:
    """Tests for LearnableWeight model."""

    def test_learnable_weight_creation(self):
        """Test creating a valid LearnableWeight."""
        from weights.config import LearnableWeight

        weight = LearnableWeight(
            default=0.7,
            bounds=(0.3, 0.9),
            learnable=True,
            learning_rate=0.01,
        )

        assert weight.default == 0.7
        assert weight.bounds == (0.3, 0.9)
        assert weight.learnable is True

    def test_learnable_weight_default_must_be_in_bounds(self):
        """Test that default must be within bounds."""
        from weights.config import LearnableWeight

        with pytest.raises(ValidationError):
            LearnableWeight(
                default=0.1,  # Outside bounds
                bounds=(0.3, 0.9),
            )


class TestRetrievalWeights:
    """Tests for RetrievalWeights model."""

    def test_retrieval_weights_defaults(self):
        """Test default values for RetrievalWeights."""
        from weights.config import RetrievalWeights

        weights = RetrievalWeights()

        assert weights.alpha.default == 0.7
        assert weights.over_retrieve_factor == 3
        assert weights.max_graph_hops == 3

    def test_retrieval_weights_custom_alpha(self):
        """Test custom alpha value."""
        from weights.config import RetrievalWeights, LearnableWeight

        weights = RetrievalWeights(
            alpha=LearnableWeight(default=0.5, bounds=(0.3, 0.9))
        )

        assert weights.alpha.default == 0.5


class TestRLCFAuthorityWeights:
    """Tests for RLCFAuthorityWeights model."""

    def test_rlcf_weights_defaults(self):
        """Test default RLCF authority weights."""
        from weights.config import RLCFAuthorityWeights

        weights = RLCFAuthorityWeights()

        # Default weights should be set
        assert weights.baseline_credentials.default == 0.4
        assert weights.track_record.default == 0.4
        assert weights.recent_performance.default == 0.2


class TestGatingWeights:
    """Tests for GatingWeights model."""

    def test_gating_weights_has_expert_priors(self):
        """Test that gating weights include expert priors."""
        from weights.config import GatingWeights

        weights = GatingWeights()

        assert "LiteralExpert" in weights.expert_priors
        assert "SystemicExpert" in weights.expert_priors
        assert "PrinciplesExpert" in weights.expert_priors
        assert "PrecedentExpert" in weights.expert_priors

    def test_gating_weights_query_modifiers(self):
        """Test query type modifiers are present."""
        from weights.config import GatingWeights

        weights = GatingWeights()

        assert "definitorio" in weights.query_type_modifiers
        assert "interpretativo" in weights.query_type_modifiers


class TestWeightConfig:
    """Tests for WeightConfig model."""

    def test_weight_config_creation(self, sample_weight_config):
        """Test creating a full WeightConfig."""
        from weights.config import WeightConfig

        config = WeightConfig(**sample_weight_config)

        assert config.version == "2.0"
        assert config.retrieval.alpha.default == 0.7

    def test_weight_config_get_retrieval_alpha(self):
        """Test get_retrieval_alpha helper method."""
        from weights.config import WeightConfig

        config = WeightConfig()

        alpha = config.get_retrieval_alpha()

        assert alpha == 0.7  # Default value

    def test_weight_config_get_gating_prior(self):
        """Test get_gating_prior helper method."""
        from weights.config import WeightConfig

        config = WeightConfig()

        prior = config.get_gating_prior("LiteralExpert")

        assert prior == 0.25  # Default uniform prior

    def test_weight_config_get_gating_prior_unknown_expert(self):
        """Test get_gating_prior for unknown expert returns default."""
        from weights.config import WeightConfig

        config = WeightConfig()

        prior = config.get_gating_prior("UnknownExpert")

        assert prior == 0.25  # Default fallback


class TestExperimentConfig:
    """Tests for ExperimentConfig model."""

    def test_experiment_config_requires_two_variants(self):
        """Test that experiment needs at least 2 variants."""
        from weights.config import ExperimentConfig, ExperimentVariant, WeightConfig

        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test_experiment",
                variants=[
                    ExperimentVariant(
                        name="control",
                        weights=WeightConfig(),
                        allocation_ratio=1.0,
                    )
                ],  # Only 1 variant
            )

    def test_experiment_config_allocation_must_sum_to_one(self):
        """Test that allocation ratios must sum to 1."""
        from weights.config import ExperimentConfig, ExperimentVariant, WeightConfig

        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test_experiment",
                variants=[
                    ExperimentVariant(
                        name="control",
                        weights=WeightConfig(),
                        allocation_ratio=0.3,  # Total 0.6, not 1.0
                    ),
                    ExperimentVariant(
                        name="treatment",
                        weights=WeightConfig(),
                        allocation_ratio=0.3,
                    ),
                ],
            )
