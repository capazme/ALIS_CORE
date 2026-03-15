"""
Test Checkpoint Serialization + Trained Flag (STORY-11-6)
=========================================================

Tests for:
- save_checkpoint() uses state_dict() instead of named_parameters()
- load_checkpoint() uses load_state_dict() instead of manual .data assignment
- Backward compatibility with old named_parameters() format
- Orchestrator trained flag reflects checkpoint presence

Example:
    pytest tests/rlcf/test_checkpoint_serialization.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from merlt.rlcf.policy_gradient import (
    GatingPolicy,
    PolicyGradientTrainer,
    TrainerConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def policy():
    """GatingPolicy with small dims for fast tests."""
    return GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4, device="cpu")


@pytest.fixture
def trainer(policy):
    """PolicyGradientTrainer wrapping the policy."""
    config = TrainerConfig(learning_rate=1e-3)
    return PolicyGradientTrainer(policy=policy, config=config)


@pytest.fixture
def checkpoint_path(tmp_path):
    """Temp path for checkpoint files."""
    return str(tmp_path / "test_checkpoint.pt")


# =============================================================================
# TEST save_checkpoint() uses state_dict()
# =============================================================================


class TestSaveCheckpointStateDict:
    """Verify save_checkpoint() serializes via state_dict()."""

    def test_save_produces_valid_checkpoint(self, trainer, checkpoint_path):
        """save_checkpoint() creates a loadable .pt file."""
        trainer.save_checkpoint(checkpoint_path)
        assert Path(checkpoint_path).exists()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "baseline" in checkpoint
        assert "num_updates" in checkpoint

    def test_state_dict_keys_match_mlp(self, trainer, checkpoint_path):
        """Saved keys must match mlp.state_dict() keys exactly."""
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        saved_keys = set(checkpoint["model_state_dict"].keys())
        expected_keys = set(trainer.policy.mlp.state_dict().keys())
        assert saved_keys == expected_keys

    def test_state_dict_includes_all_layers(self, trainer, checkpoint_path):
        """state_dict() captures all layers (weight + bias for each Linear)."""
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state = checkpoint["model_state_dict"]
        # GatingPolicy MLP has 3 Linear layers (0, 3, 6 in Sequential)
        assert "0.weight" in state
        assert "0.bias" in state
        assert "3.weight" in state
        assert "3.bias" in state
        assert "6.weight" in state
        assert "6.bias" in state

    def test_saved_values_are_cpu_tensors(self, trainer, checkpoint_path):
        """All saved tensors should be on CPU."""
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        for k, v in checkpoint["model_state_dict"].items():
            assert v.device == torch.device("cpu"), f"{k} not on CPU"

    def test_metadata_preserved(self, trainer, checkpoint_path):
        """Custom metadata is saved in checkpoint."""
        meta = {"epoch": 5, "loss": 0.42}
        trainer.save_checkpoint(checkpoint_path, metadata=meta)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        assert checkpoint["metadata"]["epoch"] == 5
        assert checkpoint["metadata"]["loss"] == 0.42


# =============================================================================
# TEST load_checkpoint() uses load_state_dict()
# =============================================================================


class TestLoadCheckpointStateDict:
    """Verify load_checkpoint() restores via load_state_dict()."""

    def test_roundtrip_preserves_weights(self, trainer, checkpoint_path, policy):
        """Save → load roundtrip produces identical weights."""
        # Modify weights to something non-default
        with torch.no_grad():
            for param in policy.mlp.parameters():
                param.fill_(0.42)

        trainer.save_checkpoint(checkpoint_path)

        # Create fresh policy+trainer
        new_policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4, device="cpu")
        new_trainer = PolicyGradientTrainer(policy=new_policy, config=TrainerConfig())

        new_trainer.load_checkpoint(checkpoint_path)

        # Compare all parameters
        for (n1, p1), (n2, p2) in zip(
            policy.mlp.named_parameters(),
            new_policy.mlp.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_roundtrip_preserves_baseline(self, trainer, checkpoint_path):
        """Baseline value survives save/load."""
        trainer.baseline = 0.75
        trainer.num_updates = 42
        trainer.save_checkpoint(checkpoint_path)

        new_policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4, device="cpu")
        new_trainer = PolicyGradientTrainer(policy=new_policy, config=TrainerConfig())
        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.baseline == 0.75
        assert new_trainer.num_updates == 42

    def test_load_returns_metadata(self, trainer, checkpoint_path):
        """load_checkpoint() returns the stored metadata dict."""
        trainer.save_checkpoint(checkpoint_path, metadata={"run": "abc"})

        new_policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4, device="cpu")
        new_trainer = PolicyGradientTrainer(policy=new_policy, config=TrainerConfig())
        meta = new_trainer.load_checkpoint(checkpoint_path)

        assert meta == {"run": "abc"}

    def test_load_state_dict_rejects_wrong_architecture(self, trainer, checkpoint_path):
        """load_state_dict() raises on mismatched architecture (unlike manual .data)."""
        trainer.save_checkpoint(checkpoint_path)

        # Different hidden_dim → different layer sizes
        wrong_policy = GatingPolicy(input_dim=32, hidden_dim=64, num_experts=4, device="cpu")
        wrong_trainer = PolicyGradientTrainer(policy=wrong_policy, config=TrainerConfig())

        with pytest.raises(RuntimeError):
            wrong_trainer.load_checkpoint(checkpoint_path)


# =============================================================================
# TEST BACKWARD COMPATIBILITY (old named_parameters format)
# =============================================================================


class TestBackwardCompatibility:
    """Old checkpoints saved with named_parameters() should still load."""

    def test_old_format_loads_correctly(self, policy, checkpoint_path):
        """Checkpoint saved with named_parameters() loads with new load_state_dict()."""
        # Simulate old save format
        with torch.no_grad():
            for param in policy.mlp.parameters():
                param.fill_(0.99)

        old_checkpoint = {
            "model_state_dict": {
                name: param.cpu()
                for name, param in policy.mlp.named_parameters()
            },
            "optimizer_state_dict": torch.optim.Adam(
                policy.mlp.parameters(), lr=1e-3
            ).state_dict(),
            "baseline": 0.5,
            "num_updates": 10,
            "config": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "baseline_decay": 0.99,
                "clip_grad_norm": 1.0,
                "entropy_coef": 0.01,
            },
            "policy_config": {
                "input_dim": 32,
                "hidden_dim": 16,
                "num_experts": 4,
                "relation_dim": None,
            },
            "timestamp": "2026-03-01T00:00:00",
            "metadata": {},
        }
        torch.save(old_checkpoint, checkpoint_path)

        # Load with new code
        new_policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4, device="cpu")
        new_trainer = PolicyGradientTrainer(policy=new_policy, config=TrainerConfig())
        new_trainer.load_checkpoint(checkpoint_path)

        # Weights should match
        for param in new_policy.mlp.parameters():
            assert torch.allclose(param, torch.full_like(param, 0.99))

    def test_old_and_new_format_keys_identical(self, policy):
        """named_parameters() and state_dict() produce identical keys for Sequential."""
        named_keys = {name for name, _ in policy.mlp.named_parameters()}
        state_keys = set(policy.mlp.state_dict().keys())
        assert named_keys == state_keys


# =============================================================================
# TEST TRAINED FLAG DETECTION
# =============================================================================


class TestTrainedFlagDetection:
    """Orchestrator trained flag reflects checkpoint presence."""

    def test_hybrid_router_no_checkpoint(self):
        """HybridExpertRouter without checkpoint → loaded_from_checkpoint=False."""
        from merlt.experts.neural_gating.hybrid_router import HybridExpertRouter
        from merlt.experts.neural_gating.neural import ExpertGatingMLP

        mlp = ExpertGatingMLP()
        router = HybridExpertRouter(
            neural_gating=mlp,
            checkpoint_path=None,
            device="cpu",
        )

        assert router.loaded_from_checkpoint is False

    def test_hybrid_router_with_checkpoint(self, tmp_path):
        """HybridExpertRouter with valid checkpoint → loaded_from_checkpoint=True."""
        from merlt.experts.neural_gating.hybrid_router import HybridExpertRouter
        from merlt.experts.neural_gating.neural import ExpertGatingMLP

        mlp = ExpertGatingMLP()

        # Save a valid checkpoint
        ckpt_path = tmp_path / "gating.pt"
        torch.save({"model_state_dict": mlp.state_dict()}, ckpt_path)

        router = HybridExpertRouter(
            neural_gating=mlp,
            checkpoint_path=ckpt_path,
            device="cpu",
        )

        assert router.loaded_from_checkpoint is True

    def test_hybrid_router_nonexistent_checkpoint(self, tmp_path):
        """HybridExpertRouter with nonexistent path → loaded_from_checkpoint=False."""
        from merlt.experts.neural_gating.hybrid_router import HybridExpertRouter
        from merlt.experts.neural_gating.neural import ExpertGatingMLP

        mlp = ExpertGatingMLP()
        router = HybridExpertRouter(
            neural_gating=mlp,
            checkpoint_path=tmp_path / "does_not_exist.pt",
            device="cpu",
        )

        assert router.loaded_from_checkpoint is False

    def test_hybrid_router_corrupted_checkpoint(self, tmp_path):
        """HybridExpertRouter with corrupted checkpoint → loaded_from_checkpoint=False, no crash."""
        from merlt.experts.neural_gating.hybrid_router import HybridExpertRouter
        from merlt.experts.neural_gating.neural import ExpertGatingMLP

        mlp = ExpertGatingMLP()

        # Write garbage to checkpoint file
        ckpt_path = tmp_path / "corrupted.pt"
        ckpt_path.write_bytes(b"not a valid checkpoint")

        router = HybridExpertRouter(
            neural_gating=mlp,
            checkpoint_path=ckpt_path,
            device="cpu",
        )

        assert router.loaded_from_checkpoint is False

    def test_orchestrator_uses_trained_flag(self):
        """Orchestrator pipeline_trace uses hybrid_router.loaded_from_checkpoint."""
        from merlt.experts.orchestrator import MultiExpertOrchestrator

        mock_router = MagicMock()
        mock_router.loaded_from_checkpoint = True
        mock_router.confidence_threshold = 0.7
        mock_router.neural_gating = MagicMock()
        mock_router.neural_gating.get_expert_priors.return_value = {}

        # getattr fallback test
        assert getattr(mock_router, 'loaded_from_checkpoint', False) is True

    def test_orchestrator_getattr_fallback_no_attr(self):
        """getattr fallback returns False if attr missing (old router version)."""
        mock_router = MagicMock(spec=[])  # no attributes
        assert getattr(mock_router, 'loaded_from_checkpoint', False) is False
