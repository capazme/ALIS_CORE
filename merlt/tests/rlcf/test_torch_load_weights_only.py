"""
Test torch.load weights_only parameter (STORY-12-4)
====================================================

Tests for:
- weights_only=True on inference-only checkpoint loads
- weights_only=False on trainer checkpoint loads (optimizer state)
- Audit: all torch.load calls have explicit weights_only parameter

Example:
    pytest tests/rlcf/test_torch_load_weights_only.py -v
"""

import pytest
import torch
from pathlib import Path

from merlt.rlcf.policy_gradient import GatingPolicy, TraversalPolicy, PolicyGradientTrainer
from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def pm(checkpoint_dir):
    """PolicyManager with tmp checkpoint dir."""
    config = PolicyConfig(checkpoint_dir=checkpoint_dir)
    return PolicyManager(config=config)


# =============================================================================
# TEST weights_only=True (inference loads)
# =============================================================================


class TestWeightsOnlyTrue:
    """Inference checkpoint loads use weights_only=True."""

    def test_gating_policy_loads_with_weights_only(self, pm, checkpoint_dir):
        """PolicyManager._load_gating_policy() works with weights_only=True."""
        policy = GatingPolicy(input_dim=1024, hidden_dim=256, num_experts=4)
        ckpt = {
            "input_dim": 1024,
            "hidden_dim": 256,
            "num_experts": 4,
            "mlp_state_dict": policy.mlp.state_dict(),
        }
        torch.save(ckpt, checkpoint_dir / "gating_policy_latest.pt")

        loaded = pm._load_gating_policy()
        assert loaded is not None
        assert loaded.input_dim == 1024

    def test_traversal_policy_loads_with_weights_only(self, pm, checkpoint_dir):
        """PolicyManager._load_traversal_policy() works with weights_only=True."""
        policy = TraversalPolicy(input_dim=1024, relation_dim=64, hidden_dim=128)
        ckpt = {
            "input_dim": 1024,
            "relation_dim": 64,
            "hidden_dim": 128,
            "mlp_state_dict": policy.mlp.state_dict(),
            "relation_embeddings_state_dict": policy.relation_embeddings.state_dict(),
        }
        torch.save(ckpt, checkpoint_dir / "traversal_policy_latest.pt")

        loaded = pm._load_traversal_policy()
        assert loaded is not None
        assert loaded.input_dim == 1024


# =============================================================================
# TEST weights_only=False (trainer loads with optimizer)
# =============================================================================


class TestWeightsOnlyFalse:
    """Trainer checkpoint loads use weights_only=False for optimizer state."""

    def test_policy_gradient_trainer_roundtrip(self, tmp_path):
        """PolicyGradientTrainer save/load checkpoint with optimizer state."""
        policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer = PolicyGradientTrainer(policy=policy)

        ckpt_path = str(tmp_path / "pg_checkpoint.pt")
        trainer.save_checkpoint(ckpt_path)

        policy2 = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer2 = PolicyGradientTrainer(policy=policy2)
        trainer2.load_checkpoint(ckpt_path)

    def test_single_step_trainer_roundtrip(self, tmp_path):
        """SingleStepTrainer save/load checkpoint with optimizer state."""
        from merlt.rlcf.single_step_trainer import SingleStepTrainer

        policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer = SingleStepTrainer(policy=policy)

        ckpt_path = str(tmp_path / "ss_checkpoint.pt")
        trainer.save_checkpoint(ckpt_path)

        policy2 = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer2 = SingleStepTrainer(policy=policy2)
        trainer2.load_checkpoint(ckpt_path)

    def test_ppo_trainer_roundtrip(self, tmp_path):
        """PPOTrainer save/load checkpoint with optimizer state."""
        from merlt.rlcf.ppo_trainer import PPOTrainer

        policy = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer = PPOTrainer(policy=policy)

        ckpt_path = str(tmp_path / "ppo_checkpoint.pt")
        trainer.save_checkpoint(ckpt_path)

        policy2 = GatingPolicy(input_dim=32, hidden_dim=16, num_experts=4)
        trainer2 = PPOTrainer(policy=policy2)
        trainer2.load_checkpoint(ckpt_path)


# =============================================================================
# TEST audit completeness
# =============================================================================


class TestAuditCompleteness:
    """Verify all torch.load calls have explicit weights_only parameter."""

    def test_no_torch_load_without_weights_only(self):
        """All torch.load calls in merlt/ have explicit weights_only param."""
        import re

        merlt_dir = Path(__file__).parent.parent.parent / "merlt"
        pattern = re.compile(r"torch\.load\([^)]*\)")

        missing = []
        for py_file in merlt_dir.rglob("*.py"):
            content = py_file.read_text()
            for match in pattern.finditer(content):
                call = match.group()
                if "weights_only" not in call:
                    missing.append(f"{py_file.relative_to(merlt_dir)}:{call}")

        assert not missing, f"torch.load without weights_only:\n" + "\n".join(missing)
