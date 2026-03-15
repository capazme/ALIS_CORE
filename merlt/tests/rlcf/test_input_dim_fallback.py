"""
Test input_dim fallback correctness (STORY-12-1)
=================================================

Verifies that PolicyManager checkpoint fallback uses 1024 (E5-large)
instead of the legacy 768 (BERT-base) default.

Example:
    pytest tests/rlcf/test_input_dim_fallback.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from merlt.rlcf.policy_gradient import GatingPolicy, TraversalPolicy
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
# TEST PolicyManager FALLBACK DEFAULTS
# =============================================================================


class TestGatingPolicyFallback:
    """Gating policy checkpoint without input_dim key uses 1024."""

    def test_fallback_creates_1024_dim_policy(self, pm, checkpoint_dir):
        """Checkpoint missing input_dim key -> ExpertGatingMLP(input_dim=1024)."""
        from merlt.experts.neural_gating.neural import ExpertGatingMLP, GatingConfig

        # Save a checkpoint WITHOUT input_dim key (uses model_state_dict format)
        mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
        ckpt = {
            "model_state_dict": mlp.state_dict(),
            # NOTE: no "input_dim" key — triggers fallback to default 1024
        }
        ckpt_path = checkpoint_dir / "gating_policy_latest.pt"
        torch.save(ckpt, ckpt_path)

        policy = pm._load_gating_policy()
        assert policy is not None
        assert policy.config.input_dim == 1024

    def test_explicit_input_dim_respected(self, pm, checkpoint_dir):
        """Checkpoint with explicit input_dim=512 -> uses 512."""
        from merlt.experts.neural_gating.neural import ExpertGatingMLP, GatingConfig

        mlp = ExpertGatingMLP(GatingConfig(input_dim=512))
        ckpt = {
            "model_state_dict": mlp.state_dict(),
            "input_dim": 512,
        }
        ckpt_path = checkpoint_dir / "gating_policy_latest.pt"
        torch.save(ckpt, ckpt_path)

        policy = pm._load_gating_policy()
        assert policy is not None
        assert policy.config.input_dim == 512

    def test_no_checkpoint_returns_none(self, pm):
        """No checkpoint file -> returns None."""
        policy = pm._load_gating_policy()
        assert policy is None


class TestTraversalPolicyFallback:
    """Traversal policy checkpoint without input_dim key uses 1024."""

    def test_fallback_creates_1024_dim_policy(self, pm, checkpoint_dir):
        """Checkpoint missing input_dim -> TraversalPolicy(input_dim=1024)."""
        tp = TraversalPolicy(input_dim=1024, relation_dim=64, hidden_dim=128)
        ckpt = {
            "mlp_state_dict": tp.mlp.state_dict(),
            "relation_embeddings_state_dict": tp.relation_embeddings.state_dict(),
            "relation_dim": 64,
            "hidden_dim": 128,
            # NOTE: no "input_dim" key
        }
        ckpt_path = checkpoint_dir / "traversal_policy_latest.pt"
        torch.save(ckpt, ckpt_path)

        policy = pm._load_traversal_policy()
        assert policy is not None
        assert policy.input_dim == 1024

    def test_explicit_input_dim_respected(self, pm, checkpoint_dir):
        """Checkpoint with explicit input_dim -> uses that value."""
        tp = TraversalPolicy(input_dim=256, relation_dim=64, hidden_dim=128)
        ckpt = {
            "mlp_state_dict": tp.mlp.state_dict(),
            "relation_embeddings_state_dict": tp.relation_embeddings.state_dict(),
            "input_dim": 256,
            "relation_dim": 64,
            "hidden_dim": 128,
        }
        ckpt_path = checkpoint_dir / "traversal_policy_latest.pt"
        torch.save(ckpt, ckpt_path)

        policy = pm._load_traversal_policy()
        assert policy is not None
        assert policy.input_dim == 256


# =============================================================================
# TEST ValueNetwork DEFAULT
# =============================================================================


class TestValueNetworkDefault:
    """ValueNetwork default input_dim is 1024."""

    def test_default_input_dim(self):
        from merlt.rlcf.ppo_trainer import ValueNetwork
        vn = ValueNetwork()
        assert vn.input_dim == 1024

    def test_custom_input_dim(self):
        from merlt.rlcf.ppo_trainer import ValueNetwork
        vn = ValueNetwork(input_dim=512)
        assert vn.input_dim == 512


# =============================================================================
# TEST NO STALE 768 REFERENCES
# =============================================================================


class TestNo768References:
    """Verify no stale 768 defaults remain in critical code paths."""

    def test_gating_policy_default_is_1024(self):
        """GatingPolicy default input_dim is 1024."""
        policy = GatingPolicy()
        assert policy.input_dim == 1024

    def test_traversal_policy_default_is_1024(self):
        """TraversalPolicy default input_dim is 1024."""
        policy = TraversalPolicy()
        assert policy.input_dim == 1024

    def test_prompt_policy_default_is_1024(self):
        """PromptPolicy default input_dim is 1024."""
        from merlt.rlcf.prompt_policy import PromptPolicy
        policy = PromptPolicy()
        assert policy.input_dim == 1024
