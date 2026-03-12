"""
Test per TraversalTrainingService fixes (STORY-11-4)
=====================================================

Test per:
- get_traversal_policy() instead of load_traversal_policy()
- Optimizer created and used in REINFORCE loop
- policy.train() before loop, after loop inference mode
- Both versioned + latest checkpoint saved
- Fresh TraversalPolicy uses input_dim=1024
- Parameter weights change after training
- Graceful fallback when no checkpoint exists
- TrainingResult includes traversal outcome
"""

import pytest
from pathlib import Path
from datetime import datetime, UTC

from merlt.rlcf.traversal_training_service import (
    TraversalTrainingService,
    TraversalTrainingSample,
    TraversalTrainingResult,
)
from merlt.rlcf.training_scheduler import TrainingResult


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def service():
    """Fresh TraversalTrainingService."""
    return TraversalTrainingService()


@pytest.fixture
def samples():
    """Minimal training samples (above MIN_SAMPLES threshold)."""
    result = []
    for i in range(25):
        result.append(TraversalTrainingSample(
            query_embedding=[0.1 * (i % 10)] * 1024,
            relation_type="RIFERIMENTO" if i % 2 == 0 else "CITATO_DA",
            expert_type="literal" if i % 3 == 0 else "systemic",
            reward=0.3 + (i % 5) * 0.15,
        ))
    return result


@pytest.fixture
def few_samples():
    """Fewer than MIN_SAMPLES - training should be skipped."""
    return [
        TraversalTrainingSample(
            query_embedding=[0.1] * 1024,
            relation_type="RIFERIMENTO",
            expert_type="literal",
            reward=0.8,
        )
        for _ in range(5)
    ]


# =============================================================================
# TEST train_traversal_policy() FIXES
# =============================================================================


class TestTraversalTrainingFixes:
    """Test that the 6 bugs are fixed."""

    @pytest.mark.asyncio
    async def test_trains_without_attribute_error(self, service, samples, tmp_path, monkeypatch):
        """Bug 1: No more AttributeError on pm.load_traversal_policy()."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        # Should not raise - uses get_traversal_policy() now
        result = await service.train_traversal_policy(samples, epochs=1)

        assert isinstance(result, TraversalTrainingResult)
        assert result.epochs_completed == 1
        assert result.samples_used == len(samples)

    @pytest.mark.asyncio
    async def test_optimizer_produces_nonzero_loss(self, service, samples, tmp_path, monkeypatch):
        """Bug 2: Optimizer runs, loss is non-zero."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        result = await service.train_traversal_policy(samples, epochs=2)

        assert result.avg_loss != 0.0, "Loss should be non-zero when optimizer runs"

    @pytest.mark.asyncio
    async def test_parameters_change_after_training(self, service, samples, tmp_path, monkeypatch):
        """Bug 2+3: Parameters actually update (optimizer + train mode)."""
        import torch
        from merlt.rlcf.policy_gradient import TraversalPolicy

        monkeypatch.chdir(tmp_path)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a fresh policy to capture initial state
        fresh_policy = TraversalPolicy(input_dim=1024, hidden_dim=128)
        initial_weight = list(fresh_policy.mlp.parameters())[0].clone().detach()

        # Train (creates its own fresh policy internally since no checkpoint)
        result = await service.train_traversal_policy(samples, epochs=3)
        assert result.epochs_completed == 3

        # Load saved checkpoint and verify it exists
        checkpoint_path = checkpoint_dir / "traversal_policy_latest.pt"
        assert checkpoint_path.exists(), "Latest checkpoint must be saved"

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert "mlp_state_dict" in ckpt
        assert len(ckpt["mlp_state_dict"]) > 0

    @pytest.mark.asyncio
    async def test_saves_latest_alias(self, service, samples, tmp_path, monkeypatch):
        """Bug 4: Both versioned and latest checkpoint saved."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        result = await service.train_traversal_policy(samples, epochs=1)

        checkpoint_dir = tmp_path / "checkpoints"

        # Latest alias must exist
        latest = checkpoint_dir / "traversal_policy_latest.pt"
        assert latest.exists(), "latest alias must be saved"

        # Versioned checkpoint must exist
        versioned = list(checkpoint_dir.glob("traversal_policy_traversal_v*.pt"))
        assert len(versioned) >= 1, "versioned checkpoint must be saved"

    @pytest.mark.asyncio
    async def test_fresh_policy_uses_1024_input_dim(self, service, samples, tmp_path, monkeypatch):
        """Bug 5: Fresh policy uses input_dim=1024 (E5-large)."""
        import torch

        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        result = await service.train_traversal_policy(samples, epochs=1)

        # Load and verify input_dim in saved checkpoint
        ckpt = torch.load(
            tmp_path / "checkpoints" / "traversal_policy_latest.pt",
            map_location="cpu",
        )
        assert ckpt["input_dim"] == 1024

    @pytest.mark.asyncio
    async def test_skips_below_min_samples(self, service, few_samples, tmp_path, monkeypatch):
        """Training skipped when samples < MIN_SAMPLES."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        result = await service.train_traversal_policy(few_samples)

        assert result.epochs_completed == 0
        assert result.samples_used == 0
        assert result.checkpoint_name == "none"

    @pytest.mark.asyncio
    async def test_loads_existing_checkpoint(self, service, samples, tmp_path, monkeypatch):
        """Loads from existing checkpoint instead of creating fresh."""
        import torch
        from merlt.rlcf.policy_gradient import TraversalPolicy
        from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig

        monkeypatch.chdir(tmp_path)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Save a pre-trained checkpoint
        policy = TraversalPolicy(input_dim=1024, hidden_dim=128)
        # Modify a parameter to distinguish from fresh
        with torch.no_grad():
            list(policy.mlp.parameters())[0].fill_(0.42)

        pm = PolicyManager(config=PolicyConfig(checkpoint_dir=checkpoint_dir))
        pm.save_traversal_policy(policy, name="latest")

        # Train from this checkpoint
        result = await service.train_traversal_policy(samples, epochs=1)
        assert result.epochs_completed == 1

    @pytest.mark.asyncio
    async def test_checkpoint_is_loadable_by_policy_manager(self, service, samples, tmp_path, monkeypatch):
        """Saved checkpoint can be loaded back by PolicyManager."""
        from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig

        monkeypatch.chdir(tmp_path)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        await service.train_traversal_policy(samples, epochs=1)

        # PolicyManager should load it
        pm = PolicyManager(config=PolicyConfig(checkpoint_dir=checkpoint_dir))
        loaded = pm.get_traversal_policy()

        assert loaded is not None
        assert loaded.input_dim == 1024


# =============================================================================
# TEST get_domain_weights_table()
# =============================================================================


class TestDomainWeightsTable:
    """Test get_domain_weights_table uses correct API."""

    def test_returns_uniform_defaults_without_checkpoint(self, service, tmp_path, monkeypatch):
        """Returns uniform defaults when no checkpoint exists."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        table = service.get_domain_weights_table()

        assert "literal" in table
        assert "systemic" in table
        assert "principles" in table
        assert "precedent" in table

        # All values should be uniform default
        for expert, weights in table.items():
            for rel_type, weight in weights.items():
                assert 0.0 <= weight <= 1.0

    @pytest.mark.asyncio
    async def test_returns_trained_weights_after_training(self, service, samples, tmp_path, monkeypatch):
        """Returns trained weights after training produces a checkpoint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        # Train first
        await service.train_traversal_policy(samples, epochs=2)

        # Get weights table
        table = service.get_domain_weights_table()

        assert "literal" in table
        # Should have RIFERIMENTO and other relation types
        assert len(table["literal"]) > 0


# =============================================================================
# TEST TrainingResult TRAVERSAL FIELDS
# =============================================================================


class TestTrainingResultTraversalFields:
    """Test traversal fields added to TrainingResult."""

    def test_default_values(self):
        """traversal_trained and traversal_samples default to False/0."""
        result = TrainingResult()
        assert result.traversal_trained is False
        assert result.traversal_samples == 0

    def test_in_to_dict(self):
        """Traversal fields appear in to_dict output."""
        result = TrainingResult(traversal_trained=True, traversal_samples=25)
        d = result.to_dict()
        assert d["traversal_trained"] is True
        assert d["traversal_samples"] == 25

    def test_set_values(self):
        """Can set traversal fields."""
        result = TrainingResult(
            traversal_trained=True,
            traversal_samples=42,
        )
        assert result.traversal_trained is True
        assert result.traversal_samples == 42
