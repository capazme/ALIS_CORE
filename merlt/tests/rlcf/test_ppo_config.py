"""
Test per PPO Config, ValueNetwork, PPOBuffer, PPOMetrics
=========================================================

Test per le strutture dati e componenti di supporto del PPO trainer.
"""

import pytest
import torch

from merlt.rlcf.ppo_trainer import (
    PPOConfig,
    ValueNetwork,
    PPOBuffer,
    PPOMetrics,
)


# =============================================================================
# TEST PPO CONFIG
# =============================================================================

class TestPPOConfig:
    """Test per PPOConfig."""

    def test_default_config(self):
        """Test config con valori default."""
        config = PPOConfig()

        assert config.clip_ratio == 0.2
        assert config.num_epochs == 4
        assert config.learning_rate == 3e-4
        assert config.value_coef == 0.5
        assert config.entropy_coef == 0.01

    def test_custom_config(self):
        """Test config con valori custom."""
        config = PPOConfig(
            clip_ratio=0.3,
            num_epochs=10,
            learning_rate=1e-3,
            target_kl=0.02
        )

        assert config.clip_ratio == 0.3
        assert config.num_epochs == 10
        assert config.target_kl == 0.02

    def test_to_dict(self):
        """Test serializzazione config."""
        config = PPOConfig(clip_ratio=0.25)
        data = config.to_dict()

        assert data["clip_ratio"] == 0.25
        assert "learning_rate" in data
        assert "gae_lambda" in data


# =============================================================================
# TEST VALUE NETWORK
# =============================================================================

class TestValueNetwork:
    """Test per ValueNetwork."""

    def test_create_network(self):
        """Test creazione network."""
        net = ValueNetwork(input_dim=64, hidden_dim=32)

        assert net.input_dim == 64
        assert net.hidden_dim == 32
        assert net.device in ["cpu", "cuda", "mps"]

    def test_forward(self):
        """Test forward pass."""
        net = ValueNetwork(input_dim=64, hidden_dim=32, device="cpu")
        net.train()

        state = torch.randn(8, 64)  # batch of 8
        values = net.forward(state)

        assert values.shape == (8, 1)

    def test_forward_single(self):
        """Test forward pass singolo sample."""
        net = ValueNetwork(input_dim=64, hidden_dim=32, device="cpu")

        state = torch.randn(1, 64)
        values = net.forward(state)

        assert values.shape == (1, 1)

    def test_parameters(self):
        """Test accesso parametri."""
        net = ValueNetwork(input_dim=64, hidden_dim=32)
        params = list(net.parameters())

        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_to_device(self):
        """Test spostamento su device."""
        net = ValueNetwork(input_dim=64, device="cpu")
        net = net.to("cpu")

        assert net.device == "cpu"

    def test_train_eval_mode(self):
        """Test train/eval mode."""
        net = ValueNetwork(input_dim=64)

        net.train()
        assert net.mlp.training is True

        net.eval()
        assert net.mlp.training is False


# =============================================================================
# TEST PPO BUFFER
# =============================================================================

class TestPPOBuffer:
    """Test per PPOBuffer."""

    def test_create_buffer(self):
        """Test creazione buffer."""
        buffer = PPOBuffer(gamma=0.99, gae_lambda=0.95)

        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95
        assert len(buffer) == 0

    def test_add_experience(self):
        """Test aggiunta esperienza."""
        buffer = PPOBuffer()

        state = torch.randn(64)
        action = torch.randn(4)

        buffer.add(
            state=state,
            action=action,
            reward=0.8,
            log_prob=-0.5,
            value=0.7,
            done=True
        )

        assert len(buffer) == 1

    def test_add_multiple(self):
        """Test aggiunta multiple esperienze."""
        buffer = PPOBuffer()

        for i in range(10):
            buffer.add(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=i / 10,
                log_prob=-0.5,
                value=0.5,
                done=True
            )

        assert len(buffer) == 10

    def test_compute_advantages_simple(self):
        """Test calcolo advantages semplice (episodi singoli)."""
        buffer = PPOBuffer(gamma=0.99)

        # Aggiungi esperienze con done=True (episodi singoli)
        for i in range(5):
            buffer.add(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=1.0,
                log_prob=-0.5,
                value=0.5,  # V(s) = 0.5
                done=True
            )

        buffer.compute_advantages()

        # Per episodi singoli con done=True:
        # advantage = reward - value = 1.0 - 0.5 = 0.5
        assert len(buffer.advantages) == 5
        for adv in buffer.advantages:
            assert abs(adv - 0.5) < 0.01

    def test_compute_gae(self):
        """Test calcolo GAE."""
        buffer = PPOBuffer(gamma=0.99, gae_lambda=0.95)

        # Episodio multi-step
        buffer.add(torch.randn(64), torch.randn(4), reward=1.0, log_prob=-0.5, value=0.5, done=False)
        buffer.add(torch.randn(64), torch.randn(4), reward=1.0, log_prob=-0.5, value=0.6, done=False)
        buffer.add(torch.randn(64), torch.randn(4), reward=1.0, log_prob=-0.5, value=0.7, done=True)

        buffer.compute_gae(last_value=0.0)

        assert len(buffer.advantages) == 3
        assert len(buffer.returns) == 3

    def test_get_batch(self):
        """Test conversione in batch tensors."""
        buffer = PPOBuffer()

        for i in range(5):
            buffer.add(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=0.5,
                log_prob=-0.5,
                value=0.5,
                done=True
            )

        buffer.compute_advantages()
        batch = buffer.get_batch(device="cpu")

        assert "states" in batch
        assert "actions" in batch
        assert "old_log_probs" in batch
        assert "advantages" in batch
        assert "returns" in batch

        assert batch["states"].shape == (5, 64)
        assert batch["actions"].shape == (5, 4)

    def test_clear(self):
        """Test svuotamento buffer."""
        buffer = PPOBuffer()

        for i in range(10):
            buffer.add(torch.randn(64), torch.randn(4), 0.5, -0.5, 0.5, True)

        assert len(buffer) == 10

        buffer.clear()

        assert len(buffer) == 0


# =============================================================================
# TEST PPO METRICS
# =============================================================================

class TestPPOMetrics:
    """Test per PPOMetrics."""

    def test_default_metrics(self):
        """Test metriche default."""
        metrics = PPOMetrics()

        assert metrics.policy_loss == 0.0
        assert metrics.value_loss == 0.0
        assert metrics.num_updates == 0

    def test_custom_metrics(self):
        """Test metriche custom."""
        metrics = PPOMetrics(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=0.1,
            clip_fraction=0.15,
            num_updates=10
        )

        assert metrics.policy_loss == 0.5
        assert metrics.clip_fraction == 0.15

    def test_to_dict(self):
        """Test serializzazione metriche."""
        metrics = PPOMetrics(
            policy_loss=0.123456789,
            value_loss=0.987654321
        )

        data = metrics.to_dict()

        assert data["policy_loss"] == 0.123457  # rounded
        assert data["value_loss"] == 0.987654
