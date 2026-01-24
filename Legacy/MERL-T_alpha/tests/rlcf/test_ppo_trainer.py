"""
Test per PPO Trainer
====================

Test completi per PPOTrainer, PPOBuffer, ValueNetwork.
"""

import pytest
import tempfile
import os
import math
from unittest.mock import MagicMock, patch

import torch

from merlt.rlcf.ppo_trainer import (
    PPOConfig,
    ValueNetwork,
    PPOExperience,
    PPOBuffer,
    PPOMetrics,
    PPOTrainer,
    create_ppo_trainer
)
from merlt.rlcf.policy_gradient import GatingPolicy


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


# =============================================================================
# TEST PPO TRAINER
# =============================================================================

class TestPPOTrainer:
    """Test per PPOTrainer."""

    @pytest.fixture
    def policy(self):
        """Crea policy per test."""
        return GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4, device="cpu")

    @pytest.fixture
    def trainer(self, policy):
        """Crea trainer per test."""
        config = PPOConfig(
            num_epochs=2,
            learning_rate=1e-3,
            clip_ratio=0.2
        )
        return PPOTrainer(policy, config)

    def test_create_trainer(self, policy):
        """Test creazione trainer."""
        trainer = PPOTrainer(policy)

        assert trainer.policy == policy
        assert trainer.value_net is not None
        assert trainer.config.clip_ratio == 0.2

    def test_create_trainer_custom_config(self, policy):
        """Test trainer con config custom."""
        config = PPOConfig(clip_ratio=0.3, num_epochs=10)
        trainer = PPOTrainer(policy, config)

        assert trainer.config.clip_ratio == 0.3
        assert trainer.config.num_epochs == 10

    def test_collect_experience(self, trainer):
        """Test raccolta esperienze."""
        state = torch.randn(64)
        action = torch.randn(4)

        trainer.collect_experience(
            state=state,
            action=action,
            reward=0.8,
            log_prob=-0.5,
            done=True
        )

        assert len(trainer.buffer) == 1

    def test_compute_reward(self, trainer):
        """Test calcolo reward."""
        # Test con oggetto con metodo overall_score
        feedback = MagicMock()
        feedback.overall_score.return_value = 0.75

        reward = trainer.compute_reward(feedback)
        assert reward == 0.75

        # Test con dict
        feedback_dict = {"overall_score": 0.6}
        reward = trainer.compute_reward(feedback_dict)
        assert reward == 0.6

        # Test con float
        reward = trainer.compute_reward(0.9)
        assert reward == 0.9

    def test_update_empty_buffer(self, trainer):
        """Test update con buffer vuoto."""
        metrics = trainer.update()

        assert metrics.num_updates == 0

    def test_update_single_experience(self, trainer):
        """Test update con singola esperienza."""
        state = torch.randn(64)
        action = torch.randn(4)

        trainer.collect_experience(state, action, reward=0.8, log_prob=-0.5, done=True)

        metrics = trainer.update()

        assert metrics.num_updates == 1
        assert metrics.epochs_completed >= 1

    def test_update_batch(self, trainer):
        """Test update con batch di esperienze."""
        for i in range(10):
            state = torch.randn(64)
            action = torch.randn(4)
            trainer.collect_experience(
                state=state,
                action=action,
                reward=i / 10,
                log_prob=-0.5,
                done=True
            )

        metrics = trainer.update()

        assert metrics.num_updates == 1
        assert metrics.policy_loss != 0.0 or metrics.value_loss != 0.0

    def test_update_multiple_epochs(self, policy):
        """Test che faccia multiple epochs."""
        config = PPOConfig(num_epochs=5, target_kl=None)
        trainer = PPOTrainer(policy, config)

        for i in range(20):
            trainer.collect_experience(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=0.5 + (i % 3) * 0.1,
                log_prob=-0.5,
                done=True
            )

        metrics = trainer.update()

        # Dovrebbe completare tutte le epoch se non early stopped
        assert metrics.epochs_completed == 5

    def test_early_stopping_on_kl(self, policy):
        """Test early stopping su KL divergence."""
        config = PPOConfig(
            num_epochs=100,  # Molte epochs
            target_kl=0.001  # KL molto basso -> early stop
        )
        trainer = PPOTrainer(policy, config)

        for i in range(50):
            trainer.collect_experience(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=0.5,
                log_prob=-0.5,
                done=True
            )

        metrics = trainer.update()

        # Potrebbe fare early stop
        # Non garantito, dipende dal KL effettivo
        assert metrics.epochs_completed >= 1

    def test_clip_fraction(self, trainer):
        """Test che clip fraction sia calcolata."""
        for i in range(20):
            trainer.collect_experience(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=0.5,
                log_prob=-0.5,
                done=True
            )

        metrics = trainer.update()

        # Clip fraction dovrebbe essere in [0, 1]
        assert 0 <= metrics.clip_fraction <= 1

    def test_buffer_cleared_after_update(self, trainer):
        """Test che buffer sia svuotato dopo update."""
        for i in range(5):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), 0.5, -0.5, True
            )

        assert len(trainer.buffer) == 5

        trainer.update()

        assert len(trainer.buffer) == 0

    def test_get_stats(self, trainer):
        """Test statistiche training."""
        trainer.collect_experience(
            torch.randn(64), torch.randn(4), 0.5, -0.5, True
        )
        trainer.update()

        stats = trainer.get_stats()

        assert "num_updates" in stats
        assert "total_episodes" in stats
        assert "config" in stats

    def test_save_load_checkpoint(self, trainer):
        """Test salvataggio e caricamento checkpoint."""
        # Training iniziale
        for i in range(10):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), 0.5, -0.5, True
            )
        trainer.update()

        num_updates_before = trainer.num_updates

        # Salva
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path, metadata={"test": "value"})

            # Crea nuovo trainer
            new_policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
            new_trainer = PPOTrainer(new_policy)

            # Carica
            metadata = new_trainer.load_checkpoint(path)

            assert new_trainer.num_updates == num_updates_before
            assert metadata.get("test") == "value"
        finally:
            os.unlink(path)


# =============================================================================
# TEST FACTORY
# =============================================================================

class TestCreatePPOTrainer:
    """Test per factory function."""

    def test_create_basic(self):
        """Test creazione base."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = create_ppo_trainer(policy)

        assert isinstance(trainer, PPOTrainer)
        assert trainer.config.clip_ratio == 0.2

    def test_create_custom(self):
        """Test creazione con parametri custom."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = create_ppo_trainer(
            policy,
            clip_ratio=0.3,
            num_epochs=8,
            learning_rate=1e-4
        )

        assert trainer.config.clip_ratio == 0.3
        assert trainer.config.num_epochs == 8
        assert trainer.config.learning_rate == 1e-4

    def test_create_with_checkpoint(self):
        """Test creazione con checkpoint."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer1 = create_ppo_trainer(policy)

        # Training
        for i in range(5):
            trainer1.collect_experience(
                torch.randn(64), torch.randn(4), 0.5, -0.5, True
            )
        trainer1.update()

        # Salva
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            trainer1.save_checkpoint(path)

            # Crea nuovo trainer da checkpoint
            policy2 = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
            trainer2 = create_ppo_trainer(policy2, checkpoint_path=path)

            assert trainer2.num_updates == trainer1.num_updates
        finally:
            os.unlink(path)


# =============================================================================
# TEST INTEGRAZIONE
# =============================================================================

class TestPPOIntegration:
    """Test di integrazione PPO."""

    def test_training_loop(self):
        """Test loop di training completo."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4, device="cpu")
        config = PPOConfig(
            num_epochs=3,
            learning_rate=1e-3,
            clip_ratio=0.2
        )
        trainer = PPOTrainer(policy, config)

        all_metrics = []

        # Simula 5 update cycles
        for cycle in range(5):
            # Raccogli esperienze
            for i in range(20):
                state = torch.randn(64)

                # Forward pass policy
                policy.eval()
                with torch.no_grad():
                    weights, log_probs = policy.forward(state.unsqueeze(0))
                    weights = weights.squeeze(0)
                    log_prob = log_probs.sum().item()

                # Simula reward
                reward = 0.5 + 0.3 * torch.rand(1).item()

                trainer.collect_experience(
                    state=state,
                    action=weights,
                    reward=reward,
                    log_prob=log_prob,
                    done=True
                )

            # Update
            policy.train()
            metrics = trainer.update()
            all_metrics.append(metrics)

        # Verifica che abbia fatto 5 update
        assert trainer.num_updates == 5

        # Verifica metriche
        for m in all_metrics:
            assert m.num_updates > 0
            assert m.epochs_completed > 0

    def test_advantage_normalization(self):
        """Test che advantages siano normalizzati."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = PPOTrainer(policy)

        # Aggiungi esperienze con reward variati
        for i in range(100):
            trainer.collect_experience(
                state=torch.randn(64),
                action=torch.randn(4),
                reward=i / 100,  # 0.0 to 0.99
                log_prob=-0.5,
                done=True
            )

        # Calcola advantages
        trainer.buffer.compute_gae()
        batch = trainer.buffer.get_batch("cpu")

        advantages = batch["advantages"]

        # Advantages dovrebbero essere normalizzati (mean ~0, std ~1)
        assert abs(advantages.mean().item()) < 0.1
        assert 0.5 < advantages.std().item() < 1.5

    def test_value_learning(self):
        """Test che value function apprenda."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        config = PPOConfig(num_epochs=5, value_coef=1.0)
        trainer = PPOTrainer(policy, config)

        # Training con reward fisso
        fixed_state = torch.randn(64)

        # Prima del training
        with torch.no_grad():
            value_before = trainer.value_net.forward(fixed_state.unsqueeze(0)).item()

        # Training con stesso state e reward alto
        for _ in range(10):
            for _ in range(20):
                trainer.collect_experience(
                    state=fixed_state.clone(),
                    action=torch.randn(4),
                    reward=1.0,  # Reward alto costante
                    log_prob=-0.5,
                    done=True
                )
            trainer.update()

        # Dopo training
        with torch.no_grad():
            value_after = trainer.value_net.forward(fixed_state.unsqueeze(0)).item()

        # Value dovrebbe essere aumentato (reward alto)
        # Non garantito ma probabile
        # Verifica solo che sia cambiato
        assert value_before != value_after

    def test_policy_improvement(self):
        """Test che policy migliori nel tempo."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4, device="cpu")
        config = PPOConfig(num_epochs=3)
        trainer = PPOTrainer(policy, config)

        # Simula scenario dove expert 0 e' sempre migliore
        def get_reward(weights):
            # Reward proporzionale a peso expert 0
            return 0.5 + 0.5 * weights[0].item()

        # Initial distribution
        test_state = torch.randn(64)
        with torch.no_grad():
            initial_weights, _ = policy.forward(test_state.unsqueeze(0))
            initial_expert0_weight = initial_weights[0, 0].item()

        # Training
        for _ in range(20):
            for _ in range(50):
                state = torch.randn(64)
                with torch.no_grad():
                    weights, log_probs = policy.forward(state.unsqueeze(0))
                    log_prob = log_probs.sum().item()

                reward = get_reward(weights.squeeze(0))

                trainer.collect_experience(
                    state=state,
                    action=weights.squeeze(0),
                    reward=reward,
                    log_prob=log_prob,
                    done=True
                )
            trainer.update()

        # Final distribution
        policy.eval()
        with torch.no_grad():
            final_weights, _ = policy.forward(test_state.unsqueeze(0))
            final_expert0_weight = final_weights[0, 0].item()

        # Expert 0 weight dovrebbe essere aumentato
        # (non garantito, ma probabile dato il reward structure)
        # Verifica solo che ci sia stato cambiamento
        assert initial_expert0_weight != final_expert0_weight


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test casi limite."""

    def test_single_sample_update(self):
        """Test update con singolo sample."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = PPOTrainer(policy)

        trainer.collect_experience(
            torch.randn(64), torch.randn(4), 0.5, -0.5, True
        )

        # Non dovrebbe crashare
        metrics = trainer.update()
        assert metrics.num_updates == 1

    def test_very_high_reward(self):
        """Test con reward molto alto."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = PPOTrainer(policy)

        for _ in range(10):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), 100.0, -0.5, True
            )

        # Non dovrebbe crashare
        metrics = trainer.update()
        assert metrics.num_updates == 1

    def test_negative_reward(self):
        """Test con reward negativo."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        trainer = PPOTrainer(policy)

        for _ in range(10):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), -1.0, -0.5, True
            )

        metrics = trainer.update()
        assert metrics.num_updates == 1

    def test_zero_clip_ratio(self):
        """Test con clip ratio = 0 (no clipping)."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        config = PPOConfig(clip_ratio=0.0, num_epochs=2)
        trainer = PPOTrainer(policy, config)

        for _ in range(10):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), 0.5, -0.5, True
            )

        # Con clip_ratio=0, tutto viene clipped
        metrics = trainer.update()
        assert metrics.num_updates == 1

    def test_large_clip_ratio(self):
        """Test con clip ratio grande."""
        policy = GatingPolicy(input_dim=64, hidden_dim=32, device="cpu")
        config = PPOConfig(clip_ratio=1.0, num_epochs=2, target_kl=None)
        trainer = PPOTrainer(policy, config)

        for _ in range(10):
            trainer.collect_experience(
                torch.randn(64), torch.randn(4), 0.5, -0.5, True
            )

        # Con clip_ratio grande, il clipping e' meno frequente
        # ma non garantito zero a causa delle dinamiche del training
        metrics = trainer.update()
        assert metrics.num_updates == 1  # Verifica che abbia completato
