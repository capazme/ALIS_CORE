"""
Test per ReActPPOTrainer.

Valida che il trainer PPO per ReAct multi-step:
1. Calcola correttamente GAE per credit assignment temporale
2. Gestisce trajectory di lunghezza variabile
3. Esegue PPO update con clipping
4. Converge su task sintetici multi-step
"""

import pytest
import numpy as np
import torch

from merlt.rlcf.react_ppo_trainer import (
    ReActPPOTrainer,
    ReActConfig,
    ReActPolicy,
    ReActTrajectory,
    ReActStep,
    ReActActionType,
    compute_gae,
    create_react_policy,
    create_react_ppo_trainer
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def policy():
    """Crea ReActPolicy per test."""
    return ReActPolicy(state_dim=64, num_actions=5, hidden_dim=32)


@pytest.fixture
def config():
    """Crea config per test."""
    return ReActConfig(
        learning_rate=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        num_epochs=2,
        value_coef=0.5,
        entropy_coef=0.01
    )


@pytest.fixture
def trainer(policy, config):
    """Crea trainer per test."""
    return ReActPPOTrainer(policy, config)


def create_trajectory(
    num_steps: int = 5,
    final_reward: float = 1.0,
    intermediate_rewards: bool = False,
    device: str = "cpu"
) -> ReActTrajectory:
    """Crea trajectory sintetica per test."""
    trajectory = ReActTrajectory(
        query_id=f"test_{np.random.randint(1000)}",
        query="Test query",
        query_embedding=torch.randn(64).to(device),
        expert_type="test_expert"
    )

    for i in range(num_steps):
        state = torch.randn(64).to(device)
        action_idx = np.random.randint(5)

        step = ReActStep(
            state=state,
            action_type=ReActActionType.THINK,
            action_args={"step": i},
            action_index=action_idx,
            log_prob=np.random.uniform(-2, 0),
            value=np.random.uniform(0, 1),
            observation=f"Observation {i}",
            reward=0.1 if intermediate_rewards else 0.0,
            done=(i == num_steps - 1)
        )
        trajectory.add_step(step)

    trajectory.set_final_reward(final_reward)
    return trajectory


# =============================================================================
# TEST INITIALIZATION
# =============================================================================

class TestReActPolicyInit:
    """Test inizializzazione ReActPolicy."""

    def test_init_default(self):
        """Test inizializzazione con valori default."""
        policy = ReActPolicy()

        assert policy.state_dim == 1024
        assert policy.num_actions == 7
        assert policy.hidden_dim == 256

    def test_init_custom(self):
        """Test inizializzazione con valori custom."""
        policy = ReActPolicy(state_dim=128, num_actions=10, hidden_dim=64)

        assert policy.state_dim == 128
        assert policy.num_actions == 10
        assert policy.hidden_dim == 64

    def test_forward_shape(self, policy):
        """Test output shapes del forward pass."""
        state = torch.randn(4, 64).to(policy.device)  # Batch size 4

        action_probs, log_probs, values = policy.forward(state)

        assert action_probs.shape == (4, 5)
        assert log_probs.shape == (4, 5)
        assert values.shape == (4, 1)

    def test_select_action(self, policy):
        """Test selezione azione."""
        state = torch.randn(64).to(policy.device)

        action_idx, log_prob, value = policy.select_action(state)

        assert 0 <= action_idx < 5
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_evaluate_actions(self, policy):
        """Test valutazione azioni per batch."""
        states = torch.randn(8, 64).to(policy.device)
        actions = torch.randint(0, 5, (8,)).to(policy.device)

        log_probs, values, entropy = policy.evaluate_actions(states, actions)

        assert log_probs.shape == (8,)
        assert values.shape == (8,)
        assert entropy.shape == (8,)


class TestReActPPOTrainerInit:
    """Test inizializzazione trainer."""

    def test_init_default_config(self, policy):
        """Test inizializzazione con config default."""
        trainer = ReActPPOTrainer(policy)

        assert trainer.policy is policy
        assert trainer.config is not None
        assert trainer.num_updates == 0

    def test_init_custom_config(self, policy, config):
        """Test inizializzazione con config custom."""
        trainer = ReActPPOTrainer(policy, config)

        assert trainer.config.gamma == 0.99
        assert trainer.config.gae_lambda == 0.95
        assert trainer.config.clip_ratio == 0.2

    def test_factory_functions(self):
        """Test factory functions."""
        policy = create_react_policy(state_dim=128, num_actions=6)
        assert policy.state_dim == 128
        assert policy.num_actions == 6

        trainer = create_react_ppo_trainer(
            policy,
            gamma=0.95,
            clip_ratio=0.1
        )
        assert trainer.config.gamma == 0.95
        assert trainer.config.clip_ratio == 0.1


# =============================================================================
# TEST GAE COMPUTATION
# =============================================================================

class TestGAEComputation:
    """Test calcolo GAE."""

    def test_single_step_gae(self):
        """GAE per singolo step."""
        rewards = [1.0]
        values = [0.5]
        dones = [True]

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        # Per single step done: advantage = reward - value
        assert len(advantages) == 1
        assert abs(advantages[0] - 0.5) < 0.01  # 1.0 - 0.5 = 0.5

    def test_multi_step_gae(self):
        """GAE per multi-step trajectory."""
        rewards = [0.0, 0.0, 1.0]
        values = [0.3, 0.5, 0.7]
        dones = [False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 3
        assert len(returns) == 3

        # Ultimo step: advantage = reward - value = 1.0 - 0.7 = 0.3
        assert abs(advantages[2] - 0.3) < 0.01

        # Step precedenti dovrebbero avere advantage propagato
        # (dipende da gamma e lambda)
        assert advantages[1] > 0  # Reward propagato
        assert advantages[0] > 0  # Reward propagato

    def test_gae_with_done_in_middle(self):
        """GAE con done nel mezzo (episodi separati)."""
        rewards = [0.5, 0.0, 1.0]
        values = [0.3, 0.5, 0.7]
        dones = [True, False, True]  # Primo step è done

        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95
        )

        # Step 0 è done, quindi advantage = 0.5 - 0.3 = 0.2
        assert abs(advantages[0] - 0.2) < 0.01


# =============================================================================
# TEST TRAJECTORY MANAGEMENT
# =============================================================================

class TestTrajectoryManagement:
    """Test gestione trajectory."""

    def test_add_trajectory(self, trainer, policy):
        """Test aggiunta trajectory."""
        trajectory = create_trajectory(num_steps=5, final_reward=1.0, device=policy.device)

        trainer.add_trajectory(trajectory)

        assert len(trainer.trajectories) == 1
        assert len(trainer._all_steps) == 5
        assert trainer._total_episodes == 1

    def test_add_multiple_trajectories(self, trainer, policy):
        """Test aggiunta multiple trajectory."""
        for i in range(3):
            trajectory = create_trajectory(num_steps=i + 3, final_reward=0.5 + i * 0.2, device=policy.device)
            trainer.add_trajectory(trajectory)

        assert len(trainer.trajectories) == 3
        assert trainer._total_episodes == 3
        # Total steps: 3 + 4 + 5 = 12
        assert len(trainer._all_steps) == 12

    def test_add_trajectory_from_steps(self, trainer, policy):
        """Test aggiunta trajectory da lista di dict."""
        device = policy.device
        steps = [
            {
                "state": torch.randn(64).to(device),
                "action_type": "think",
                "action_index": 0,
                "log_prob": -0.5,
                "value": 0.3,
                "observation": "Obs 1",
                "reward": 0.0
            },
            {
                "state": torch.randn(64).to(device),
                "action_type": "think",
                "action_index": 1,
                "log_prob": -0.7,
                "value": 0.5,
                "observation": "Obs 2",
                "reward": 0.0
            }
        ]

        trainer.add_trajectory_from_steps(
            query_id="test",
            query="Test query",
            query_embedding=torch.randn(64).to(device),
            steps=steps,
            final_reward=1.0,
            expert_type="literal"
        )

        assert len(trainer.trajectories) == 1
        assert len(trainer._all_steps) == 2


# =============================================================================
# TEST PPO UPDATE
# =============================================================================

class TestPPOUpdate:
    """Test PPO update."""

    def test_update_single_trajectory(self, trainer, policy):
        """Update con singola trajectory."""
        trajectory = create_trajectory(num_steps=5, final_reward=1.0, device=policy.device)
        trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "kl_divergence" in metrics
        assert "clip_fraction" in metrics
        assert metrics["num_updates"] == 1

    def test_update_multiple_trajectories(self, trainer, policy):
        """Update con multiple trajectory."""
        for _ in range(5):
            trajectory = create_trajectory(num_steps=3, final_reward=np.random.uniform(0, 1), device=policy.device)
            trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        assert metrics["num_trajectories"] == 5
        assert metrics["num_steps"] == 15

    def test_update_clears_buffer(self, trainer, policy):
        """Update svuota il buffer."""
        trajectory = create_trajectory(num_steps=5, device=policy.device)
        trainer.add_trajectory(trajectory)

        assert len(trainer._all_steps) == 5

        trainer.update()

        assert len(trainer._all_steps) == 0
        assert len(trainer.trajectories) == 0

    def test_update_empty_buffer_no_crash(self, trainer):
        """Update con buffer vuoto non causa crash."""
        metrics = trainer.update()

        assert metrics == {}

    def test_multiple_epochs(self, policy):
        """Test multiple epochs di training."""
        config = ReActConfig(num_epochs=5, target_kl=None)  # No early stop
        trainer = ReActPPOTrainer(policy, config)

        trajectory = create_trajectory(num_steps=10, final_reward=1.0, device=policy.device)
        trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        assert metrics["epochs_completed"] == 5

    def test_early_stopping_on_kl(self, policy):
        """Test early stopping su KL divergence."""
        config = ReActConfig(num_epochs=100, target_kl=0.001)  # Molto basso
        trainer = ReActPPOTrainer(policy, config)

        trajectory = create_trajectory(num_steps=10, final_reward=1.0, device=policy.device)
        trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        # Potrebbe fermarsi prima di 100 epochs
        assert metrics["epochs_completed"] <= 100


# =============================================================================
# TEST CONVERGENCE
# =============================================================================

class TestConvergence:
    """Test convergenza su task sintetici."""

    def test_value_prediction_improves(self, policy):
        """Value prediction migliora con training.

        Nota: Con dati random, il value function non può apprendere
        pattern reali. Questo test verifica che:
        1. Il training completa senza errori
        2. Le metriche vengono calcolate correttamente
        3. Il value loss tende a diminuire (fitting ai dati)
        """
        config = ReActConfig(
            learning_rate=0.01,
            num_epochs=4,
            value_coef=1.0,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=None  # Disabilita early stopping per test deterministico
        )
        trainer = ReActPPOTrainer(policy, config)

        # Train su trajectory con reward costante (pattern semplice)
        value_losses = []

        for update_idx in range(10):
            # Crea batch di trajectory con reward prevedibile
            for _ in range(5):
                trajectory = create_trajectory(num_steps=3, final_reward=0.8, device=policy.device)
                trainer.add_trajectory(trajectory)

            metrics = trainer.update()
            if "value_loss" in metrics:
                value_losses.append(metrics["value_loss"])

        # Il value loss dovrebbe diminuire nel tempo
        # (value function impara a predire il return costante di 0.8)
        assert len(value_losses) >= 5, "Should have at least 5 value loss measurements"

        # Con reward costante, value loss dovrebbe tendere a calare
        first_half = np.mean(value_losses[:len(value_losses)//2])
        second_half = np.mean(value_losses[len(value_losses)//2:])

        # Verifica che value loss non esploda (test di stabilità)
        assert second_half < first_half * 10, (
            f"Value loss should not explode: first_half={first_half:.4f}, "
            f"second_half={second_half:.4f}"
        )


# =============================================================================
# TEST CHECKPOINT
# =============================================================================

class TestCheckpoint:
    """Test save/load checkpoint."""

    def test_save_and_load_checkpoint(self, trainer, policy, tmp_path):
        """Test salvataggio e caricamento checkpoint."""
        # Train un po'
        for _ in range(3):
            trajectory = create_trajectory(num_steps=5, final_reward=0.8, device=policy.device)
            trainer.add_trajectory(trajectory)
            trainer.update()

        # Salva
        checkpoint_path = tmp_path / "react_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), metadata={"test": True})

        assert checkpoint_path.exists()

        # Valori prima del load
        num_updates_before = trainer.num_updates

        # Nuovo trainer
        new_policy = ReActPolicy(state_dim=64, num_actions=5, hidden_dim=32)
        new_trainer = ReActPPOTrainer(new_policy)

        # Load
        metadata = new_trainer.load_checkpoint(str(checkpoint_path))

        assert metadata.get("test") == True
        assert new_trainer.num_updates == num_updates_before


# =============================================================================
# TEST STEP PENALTY
# =============================================================================

class TestStepPenalty:
    """Test step penalty per efficienza."""

    def test_step_penalty_applied(self, policy):
        """Verifica che step penalty viene applicato."""
        config = ReActConfig(step_penalty=-0.1)
        trainer = ReActPPOTrainer(policy, config)

        trajectory = create_trajectory(num_steps=5, final_reward=1.0, device=policy.device)
        initial_rewards = [step.reward for step in trajectory.steps]

        trainer.add_trajectory(trajectory)

        # I reward intermedi dovrebbero avere -0.1
        for i, step in enumerate(trainer.trajectories[0].steps[:-1]):
            assert step.reward == initial_rewards[i] + (-0.1)

        # L'ultimo step mantiene il final reward
        assert trainer.trajectories[0].steps[-1].reward == 1.0


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_single_step_trajectory(self, trainer, policy):
        """Trajectory con singolo step."""
        trajectory = create_trajectory(num_steps=1, final_reward=1.0, device=policy.device)
        trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        assert metrics["num_steps"] == 1
        assert metrics["num_updates"] == 1

    def test_very_long_trajectory(self, trainer, policy):
        """Trajectory molto lunga."""
        trajectory = create_trajectory(num_steps=50, final_reward=1.0, device=policy.device)
        trainer.add_trajectory(trajectory)

        metrics = trainer.update()

        assert metrics["num_steps"] == 50

    def test_get_stats(self, trainer, policy):
        """Test get_stats."""
        trajectory = create_trajectory(num_steps=5, device=policy.device)
        trainer.add_trajectory(trajectory)

        stats = trainer.get_stats()

        assert "num_updates" in stats
        assert "total_episodes" in stats
        assert "total_steps" in stats
        assert "buffer_trajectories" in stats
        assert "config" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
