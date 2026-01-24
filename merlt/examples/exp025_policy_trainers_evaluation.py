#!/usr/bin/env python3
"""
EXP-025: Policy Trainers Evaluation
====================================

Esperimento dettagliato per validare i due nuovi trainer:
1. SingleStepTrainer (REINFORCE per routing single-step)
2. ReActPPOTrainer (PPO per Expert multi-step reasoning)

Scenario 1: Routing Task (SingleStepTrainer)
- Task: Classificare query in 4 categorie basate su embedding
- Metrica: Accuracy, convergenza, reward medio

Scenario 2: ReAct Task (ReActPPOTrainer)
- Task: Raggiungere goal in N step con reward sparso
- Metrica: Episode length, success rate, value loss

Usage:
    python scripts/exp025_policy_trainers_evaluation.py
    python scripts/exp025_policy_trainers_evaluation.py --scenario routing
    python scripts/exp025_policy_trainers_evaluation.py --scenario react
    python scripts/exp025_policy_trainers_evaluation.py --all --save-plots
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from merlt.rlcf.policy_gradient import GatingPolicy
from merlt.rlcf.single_step_trainer import (
    SingleStepTrainer,
    SingleStepConfig,
    create_single_step_trainer,
)
from merlt.rlcf.react_ppo_trainer import (
    ReActPPOTrainer,
    ReActConfig,
    ReActPolicy,
    ReActTrajectory,
    ReActStep,
    ReActActionType,
    create_react_policy,
    create_react_ppo_trainer,
)
from merlt.rlcf.execution_trace import ExecutionTrace
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
)

# ============================================================================
# Setup Logging
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "exp025_policy_trainers.log"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ExperimentConfig:
    """Configurazione esperimento."""
    # General
    seed: int = 42
    device: str = "auto"

    # SingleStepTrainer config
    routing_input_dim: int = 64
    routing_hidden_dim: int = 32
    routing_num_experts: int = 4
    routing_learning_rate: float = 0.01
    routing_baseline_decay: float = 0.95
    routing_entropy_coef: float = 0.01
    routing_train_episodes: int = 500
    routing_eval_episodes: int = 100

    # ReActPPOTrainer config
    react_state_dim: int = 64
    react_num_actions: int = 5
    react_hidden_dim: int = 32
    react_learning_rate: float = 0.005
    react_gamma: float = 0.99
    react_gae_lambda: float = 0.95
    react_clip_ratio: float = 0.2
    react_num_epochs: int = 4
    react_value_coef: float = 0.5
    react_entropy_coef: float = 0.01
    react_step_penalty: float = -0.05
    react_train_episodes: int = 200
    react_eval_episodes: int = 50
    react_max_steps: int = 10


@dataclass
class RoutingMetrics:
    """Metriche per esperimento routing."""
    train_accuracies: List[float] = field(default_factory=list)
    train_rewards: List[float] = field(default_factory=list)
    eval_accuracy: float = 0.0
    eval_reward: float = 0.0
    expert_distribution: Dict[str, float] = field(default_factory=dict)
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    convergence_episode: int = 0
    final_entropy: float = 0.0


@dataclass
class ReActMetrics:
    """Metriche per esperimento ReAct."""
    train_rewards: List[float] = field(default_factory=list)
    train_lengths: List[float] = field(default_factory=list)
    train_success_rates: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    eval_reward: float = 0.0
    eval_length: float = 0.0
    eval_success_rate: float = 0.0
    random_baseline: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Scenario 1: Routing Task (SingleStepTrainer)
# ============================================================================


class RoutingEnvironment:
    """
    Ambiente sintetico per routing task.

    Il task e' classificare embeddings in 4 categorie basate su
    pattern semplici (quadrante nel piano delle prime 2 dimensioni).
    """

    def __init__(self, input_dim: int = 64, num_experts: int = 4, noise: float = 0.1):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.noise = noise
        self.expert_names = ["literal", "systemic", "principles", "precedent"]

    def generate_query(self) -> Tuple[np.ndarray, int]:
        """
        Genera query con expert target basato su pattern.

        Pattern:
        - Expert 0 (literal): x[0] > 0 AND x[1] > 0
        - Expert 1 (systemic): x[0] < 0 AND x[1] > 0
        - Expert 2 (principles): x[0] < 0 AND x[1] < 0
        - Expert 3 (precedent): x[0] > 0 AND x[1] < 0
        """
        # Genera embedding casuale
        embedding = np.random.randn(self.input_dim).astype(np.float32)

        # Determina expert target basato su quadrante
        if embedding[0] > 0 and embedding[1] > 0:
            target = 0  # literal
        elif embedding[0] < 0 and embedding[1] > 0:
            target = 1  # systemic
        elif embedding[0] < 0 and embedding[1] < 0:
            target = 2  # principles
        else:
            target = 3  # precedent

        return embedding, target

    def compute_reward(self, chosen_expert: int, target_expert: int) -> float:
        """
        Calcola reward per la scelta.

        Reward:
        - 1.0: Scelta corretta
        - 0.3: Expert adiacente (quadrante vicino)
        - 0.0: Scelta sbagliata
        """
        if chosen_expert == target_expert:
            return 1.0
        elif abs(chosen_expert - target_expert) == 1 or abs(chosen_expert - target_expert) == 3:
            return 0.3  # Quadrante adiacente
        else:
            return 0.0


def create_trace_from_embedding(
    query_embedding: np.ndarray,
    weights: np.ndarray,
    expert_names: List[str],
) -> ExecutionTrace:
    """Crea ExecutionTrace da embedding e weights."""
    trace = ExecutionTrace(query_id=f"q_{np.random.randint(10000)}")
    log_probs = np.log(weights + 1e-8)

    for i, expert_type in enumerate(expert_names):
        trace.add_expert_selection(
            expert_type=expert_type,
            weight=float(weights[i]),
            log_prob=float(log_probs[i]),
            metadata={
                "source": "gating_policy",
                "query_embedding": query_embedding.tolist(),
                "action_index": i,
            },
        )

    return trace


def create_feedback_from_reward(reward: float) -> MultilevelFeedback:
    """Crea MultilevelFeedback da reward scalare."""
    return MultilevelFeedback(
        query_id="feedback",
        retrieval_feedback=RetrievalFeedback(precision=reward, recall=reward),
        reasoning_feedback=ReasoningFeedback(logical_coherence=reward, legal_soundness=reward),
        synthesis_feedback=SynthesisFeedback(clarity=reward, usefulness=reward),
    )


def run_routing_experiment(config: ExperimentConfig) -> RoutingMetrics:
    """
    Esegue esperimento completo per SingleStepTrainer.
    """
    logger.info("=" * 60)
    logger.info("SCENARIO 1: ROUTING TASK (SingleStepTrainer)")
    logger.info("=" * 60)

    # Setup
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Device: {device}")

    # Crea policy e trainer
    policy = GatingPolicy(
        input_dim=config.routing_input_dim,
        hidden_dim=config.routing_hidden_dim,
        num_experts=config.routing_num_experts,
    )

    trainer = create_single_step_trainer(
        policy,
        learning_rate=config.routing_learning_rate,
        baseline_decay=config.routing_baseline_decay,
        entropy_coef=config.routing_entropy_coef,
    )

    env = RoutingEnvironment(
        input_dim=config.routing_input_dim,
        num_experts=config.routing_num_experts,
    )

    metrics = RoutingMetrics()

    # -------------------------------------------------------------------------
    # FASE 1: Baseline (Random Policy)
    # -------------------------------------------------------------------------
    logger.info("\n--- FASE 1: Baseline (Random Policy) ---")

    baseline_correct = 0
    baseline_rewards = []

    for _ in range(config.routing_eval_episodes):
        embedding, target = env.generate_query()

        # Random choice
        chosen = np.random.randint(config.routing_num_experts)
        reward = env.compute_reward(chosen, target)

        if chosen == target:
            baseline_correct += 1
        baseline_rewards.append(reward)

    baseline_accuracy = baseline_correct / config.routing_eval_episodes
    baseline_reward = np.mean(baseline_rewards)

    logger.info(f"Baseline Accuracy: {baseline_accuracy:.2%}")
    logger.info(f"Baseline Avg Reward: {baseline_reward:.3f}")

    metrics.baseline_comparison["random_accuracy"] = baseline_accuracy
    metrics.baseline_comparison["random_reward"] = baseline_reward

    # -------------------------------------------------------------------------
    # FASE 2: Training
    # -------------------------------------------------------------------------
    logger.info(f"\n--- FASE 2: Training ({config.routing_train_episodes} episodes) ---")

    window_size = 50
    recent_correct = []
    expert_counts = {name: 0 for name in env.expert_names}

    for ep in range(config.routing_train_episodes):
        embedding, target = env.generate_query()

        # Forward pass
        with torch.no_grad():
            input_tensor = torch.tensor(embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        # Sample action
        chosen = int(np.argmax(weights))
        reward = env.compute_reward(chosen, target)
        correct = 1 if chosen == target else 0

        # Track metrics
        recent_correct.append(correct)
        if len(recent_correct) > window_size:
            recent_correct.pop(0)

        expert_counts[env.expert_names[chosen]] += 1

        # Train
        trace = create_trace_from_embedding(embedding, weights, env.expert_names)
        feedback = create_feedback_from_reward(reward)
        train_metrics = trainer.update(trace, feedback)

        # Log periodically
        if (ep + 1) % 100 == 0:
            rolling_acc = sum(recent_correct) / len(recent_correct)
            metrics.train_accuracies.append(rolling_acc)
            metrics.train_rewards.append(reward)

            logger.info(
                f"Episode {ep+1:4d} | "
                f"Rolling Acc: {rolling_acc:.2%} | "
                f"Reward: {reward:.2f} | "
                f"Baseline: {trainer.baseline:.3f} | "
                f"Grad Norm: {train_metrics.get('grad_norm', 0):.4f}"
            )

    # Expert distribution
    total_episodes = config.routing_train_episodes
    metrics.expert_distribution = {
        name: count / total_episodes for name, count in expert_counts.items()
    }

    logger.info("\nExpert Distribution durante training:")
    for name, ratio in metrics.expert_distribution.items():
        logger.info(f"  {name}: {ratio:.2%}")

    # -------------------------------------------------------------------------
    # FASE 3: Evaluation (Trained Policy)
    # -------------------------------------------------------------------------
    logger.info(f"\n--- FASE 3: Evaluation ({config.routing_eval_episodes} episodes) ---")

    eval_correct = 0
    eval_rewards = []

    for _ in range(config.routing_eval_episodes):
        embedding, target = env.generate_query()

        with torch.no_grad():
            input_tensor = torch.tensor(embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        chosen = int(np.argmax(weights))
        reward = env.compute_reward(chosen, target)

        if chosen == target:
            eval_correct += 1
        eval_rewards.append(reward)

    metrics.eval_accuracy = eval_correct / config.routing_eval_episodes
    metrics.eval_reward = np.mean(eval_rewards)

    logger.info(f"Eval Accuracy: {metrics.eval_accuracy:.2%}")
    logger.info(f"Eval Avg Reward: {metrics.eval_reward:.3f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    improvement_acc = metrics.eval_accuracy - baseline_accuracy
    improvement_reward = metrics.eval_reward - baseline_reward

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: ROUTING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Baseline Accuracy:  {baseline_accuracy:.2%}")
    logger.info(f"Trained Accuracy:   {metrics.eval_accuracy:.2%}")
    logger.info(f"Improvement:        {improvement_acc:+.2%} ({improvement_acc/baseline_accuracy*100:+.1f}% relative)")
    logger.info(f"Baseline Reward:    {baseline_reward:.3f}")
    logger.info(f"Trained Reward:     {metrics.eval_reward:.3f}")
    logger.info(f"Improvement:        {improvement_reward:+.3f}")

    metrics.baseline_comparison["improvement_accuracy"] = improvement_acc
    metrics.baseline_comparison["improvement_reward"] = improvement_reward

    # Get final entropy
    stats = trainer.get_stats()
    metrics.final_entropy = stats.get("reward_variance", 0.0)

    return metrics


# ============================================================================
# Scenario 2: ReAct Task (ReActPPOTrainer)
# ============================================================================


class ReActEnvironment:
    """
    Ambiente sintetico per ReAct multi-step reasoning.

    Il task e' navigare verso un goal in uno spazio discreto,
    usando azioni: LEFT, RIGHT, UP, DOWN, FINISH.

    - Reward: +1 se raggiunge goal, 0 altrimenti
    - Step penalty: -0.05 per ogni step (incentiva efficienza)
    - Max steps: 10
    """

    def __init__(
        self,
        state_dim: int = 64,
        grid_size: int = 5,
        max_steps: int = 10,
        step_penalty: float = -0.05,
    ):
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_penalty = step_penalty

        # Actions: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=FINISH
        self.num_actions = 5
        self.action_names = ["LEFT", "RIGHT", "UP", "DOWN", "FINISH"]

        # State encoding matrix (random but fixed)
        np.random.seed(123)
        self.state_encoder = np.random.randn(grid_size * grid_size, state_dim).astype(np.float32)
        np.random.seed(None)

    def reset(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Reset environment, return initial state, position, goal."""
        # Random start and goal positions
        self.pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.goal = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))

        # Make sure start != goal
        while self.goal == self.pos:
            self.goal = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))

        self.steps = 0
        return self._get_state(), self.pos, self.goal

    def _get_state(self) -> np.ndarray:
        """Encode current state as vector."""
        # Encode position and goal
        pos_idx = self.pos[0] * self.grid_size + self.pos[1]
        goal_idx = self.goal[0] * self.grid_size + self.goal[1]

        state = self.state_encoder[pos_idx] + 0.5 * self.state_encoder[goal_idx]

        # Add distance information
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        state[0] = dx / self.grid_size
        state[1] = dy / self.grid_size

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action, return new_state, reward, done, info."""
        self.steps += 1

        # Execute action
        x, y = self.pos

        if action == 0:  # LEFT
            x = max(0, x - 1)
        elif action == 1:  # RIGHT
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # UP
            y = max(0, y - 1)
        elif action == 3:  # DOWN
            y = min(self.grid_size - 1, y + 1)
        elif action == 4:  # FINISH
            pass

        self.pos = (x, y)

        # Check termination
        reached_goal = self.pos == self.goal
        finish_action = action == 4
        max_steps_reached = self.steps >= self.max_steps

        done = reached_goal or finish_action or max_steps_reached

        # Compute reward
        if reached_goal:
            reward = 1.0
        elif finish_action and not reached_goal:
            reward = -0.5  # Penalty for wrong FINISH
        else:
            reward = self.step_penalty  # Step penalty

        info = {
            "reached_goal": reached_goal,
            "position": self.pos,
            "goal": self.goal,
            "steps": self.steps,
        }

        return self._get_state(), reward, done, info

    def optimal_steps(self) -> int:
        """Calculate optimal number of steps to reach goal."""
        return abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])


def run_react_experiment(config: ExperimentConfig) -> ReActMetrics:
    """
    Esegue esperimento completo per ReActPPOTrainer.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 2: REACT TASK (ReActPPOTrainer)")
    logger.info("=" * 60)

    # Setup
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Device: {device}")

    # Crea policy e trainer
    policy = create_react_policy(
        state_dim=config.react_state_dim,
        num_actions=5,  # LEFT, RIGHT, UP, DOWN, FINISH
        hidden_dim=config.react_hidden_dim,
    )

    react_config = ReActConfig(
        learning_rate=config.react_learning_rate,
        gamma=config.react_gamma,
        gae_lambda=config.react_gae_lambda,
        clip_ratio=config.react_clip_ratio,
        num_epochs=config.react_num_epochs,
        value_coef=config.react_value_coef,
        entropy_coef=config.react_entropy_coef,
        step_penalty=config.react_step_penalty,
    )

    trainer = ReActPPOTrainer(policy, react_config)

    env = ReActEnvironment(
        state_dim=config.react_state_dim,
        max_steps=config.react_max_steps,
        step_penalty=config.react_step_penalty,
    )

    metrics = ReActMetrics()

    # -------------------------------------------------------------------------
    # FASE 1: Baseline (Random Policy)
    # -------------------------------------------------------------------------
    logger.info("\n--- FASE 1: Baseline (Random Policy) ---")

    baseline_rewards = []
    baseline_lengths = []
    baseline_successes = []

    for _ in range(config.react_eval_episodes):
        state, _, _ = env.reset()
        episode_reward = 0

        for step in range(config.react_max_steps):
            action = np.random.randint(env.num_actions)
            state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                break

        baseline_rewards.append(episode_reward)
        baseline_lengths.append(info["steps"])
        baseline_successes.append(1 if info["reached_goal"] else 0)

    metrics.random_baseline = {
        "avg_reward": float(np.mean(baseline_rewards)),
        "avg_length": float(np.mean(baseline_lengths)),
        "success_rate": float(np.mean(baseline_successes)),
    }

    logger.info(f"Random Policy - Avg Reward: {metrics.random_baseline['avg_reward']:.3f}")
    logger.info(f"Random Policy - Avg Length: {metrics.random_baseline['avg_length']:.1f}")
    logger.info(f"Random Policy - Success Rate: {metrics.random_baseline['success_rate']:.2%}")

    # -------------------------------------------------------------------------
    # FASE 2: Training
    # -------------------------------------------------------------------------
    logger.info(f"\n--- FASE 2: Training ({config.react_train_episodes} episodes) ---")

    batch_size = 10  # Episodes per update
    window_size = 20
    recent_rewards = []
    recent_lengths = []
    recent_successes = []

    for ep in range(config.react_train_episodes):
        # Collect trajectory
        state, pos, goal = env.reset()

        trajectory = ReActTrajectory(
            query_id=f"ep_{ep}",
            query=f"Navigate from {pos} to {goal}",
            query_embedding=torch.randn(config.react_state_dim),
            expert_type="navigation",
        )

        episode_reward = 0

        for step_idx in range(config.react_max_steps):
            state_tensor = torch.tensor(state, device=policy.device)

            # Select action
            action_idx, log_prob, value = policy.select_action(state_tensor)

            # Execute action
            next_state, reward, done, info = env.step(action_idx)
            episode_reward += reward

            # Create step
            react_step = ReActStep(
                state=state_tensor,
                action_type=ReActActionType.ACT,
                action_args={"action": env.action_names[action_idx]},
                action_index=action_idx,
                log_prob=log_prob,
                value=value,
                observation=f"Moved to {env.pos}",
                reward=reward,
                done=done,
            )
            trajectory.add_step(react_step)

            state = next_state

            if done:
                break

        # Set final reward
        final_reward = 1.0 if info["reached_goal"] else 0.0
        trajectory.set_final_reward(final_reward)

        # Add to trainer
        trainer.add_trajectory(trajectory)

        # Track metrics
        recent_rewards.append(episode_reward)
        recent_lengths.append(info["steps"])
        recent_successes.append(1 if info["reached_goal"] else 0)

        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
            recent_lengths.pop(0)
            recent_successes.pop(0)

        # Update every batch_size episodes
        if (ep + 1) % batch_size == 0:
            update_metrics = trainer.update()

            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)

            metrics.train_rewards.append(avg_reward)
            metrics.train_lengths.append(avg_length)
            metrics.train_success_rates.append(success_rate)

            if "value_loss" in update_metrics:
                metrics.value_losses.append(update_metrics["value_loss"])
            if "policy_loss" in update_metrics:
                metrics.policy_losses.append(update_metrics["policy_loss"])

            if (ep + 1) % 50 == 0:
                logger.info(
                    f"Episode {ep+1:4d} | "
                    f"Reward: {avg_reward:.3f} | "
                    f"Length: {avg_length:.1f} | "
                    f"Success: {success_rate:.2%} | "
                    f"VLoss: {update_metrics.get('value_loss', 0):.4f} | "
                    f"PLoss: {update_metrics.get('policy_loss', 0):.4f}"
                )

    # -------------------------------------------------------------------------
    # FASE 3: Evaluation (Trained Policy)
    # -------------------------------------------------------------------------
    logger.info(f"\n--- FASE 3: Evaluation ({config.react_eval_episodes} episodes) ---")

    eval_rewards = []
    eval_lengths = []
    eval_successes = []

    for _ in range(config.react_eval_episodes):
        state, _, _ = env.reset()
        episode_reward = 0

        for step in range(config.react_max_steps):
            state_tensor = torch.tensor(state, device=policy.device)

            # Greedy action selection
            action_idx, _, _ = policy.select_action(state_tensor, deterministic=True)
            state, reward, done, info = env.step(action_idx)
            episode_reward += reward

            if done:
                break

        eval_rewards.append(episode_reward)
        eval_lengths.append(info["steps"])
        eval_successes.append(1 if info["reached_goal"] else 0)

    metrics.eval_reward = float(np.mean(eval_rewards))
    metrics.eval_length = float(np.mean(eval_lengths))
    metrics.eval_success_rate = float(np.mean(eval_successes))

    logger.info(f"Trained Policy - Avg Reward: {metrics.eval_reward:.3f}")
    logger.info(f"Trained Policy - Avg Length: {metrics.eval_length:.1f}")
    logger.info(f"Trained Policy - Success Rate: {metrics.eval_success_rate:.2%}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: REACT EXPERIMENT")
    logger.info("=" * 60)

    improvement_reward = metrics.eval_reward - metrics.random_baseline["avg_reward"]
    improvement_length = metrics.random_baseline["avg_length"] - metrics.eval_length
    improvement_success = metrics.eval_success_rate - metrics.random_baseline["success_rate"]

    logger.info(f"{'Metric':<20} {'Random':<12} {'Trained':<12} {'Delta':<12}")
    logger.info("-" * 56)
    logger.info(f"{'Avg Reward':<20} {metrics.random_baseline['avg_reward']:<12.3f} {metrics.eval_reward:<12.3f} {improvement_reward:+.3f}")
    logger.info(f"{'Avg Length':<20} {metrics.random_baseline['avg_length']:<12.1f} {metrics.eval_length:<12.1f} {-improvement_length:+.1f}")
    logger.info(f"{'Success Rate':<20} {metrics.random_baseline['success_rate']:<12.2%} {metrics.eval_success_rate:<12.2%} {improvement_success:+.2%}")

    return metrics


# ============================================================================
# Visualization
# ============================================================================


def plot_routing_results(metrics: RoutingMetrics, save_path: Optional[Path] = None):
    """Plot risultati esperimento routing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training curve
    if metrics.train_accuracies:
        ax = axes[0]
        episodes = [(i + 1) * 100 for i in range(len(metrics.train_accuracies))]
        ax.plot(episodes, metrics.train_accuracies, 'b-', linewidth=2, label='Trained')
        ax.axhline(y=metrics.baseline_comparison.get("random_accuracy", 0.25),
                   color='r', linestyle='--', label='Random Baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rolling Accuracy')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Expert distribution
    ax = axes[1]
    if metrics.expert_distribution:
        experts = list(metrics.expert_distribution.keys())
        values = list(metrics.expert_distribution.values())
        bars = ax.bar(experts, values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
        ax.axhline(y=0.25, color='gray', linestyle='--', label='Uniform')
        ax.set_ylabel('Usage Ratio')
        ax.set_title('Expert Distribution')
        ax.set_ylim(0, 0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.1%}', ha='center', fontsize=9)

    # Comparison
    ax = axes[2]
    categories = ['Accuracy', 'Avg Reward']
    baseline_vals = [
        metrics.baseline_comparison.get("random_accuracy", 0.25),
        metrics.baseline_comparison.get("random_reward", 0.25),
    ]
    trained_vals = [metrics.eval_accuracy, metrics.eval_reward]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Random', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, trained_vals, width, label='Trained', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value')
    ax.set_title('Baseline vs Trained')
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_react_results(metrics: ReActMetrics, save_path: Optional[Path] = None):
    """Plot risultati esperimento ReAct."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training rewards
    ax = axes[0, 0]
    if metrics.train_rewards:
        ax.plot(metrics.train_rewards, 'b-', linewidth=2, label='Trained')
        ax.axhline(y=metrics.random_baseline.get("avg_reward", 0),
                   color='r', linestyle='--', label='Random')
        ax.set_xlabel('Update')
        ax.set_ylabel('Avg Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Success rate
    ax = axes[0, 1]
    if metrics.train_success_rates:
        ax.plot(metrics.train_success_rates, 'g-', linewidth=2, label='Trained')
        ax.axhline(y=metrics.random_baseline.get("success_rate", 0),
                   color='r', linestyle='--', label='Random')
        ax.set_xlabel('Update')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[1, 0]
    if metrics.value_losses:
        ax.plot(metrics.value_losses, 'b-', linewidth=2, label='Value Loss')
    if metrics.policy_losses:
        ax.plot(metrics.policy_losses, 'r-', linewidth=2, label='Policy Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final comparison
    ax = axes[1, 1]
    categories = ['Reward', 'Success Rate']
    baseline_vals = [
        metrics.random_baseline.get("avg_reward", 0),
        metrics.random_baseline.get("success_rate", 0),
    ]
    trained_vals = [metrics.eval_reward, metrics.eval_success_rate]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Random', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, trained_vals, width, label='Trained', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value')
    ax.set_title('Baseline vs Trained')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="EXP-025: Policy Trainers Evaluation")
    parser.add_argument("--scenario", choices=["routing", "react", "all"], default="all",
                       help="Which scenario to run")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    config = ExperimentConfig(seed=args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else LOG_DIR / "exp025_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXP-025: POLICY TRAINERS EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("")

    results = {}
    start_time = time.time()

    # Run experiments
    if args.scenario in ["routing", "all"]:
        routing_metrics = run_routing_experiment(config)
        results["routing"] = asdict(routing_metrics)

        if args.save_plots:
            plot_routing_results(routing_metrics, output_dir / "routing_results.png")

    if args.scenario in ["react", "all"]:
        react_metrics = run_react_experiment(config)
        results["react"] = asdict(react_metrics)

        if args.save_plots:
            plot_react_results(react_metrics, output_dir / "react_results.png")

    elapsed = time.time() - start_time

    # Save results
    results_file = output_dir / "exp025_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Results saved to: {results_file}")

    # Final summary
    if "routing" in results:
        r = results["routing"]
        logger.info(f"\nROUTING: {r['eval_accuracy']:.2%} accuracy ({r['baseline_comparison']['improvement_accuracy']:+.2%} vs random)")

    if "react" in results:
        r = results["react"]
        improvement = r["eval_success_rate"] - r["random_baseline"]["success_rate"]
        logger.info(f"REACT: {r['eval_success_rate']:.2%} success rate ({improvement:+.2%} vs random)")


if __name__ == "__main__":
    main()
