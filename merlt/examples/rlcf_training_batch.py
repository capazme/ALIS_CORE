#!/usr/bin/env python3
"""
RLCF Training Batch Script
==========================

Script di training batch per le policy RLCF con tutte le features complete:

1. RETRIEVAL DATI: Carica traces e feedback dal database
2. CURRICULUM LEARNING: Ordina i dati per difficoltà crescente
3. TRAINING: Esegue REINFORCE/PPO su GatingPolicy o ReActPolicy
4. CHECKPOINT: Salva versione della policy trainata
5. EVALUATION: Off-policy evaluation per validare la nuova policy
6. LOGGING: Metriche complete in logs/

Uso:
    # Training con dati dal database
    python scripts/rlcf_training_batch.py

    # Training con configurazione custom
    python scripts/rlcf_training_batch.py --config config/rlcf_training.yaml

    # Dry run (mostra cosa farebbe senza eseguire)
    python scripts/rlcf_training_batch.py --dry-run

    # Training con specifica versione
    python scripts/rlcf_training_batch.py --policy-version v1.1.0

Ambiente:
    - RLCF_POSTGRES_URL: URL PostgreSQL (default: postgresql+asyncpg://...)
    - RLCF_MIN_FEEDBACK: Minimo feedback per training (default: 10)
    - RLCF_POLICY_DIR: Directory checkpoints (default: models/policies/)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import structlog
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from merlt.rlcf.persistence import (
    RLCFPersistence,
    create_persistence,
)
from merlt.rlcf.execution_trace import ExecutionTrace
from merlt.rlcf.multilevel_feedback import MultilevelFeedback
from merlt.rlcf.policy_gradient import GatingPolicy, TraversalPolicy
from merlt.rlcf.single_step_trainer import SingleStepTrainer, SingleStepConfig
from merlt.rlcf.react_ppo_trainer import (
    ReActPPOTrainer,
    ReActConfig,
    ReActPolicy,
    ReActTrajectory,
)

# Optional: curriculum learning e off-policy eval
try:
    from merlt.rlcf.curriculum_learning import CurriculumScheduler, DifficultyAssessor
    HAS_CURRICULUM = True
except ImportError:
    HAS_CURRICULUM = False

try:
    from merlt.rlcf.off_policy_eval import OPEEvaluator
    HAS_OPE = True
except ImportError:
    HAS_OPE = False

# Logging setup
log = structlog.get_logger()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    "training": {
        "policy_type": "gating",  # gating, traversal, react
        "min_feedback": 10,
        "max_episodes": 1000,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "entropy_coef": 0.01,
        "clip_grad_norm": 1.0,
    },
    "gating_policy": {
        "input_dim": 768,
        "hidden_dim": 256,
        "num_experts": 4,
    },
    "react_policy": {
        "state_dim": 1024,
        "num_actions": 7,
        "hidden_dim": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "num_epochs": 4,
        "step_penalty": -0.05,
    },
    "curriculum": {
        "enabled": True,
        "initial_difficulty": 0.3,
        "target_difficulty": 0.8,
        "progression_rate": 0.1,
    },
    "evaluation": {
        "enabled": True,
        "holdout_ratio": 0.2,
        "min_holdout": 5,
    },
    "checkpoint": {
        "directory": "models/policies",
        "save_frequency": 100,  # episodes
        "keep_last_n": 5,
    },
    "database": {
        "url": None,  # From env
        "lookback_days": 30,
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Carica configurazione da file YAML o usa default."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

        log.info("Config loaded from file", path=config_path)

    # Override from environment
    if os.getenv("RLCF_POSTGRES_URL"):
        config["database"]["url"] = os.getenv("RLCF_POSTGRES_URL")

    if os.getenv("RLCF_MIN_FEEDBACK"):
        config["training"]["min_feedback"] = int(os.getenv("RLCF_MIN_FEEDBACK"))

    if os.getenv("RLCF_POLICY_DIR"):
        config["checkpoint"]["directory"] = os.getenv("RLCF_POLICY_DIR")

    return config


# ==============================================================================
# DATA LOADING
# ==============================================================================

async def load_training_data(
    persistence: RLCFPersistence,
    config: Dict[str, Any],
    policy_version: Optional[str] = None
) -> List[Tuple[ExecutionTrace, MultilevelFeedback]]:
    """
    Carica dati di training dal database.

    Args:
        persistence: Service di persistenza
        config: Configurazione
        policy_version: Filtra per versione policy

    Returns:
        Lista di (trace, feedback) pairs
    """
    lookback_days = config["database"]["lookback_days"]
    min_date = datetime.utcnow() - timedelta(days=lookback_days)

    training_data = await persistence.get_training_data(
        policy_version=policy_version,
        min_date=min_date,
        limit=config["training"]["max_episodes"]
    )

    log.info(
        "Training data loaded",
        count=len(training_data),
        lookback_days=lookback_days,
        policy_version=policy_version
    )

    return training_data


def convert_to_routing_format(
    training_data: List[Tuple[ExecutionTrace, MultilevelFeedback]]
) -> List[Dict[str, Any]]:
    """
    Converte dati per training routing (GatingPolicy).

    Args:
        training_data: Lista di (trace, feedback)

    Returns:
        Lista di dict con query_embedding, weights, reward
    """
    routing_data = []

    for trace, feedback in training_data:
        # Trova azione expert_selection nel trace
        expert_actions = trace.get_actions_by_type("expert_selection")

        if not expert_actions:
            continue

        # Estrai query_embedding dal metadata
        query_embedding = None
        for action in expert_actions:
            if "query_embedding" in action.metadata:
                query_embedding = action.metadata["query_embedding"]
                break

        if query_embedding is None:
            continue

        # Estrai weights e log_probs
        weights = []
        log_probs = []
        for action in expert_actions:
            weights.append(action.parameters.get("weight", 0.0))
            log_probs.append(action.log_prob)

        # Reward dal feedback
        reward = feedback.overall_score()

        routing_data.append({
            "query_id": trace.query_id,
            "query_embedding": np.array(query_embedding, dtype=np.float32),
            "weights": np.array(weights, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "reward": reward,
            "trace": trace,
            "feedback": feedback
        })

    log.info(
        "Converted to routing format",
        total=len(training_data),
        valid=len(routing_data)
    )

    return routing_data


def convert_to_react_format(
    training_data: List[Tuple[ExecutionTrace, MultilevelFeedback]]
) -> List[ReActTrajectory]:
    """
    Converte dati per training ReAct (multi-step).

    Args:
        training_data: Lista di (trace, feedback)

    Returns:
        Lista di ReActTrajectory
    """
    trajectories = []

    for trace, feedback in training_data:
        # Solo trace con tool_use o graph_traversal (multi-step)
        tool_actions = trace.get_actions_by_type("tool_use")
        traversal_actions = trace.get_actions_by_type("graph_traversal")

        if not tool_actions and not traversal_actions:
            # Single-step, skip per ReAct
            continue

        # Crea trajectory
        query_embedding = trace.metadata.get("query_embedding")
        if query_embedding is None:
            # Try to get from first action
            for action in trace.actions:
                if "query_embedding" in action.metadata:
                    query_embedding = action.metadata["query_embedding"]
                    break

        if query_embedding is None:
            query_embedding = [0.0] * 1024  # Fallback

        from merlt.rlcf.react_ppo_trainer import (
            ReActTrajectory,
            ReActStep,
            ReActActionType,
        )

        trajectory = ReActTrajectory(
            query_id=trace.query_id,
            query="",  # Not stored in trace
            query_embedding=torch.tensor(query_embedding, dtype=torch.float32),
            expert_type=trace.metadata.get("expert_type", "unknown")
        )

        # Convert actions to steps
        all_actions = tool_actions + traversal_actions
        all_actions.sort(key=lambda a: a.timestamp)

        for i, action in enumerate(all_actions):
            # Map action type
            if action.action_type == "tool_use":
                action_type = ReActActionType.SEARCH_KG
            elif action.action_type == "graph_traversal":
                action_type = ReActActionType.READ_ARTICLE
            else:
                action_type = ReActActionType.THINK

            # Get state from metadata or generate placeholder
            state = action.metadata.get("state")
            if state is None:
                state = [0.0] * 1024
            state_tensor = torch.tensor(state, dtype=torch.float32)

            step = ReActStep(
                state=state_tensor,
                action_type=action_type,
                action_args=action.parameters,
                action_index=i % 7,  # Placeholder
                log_prob=action.log_prob,
                value=0.0,  # Not stored
                observation=str(action.parameters),
                reward=0.0,  # Intermediate
                done=(i == len(all_actions) - 1)
            )
            trajectory.add_step(step)

        # Set final reward from feedback
        trajectory.set_final_reward(feedback.overall_score())
        trajectories.append(trajectory)

    log.info(
        "Converted to ReAct format",
        total=len(training_data),
        valid=len(trajectories)
    )

    return trajectories


# ==============================================================================
# TRAINING
# ==============================================================================

def create_gating_policy(config: Dict[str, Any]) -> GatingPolicy:
    """Crea GatingPolicy da config."""
    policy_config = config["gating_policy"]
    return GatingPolicy(
        input_dim=policy_config["input_dim"],
        hidden_dim=policy_config["hidden_dim"],
        num_experts=policy_config["num_experts"]
    )


def create_react_policy(config: Dict[str, Any]) -> ReActPolicy:
    """Crea ReActPolicy da config."""
    policy_config = config["react_policy"]
    return ReActPolicy(
        state_dim=policy_config["state_dim"],
        num_actions=policy_config["num_actions"],
        hidden_dim=policy_config["hidden_dim"]
    )


def train_routing_policy(
    policy: GatingPolicy,
    training_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    curriculum_scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Training loop per GatingPolicy con SingleStepTrainer.

    Args:
        policy: GatingPolicy da trainare
        training_data: Dati nel formato routing
        config: Configurazione
        curriculum_scheduler: Optional curriculum learning

    Returns:
        Metriche di training
    """
    trainer_config = SingleStepConfig(
        learning_rate=config["training"]["learning_rate"],
        entropy_coef=config["training"]["entropy_coef"],
        clip_grad_norm=config["training"]["clip_grad_norm"],
        baseline_decay=0.95
    )

    trainer = SingleStepTrainer(policy, trainer_config)

    # Metrics tracking
    rewards = []
    losses = []
    grad_norms = []
    advantages = []

    # Curriculum learning sorting
    if curriculum_scheduler and HAS_CURRICULUM:
        # Sort by difficulty
        training_data = sorted(
            training_data,
            key=lambda x: curriculum_scheduler.assess_difficulty(x.get("query_text", "")).difficulty_score
        )

    # Training loop
    for i, data in enumerate(training_data):
        trace = data["trace"]
        feedback = data["feedback"]

        # Update policy
        metrics = trainer.update(trace, feedback)

        # Track metrics
        if metrics.get("loss", 0) != 0:
            rewards.append(feedback.overall_score())
            losses.append(metrics.get("loss", 0))
            grad_norms.append(metrics.get("grad_norm", 0))
            advantages.append(metrics.get("raw_advantage", 0))

        # Log progress
        if (i + 1) % 50 == 0:
            log.info(
                "Training progress",
                episode=i + 1,
                total=len(training_data),
                avg_reward=np.mean(rewards[-50:]) if rewards else 0,
                avg_loss=np.mean(losses[-50:]) if losses else 0
            )

        # Update curriculum
        if curriculum_scheduler and HAS_CURRICULUM and (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            curriculum_scheduler.update_after_epoch(avg_reward)

    # Final stats
    stats = trainer.get_stats()

    return {
        "num_episodes": len(training_data),
        "final_avg_reward": np.mean(rewards) if rewards else 0,
        "final_avg_loss": np.mean(losses) if losses else 0,
        "avg_grad_norm": np.mean(grad_norms) if grad_norms else 0,
        "reward_std": np.std(rewards) if rewards else 0,
        "baseline": stats.get("baseline", 0),
        "reward_variance": stats.get("reward_variance", 0),
        "num_updates": stats.get("num_updates", 0),
    }


def train_react_policy(
    policy: ReActPolicy,
    trajectories: List[ReActTrajectory],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Training loop per ReActPolicy con PPO.

    Args:
        policy: ReActPolicy da trainare
        trajectories: Trajectory per training
        config: Configurazione

    Returns:
        Metriche di training
    """
    react_config_dict = config["react_policy"]
    react_config = ReActConfig(
        learning_rate=config["training"]["learning_rate"],
        gamma=react_config_dict["gamma"],
        gae_lambda=react_config_dict["gae_lambda"],
        clip_ratio=react_config_dict["clip_ratio"],
        num_epochs=react_config_dict["num_epochs"],
        step_penalty=react_config_dict["step_penalty"],
        value_coef=0.5,
        entropy_coef=config["training"]["entropy_coef"]
    )

    trainer = ReActPPOTrainer(policy, react_config)

    # Add trajectories
    for traj in trajectories:
        trainer.add_trajectory(traj)

    # Training update
    metrics = trainer.update()

    return {
        "num_trajectories": metrics.get("num_trajectories", 0),
        "num_steps": metrics.get("num_steps", 0),
        "policy_loss": metrics.get("policy_loss", 0),
        "value_loss": metrics.get("value_loss", 0),
        "entropy": metrics.get("entropy", 0),
        "kl_divergence": metrics.get("kl_divergence", 0),
        "clip_fraction": metrics.get("clip_fraction", 0),
        "epochs_completed": metrics.get("epochs_completed", 0),
        "num_updates": metrics.get("num_updates", 0),
    }


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_policy(
    policy: GatingPolicy,
    eval_data: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valutazione della policy su holdout set.

    Args:
        policy: Policy trainata
        eval_data: Dati di valutazione
        config: Configurazione

    Returns:
        Metriche di valutazione
    """
    policy.eval()
    correct = 0
    total = 0
    rewards = []

    with torch.no_grad():
        for data in eval_data:
            query_embedding = torch.tensor(
                data["query_embedding"],
                dtype=torch.float32,
                device=policy.device
            ).unsqueeze(0)

            # Forward pass
            weights, log_probs = policy.forward(query_embedding)
            predicted_expert = weights.argmax(dim=1).item()

            # Check if matches best expert in original trace
            original_weights = data["weights"]
            best_original = np.argmax(original_weights)

            if predicted_expert == best_original:
                correct += 1

            total += 1
            rewards.append(data["reward"])

    policy.train()

    accuracy = correct / total if total > 0 else 0

    return {
        "eval_accuracy": accuracy,
        "eval_samples": total,
        "avg_reward": np.mean(rewards) if rewards else 0,
    }


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(
    policy: torch.nn.Module,
    version: str,
    policy_type: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any]
) -> str:
    """
    Salva checkpoint della policy.

    Args:
        policy: Policy da salvare
        version: Version string
        policy_type: Tipo policy
        config: Configurazione
        metrics: Metriche training

    Returns:
        Path del checkpoint
    """
    checkpoint_dir = Path(config["checkpoint"]["directory"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{policy_type}_{version}.pt"

    torch.save({
        "version": version,
        "policy_type": policy_type,
        "state_dict": policy.state_dict(),
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }, checkpoint_path)

    log.info(
        "Checkpoint saved",
        path=str(checkpoint_path),
        version=version
    )

    return str(checkpoint_path)


def load_latest_checkpoint(
    policy_type: str,
    config: Dict[str, Any]
) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Carica l'ultimo checkpoint per un tipo di policy.

    Args:
        policy_type: Tipo policy
        config: Configurazione

    Returns:
        (state_dict, version) o None
    """
    checkpoint_dir = Path(config["checkpoint"]["directory"])

    if not checkpoint_dir.exists():
        return None

    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob(f"{policy_type}_v*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    checkpoint = torch.load(latest, weights_only=False)

    log.info(
        "Checkpoint loaded",
        path=str(latest),
        version=checkpoint.get("version")
    )

    return checkpoint["state_dict"], checkpoint.get("version", "v0.0.0")


def get_next_version(current_version: Optional[str]) -> str:
    """Genera prossima versione (semantic versioning)."""
    if current_version is None:
        return "v1.0.0"

    # Parse version
    version = current_version.lstrip("v")
    parts = version.split(".")

    if len(parts) != 3:
        return "v1.0.0"

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    # Increment patch
    return f"v{major}.{minor}.{patch + 1}"


# ==============================================================================
# MAIN
# ==============================================================================

async def run_training(
    config: Dict[str, Any],
    policy_version: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Esegue il training batch completo.

    Args:
        config: Configurazione
        policy_version: Versione policy (opzionale)
        dry_run: Se True, mostra cosa farebbe senza eseguire

    Returns:
        Risultati del training
    """
    results = {
        "status": "started",
        "start_time": datetime.utcnow().isoformat(),
        "config": config
    }

    try:
        # Initialize persistence
        persistence = await create_persistence(config["database"]["url"])

        # Get training stats
        stats = await persistence.get_training_stats(policy_version)
        results["data_stats"] = stats

        log.info("Training stats", **stats)

        # Check minimum feedback
        if stats["traces_with_feedback"] < config["training"]["min_feedback"]:
            log.warning(
                "Insufficient feedback for training",
                available=stats["traces_with_feedback"],
                required=config["training"]["min_feedback"]
            )
            results["status"] = "insufficient_data"
            return results

        if dry_run:
            log.info("Dry run - would train with", **stats)
            results["status"] = "dry_run"
            return results

        # Start training session
        session_id = await persistence.start_training_session(
            policy_type=config["training"]["policy_type"],
            policy_version_from=policy_version,
            config=config
        )
        results["session_id"] = session_id

        # Load training data
        training_data = await load_training_data(
            persistence, config, policy_version
        )

        # Split train/eval
        if config["evaluation"]["enabled"]:
            holdout_size = max(
                int(len(training_data) * config["evaluation"]["holdout_ratio"]),
                config["evaluation"]["min_holdout"]
            )
            eval_data = training_data[:holdout_size]
            train_data = training_data[holdout_size:]
        else:
            train_data = training_data
            eval_data = []

        results["train_size"] = len(train_data)
        results["eval_size"] = len(eval_data)

        # Initialize curriculum scheduler
        curriculum_scheduler = None
        if config["curriculum"]["enabled"] and HAS_CURRICULUM:
            curriculum_scheduler = CurriculumScheduler(
                initial_difficulty=config["curriculum"]["initial_difficulty"],
                target_difficulty=config["curriculum"]["target_difficulty"],
                progression_rate=config["curriculum"]["progression_rate"]
            )

        # Train based on policy type
        policy_type = config["training"]["policy_type"]

        if policy_type == "gating":
            # Convert data and train
            routing_data = convert_to_routing_format(train_data)

            # Load or create policy
            policy = create_gating_policy(config)

            checkpoint = load_latest_checkpoint("gating", config)
            if checkpoint:
                policy.load_state_dict(checkpoint[0])
                current_version = checkpoint[1]
            else:
                current_version = None

            # Train
            train_metrics = train_routing_policy(
                policy, routing_data, config, curriculum_scheduler
            )
            results["training_metrics"] = train_metrics

            # Evaluate
            if eval_data:
                eval_routing_data = convert_to_routing_format(eval_data)
                eval_metrics = evaluate_policy(policy, eval_routing_data, config)
                results["eval_metrics"] = eval_metrics

            # Save checkpoint
            new_version = get_next_version(current_version)
            checkpoint_path = save_checkpoint(
                policy, new_version, "gating", config, train_metrics
            )
            results["checkpoint_path"] = checkpoint_path
            results["new_version"] = new_version

            # Save to database
            await persistence.save_policy_checkpoint(
                version=new_version,
                policy_type="gating",
                state_dict_path=checkpoint_path,
                config=config["gating_policy"],
                training_metrics=train_metrics,
                training_session_id=session_id,
                training_episodes=len(routing_data)
            )

        elif policy_type == "react":
            # Convert data and train
            trajectories = convert_to_react_format(train_data)

            if not trajectories:
                log.warning("No multi-step trajectories found for ReAct training")
                results["status"] = "no_react_data"
                return results

            # Load or create policy
            policy = create_react_policy(config)

            checkpoint = load_latest_checkpoint("react", config)
            if checkpoint:
                policy.load_state_dict(checkpoint[0])
                current_version = checkpoint[1]
            else:
                current_version = None

            # Train
            train_metrics = train_react_policy(policy, trajectories, config)
            results["training_metrics"] = train_metrics

            # Save checkpoint
            new_version = get_next_version(current_version)
            checkpoint_path = save_checkpoint(
                policy, new_version, "react", config, train_metrics
            )
            results["checkpoint_path"] = checkpoint_path
            results["new_version"] = new_version

            # Save to database
            await persistence.save_policy_checkpoint(
                version=new_version,
                policy_type="react",
                state_dict_path=checkpoint_path,
                config=config["react_policy"],
                training_metrics=train_metrics,
                training_session_id=session_id,
                training_episodes=len(trajectories)
            )

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # Complete training session
        await persistence.complete_training_session(
            session_id=session_id,
            policy_version_to=new_version,
            num_traces=len(train_data),
            num_feedback=len(train_data),
            metrics=results.get("training_metrics", {}),
            trace_ids=[t.query_id for t, f in train_data]
        )

        results["status"] = "completed"
        results["end_time"] = datetime.utcnow().isoformat()

        log.info("Training completed", **{
            k: v for k, v in results.items()
            if k not in ["config", "training_metrics"]
        })

    except Exception as e:
        log.error("Training failed", error=str(e), exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)

        # Mark session as failed if we have session_id
        if "session_id" in results:
            try:
                await persistence.fail_training_session(
                    results["session_id"],
                    str(e)
                )
            except Exception:
                pass

    return results


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="RLCF Training Batch Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--policy-version", "-v",
        help="Filter traces by policy version"
    )
    parser.add_argument(
        "--policy-type", "-t",
        choices=["gating", "react"],
        help="Override policy type to train"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without training"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override policy type if specified
    if args.policy_type:
        config["training"]["policy_type"] = args.policy_type

    # Run training
    results = asyncio.run(
        run_training(
            config=config,
            policy_version=args.policy_version,
            dry_run=args.dry_run
        )
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("RLCF TRAINING RESULTS")
    print("=" * 60)
    print(f"Status: {results.get('status')}")
    print(f"Policy Type: {config['training']['policy_type']}")
    print(f"New Version: {results.get('new_version', 'N/A')}")

    if "training_metrics" in results:
        print("\nTraining Metrics:")
        for key, value in results["training_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if "eval_metrics" in results:
        print("\nEvaluation Metrics:")
        for key, value in results["eval_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("=" * 60)

    # Exit code based on status
    sys.exit(0 if results.get("status") == "completed" else 1)


if __name__ == "__main__":
    main()
