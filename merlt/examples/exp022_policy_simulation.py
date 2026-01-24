#!/usr/bin/env python3
"""
EXP-022: Policy Gradient Simulation
====================================

Script principale per simulare e valutare il sistema GatingPolicy
vs routing rule-based tradizionale.

Esegue 3 fasi:
1. Baseline (100 query): routing rule-based
2. Training (500 query): policy gradient con feedback
3. Evaluation (100 query): policy finale vs baseline

Usage:
    python scripts/exp022_policy_simulation.py
    python scripts/exp022_policy_simulation.py --config path/to/config.yaml
"""

import asyncio
import argparse
import json
import logging
import numpy as np
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from merlt.experts.router import ExpertRouter, RoutingDecision
from merlt.experts.base import ExpertContext


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "docs/experiments/EXP-022_policy_gradient_simulation/config.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "docs/experiments/EXP-022_policy_gradient_simulation/results"
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class SimulationMetrics:
    """Metriche aggregate per una fase della simulazione."""
    phase: str
    num_queries: int
    avg_reward: float
    expert_usage: Dict[str, float]
    load_balance_score: float
    policy_entropy: float
    routing_decisions: List[Dict[str, Any]] = field(default_factory=list)
    reward_trend: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QuerySample:
    """Sample di query sintetica per la simulazione."""
    query_text: str
    query_type: str  # definitional, relational, normative, jurisprudential
    embedding: np.ndarray
    gold_expert: Optional[str] = None  # Expert ottimale per questa query


@dataclass
class SyntheticFeedback:
    """Feedback sintetico per simulazione (semplificato)."""
    quantitative_score: float
    qualitative_label: str
    feedback_type: str = "synthetic"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Policy Gradient Implementation (Simplified)
# ============================================================================

class GatingPolicy:
    """
    Neural Policy Gradient per routing expert.

    Implementazione semplificata per simulazione.
    Usa un network lineare per mappare embedding -> expert weights.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_experts: int = 4,
        learning_rate: float = 0.0001,
        baseline_decay: float = 0.99,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.device = device

        # Simple linear policy (embedding -> logits)
        self.weights = np.random.randn(input_dim, num_experts) * 0.01
        self.bias = np.zeros(num_experts)

        # Baseline per REINFORCE con baseline
        self.baseline = 0.0

        # Expert names mapping
        self.expert_names = ["literal", "systemic", "principles", "precedent"]

        logging.info(
            f"Initialized GatingPolicy: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_experts={num_experts}, "
            f"lr={learning_rate}, device={device}"
        )

    def forward(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Forward pass: embedding -> expert weights.

        Args:
            embedding: Query embedding (768-dim)

        Returns:
            Dict expert_name -> weight (softmax probabilities)
        """
        # Linear projection
        logits = np.dot(embedding, self.weights) + self.bias

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        return {name: float(prob) for name, prob in zip(self.expert_names, probs)}

    def sample_expert(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Sample expert basato sulle probabilità.

        Returns:
            Tuple (expert_name, probability)
        """
        weights = self.forward(embedding)
        expert_names = list(weights.keys())
        probs = list(weights.values())

        selected = np.random.choice(expert_names, p=probs)
        return selected, weights[selected]

    def update(self, embedding: np.ndarray, expert: str, reward: float):
        """
        Policy gradient update (REINFORCE con baseline).

        Args:
            embedding: Query embedding
            expert: Expert selezionato
            reward: Feedback reward
        """
        # Compute advantage
        advantage = reward - self.baseline

        # Forward pass per ottenere probabilità attuali
        weights = self.forward(embedding)
        selected_prob = weights[expert]

        # Gradient: ∇log π(expert|embedding) * advantage
        expert_idx = self.expert_names.index(expert)

        # One-hot encoding dell'expert selezionato
        target = np.zeros(self.num_experts)
        target[expert_idx] = 1.0

        # Gradient computation (simplified)
        probs = np.array([weights[name] for name in self.expert_names])
        grad = (target - probs) * advantage

        # Update weights
        self.weights += self.learning_rate * np.outer(embedding, grad)
        self.bias += self.learning_rate * grad

        # Update baseline (exponential moving average)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        logging.debug(
            f"Policy update: expert={expert}, reward={reward:.3f}, "
            f"advantage={advantage:.3f}, baseline={self.baseline:.3f}"
        )


# ============================================================================
# Simulator
# ============================================================================

class PolicySimulator:
    """
    Simulatore per esperimento policy gradient.

    Gestisce le 3 fasi: baseline, training, evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policy: Optional[GatingPolicy] = None
        self.router: Optional[ExpertRouter] = None

        # Metriche per fase
        self.metrics: Dict[str, SimulationMetrics] = {}

        # Setup logging
        self._setup_logging()

        # Setup random seed
        seed = config["simulation"]["random_seed"]
        np.random.seed(seed)
        logging.info(f"Random seed: {seed}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config["output"]["logging"]
        log_dir = DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_config["file"].split("/")[-1]

        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 80)
        logging.info(f"EXP-022 Policy Gradient Simulation - {datetime.now()}")
        logging.info("=" * 80)

    def setup_policy(self):
        """Inizializza policy e router."""
        policy_config = self.config["policy"]

        self.policy = GatingPolicy(
            input_dim=policy_config["input_dim"],
            hidden_dim=policy_config["hidden_dim"],
            num_experts=policy_config["num_experts"],
            learning_rate=policy_config["learning_rate"],
            baseline_decay=policy_config["baseline_decay"],
            device=policy_config["device"]
        )

        self.router = ExpertRouter()

        logging.info("Policy and router initialized")

    def generate_synthetic_query(self) -> QuerySample:
        """
        Genera query sintetica con embedding random.

        Returns:
            QuerySample con embedding 768-dim
        """
        # Sampling tipo di query
        query_types = list(self.config["simulation"]["synthetic_feedback"]["query_types"].keys())
        query_type_dist = self.config["simulation"]["synthetic_feedback"]["query_type_distribution"]

        query_type = np.random.choice(
            query_types,
            p=[query_type_dist[qt] for qt in query_types]
        )

        # Template query per tipo
        query_templates = {
            "definitional": "Cos'è la {concept}?",
            "relational": "Qual è la relazione tra {concept1} e {concept2}?",
            "normative": "Qual è la ratio della norma sulla {concept}?",
            "jurisprudential": "Qual è l'orientamento della Cassazione sulla {concept}?"
        }

        concepts = ["responsabilità", "buona fede", "risarcimento", "prescrizione", "legittima difesa"]
        query_text = query_templates[query_type].format(
            concept=np.random.choice(concepts),
            concept1=np.random.choice(concepts),
            concept2=np.random.choice(concepts)
        )

        # Embedding random (768-dim, normalized)
        embedding = np.random.randn(self.config["policy"]["input_dim"])
        embedding = embedding / np.linalg.norm(embedding)

        # Gold expert basato su affinità query-expert
        affinities = self.config["simulation"]["synthetic_feedback"]["query_types"][query_type]
        gold_expert = max(affinities, key=affinities.get)

        return QuerySample(
            query_text=query_text,
            query_type=query_type,
            embedding=embedding,
            gold_expert=gold_expert
        )

    def generate_synthetic_feedback(
        self,
        query: QuerySample,
        expert: str
    ) -> SyntheticFeedback:
        """
        Genera feedback sintetico basato su query-expert match.

        Args:
            query: Query sample
            expert: Expert selezionato

        Returns:
            MultilevelFeedback con reward simulato
        """
        # Base quality dell'expert
        expert_quality = self.config["simulation"]["expert_quality"][expert]

        # Match query-expert
        query_type_affinities = self.config["simulation"]["synthetic_feedback"]["query_types"][query.query_type]
        match_score = query_type_affinities[expert]

        # Noise
        noise_std = self.config["simulation"]["synthetic_feedback"]["noise_std"]
        noise = np.random.normal(0, noise_std)

        # Final reward: base_quality * match_score + noise
        reward = expert_quality * match_score + noise
        reward = np.clip(reward, 0.0, 1.0)  # Clamp [0, 1]

        # Converti in feedback qualitativo
        if reward >= 0.7:
            qualitative = "excellent"
        elif reward >= 0.5:
            qualitative = "good"
        elif reward >= 0.3:
            qualitative = "fair"
        else:
            qualitative = "poor"

        return SyntheticFeedback(
            quantitative_score=reward,
            qualitative_label=qualitative,
            feedback_type="synthetic",
            metadata={
                "query_type": query.query_type,
                "expert": expert,
                "expert_quality": expert_quality,
                "match_score": match_score,
                "gold_expert": query.gold_expert
            }
        )

    async def run_baseline_phase(self) -> SimulationMetrics:
        """
        Fase 1: Baseline con routing rule-based.

        Returns:
            SimulationMetrics per la fase baseline
        """
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: BASELINE (Rule-Based Routing)")
        logging.info("=" * 80)

        num_queries = self.config["experiment"]["phases"]["baseline"]["num_queries"]

        expert_counts = {"literal": 0, "systemic": 0, "principles": 0, "precedent": 0}
        rewards = []
        routing_decisions = []

        for i in range(num_queries):
            # Genera query
            query = self.generate_synthetic_query()

            # Routing rule-based
            context = ExpertContext(query_text=query.query_text)
            decision: RoutingDecision = await self.router.route(context)

            # Seleziona top expert
            top_expert = max(decision.expert_weights, key=decision.expert_weights.get)
            expert_counts[top_expert] += 1

            # Genera feedback (anche se non usato per training in baseline)
            feedback = self.generate_synthetic_feedback(query, top_expert)
            rewards.append(feedback.quantitative_score)

            routing_decisions.append({
                "iteration": i,
                "query_type": query.query_type,
                "selected_expert": top_expert,
                "expert_weights": decision.expert_weights,
                "reward": feedback.quantitative_score,
                "gold_expert": query.gold_expert,
                "correct": top_expert == query.gold_expert
            })

            if (i + 1) % 20 == 0:
                logging.info(f"Baseline progress: {i + 1}/{num_queries} queries")

        # Compute metrics
        expert_usage = {k: v / num_queries for k, v in expert_counts.items()}
        avg_reward = np.mean(rewards)
        load_balance_score = self._compute_load_balance(expert_usage)
        policy_entropy = self._compute_entropy(expert_usage)

        metrics = SimulationMetrics(
            phase="baseline",
            num_queries=num_queries,
            avg_reward=avg_reward,
            expert_usage=expert_usage,
            load_balance_score=load_balance_score,
            policy_entropy=policy_entropy,
            routing_decisions=routing_decisions,
            reward_trend=rewards
        )

        logging.info(f"\nBaseline Results:")
        logging.info(f"  Avg Reward: {avg_reward:.3f}")
        logging.info(f"  Expert Usage: {expert_usage}")
        logging.info(f"  Load Balance: {load_balance_score:.3f}")
        logging.info(f"  Entropy: {policy_entropy:.3f}")

        return metrics

    async def run_training_phase(self) -> SimulationMetrics:
        """
        Fase 2: Training con policy gradient.

        Returns:
            SimulationMetrics per la fase training
        """
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: TRAINING (Policy Gradient)")
        logging.info("=" * 80)

        num_queries = self.config["experiment"]["phases"]["training"]["num_queries"]
        feedback_rate = self.config["experiment"]["phases"]["training"]["feedback_rate"]

        expert_counts = {"literal": 0, "systemic": 0, "principles": 0, "precedent": 0}
        rewards = []
        routing_decisions = []

        for i in range(num_queries):
            # Genera query
            query = self.generate_synthetic_query()

            # Routing con policy
            expert_weights = self.policy.forward(query.embedding)
            selected_expert, prob = self.policy.sample_expert(query.embedding)
            expert_counts[selected_expert] += 1

            # Genera feedback (con feedback_rate probabilità)
            if np.random.random() < feedback_rate:
                feedback = self.generate_synthetic_feedback(query, selected_expert)

                # Update policy
                self.policy.update(query.embedding, selected_expert, feedback.quantitative_score)

                reward = feedback.quantitative_score
            else:
                # Nessun feedback
                reward = None

            if reward is not None:
                rewards.append(reward)

            routing_decisions.append({
                "iteration": i,
                "query_type": query.query_type,
                "selected_expert": selected_expert,
                "expert_weights": expert_weights,
                "reward": reward,
                "gold_expert": query.gold_expert,
                "correct": selected_expert == query.gold_expert,
                "has_feedback": reward is not None
            })

            if (i + 1) % 50 == 0:
                recent_avg = np.mean(rewards[-50:]) if rewards else 0.0
                logging.info(
                    f"Training progress: {i + 1}/{num_queries} queries | "
                    f"Recent Avg Reward: {recent_avg:.3f} | "
                    f"Baseline: {self.policy.baseline:.3f}"
                )

        # Compute metrics
        expert_usage = {k: v / num_queries for k, v in expert_counts.items()}
        avg_reward = np.mean(rewards) if rewards else 0.0
        load_balance_score = self._compute_load_balance(expert_usage)
        policy_entropy = self._compute_entropy(expert_usage)

        metrics = SimulationMetrics(
            phase="training",
            num_queries=num_queries,
            avg_reward=avg_reward,
            expert_usage=expert_usage,
            load_balance_score=load_balance_score,
            policy_entropy=policy_entropy,
            routing_decisions=routing_decisions,
            reward_trend=rewards
        )

        logging.info(f"\nTraining Results:")
        logging.info(f"  Avg Reward: {avg_reward:.3f}")
        logging.info(f"  Expert Usage: {expert_usage}")
        logging.info(f"  Load Balance: {load_balance_score:.3f}")
        logging.info(f"  Entropy: {policy_entropy:.3f}")
        logging.info(f"  Final Baseline: {self.policy.baseline:.3f}")

        return metrics

    async def run_evaluation_phase(self) -> SimulationMetrics:
        """
        Fase 3: Evaluation con policy frozen.

        Returns:
            SimulationMetrics per la fase evaluation
        """
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 3: EVALUATION (Policy Frozen)")
        logging.info("=" * 80)

        num_queries = self.config["experiment"]["phases"]["evaluation"]["num_queries"]

        expert_counts = {"literal": 0, "systemic": 0, "principles": 0, "precedent": 0}
        rewards = []
        routing_decisions = []

        for i in range(num_queries):
            # Genera query
            query = self.generate_synthetic_query()

            # Routing con policy (frozen, no updates)
            expert_weights = self.policy.forward(query.embedding)
            selected_expert = max(expert_weights, key=expert_weights.get)  # Greedy selection
            expert_counts[selected_expert] += 1

            # Genera feedback (solo per valutazione)
            feedback = self.generate_synthetic_feedback(query, selected_expert)
            rewards.append(feedback.quantitative_score)

            routing_decisions.append({
                "iteration": i,
                "query_type": query.query_type,
                "selected_expert": selected_expert,
                "expert_weights": expert_weights,
                "reward": feedback.quantitative_score,
                "gold_expert": query.gold_expert,
                "correct": selected_expert == query.gold_expert
            })

            if (i + 1) % 20 == 0:
                logging.info(f"Evaluation progress: {i + 1}/{num_queries} queries")

        # Compute metrics
        expert_usage = {k: v / num_queries for k, v in expert_counts.items()}
        avg_reward = np.mean(rewards)
        load_balance_score = self._compute_load_balance(expert_usage)
        policy_entropy = self._compute_entropy(expert_usage)

        metrics = SimulationMetrics(
            phase="evaluation",
            num_queries=num_queries,
            avg_reward=avg_reward,
            expert_usage=expert_usage,
            load_balance_score=load_balance_score,
            policy_entropy=policy_entropy,
            routing_decisions=routing_decisions,
            reward_trend=rewards
        )

        logging.info(f"\nEvaluation Results:")
        logging.info(f"  Avg Reward: {avg_reward:.3f}")
        logging.info(f"  Expert Usage: {expert_usage}")
        logging.info(f"  Load Balance: {load_balance_score:.3f}")
        logging.info(f"  Entropy: {policy_entropy:.3f}")

        return metrics

    def compute_final_metrics(self):
        """Computa metriche finali e confronto baseline vs evaluation."""
        logging.info("\n" + "=" * 80)
        logging.info("FINAL ANALYSIS")
        logging.info("=" * 80)

        baseline = self.metrics["baseline"]
        evaluation = self.metrics["evaluation"]

        reward_improvement = (evaluation.avg_reward - baseline.avg_reward) / baseline.avg_reward * 100

        logging.info(f"\nReward Improvement:")
        logging.info(f"  Baseline: {baseline.avg_reward:.3f}")
        logging.info(f"  Evaluation: {evaluation.avg_reward:.3f}")
        logging.info(f"  Improvement: {reward_improvement:+.1f}%")

        logging.info(f"\nLoad Balance:")
        logging.info(f"  Baseline: {baseline.load_balance_score:.3f}")
        logging.info(f"  Evaluation: {evaluation.load_balance_score:.3f}")

        logging.info(f"\nPolicy Entropy:")
        logging.info(f"  Baseline: {baseline.policy_entropy:.3f}")
        logging.info(f"  Evaluation: {evaluation.policy_entropy:.3f}")

        # Success criteria
        success_criteria = {
            "reward_improvement_pct": reward_improvement,
            "target_improvement": 10.0,
            "reward_success": reward_improvement >= 10.0,

            "load_balance_score": evaluation.load_balance_score,
            "target_load_balance": 0.75,
            "load_balance_success": evaluation.load_balance_score >= 0.75,

            "policy_entropy": evaluation.policy_entropy,
            "target_entropy": 1.0,
            "entropy_success": evaluation.policy_entropy >= 1.0,
        }

        all_success = all([
            success_criteria["reward_success"],
            success_criteria["load_balance_success"],
            success_criteria["entropy_success"]
        ])

        logging.info(f"\n{'='*80}")
        if all_success:
            logging.info("SUCCESS: All criteria met!")
        else:
            logging.info("PARTIAL SUCCESS: Some criteria not met")

        for key, value in success_criteria.items():
            if not key.endswith("_success"):
                continue
            status = "✓" if value else "✗"
            logging.info(f"  {status} {key}: {value}")

        return success_criteria

    def export_results(self):
        """Esporta risultati in JSON."""
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export metrics per fase
        metrics_path = output_dir / "metrics.json"
        metrics_data = {
            phase: {
                k: v for k, v in asdict(metrics).items()
                if k not in ["routing_decisions", "reward_trend"]
            }
            for phase, metrics in self.metrics.items()
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logging.info(f"Exported metrics to {metrics_path}")

        # Export reward trend
        reward_trend_path = output_dir / "reward_trend.json"
        reward_trend_data = {
            phase: metrics.reward_trend
            for phase, metrics in self.metrics.items()
        }

        with open(reward_trend_path, "w") as f:
            json.dump(reward_trend_data, f, indent=2)

        logging.info(f"Exported reward trend to {reward_trend_path}")

        # Export expert usage
        expert_usage_path = output_dir / "expert_usage.json"
        expert_usage_data = {
            phase: metrics.expert_usage
            for phase, metrics in self.metrics.items()
        }

        with open(expert_usage_path, "w") as f:
            json.dump(expert_usage_data, f, indent=2)

        logging.info(f"Exported expert usage to {expert_usage_path}")

        # Export routing decisions
        for phase, metrics in self.metrics.items():
            decisions_path = output_dir / f"routing_decisions_{phase}.json"
            with open(decisions_path, "w") as f:
                json.dump(metrics.routing_decisions, f, indent=2)

            logging.info(f"Exported {phase} routing decisions to {decisions_path}")

        # Export convergence analysis
        convergence_path = output_dir / "convergence.json"
        convergence_data = self._analyze_convergence()

        with open(convergence_path, "w") as f:
            json.dump(convergence_data, f, indent=2)

        logging.info(f"Exported convergence analysis to {convergence_path}")

    def _compute_load_balance(self, expert_usage: Dict[str, float]) -> float:
        """
        Compute Load Balance Score.

        LBS = 1 - std(usage) / mean(usage)
        """
        values = list(expert_usage.values())
        mean_usage = np.mean(values)
        std_usage = np.std(values)

        if mean_usage == 0:
            return 0.0

        lbs = 1 - (std_usage / mean_usage)
        return max(0.0, lbs)

    def _compute_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Compute entropy della distribuzione.

        H(p) = -Σ p_i log(p_i)
        """
        probs = list(distribution.values())
        probs = [p for p in probs if p > 0]  # Evita log(0)

        if not probs:
            return 0.0

        entropy = -np.sum([p * np.log(p) for p in probs])
        return entropy

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analizza convergenza della policy durante training."""
        training_metrics = self.metrics["training"]

        window = self.config["metrics"]["convergence"]["window"]

        if len(training_metrics.reward_trend) < window:
            return {"error": "Not enough data for convergence analysis"}

        # Ultime N iterazioni
        final_rewards = training_metrics.reward_trend[-window:]

        # Variance
        variance = np.var(final_rewards)

        # Trend (regressione lineare)
        x = np.arange(len(final_rewards))
        slope, intercept = np.polyfit(x, final_rewards, 1)

        # Entropy nelle ultime N decisioni
        final_decisions = training_metrics.routing_decisions[-window:]
        expert_counts = {"literal": 0, "systemic": 0, "principles": 0, "precedent": 0}
        for decision in final_decisions:
            expert_counts[decision["selected_expert"]] += 1

        expert_usage = {k: v / window for k, v in expert_counts.items()}
        entropy = self._compute_entropy(expert_usage)

        return {
            "window_size": window,
            "final_rewards": [float(r) for r in final_rewards],
            "variance": float(variance),
            "trend_slope": float(slope),
            "trend_intercept": float(intercept),
            "entropy": float(entropy),
            "expert_usage": expert_usage,
            "converged": bool(variance < self.config["metrics"]["convergence"]["max_variance"])
        }

    async def run(self):
        """Esegue l'intero esperimento."""
        logging.info("Starting EXP-022 Policy Gradient Simulation")

        # Setup
        self.setup_policy()

        # Run phases
        self.metrics["baseline"] = await self.run_baseline_phase()
        self.metrics["training"] = await self.run_training_phase()
        self.metrics["evaluation"] = await self.run_evaluation_phase()

        # Analyze
        success_criteria = self.compute_final_metrics()

        # Export
        self.export_results()

        logging.info("\nExperiment completed!")

        return success_criteria


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="EXP-022: Policy Gradient Simulation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file"
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Run simulation
    simulator = PolicySimulator(config)
    success_criteria = await simulator.run()

    # Exit code based on success
    if all([
        success_criteria["reward_success"],
        success_criteria["load_balance_success"],
        success_criteria["entropy_success"]
    ]):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
