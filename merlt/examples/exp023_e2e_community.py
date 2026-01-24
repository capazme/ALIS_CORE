#!/usr/bin/env python3
"""
EXP-023: End-to-End Community Simulation
========================================

Esperimento completo RLCF con:
- GatingPolicy per expert selection
- TraversalPolicy per graph traversal weights
- Community di 20 utenti sintetici
- Training REINFORCE con baseline

Usage:
    python scripts/exp023_e2e_community.py
    python scripts/exp023_e2e_community.py --config path/to/config.yaml
    python scripts/exp023_e2e_community.py --dry-run
"""

import asyncio
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QuerySample:
    """Singola query per l'esperimento."""
    query_id: str
    text: str
    query_type: str  # definitional, interpretive, procedural, jurisprudential
    domain: str  # civile, penale, costituzionale
    expected_expert: Optional[str] = None  # Gold standard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticUser:
    """Utente sintetico con authority tracking."""
    user_id: str
    profile: str
    baseline_authority: float
    current_authority: float
    feedback_bias: float
    noise_std: float
    track_record: float = 0.5
    feedbacks_given: int = 0
    domains: List[str] = field(default_factory=list)

    def generate_feedback(self, quality_score: float, rng: random.Random) -> float:
        """Genera feedback con bias e noise."""
        # Applica bias del profilo
        biased = quality_score + self.feedback_bias
        # Aggiungi noise
        noisy = biased + rng.gauss(0, self.noise_std)
        # Clamp a [0, 1]
        return max(0.0, min(1.0, noisy))

    def update_authority(self, feedback_accuracy: float, lambda_factor: float = 0.15):
        """Aggiorna authority basata su accuratezza feedback."""
        # Exponential moving average
        self.track_record = (1 - lambda_factor) * self.track_record + lambda_factor * feedback_accuracy
        # Combine baseline, track_record
        self.current_authority = 0.4 * self.baseline_authority + 0.35 * self.track_record + 0.25 * feedback_accuracy
        self.feedbacks_given += 1


@dataclass
class ExpertResponse:
    """Risposta da un expert (simulata o reale)."""
    expert_type: str
    interpretation: str
    confidence: float
    legal_basis: List[str]
    graph_score: float = 0.5
    execution_time_ms: float = 0.0
    trace: Optional[Dict[str, Any]] = None


@dataclass
class PhaseMetrics:
    """Metriche aggregate per fase."""
    phase: str
    num_queries: int
    avg_reward: float
    std_reward: float
    expert_usage: Dict[str, float]
    load_balance_score: float
    policy_entropy: float
    avg_confidence: float
    avg_graph_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IterationMetrics:
    """Metriche per singola iterazione di training."""
    iteration: int
    avg_reward: float
    policy_entropy: float
    expert_usage: Dict[str, float]
    authority_snapshot: Dict[str, float]
    loss: float = 0.0


# ============================================================================
# Query Generator
# ============================================================================

class QueryGenerator:
    """Genera query per l'esperimento."""

    QUERY_TEMPLATES = {
        "definitional": [
            "Cos'è {concept}?",
            "Qual è la definizione di {concept}?",
            "Come si definisce {concept} nel diritto {domain}?",
        ],
        "interpretive": [
            "Come si interpreta l'art. {article}?",
            "Qual è il significato dell'art. {article}?",
            "Come va letto l'art. {article} in combinato disposto?",
        ],
        "procedural": [
            "Quali sono i termini per {action}?",
            "Come si procede per {action}?",
            "Qual è la procedura per {action}?",
        ],
        "jurisprudential": [
            "Come è stata applicata la norma su {concept}?",
            "Qual è l'orientamento giurisprudenziale su {concept}?",
            "Come ha interpretato la Cassazione {concept}?",
        ],
    }

    CONCEPTS = {
        "civile": [
            "la risoluzione del contratto",
            "l'inadempimento contrattuale",
            "la responsabilità del debitore",
            "il risarcimento del danno",
            "la clausola penale",
            "l'eccezione di inadempimento",
            "la diffida ad adempiere",
            "il termine essenziale",
            "la caparra confirmatoria",
            "l'impossibilità sopravvenuta",
        ],
        "penale": [
            "la legittima difesa",
            "lo stato di necessità",
            "il concorso di persone",
            "il tentativo",
            "la recidiva",
        ],
        "costituzionale": [
            "il principio di uguaglianza",
            "la libertà personale",
            "il diritto alla difesa",
            "la presunzione di innocenza",
            "il giusto processo",
        ],
    }

    ARTICLES = {
        "civile": ["1453", "1454", "1455", "1456", "1457", "1460", "1461", "1462"],
        "penale": ["52", "54", "110", "56"],
        "costituzionale": ["3", "13", "24", "27", "111"],
    }

    ACTIONS = {
        "civile": [
            "la risoluzione giudiziale",
            "l'azione di risarcimento",
            "l'esecuzione forzata",
        ],
        "penale": [
            "la querela",
            "la costituzione di parte civile",
        ],
        "costituzionale": [
            "il ricorso in Cassazione",
            "l'eccezione di incostituzionalità",
        ],
    }

    # Gold standard: quale expert dovrebbe rispondere
    EXPERT_MAP = {
        "definitional": "literal",
        "interpretive": "systemic",
        "procedural": "systemic",
        "jurisprudential": "precedent",
    }

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.query_counter = 0

    def generate_query(
        self,
        query_type: Optional[str] = None,
        domain: Optional[str] = None
    ) -> QuerySample:
        """Genera una singola query."""
        # Seleziona tipo e dominio se non specificati
        if query_type is None:
            types = self.config.get("types", {})
            query_type = self.rng.choices(
                list(types.keys()),
                weights=list(types.values())
            )[0]

        if domain is None:
            domains = self.config.get("distribution", {})
            domain = self.rng.choices(
                list(domains.keys()),
                weights=list(domains.values())
            )[0]

        # Genera testo query
        template = self.rng.choice(self.QUERY_TEMPLATES[query_type])

        if "{concept}" in template:
            concept = self.rng.choice(self.CONCEPTS.get(domain, ["concetto generico"]))
            text = template.format(concept=concept, domain=domain)
        elif "{article}" in template:
            article = self.rng.choice(self.ARTICLES.get(domain, ["1"]))
            text = template.format(article=article)
        elif "{action}" in template:
            action = self.rng.choice(self.ACTIONS.get(domain, ["l'azione"]))
            text = template.format(action=action)
        else:
            text = template

        self.query_counter += 1

        return QuerySample(
            query_id=f"q_{self.query_counter:04d}",
            text=text,
            query_type=query_type,
            domain=domain,
            expected_expert=self.EXPERT_MAP.get(query_type),
            metadata={"template": template}
        )

    def generate_batch(self, n: int) -> List[QuerySample]:
        """Genera batch di query."""
        return [self.generate_query() for _ in range(n)]


# ============================================================================
# Community Manager
# ============================================================================

class CommunityManager:
    """Gestisce la community di utenti sintetici."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.users: Dict[str, SyntheticUser] = {}
        self._create_community()

    def _create_community(self):
        """Crea utenti sintetici basati su config."""
        profiles = self.config.get("profiles", {})
        user_id = 0

        for profile_name, profile_config in profiles.items():
            count = profile_config.get("count", 1)
            for _ in range(count):
                user = SyntheticUser(
                    user_id=f"user_{user_id:03d}",
                    profile=profile_name,
                    baseline_authority=profile_config.get("baseline_authority", 0.5),
                    current_authority=profile_config.get("baseline_authority", 0.5),
                    feedback_bias=profile_config.get("feedback_bias", 0.0),
                    noise_std=profile_config.get("noise_std", 0.1),
                    domains=profile_config.get("domains", []),
                )
                self.users[user.user_id] = user
                user_id += 1

        log.info(f"Created community with {len(self.users)} users")

    def select_user(self, domain: Optional[str] = None) -> SyntheticUser:
        """Seleziona un utente casuale (opzionalmente per dominio)."""
        candidates = list(self.users.values())

        # Se dominio specificato, preferisci specialisti
        if domain:
            specialists = [u for u in candidates if domain in u.domains]
            if specialists and self.rng.random() < 0.6:
                candidates = specialists

        return self.rng.choice(candidates)

    def get_authority_snapshot(self) -> Dict[str, float]:
        """Snapshot delle authority correnti."""
        return {uid: u.current_authority for uid, u in self.users.items()}

    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        """Statistiche per profilo."""
        stats = defaultdict(lambda: {"count": 0, "avg_authority": 0.0})
        for user in self.users.values():
            stats[user.profile]["count"] += 1
            stats[user.profile]["avg_authority"] += user.current_authority

        for profile in stats:
            if stats[profile]["count"] > 0:
                stats[profile]["avg_authority"] /= stats[profile]["count"]

        return dict(stats)


# ============================================================================
# Simplified Policy (for simulation without full neural network)
# ============================================================================

class SimulatedGatingPolicy:
    """GatingPolicy semplificata per simulazione."""

    EXPERTS = ["literal", "systemic", "principles", "precedent"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Inizializza logits random
        self.logits = self.rng.standard_normal(4) * 0.1
        self.learning_rate = 0.05  # Increased for faster learning
        self.baseline = 0.0
        self.baseline_decay = 0.95  # Faster baseline adaptation

    def forward(self, embedding: Optional[List[float]] = None) -> np.ndarray:
        """Calcola probabilità expert."""
        # Softmax
        exp_logits = np.exp(self.logits - np.max(self.logits))
        probs = exp_logits / exp_logits.sum()
        return probs

    def sample_expert(self, embedding: Optional[List[float]] = None) -> Tuple[str, float, int]:
        """Campiona expert con log_prob."""
        probs = self.forward(embedding)
        idx = self.rng.choice(4, p=probs)
        log_prob = np.log(probs[idx] + 1e-10)
        return self.EXPERTS[idx], float(log_prob), int(idx)

    def entropy(self) -> float:
        """Calcola entropia della distribuzione."""
        probs = self.forward()
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def update(self, expert_idx: int, reward: float, log_prob: float):
        """REINFORCE update."""
        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # Advantage
        advantage = reward - self.baseline

        # Policy gradient (simplified)
        grad = np.zeros(4)
        probs = self.forward()

        # Gradient: advantage * (one_hot - probs)
        one_hot = np.zeros(4)
        one_hot[expert_idx] = 1.0
        grad = advantage * (one_hot - probs)

        # Update
        self.logits += self.learning_rate * grad


class SimulatedTraversalPolicy:
    """TraversalPolicy semplificata per simulazione."""

    RELATIONS = ["RIFERIMENTO", "CITATO_DA", "MODIFICA", "DEROGA", "DEFINISCE"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Pesi per relazione (inizializzati vicino a 0.5)
        self.weights = {rel: 0.5 + self.rng.standard_normal() * 0.1 for rel in self.RELATIONS}
        self.learning_rate = 0.005

    def get_weight(self, relation: str, embedding: Optional[List[float]] = None) -> Tuple[float, float]:
        """Ritorna peso e log_prob per relazione."""
        base_weight = self.weights.get(relation, 0.5)
        # Sigmoid per bounded output
        weight = 1.0 / (1.0 + np.exp(-base_weight))
        # Log prob (approssimato come log di Bernoulli)
        log_prob = float(np.log(weight + 1e-10))
        return float(weight), log_prob

    def update(self, relation: str, reward: float):
        """Update peso relazione."""
        if relation in self.weights:
            # Gradient verso reward
            self.weights[relation] += self.learning_rate * reward * 0.1


# ============================================================================
# Expert System Simulator
# ============================================================================

class ExpertSystemSimulator:
    """Simula il sistema di expert."""

    EXPERT_QUALITY = {
        "literal": 0.75,
        "systemic": 0.70,
        "principles": 0.65,
        "precedent": 0.80,
    }

    EXPERT_MATCH = {
        "literal": {"definitional": 0.9, "interpretive": 0.6, "procedural": 0.5, "jurisprudential": 0.4},
        "systemic": {"definitional": 0.6, "interpretive": 0.9, "procedural": 0.8, "jurisprudential": 0.5},
        "principles": {"definitional": 0.5, "interpretive": 0.7, "procedural": 0.4, "jurisprudential": 0.6},
        "precedent": {"definitional": 0.4, "interpretive": 0.5, "procedural": 0.6, "jurisprudential": 0.95},
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_response(
        self,
        query: QuerySample,
        expert_type: str,
        graph_weight: float = 0.5
    ) -> ExpertResponse:
        """Genera risposta simulata."""
        # Quality score basato su expert e match con query type
        base_quality = self.EXPERT_QUALITY.get(expert_type, 0.6)
        match_score = self.EXPERT_MATCH.get(expert_type, {}).get(query.query_type, 0.5)

        # Confidence = base * match + noise
        confidence = base_quality * match_score + self.rng.gauss(0, 0.05)
        confidence = max(0.1, min(1.0, confidence))

        # Graph score influenzato dal peso traversal
        graph_score = graph_weight * 0.8 + self.rng.gauss(0, 0.1)
        graph_score = max(0.0, min(1.0, graph_score))

        return ExpertResponse(
            expert_type=expert_type,
            interpretation=f"[Simulato] Risposta di {expert_type} a: {query.text[:50]}...",
            confidence=confidence,
            legal_basis=[f"Art. simulato per {query.domain}"],
            graph_score=graph_score,
            execution_time_ms=self.rng.uniform(100, 500),
        )


# ============================================================================
# Reward Calculator
# ============================================================================

class RewardCalculator:
    """Calcola reward per RLCF."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute_reward(
        self,
        response: ExpertResponse,
        query: QuerySample,
        feedback: float,
        user_authority: float
    ) -> float:
        """
        Calcola reward pesato.

        Formula: reward = feedback * authority * quality_multiplier
        """
        # Quality multiplier basato su confidence e graph_score
        quality_mult = 0.5 * response.confidence + 0.5 * response.graph_score

        # Reward = feedback pesato per authority
        reward = feedback * user_authority * quality_mult

        # Bonus se expert corretto
        if query.expected_expert and response.expert_type == query.expected_expert:
            reward += 0.1

        return float(np.clip(reward, 0.0, 1.0))


# ============================================================================
# Experiment Runner
# ============================================================================

class EXP023Runner:
    """Runner principale per EXP-023."""

    def __init__(self, config_path: Optional[Path] = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        self.seed = self.config.get("experiment", {}).get("random_seed", 42)

        # Initialize components
        self.query_generator = QueryGenerator(
            self.config.get("queries", {}),
            seed=self.seed
        )
        self.community = CommunityManager(
            self.config.get("community", {}),
            seed=self.seed + 1
        )
        self.gating_policy = SimulatedGatingPolicy(seed=self.seed + 2)
        self.traversal_policy = SimulatedTraversalPolicy(seed=self.seed + 3)
        self.expert_system = ExpertSystemSimulator(seed=self.seed + 4)
        self.reward_calculator = RewardCalculator(self.config)

        # Results storage
        self.results: Dict[str, Any] = {
            "experiment": self.config.get("experiment", {}),
            "phases": {},
            "iterations": [],
            "authority_evolution": [],
            "expert_usage_evolution": [],
        }

        # RNG for feedback decisions
        self.feedback_rng = random.Random(self.seed + 100)

        # Output directory
        self.output_dir = Path(
            self.config.get("output", {}).get(
                "directory",
                "docs/experiments/EXP-023_e2e_community_simulation/results"
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Carica configurazione da YAML."""
        if config_path is None:
            config_path = PROJECT_ROOT / "docs/experiments/EXP-023_e2e_community_simulation/config.yaml"

        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            log.warning(f"Config not found at {config_path}, using defaults")
            return {}

    def run_baseline_phase(self) -> PhaseMetrics:
        """Fase 1: Baseline con policy frozen."""
        log.info("=" * 60)
        log.info("FASE 1: BASELINE")
        log.info("=" * 60)

        phase_config = self.config.get("phases", {}).get("baseline", {})
        num_queries = phase_config.get("num_queries", 20)

        # Generate queries
        queries = self.query_generator.generate_batch(num_queries)
        log.info(f"Generated {len(queries)} baseline queries")

        # Process queries
        rewards = []
        expert_counts = defaultdict(int)

        for query in queries:
            # Sample expert (policy frozen = random weights)
            expert, log_prob, expert_idx = self.gating_policy.sample_expert()
            expert_counts[expert] += 1

            # Get traversal weight
            graph_weight, _ = self.traversal_policy.get_weight("RIFERIMENTO")

            # Generate response
            response = self.expert_system.generate_response(query, expert, graph_weight)

            # Compute "ground truth" reward (no user feedback in baseline)
            reward = response.confidence * response.graph_score
            rewards.append(reward)

            if not self.dry_run:
                log.debug(f"  {query.query_id}: expert={expert}, reward={reward:.3f}")

        # Compute metrics
        metrics = self._compute_phase_metrics("baseline", queries, rewards, expert_counts)

        self.results["phases"]["baseline"] = asdict(metrics)
        log.info(f"Baseline avg_reward: {metrics.avg_reward:.4f}")
        log.info(f"Baseline expert_usage: {metrics.expert_usage}")

        return metrics

    def run_training_phase(self) -> List[IterationMetrics]:
        """Fase 2: Training con feedback."""
        log.info("=" * 60)
        log.info("FASE 2: TRAINING")
        log.info("=" * 60)

        phase_config = self.config.get("phases", {}).get("training", {})
        num_iterations = phase_config.get("num_iterations", 10)
        queries_per_iter = phase_config.get("queries_per_iteration", 20)
        feedback_rate = phase_config.get("feedback_rate", 0.8)

        iteration_metrics = []
        all_rewards = []

        for iteration in range(num_iterations):
            log.info(f"--- Iteration {iteration + 1}/{num_iterations} ---")

            # Generate queries for this iteration
            queries = self.query_generator.generate_batch(queries_per_iter)

            iter_rewards = []
            expert_counts = defaultdict(int)

            for query in queries:
                # Sample expert
                expert, log_prob, expert_idx = self.gating_policy.sample_expert()
                expert_counts[expert] += 1

                # Get traversal weight
                relation = random.choice(SimulatedTraversalPolicy.RELATIONS)
                graph_weight, trav_log_prob = self.traversal_policy.get_weight(relation)

                # Generate response
                response = self.expert_system.generate_response(query, expert, graph_weight)

                # Decide if user provides feedback
                if self.feedback_rng.random() < feedback_rate:
                    # Select user
                    user = self.community.select_user(query.domain)

                    # Quality score (ground truth)
                    quality = response.confidence * response.graph_score

                    # User feedback (biased + noisy)
                    feedback = user.generate_feedback(quality, self.feedback_rng)

                    # Compute reward
                    reward = self.reward_calculator.compute_reward(
                        response, query, feedback, user.current_authority
                    )

                    # Update policies
                    self.gating_policy.update(expert_idx, reward, log_prob)
                    self.traversal_policy.update(relation, reward)

                    # Update user authority
                    feedback_accuracy = 1.0 - abs(feedback - quality)
                    user.update_authority(feedback_accuracy)

                    iter_rewards.append(reward)
                else:
                    # No feedback, use confidence as pseudo-reward
                    iter_rewards.append(response.confidence * 0.5)

            all_rewards.extend(iter_rewards)

            # Compute iteration metrics
            iter_metric = IterationMetrics(
                iteration=iteration,
                avg_reward=float(np.mean(iter_rewards)),
                policy_entropy=self.gating_policy.entropy(),
                expert_usage={k: v / queries_per_iter for k, v in expert_counts.items()},
                authority_snapshot=self.community.get_authority_snapshot(),
            )
            iteration_metrics.append(iter_metric)

            log.info(f"  avg_reward: {iter_metric.avg_reward:.4f}, entropy: {iter_metric.policy_entropy:.4f}")

            # Store evolution data
            self.results["iterations"].append(asdict(iter_metric))
            self.results["authority_evolution"].append(self.community.get_authority_snapshot())
            self.results["expert_usage_evolution"].append(dict(expert_counts))

        # Aggregate training metrics
        training_metrics = self._compute_phase_metrics(
            "training",
            [],  # queries not stored
            all_rewards,
            defaultdict(int)  # aggregated separately
        )
        self.results["phases"]["training"] = asdict(training_metrics)

        return iteration_metrics

    def run_evaluation_phase(self) -> PhaseMetrics:
        """Fase 3: Evaluation con policy frozen."""
        log.info("=" * 60)
        log.info("FASE 3: EVALUATION")
        log.info("=" * 60)

        phase_config = self.config.get("phases", {}).get("evaluation", {})
        num_queries = phase_config.get("num_queries", 20)

        # Regenerate same queries as baseline (reset generator)
        self.query_generator.query_counter = 0
        queries = self.query_generator.generate_batch(num_queries)

        rewards = []
        expert_counts = defaultdict(int)

        for query in queries:
            # Sample expert (policy learned)
            expert, log_prob, expert_idx = self.gating_policy.sample_expert()
            expert_counts[expert] += 1

            # Get traversal weight
            graph_weight, _ = self.traversal_policy.get_weight("RIFERIMENTO")

            # Generate response
            response = self.expert_system.generate_response(query, expert, graph_weight)

            # Compute reward (same as baseline for comparison)
            reward = response.confidence * response.graph_score
            rewards.append(reward)

        metrics = self._compute_phase_metrics("evaluation", queries, rewards, expert_counts)

        self.results["phases"]["evaluation"] = asdict(metrics)
        log.info(f"Evaluation avg_reward: {metrics.avg_reward:.4f}")
        log.info(f"Evaluation expert_usage: {metrics.expert_usage}")

        return metrics

    def _compute_phase_metrics(
        self,
        phase: str,
        queries: List[QuerySample],
        rewards: List[float],
        expert_counts: Dict[str, int]
    ) -> PhaseMetrics:
        """Calcola metriche per fase."""
        total = sum(expert_counts.values()) or 1
        expert_usage = {k: v / total for k, v in expert_counts.items()}

        # Load balance score
        usage_values = list(expert_usage.values()) or [0.25] * 4
        usage_mean = np.mean(usage_values)
        usage_std = np.std(usage_values)
        load_balance = 1.0 - (usage_std / (usage_mean + 1e-10))

        return PhaseMetrics(
            phase=phase,
            num_queries=len(queries) if queries else len(rewards),
            avg_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            expert_usage=expert_usage,
            load_balance_score=float(load_balance),
            policy_entropy=self.gating_policy.entropy(),
            avg_confidence=0.0,  # Not tracked in simulation
            avg_graph_score=0.0,  # Not tracked in simulation
        )

    def run_statistical_tests(self) -> Dict[str, Any]:
        """Esegue test statistici baseline vs evaluation."""
        log.info("=" * 60)
        log.info("STATISTICAL ANALYSIS")
        log.info("=" * 60)

        baseline = self.results["phases"].get("baseline", {})
        evaluation = self.results["phases"].get("evaluation", {})

        tests = {
            "reward_improvement": {
                "baseline": baseline.get("avg_reward", 0),
                "evaluation": evaluation.get("avg_reward", 0),
                "improvement_pct": 0.0,
                "hypothesis_passed": False,
            },
            "load_balance": {
                "baseline": baseline.get("load_balance_score", 0),
                "evaluation": evaluation.get("load_balance_score", 0),
                "target": 0.75,
                "hypothesis_passed": False,
            },
            "entropy_stability": {
                "final_entropy": evaluation.get("policy_entropy", 0),
                "target_min": 1.0,
                "hypothesis_passed": False,
            }
        }

        # H1: Reward improvement
        if baseline.get("avg_reward", 0) > 0:
            improvement = (evaluation.get("avg_reward", 0) - baseline.get("avg_reward", 0)) / baseline.get("avg_reward", 1)
            tests["reward_improvement"]["improvement_pct"] = float(improvement * 100)
            tests["reward_improvement"]["hypothesis_passed"] = improvement > 0.15

        # H3: Load balance
        tests["load_balance"]["hypothesis_passed"] = evaluation.get("load_balance_score", 0) > 0.75

        # H2: Entropy stability
        tests["entropy_stability"]["hypothesis_passed"] = evaluation.get("policy_entropy", 0) > 1.0

        log.info(f"H1 (Reward +15%): {tests['reward_improvement']['hypothesis_passed']} "
                 f"(improvement: {tests['reward_improvement']['improvement_pct']:.1f}%)")
        log.info(f"H2 (Entropy > 1.0): {tests['entropy_stability']['hypothesis_passed']} "
                 f"(entropy: {tests['entropy_stability']['final_entropy']:.3f})")
        log.info(f"H3 (LBS > 0.75): {tests['load_balance']['hypothesis_passed']} "
                 f"(LBS: {tests['load_balance']['evaluation']:.3f})")

        return tests

    def save_results(self):
        """Salva risultati su file."""
        log.info("=" * 60)
        log.info("SAVING RESULTS")
        log.info("=" * 60)

        # Main metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.results["phases"], f, indent=2, default=str)

        # Full results
        with open(self.output_dir / "full_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Iteration metrics
        with open(self.output_dir / "iterations.json", "w") as f:
            json.dump(self.results["iterations"], f, indent=2, default=str)

        # Authority evolution
        with open(self.output_dir / "authority_evolution.json", "w") as f:
            json.dump(self.results["authority_evolution"], f, indent=2, default=str)

        # Community stats
        with open(self.output_dir / "community_stats.json", "w") as f:
            json.dump(self.community.get_profile_stats(), f, indent=2, default=str)

        # Statistical tests
        tests = self.run_statistical_tests()
        with open(self.output_dir / "statistical_tests.json", "w") as f:
            json.dump(tests, f, indent=2, default=str)

        log.info(f"Results saved to {self.output_dir}")

    def run(self) -> Dict[str, Any]:
        """Esegue esperimento completo."""
        log.info("=" * 60)
        log.info("EXP-023: END-TO-END COMMUNITY SIMULATION")
        log.info("=" * 60)
        log.info(f"Dry run: {self.dry_run}")
        log.info(f"Seed: {self.seed}")
        log.info(f"Community size: {len(self.community.users)}")

        start_time = datetime.now()

        # Run phases
        baseline_metrics = self.run_baseline_phase()
        training_metrics = self.run_training_phase()
        eval_metrics = self.run_evaluation_phase()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Store timing
        self.results["timing"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": duration,
        }

        # Save results
        self.save_results()

        # Summary
        log.info("=" * 60)
        log.info("EXPERIMENT COMPLETE")
        log.info("=" * 60)
        log.info(f"Duration: {duration:.1f}s")
        log.info(f"Baseline avg_reward: {baseline_metrics.avg_reward:.4f}")
        log.info(f"Evaluation avg_reward: {eval_metrics.avg_reward:.4f}")

        improvement = 0
        if baseline_metrics.avg_reward > 0:
            improvement = (eval_metrics.avg_reward - baseline_metrics.avg_reward) / baseline_metrics.avg_reward * 100

        log.info(f"Improvement: {improvement:.1f}%")

        return self.results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EXP-023: E2E Community Simulation")
    parser.add_argument("--config", type=Path, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run without verbose logging")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    runner = EXP023Runner(
        config_path=args.config,
        dry_run=args.dry_run
    )

    if args.seed:
        runner.seed = args.seed

    results = runner.run()

    return results


if __name__ == "__main__":
    main()
