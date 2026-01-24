#!/usr/bin/env python3
"""
EXP-025: E2E Policy Trainers Simulation
========================================

Simulazione rigorosa end-to-end per validare i policy trainers
con dati realistici del dominio giuridico italiano.

SCENARIO 1: ROUTING (SingleStepTrainer)
- Query giuridiche realistiche (definitional, interpretive, procedural, etc.)
- 4 Expert Types: literal, systemic, principles, precedent
- Feedback simulato basato su expert ottimale per tipo query
- Metrica: Alignment con expert ottimale, reward medio, convergenza

SCENARIO 2: REACT (ReActPPOTrainer)
- Simulazione reasoning multi-step per Expert
- Actions: RETRIEVE, THINK, CITE, SYNTHESIZE, RESPOND
- Reward basato su completeness e efficiency
- Metrica: Success rate, episode length, efficiency

SCENARIO 3: INTEGRATED
- Pipeline completa: Query → Routing → Expert Reasoning → Feedback
- Misura end-to-end performance

Usage:
    python scripts/exp025_e2e_policy_simulation.py
    python scripts/exp025_e2e_policy_simulation.py --scenario routing
    python scripts/exp025_e2e_policy_simulation.py --scenario react
    python scripts/exp025_e2e_policy_simulation.py --scenario integrated
    python scripts/exp025_e2e_policy_simulation.py --all --verbose
"""

import argparse
import json
import logging
import sys
import time
import random
from collections import defaultdict
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
)
from merlt.rlcf.react_ppo_trainer import (
    ReActPPOTrainer,
    ReActConfig,
    ReActPolicy,
    ReActTrajectory,
    ReActStep,
    ReActActionType,
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
        logging.FileHandler(LOG_DIR / "exp025_e2e_simulation.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Legal Domain Data
# ============================================================================

# Tipi di query giuridiche
QUERY_TYPES = {
    "definitional": {
        "description": "Domande su definizioni (Cos'è...)",
        "optimal_expert": "literal",
        "patterns": [
            "Cos'è {concept}?",
            "Definizione di {concept}",
            "Cosa si intende per {concept}?",
            "Qual è la definizione giuridica di {concept}?",
        ],
        "concepts": [
            "legittima difesa",
            "stato di necessità",
            "responsabilità contrattuale",
            "responsabilità extracontrattuale",
            "culpa in contrahendo",
            "buona fede",
            "dolo",
            "colpa grave",
            "forza maggiore",
            "caso fortuito",
            "inadempimento",
            "mora del debitore",
            "caparra confirmatoria",
            "fideiussione",
            "pegno",
            "ipoteca",
        ],
    },
    "interpretive": {
        "description": "Domande su interpretazione sistematica",
        "optimal_expert": "systemic",
        "patterns": [
            "Come si interpreta {norm}?",
            "Qual è la ratio di {norm}?",
            "Come si collega {norm} con {norm2}?",
            "Interpretazione sistematica di {norm}",
        ],
        "norms": [
            "l'art. 1218 c.c.",
            "l'art. 2043 c.c.",
            "l'art. 1337 c.c.",
            "l'art. 52 c.p.",
            "l'art. 2697 c.c.",
            "l'art. 1176 c.c.",
            "l'art. 1453 c.c.",
            "l'art. 2087 c.c.",
            "l'art. 1321 c.c.",
            "l'art. 1325 c.c.",
        ],
    },
    "teleological": {
        "description": "Domande sulla finalità del legislatore",
        "optimal_expert": "principles",
        "patterns": [
            "Qual è la finalità di {norm}?",
            "Perché il legislatore ha previsto {provision}?",
            "Scopo della disciplina su {topic}",
            "Ratio legis di {norm}",
        ],
        "norms": [
            "l'art. 1218 c.c.",
            "l'art. 2043 c.c.",
            "l'art. 1337 c.c.",
            "l'art. 52 c.p.",
            "l'art. 2697 c.c.",
        ],
        "provisions": [
            "la responsabilità oggettiva",
            "il termine di prescrizione",
            "l'obbligo di buona fede",
            "la forma scritta ad substantiam",
            "il diritto di recesso",
            "la clausola penale",
            "la tutela del consumatore",
            "la nullità del contratto",
        ],
        "topics": [
            "responsabilità contrattuale",
            "responsabilità extracontrattuale",
            "obbligazioni",
            "contratti",
        ],
    },
    "jurisprudential": {
        "description": "Domande su orientamenti giurisprudenziali",
        "optimal_expert": "precedent",
        "patterns": [
            "Qual è l'orientamento della Cassazione su {topic}?",
            "Come ha interpretato la giurisprudenza {issue}?",
            "Precedenti su {topic}",
            "Evoluzione giurisprudenziale di {topic}",
        ],
        "topics": [
            "danno esistenziale",
            "danno tanatologico",
            "responsabilità medica",
            "mobbing",
            "demansionamento",
            "concorso di colpa",
            "nesso causale",
            "danno da perdita di chance",
            "danno biologico",
            "danno morale",
        ],
    },
}

# Esperti disponibili
EXPERT_TYPES = ["literal", "systemic", "principles", "precedent"]

# Mapping expert -> indice
EXPERT_TO_IDX = {exp: i for i, exp in enumerate(EXPERT_TYPES)}
IDX_TO_EXPERT = {i: exp for i, exp in enumerate(EXPERT_TYPES)}


# ============================================================================
# Query Generator
# ============================================================================


@dataclass
class LegalQuery:
    """Query giuridica con metadata."""
    text: str
    query_type: str
    optimal_expert: str
    embedding: np.ndarray
    domain: str = "civile"
    difficulty: float = 0.5


class LegalQueryGenerator:
    """
    Generatore di query giuridiche realistiche.

    Genera query basate su pattern realistici del dominio giuridico italiano,
    con embedding che riflettono il tipo di query.
    """

    def __init__(self, embedding_dim: int = 768, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)

        # Pre-compute embedding templates for each query type
        # (in produzione useremmo sentence-transformers)
        self._init_embedding_templates()

    def _init_embedding_templates(self):
        """Inizializza template embedding per tipo query."""
        self.type_templates = {}

        for i, (qtype, data) in enumerate(QUERY_TYPES.items()):
            # Crea template embedding per questo tipo
            template = self.rng.randn(self.embedding_dim).astype(np.float32)
            # Rendi distinguibile il tipo
            template[i * 10:(i + 1) * 10] += 2.0  # Spike per tipo
            template /= np.linalg.norm(template)

            self.type_templates[qtype] = template

    def generate(self, query_type: Optional[str] = None) -> LegalQuery:
        """
        Genera una query giuridica.

        Args:
            query_type: Tipo specifico o None per random

        Returns:
            LegalQuery con testo, tipo, expert ottimale, embedding
        """
        if query_type is None:
            query_type = self.rng.choice(list(QUERY_TYPES.keys()))

        data = QUERY_TYPES[query_type]

        # Genera testo query
        pattern = self.rng.choice(data["patterns"])

        # Sostituisci placeholder con valori appropriati
        text = pattern
        if "{concept}" in pattern and "concepts" in data:
            concept = self.rng.choice(data["concepts"])
            text = text.replace("{concept}", concept)
        if "{norm}" in pattern and "norms" in data:
            norm = self.rng.choice(data["norms"])
            text = text.replace("{norm}", norm)
        if "{norm2}" in pattern and "norms" in data:
            norm2 = self.rng.choice(data["norms"])
            text = text.replace("{norm2}", norm2)
        if "{provision}" in pattern and "provisions" in data:
            provision = self.rng.choice(data["provisions"])
            text = text.replace("{provision}", provision)
        if "{topic}" in pattern and "topics" in data:
            topic = self.rng.choice(data["topics"])
            text = text.replace("{topic}", topic)
        if "{issue}" in pattern and "topics" in data:
            issue = self.rng.choice(data["topics"])
            text = text.replace("{issue}", issue)

        # Genera embedding
        template = self.type_templates[query_type]
        noise = self.rng.randn(self.embedding_dim).astype(np.float32) * 0.3
        embedding = template + noise
        embedding /= np.linalg.norm(embedding)

        return LegalQuery(
            text=text,
            query_type=query_type,
            optimal_expert=data["optimal_expert"],
            embedding=embedding,
            difficulty=self.rng.uniform(0.3, 0.8),
        )

    def generate_batch(self, n: int, balanced: bool = True) -> List[LegalQuery]:
        """Genera batch di query."""
        queries = []

        if balanced:
            # Distribuzione bilanciata tra tipi
            types = list(QUERY_TYPES.keys())
            for i in range(n):
                qtype = types[i % len(types)]
                queries.append(self.generate(qtype))
        else:
            # Random
            for _ in range(n):
                queries.append(self.generate())

        return queries


# ============================================================================
# Feedback Simulator
# ============================================================================


class FeedbackSimulator:
    """
    Simula feedback realistico basato su scelta expert vs ottimale.

    Il feedback dipende da:
    - Alignment expert scelto con ottimale
    - Rumore per simulare varianza reale
    - Difficoltà query
    """

    def __init__(self, noise_level: float = 0.1, seed: int = 42):
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

        # Matrice di compatibilità tra expert
        # (quanto bene un expert può rispondere a query ottimale per altro)
        self.compatibility = {
            "literal": {"literal": 1.0, "systemic": 0.6, "principles": 0.5, "precedent": 0.4},
            "systemic": {"literal": 0.5, "systemic": 1.0, "principles": 0.7, "precedent": 0.5},
            "principles": {"literal": 0.4, "systemic": 0.7, "principles": 1.0, "precedent": 0.6},
            "precedent": {"literal": 0.4, "systemic": 0.5, "principles": 0.6, "precedent": 1.0},
        }

    def compute_reward(
        self,
        chosen_expert: str,
        optimal_expert: str,
        difficulty: float = 0.5,
    ) -> float:
        """
        Calcola reward per scelta expert.

        Args:
            chosen_expert: Expert scelto dalla policy
            optimal_expert: Expert ottimale per questa query
            difficulty: Difficoltà query (0-1)

        Returns:
            Reward in [0, 1]
        """
        # Base reward da compatibilità
        base = self.compatibility[optimal_expert][chosen_expert]

        # Aggiusta per difficoltà (query difficili penalizzano di più scelte sbagliate)
        if chosen_expert != optimal_expert:
            base *= (1 - difficulty * 0.3)

        # Aggiungi rumore
        noise = self.rng.normal(0, self.noise_level)
        reward = np.clip(base + noise, 0, 1)

        return float(reward)

    def generate_feedback(
        self,
        reward: float,
        query_type: str,
    ) -> MultilevelFeedback:
        """Genera MultilevelFeedback da reward."""
        # Varia leggermente tra livelli
        retrieval_score = reward + self.rng.normal(0, 0.05)
        reasoning_score = reward + self.rng.normal(0, 0.08)
        synthesis_score = reward + self.rng.normal(0, 0.05)

        return MultilevelFeedback(
            query_id=f"q_{self.rng.randint(10000)}",
            retrieval_feedback=RetrievalFeedback(
                precision=np.clip(retrieval_score, 0, 1),
                recall=np.clip(retrieval_score * 0.9, 0, 1),
                ranking_quality=np.clip(retrieval_score * 0.95, 0, 1),
            ),
            reasoning_feedback=ReasoningFeedback(
                logical_coherence=np.clip(reasoning_score, 0, 1),
                legal_soundness=np.clip(reasoning_score * 0.95, 0, 1),
                citation_quality=np.clip(reasoning_score * 0.9, 0, 1),
            ),
            synthesis_feedback=SynthesisFeedback(
                clarity=np.clip(synthesis_score, 0, 1),
                completeness=np.clip(synthesis_score * 0.95, 0, 1),
                usefulness=np.clip(synthesis_score, 0, 1),
            ),
        )


# ============================================================================
# Scenario 1: Routing Experiment
# ============================================================================


@dataclass
class RoutingExperimentResults:
    """Risultati esperimento routing."""
    # Training metrics
    train_rewards: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    train_expert_distribution: Dict[str, List[float]] = field(default_factory=dict)

    # Evaluation metrics
    eval_accuracy: float = 0.0
    eval_reward: float = 0.0
    eval_by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Baselines
    random_accuracy: float = 0.0
    random_reward: float = 0.0
    rule_based_accuracy: float = 0.0
    rule_based_reward: float = 0.0

    # Convergence
    convergence_episode: int = 0
    final_baseline: float = 0.0

    # Statistics
    total_train_episodes: int = 0
    total_eval_episodes: int = 0
    training_time_seconds: float = 0.0


def create_trace_for_routing(
    query: LegalQuery,
    weights: np.ndarray,
) -> ExecutionTrace:
    """Crea ExecutionTrace per routing."""
    trace = ExecutionTrace(query_id=f"q_{np.random.randint(10000)}")
    log_probs = np.log(weights + 1e-8)

    for i, expert in enumerate(EXPERT_TYPES):
        trace.add_expert_selection(
            expert_type=expert,
            weight=float(weights[i]),
            log_prob=float(log_probs[i]),
            metadata={
                "source": "gating_policy",
                "query_embedding": query.embedding.tolist(),
                "action_index": i,
                "query_type": query.query_type,
            },
        )

    return trace


def run_routing_experiment(
    n_train: int = 1000,
    n_eval: int = 200,
    learning_rate: float = 0.005,
    embedding_dim: int = 768,
    hidden_dim: int = 128,
    seed: int = 42,
    verbose: bool = True,
) -> RoutingExperimentResults:
    """
    Esegue esperimento routing completo.
    """
    logger.info("=" * 70)
    logger.info("SCENARIO 1: ROUTING EXPERIMENT (SingleStepTrainer)")
    logger.info("=" * 70)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Training episodes: {n_train}")
    logger.info(f"Evaluation episodes: {n_eval}")

    # Initialize components
    query_gen = LegalQueryGenerator(embedding_dim=embedding_dim, seed=seed)
    feedback_sim = FeedbackSimulator(seed=seed + 1)

    policy = GatingPolicy(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_experts=len(EXPERT_TYPES),
    )

    trainer = SingleStepTrainer(
        policy,
        SingleStepConfig(
            learning_rate=learning_rate,
            baseline_decay=0.95,
            entropy_coef=0.01,
            clip_grad_norm=1.0,
        ),
    )

    results = RoutingExperimentResults()

    # =========================================================================
    # BASELINE 1: Random Policy
    # =========================================================================
    logger.info("\n--- BASELINE 1: Random Policy ---")

    random_correct = 0
    random_rewards = []

    for _ in range(n_eval):
        query = query_gen.generate()
        chosen_idx = np.random.randint(len(EXPERT_TYPES))
        chosen_expert = IDX_TO_EXPERT[chosen_idx]

        reward = feedback_sim.compute_reward(chosen_expert, query.optimal_expert, query.difficulty)

        if chosen_expert == query.optimal_expert:
            random_correct += 1
        random_rewards.append(reward)

    results.random_accuracy = random_correct / n_eval
    results.random_reward = float(np.mean(random_rewards))

    logger.info(f"Random Accuracy: {results.random_accuracy:.2%}")
    logger.info(f"Random Avg Reward: {results.random_reward:.3f}")

    # =========================================================================
    # BASELINE 2: Rule-Based (Heuristic)
    # =========================================================================
    logger.info("\n--- BASELINE 2: Rule-Based Heuristic ---")

    # Simple rule: map query_type to expert
    rule_mapping = {qtype: data["optimal_expert"] for qtype, data in QUERY_TYPES.items()}

    rule_correct = 0
    rule_rewards = []

    for _ in range(n_eval):
        query = query_gen.generate()
        # Rule-based choice (cheating - uses query_type directly)
        chosen_expert = rule_mapping.get(query.query_type, "literal")

        reward = feedback_sim.compute_reward(chosen_expert, query.optimal_expert, query.difficulty)

        if chosen_expert == query.optimal_expert:
            rule_correct += 1
        rule_rewards.append(reward)

    results.rule_based_accuracy = rule_correct / n_eval
    results.rule_based_reward = float(np.mean(rule_rewards))

    logger.info(f"Rule-Based Accuracy: {results.rule_based_accuracy:.2%}")
    logger.info(f"Rule-Based Avg Reward: {results.rule_based_reward:.3f}")

    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info(f"\n--- TRAINING ({n_train} episodes) ---")

    results.total_train_episodes = n_train
    start_time = time.time()

    window_size = 100
    recent_correct = []
    recent_rewards = []
    expert_counts = {exp: [] for exp in EXPERT_TYPES}
    current_counts = {exp: 0 for exp in EXPERT_TYPES}

    for ep in range(n_train):
        query = query_gen.generate()

        # Forward pass
        with torch.no_grad():
            input_tensor = torch.tensor(query.embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        # Sample action (argmax for exploitation, could add exploration)
        chosen_idx = int(np.argmax(weights))
        chosen_expert = IDX_TO_EXPERT[chosen_idx]

        # Compute reward
        reward = feedback_sim.compute_reward(chosen_expert, query.optimal_expert, query.difficulty)
        correct = 1 if chosen_expert == query.optimal_expert else 0

        # Track metrics
        recent_correct.append(correct)
        recent_rewards.append(reward)
        current_counts[chosen_expert] += 1

        if len(recent_correct) > window_size:
            recent_correct.pop(0)
            recent_rewards.pop(0)

        # Train
        trace = create_trace_for_routing(query, weights)
        feedback = feedback_sim.generate_feedback(reward, query.query_type)
        trainer.update(trace, feedback)

        # Log periodically
        if (ep + 1) % 200 == 0:
            rolling_acc = sum(recent_correct) / len(recent_correct)
            rolling_reward = sum(recent_rewards) / len(recent_rewards)

            results.train_accuracies.append(rolling_acc)
            results.train_rewards.append(rolling_reward)

            # Track expert distribution
            total = sum(current_counts.values())
            for exp in EXPERT_TYPES:
                if exp not in results.train_expert_distribution:
                    results.train_expert_distribution[exp] = []
                results.train_expert_distribution[exp].append(current_counts[exp] / total if total > 0 else 0)

            if verbose:
                logger.info(
                    f"Ep {ep+1:5d} | "
                    f"Acc: {rolling_acc:.2%} | "
                    f"Reward: {rolling_reward:.3f} | "
                    f"Baseline: {trainer.baseline:.3f}"
                )

            # Reset counts for next window
            current_counts = {exp: 0 for exp in EXPERT_TYPES}

    results.training_time_seconds = time.time() - start_time
    results.final_baseline = trainer.baseline

    logger.info(f"\nTraining completed in {results.training_time_seconds:.1f}s")

    # =========================================================================
    # EVALUATION
    # =========================================================================
    logger.info(f"\n--- EVALUATION ({n_eval} episodes) ---")

    results.total_eval_episodes = n_eval
    eval_correct = 0
    eval_rewards = []
    eval_by_type = {qtype: {"correct": 0, "total": 0, "rewards": []} for qtype in QUERY_TYPES.keys()}

    for _ in range(n_eval):
        query = query_gen.generate()

        with torch.no_grad():
            input_tensor = torch.tensor(query.embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        chosen_idx = int(np.argmax(weights))
        chosen_expert = IDX_TO_EXPERT[chosen_idx]

        reward = feedback_sim.compute_reward(chosen_expert, query.optimal_expert, query.difficulty)

        if chosen_expert == query.optimal_expert:
            eval_correct += 1
            eval_by_type[query.query_type]["correct"] += 1

        eval_rewards.append(reward)
        eval_by_type[query.query_type]["total"] += 1
        eval_by_type[query.query_type]["rewards"].append(reward)

    results.eval_accuracy = eval_correct / n_eval
    results.eval_reward = float(np.mean(eval_rewards))

    # Per-type metrics
    for qtype, data in eval_by_type.items():
        if data["total"] > 0:
            results.eval_by_query_type[qtype] = {
                "accuracy": data["correct"] / data["total"],
                "avg_reward": float(np.mean(data["rewards"])) if data["rewards"] else 0,
                "n_samples": data["total"],
            }

    logger.info(f"Trained Policy Accuracy: {results.eval_accuracy:.2%}")
    logger.info(f"Trained Policy Avg Reward: {results.eval_reward:.3f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ROUTING EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Metric':<25} {'Random':<12} {'Rule-Based':<12} {'Trained':<12}")
    logger.info("-" * 61)
    logger.info(f"{'Accuracy':<25} {results.random_accuracy:<12.2%} {results.rule_based_accuracy:<12.2%} {results.eval_accuracy:<12.2%}")
    logger.info(f"{'Avg Reward':<25} {results.random_reward:<12.3f} {results.rule_based_reward:<12.3f} {results.eval_reward:<12.3f}")

    logger.info("\nPer Query Type:")
    for qtype, metrics in results.eval_by_query_type.items():
        optimal = QUERY_TYPES[qtype]["optimal_expert"]
        logger.info(f"  {qtype:<15} → {optimal:<10} | Acc: {metrics['accuracy']:.2%} | Reward: {metrics['avg_reward']:.3f}")

    # Improvement calculation
    improvement_vs_random = results.eval_accuracy - results.random_accuracy
    improvement_vs_rule = results.eval_accuracy - results.rule_based_accuracy

    logger.info(f"\nImprovement vs Random: {improvement_vs_random:+.2%}")
    logger.info(f"Improvement vs Rule-Based: {improvement_vs_rule:+.2%}")

    return results


# ============================================================================
# Scenario 2: ReAct Experiment
# ============================================================================


@dataclass
class ReActExperimentResults:
    """Risultati esperimento ReAct."""
    # Training metrics
    train_rewards: List[float] = field(default_factory=list)
    train_lengths: List[float] = field(default_factory=list)
    train_success_rates: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)

    # Evaluation
    eval_reward: float = 0.0
    eval_length: float = 0.0
    eval_success_rate: float = 0.0
    eval_efficiency: float = 0.0

    # Baselines
    random_reward: float = 0.0
    random_success_rate: float = 0.0
    random_length: float = 0.0

    # Statistics
    total_train_episodes: int = 0
    total_eval_episodes: int = 0
    training_time_seconds: float = 0.0


# ReAct Actions for Expert reasoning
REACT_ACTIONS = {
    0: "RETRIEVE",    # Recupera fonte dal KG
    1: "THINK",       # Ragiona su fonti
    2: "CITE",        # Cita una fonte
    3: "SYNTHESIZE",  # Sintetizza risposta
    4: "RESPOND",     # Finalizza risposta
}


class ExpertReasoningEnvironment:
    """
    Ambiente per simulare reasoning multi-step di un Expert.

    Il task è costruire una risposta completa attraverso:
    1. RETRIEVE: Recuperare fonti rilevanti (ne servono almeno 2)
    2. THINK: Ragionare sulle fonti (almeno 1 volta)
    3. CITE: Citare fonti (almeno 1)
    4. SYNTHESIZE: Creare risposta coerente
    5. RESPOND: Finalizzare

    Reward:
    - +0.3 per ogni RETRIEVE utile (max 3)
    - +0.2 per ogni THINK dopo RETRIEVE
    - +0.2 per ogni CITE dopo THINK
    - +0.5 se SYNTHESIZE dopo almeno 1 CITE
    - +1.0 bonus se RESPOND con tutti i requisiti
    - -0.05 step penalty
    - -0.5 se RESPOND prematuramente
    """

    def __init__(self, state_dim: int = 128, max_steps: int = 15):
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.num_actions = len(REACT_ACTIONS)

    def reset(self, query_text: str = "") -> np.ndarray:
        """Reset environment."""
        self.steps = 0
        self.query_text = query_text

        # Track progress
        self.retrieves = 0
        self.thinks = 0
        self.cites = 0
        self.synthesized = False
        self.action_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state encoding."""
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Encode progress in first dimensions
        state[0] = self.retrieves / 3.0
        state[1] = self.thinks / 3.0
        state[2] = self.cites / 3.0
        state[3] = 1.0 if self.synthesized else 0.0
        state[4] = self.steps / self.max_steps

        # Encode action history in next dimensions
        for i, action in enumerate(self.action_history[-5:]):
            if i * 5 + 10 < self.state_dim:
                state[i * 5 + 10 + action] = 1.0

        # Random component for variability
        state[50:] = np.random.randn(self.state_dim - 50) * 0.1

        return state

    def _check_requirements(self) -> Tuple[bool, Dict[str, bool]]:
        """Check if requirements for good response are met."""
        reqs = {
            "has_retrieves": self.retrieves >= 2,
            "has_thinks": self.thinks >= 1,
            "has_cites": self.cites >= 1,
            "has_synthesis": self.synthesized,
        }
        return all(reqs.values()), reqs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action."""
        self.steps += 1
        self.action_history.append(action)

        reward = -0.05  # Step penalty
        done = False
        info = {"action": REACT_ACTIONS[action]}

        if action == 0:  # RETRIEVE
            if self.retrieves < 3:
                reward += 0.3
                self.retrieves += 1
            else:
                reward -= 0.1  # Penalty for redundant retrieves

        elif action == 1:  # THINK
            if self.retrieves > 0:
                reward += 0.2
                self.thinks += 1
            else:
                reward -= 0.1  # Can't think without sources

        elif action == 2:  # CITE
            if self.thinks > 0:
                reward += 0.2
                self.cites += 1
            else:
                reward -= 0.1  # Should think before citing

        elif action == 3:  # SYNTHESIZE
            if self.cites > 0:
                reward += 0.5
                self.synthesized = True
            else:
                reward -= 0.2  # Need citations first

        elif action == 4:  # RESPOND
            done = True
            complete, reqs = self._check_requirements()

            if complete:
                reward += 1.0  # Success bonus
                efficiency = 1.0 - (self.steps / self.max_steps)
                reward += efficiency * 0.3  # Efficiency bonus
            else:
                reward -= 0.5  # Incomplete response penalty

            info["complete"] = complete
            info["requirements"] = reqs

        # Max steps
        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True
            info["complete"] = False
            info["requirements"] = self._check_requirements()[1]

        info["steps"] = self.steps
        info["retrieves"] = self.retrieves
        info["thinks"] = self.thinks
        info["cites"] = self.cites
        info["synthesized"] = self.synthesized

        return self._get_state(), reward, done, info


def run_react_experiment(
    n_train: int = 500,
    n_eval: int = 100,
    learning_rate: float = 0.003,
    state_dim: int = 128,
    hidden_dim: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> ReActExperimentResults:
    """
    Esegue esperimento ReAct completo.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SCENARIO 2: REACT EXPERIMENT (ReActPPOTrainer)")
    logger.info("=" * 70)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Training episodes: {n_train}")
    logger.info(f"Evaluation episodes: {n_eval}")

    # Initialize
    env = ExpertReasoningEnvironment(state_dim=state_dim)

    policy = ReActPolicy(
        state_dim=state_dim,
        num_actions=env.num_actions,
        hidden_dim=hidden_dim,
    )

    config = ReActConfig(
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        num_epochs=4,
        value_coef=0.5,
        entropy_coef=0.01,
        step_penalty=0.0,  # Already handled in env
    )

    trainer = ReActPPOTrainer(policy, config)

    results = ReActExperimentResults()

    # =========================================================================
    # BASELINE: Random Policy
    # =========================================================================
    logger.info("\n--- BASELINE: Random Policy ---")

    random_rewards = []
    random_lengths = []
    random_successes = []

    for _ in range(n_eval):
        state = env.reset()
        episode_reward = 0

        for _ in range(env.max_steps):
            action = np.random.randint(env.num_actions)
            state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                break

        random_rewards.append(episode_reward)
        random_lengths.append(info["steps"])
        random_successes.append(1 if info.get("complete", False) else 0)

    results.random_reward = float(np.mean(random_rewards))
    results.random_length = float(np.mean(random_lengths))
    results.random_success_rate = float(np.mean(random_successes))

    logger.info(f"Random - Reward: {results.random_reward:.3f}")
    logger.info(f"Random - Length: {results.random_length:.1f}")
    logger.info(f"Random - Success: {results.random_success_rate:.2%}")

    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info(f"\n--- TRAINING ({n_train} episodes) ---")

    results.total_train_episodes = n_train
    start_time = time.time()

    batch_size = 20
    window_size = 50
    recent_rewards = []
    recent_lengths = []
    recent_successes = []

    for ep in range(n_train):
        state = env.reset(query_text=f"Query {ep}")

        trajectory = ReActTrajectory(
            query_id=f"ep_{ep}",
            query=f"Legal reasoning task {ep}",
            query_embedding=torch.randn(state_dim),
            expert_type="literal",
        )

        episode_reward = 0

        for step_idx in range(env.max_steps):
            state_tensor = torch.tensor(state, device=policy.device)
            action_idx, log_prob, value = policy.select_action(state_tensor)

            next_state, reward, done, info = env.step(action_idx)
            episode_reward += reward

            # Map action index to ReActActionType
            action_type_map = {
                0: ReActActionType.SEARCH_KG,      # RETRIEVE
                1: ReActActionType.THINK,          # THINK
                2: ReActActionType.READ_ARTICLE,   # CITE
                3: ReActActionType.EXTRACT_PRINCIPLE,  # SYNTHESIZE
                4: ReActActionType.FINAL_ANSWER,   # RESPOND
            }

            react_step = ReActStep(
                state=state_tensor,
                action_type=action_type_map.get(action_idx, ReActActionType.THINK),
                action_args={"action": REACT_ACTIONS[action_idx]},
                action_index=action_idx,
                log_prob=log_prob,
                value=value,
                observation=f"Step {step_idx}: {REACT_ACTIONS[action_idx]}",
                reward=reward,
                done=done,
            )
            trajectory.add_step(react_step)

            state = next_state

            if done:
                break

        # Final reward
        final_reward = 1.0 if info.get("complete", False) else 0.0
        trajectory.set_final_reward(final_reward)

        trainer.add_trajectory(trajectory)

        # Track
        recent_rewards.append(episode_reward)
        recent_lengths.append(info["steps"])
        recent_successes.append(1 if info.get("complete", False) else 0)

        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
            recent_lengths.pop(0)
            recent_successes.pop(0)

        # Update
        if (ep + 1) % batch_size == 0:
            update_metrics = trainer.update()

            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)

            results.train_rewards.append(avg_reward)
            results.train_lengths.append(avg_length)
            results.train_success_rates.append(success_rate)

            if "value_loss" in update_metrics:
                results.value_losses.append(update_metrics["value_loss"])
            if "policy_loss" in update_metrics:
                results.policy_losses.append(update_metrics["policy_loss"])

            if verbose and (ep + 1) % 100 == 0:
                logger.info(
                    f"Ep {ep+1:4d} | "
                    f"Reward: {avg_reward:.3f} | "
                    f"Length: {avg_length:.1f} | "
                    f"Success: {success_rate:.2%}"
                )

    results.training_time_seconds = time.time() - start_time
    logger.info(f"\nTraining completed in {results.training_time_seconds:.1f}s")

    # =========================================================================
    # EVALUATION
    # =========================================================================
    logger.info(f"\n--- EVALUATION ({n_eval} episodes) ---")

    results.total_eval_episodes = n_eval
    eval_rewards = []
    eval_lengths = []
    eval_successes = []

    for _ in range(n_eval):
        state = env.reset()
        episode_reward = 0

        for _ in range(env.max_steps):
            state_tensor = torch.tensor(state, device=policy.device)
            action_idx, _, _ = policy.select_action(state_tensor, deterministic=True)
            state, reward, done, info = env.step(action_idx)
            episode_reward += reward

            if done:
                break

        eval_rewards.append(episode_reward)
        eval_lengths.append(info["steps"])
        eval_successes.append(1 if info.get("complete", False) else 0)

    results.eval_reward = float(np.mean(eval_rewards))
    results.eval_length = float(np.mean(eval_lengths))
    results.eval_success_rate = float(np.mean(eval_successes))

    # Efficiency: success with fewer steps
    if results.eval_success_rate > 0:
        avg_success_length = np.mean([l for l, s in zip(eval_lengths, eval_successes) if s])
        results.eval_efficiency = 1.0 - (avg_success_length / env.max_steps)
    else:
        results.eval_efficiency = 0.0

    logger.info(f"Trained - Reward: {results.eval_reward:.3f}")
    logger.info(f"Trained - Length: {results.eval_length:.1f}")
    logger.info(f"Trained - Success: {results.eval_success_rate:.2%}")
    logger.info(f"Trained - Efficiency: {results.eval_efficiency:.2%}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("REACT EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Metric':<20} {'Random':<15} {'Trained':<15} {'Delta':<15}")
    logger.info("-" * 65)
    logger.info(f"{'Avg Reward':<20} {results.random_reward:<15.3f} {results.eval_reward:<15.3f} {results.eval_reward - results.random_reward:+.3f}")
    logger.info(f"{'Avg Length':<20} {results.random_length:<15.1f} {results.eval_length:<15.1f} {results.eval_length - results.random_length:+.1f}")
    logger.info(f"{'Success Rate':<20} {results.random_success_rate:<15.2%} {results.eval_success_rate:<15.2%} {results.eval_success_rate - results.random_success_rate:+.2%}")

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="EXP-025: E2E Policy Simulation")
    parser.add_argument("--scenario", choices=["routing", "react", "integrated", "all"], default="all")
    parser.add_argument("--n-train", type=int, default=1000, help="Training episodes")
    parser.add_argument("--n-eval", type=int, default=200, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else LOG_DIR / "exp025_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXP-025: E2E POLICY TRAINERS SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {output_dir}")

    results = {}
    start_time = time.time()

    # Run experiments
    if args.scenario in ["routing", "all"]:
        routing_results = run_routing_experiment(
            n_train=args.n_train,
            n_eval=args.n_eval,
            seed=args.seed,
            verbose=args.verbose,
        )
        results["routing"] = asdict(routing_results)

    if args.scenario in ["react", "all"]:
        react_results = run_react_experiment(
            n_train=args.n_train // 2,  # ReAct è più costoso
            n_eval=args.n_eval // 2,
            seed=args.seed,
            verbose=args.verbose,
        )
        results["react"] = asdict(react_results)

    elapsed = time.time() - start_time

    # Save results
    results_file = output_dir / f"exp025_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        logger.info(f"\nROUTING SUMMARY:")
        logger.info(f"  Trained Accuracy: {r['eval_accuracy']:.2%} (vs {r['random_accuracy']:.2%} random)")
        logger.info(f"  Trained Reward: {r['eval_reward']:.3f} (vs {r['random_reward']:.3f} random)")

    if "react" in results:
        r = results["react"]
        logger.info(f"\nREACT SUMMARY:")
        logger.info(f"  Trained Success: {r['eval_success_rate']:.2%} (vs {r['random_success_rate']:.2%} random)")
        logger.info(f"  Trained Reward: {r['eval_reward']:.3f} (vs {r['random_reward']:.3f} random)")


if __name__ == "__main__":
    main()
