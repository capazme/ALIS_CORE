#!/usr/bin/env python3
"""
E2E RLCF Training Loop
======================

Script principale per training end-to-end del sistema RLCF.

Usa le componenti reali:
- GatingPolicy da merlt.rlcf.policy_gradient
- PolicyGradientTrainer per REINFORCE
- MultiExpertOrchestrator con neural routing
- ExecutionTrace per tracciamento azioni
- MultilevelFeedback per feedback strutturato

Fasi:
1. Baseline: Valuta routing rule-based
2. Training: Policy gradient con feedback (simulati o reali)
3. Evaluation: Confronta policy trainata vs baseline

Usage:
    python scripts/e2e_rlcf_training.py
    python scripts/e2e_rlcf_training.py --epochs 10 --queries-per-epoch 50
    python scripts/e2e_rlcf_training.py --config config/training.yaml
"""

import asyncio
import argparse
import json
import logging
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from merlt.rlcf.policy_gradient import (
    GatingPolicy,
    TraversalPolicy,
    PolicyGradientTrainer,
    TrainerConfig,
)
from merlt.rlcf.execution_trace import ExecutionTrace, Action
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
    create_feedback_from_user_rating,
)
from merlt.experts.orchestrator import MultiExpertOrchestrator, OrchestratorConfig
from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig, SynthesisMode
from merlt.experts.base import ExpertContext

import structlog
log = structlog.get_logger()


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "e2e_rlcf"
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "e2e_rlcf"


@dataclass
class TrainingConfig:
    """Configurazione per E2E training."""
    # Training
    epochs: int = 5
    queries_per_epoch: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    baseline_decay: float = 0.99

    # Policy
    embedding_dim: int = 768
    hidden_dim: int = 256
    num_experts: int = 4
    device: str = "cpu"

    # Feedback
    feedback_rate: float = 0.8  # Probabilita' di feedback per query
    use_synthetic_feedback: bool = True

    # Checkpointing
    checkpoint_every_n_epochs: int = 1
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)

    # Logging
    log_every_n_queries: int = 5
    log_dir: str = str(DEFAULT_LOG_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)

    # Random seed
    seed: int = 42


@dataclass
class EpochMetrics:
    """Metriche per singola epoca."""
    epoch: int
    num_queries: int
    num_feedbacks: int
    avg_reward: float
    avg_loss: float
    expert_usage: Dict[str, float]
    baseline: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainingResults:
    """Risultati complessivi del training."""
    total_epochs: int
    total_queries: int
    total_feedbacks: int
    final_avg_reward: float
    baseline_avg_reward: float
    improvement_pct: float
    epoch_history: List[Dict[str, Any]]
    config: Dict[str, Any]
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# SAMPLE QUERIES
# =============================================================================

SAMPLE_QUERIES = [
    # Definitional
    ("Cos'è il contratto?", "definitional", "literal"),
    ("Definizione di legittima difesa", "definitional", "literal"),
    ("Cosa si intende per buona fede?", "definitional", "literal"),
    ("Cos'è la responsabilità extracontrattuale?", "definitional", "literal"),

    # Interpretive/Systemic
    ("Come si interpreta l'art. 1453 c.c.?", "interpretive", "systemic"),
    ("Rapporto tra art. 2043 e art. 2059 c.c.", "interpretive", "systemic"),
    ("Interpretazione sistematica del diritto di recesso", "interpretive", "systemic"),
    ("Connessione tra possesso e proprietà", "interpretive", "systemic"),

    # Normative/Principles
    ("Ratio legis dell'art. 1218 c.c.", "normative", "principles"),
    ("Finalità della disciplina sulla garanzia", "normative", "principles"),
    ("Principi alla base della responsabilità contrattuale", "normative", "principles"),
    ("Scopo della prescrizione nel codice civile", "normative", "principles"),

    # Jurisprudential
    ("Orientamento della Cassazione sulla clausola penale", "jurisprudential", "precedent"),
    ("Evoluzione giurisprudenziale del danno non patrimoniale", "jurisprudential", "precedent"),
    ("Precedenti sulla risoluzione per inadempimento", "jurisprudential", "precedent"),
    ("Come ha interpretato la Corte l'art. 1337 c.c.?", "jurisprudential", "precedent"),

    # Mixed
    ("Responsabilità del debitore ex art. 1218 c.c.", "mixed", "systemic"),
    ("Obblighi precontrattuali di informazione", "mixed", "principles"),
    ("Tutela del consumatore nel codice civile", "mixed", "systemic"),
    ("Effetti della risoluzione del contratto", "mixed", "literal"),
]


# =============================================================================
# MOCK EMBEDDING SERVICE
# =============================================================================

class MockEmbeddingService:
    """
    Mock embedding service per testing.

    In produzione, usare il vero EmbeddingService da merlt.storage.vectors.
    """

    def __init__(self, dim: int = 768, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self._cache: Dict[str, np.ndarray] = {}

    async def encode_query_async(self, query: str) -> np.ndarray:
        """Genera embedding deterministico per query."""
        if query not in self._cache:
            # Hash-based seed per determinismo
            query_seed = hash(query) % (2**32)
            local_rng = np.random.RandomState(query_seed)
            embedding = local_rng.randn(self.dim).astype(np.float32)
            # Normalizza
            embedding = embedding / np.linalg.norm(embedding)
            self._cache[query] = embedding
        return self._cache[query]


# =============================================================================
# SYNTHETIC FEEDBACK GENERATOR
# =============================================================================

class SyntheticFeedbackGenerator:
    """
    Genera feedback sintetici per training.

    Simula feedback basati su:
    - Query type
    - Expert selezionato
    - Noise stocastico
    """

    # Quality base per expert dato query type
    EXPERT_QUALITY = {
        "literal": {"definitional": 0.9, "interpretive": 0.6, "normative": 0.5, "jurisprudential": 0.4, "mixed": 0.6},
        "systemic": {"definitional": 0.6, "interpretive": 0.85, "normative": 0.7, "jurisprudential": 0.6, "mixed": 0.75},
        "principles": {"definitional": 0.5, "interpretive": 0.7, "normative": 0.9, "jurisprudential": 0.6, "mixed": 0.7},
        "precedent": {"definitional": 0.4, "interpretive": 0.6, "normative": 0.6, "jurisprudential": 0.9, "mixed": 0.65},
    }

    def __init__(self, noise_std: float = 0.1, seed: int = 42):
        self.noise_std = noise_std
        self.rng = random.Random(seed)

    def generate(
        self,
        query_type: str,
        expert_weights: Dict[str, float],
        trace_id: str
    ) -> MultilevelFeedback:
        """
        Genera MultilevelFeedback sintetico.

        Args:
            query_type: Tipo query (definitional, interpretive, etc.)
            expert_weights: Pesi usati per gli expert
            trace_id: ID del trace

        Returns:
            MultilevelFeedback
        """
        # Calcola quality pesata
        weighted_quality = 0.0
        for expert, weight in expert_weights.items():
            expert_quality = self.EXPERT_QUALITY.get(expert, {}).get(query_type, 0.5)
            weighted_quality += weight * expert_quality

        # Aggiungi noise
        noise = self.rng.gauss(0, self.noise_std)
        base_score = max(0.1, min(1.0, weighted_quality + noise))

        # Genera componenti feedback con variazione
        retrieval_score = max(0.1, min(1.0, base_score + self.rng.gauss(0, 0.05)))
        reasoning_score = max(0.1, min(1.0, base_score + self.rng.gauss(0, 0.05)))
        synthesis_score = max(0.1, min(1.0, base_score + self.rng.gauss(0, 0.05)))

        return MultilevelFeedback(
            query_id=trace_id,
            retrieval_feedback=RetrievalFeedback(
                precision=retrieval_score,
                recall=retrieval_score * 0.9,
                sources_relevant=int(retrieval_score * 5),
                sources_total=5,
                ranking_quality=retrieval_score
            ),
            reasoning_feedback=ReasoningFeedback(
                logical_coherence=reasoning_score,
                legal_soundness=reasoning_score * 0.95,
                citation_quality=reasoning_score * 0.9,
                interpretation_accuracy=reasoning_score,
                expert_agreement=reasoning_score * 0.85,
                reasoning_steps_clear=reasoning_score * 0.9
            ),
            synthesis_feedback=SynthesisFeedback(
                clarity=synthesis_score,
                completeness=synthesis_score * 0.9,
                usefulness=synthesis_score,
                conciseness=synthesis_score * 0.85,
                language_quality=synthesis_score * 0.9,
                structure_quality=synthesis_score * 0.95,
                user_satisfaction=synthesis_score
            ),
            overall_rating=base_score
        )


# =============================================================================
# TRAINING LOOP
# =============================================================================

class E2ERLCFTrainer:
    """
    Trainer end-to-end per RLCF.

    Coordina:
    - GatingPolicy per neural routing
    - PolicyGradientTrainer per updates
    - MultiExpertOrchestrator per execution
    - Feedback collection (synthetic o real)
    """

    def __init__(self, config: TrainingConfig):
        """
        Inizializza trainer.

        Args:
            config: TrainingConfig
        """
        self.config = config

        # Set seed
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Setup directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_components()

        # Metrics
        self.epoch_metrics: List[EpochMetrics] = []
        self.expert_usage_history: Dict[str, List[float]] = {
            "literal": [], "systemic": [], "principles": [], "precedent": []
        }

        log.info(
            "E2ERLCFTrainer initialized",
            config=asdict(config),
            checkpoint_dir=config.checkpoint_dir
        )

    def _init_components(self):
        """Inizializza componenti ML."""
        # Mock embedding service
        self.embedding_service = MockEmbeddingService(
            dim=self.config.embedding_dim,
            seed=self.config.seed
        )

        # GatingPolicy
        self.gating_policy = GatingPolicy(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_experts=self.config.num_experts,
            device=self.config.device
        )

        # Trainer
        trainer_config = TrainerConfig(
            learning_rate=self.config.learning_rate,
            baseline_decay=self.config.baseline_decay
        )
        self.trainer = PolicyGradientTrainer(
            policy=self.gating_policy,
            config=trainer_config
        )

        # Synthesizer (mock - no AI service)
        synth_config = SynthesisConfig(mode=SynthesisMode.CONVERGENT)
        self.synthesizer = AdaptiveSynthesizer(
            ai_service=None,
            config=synth_config
        )

        # Orchestrator
        self.orchestrator = MultiExpertOrchestrator(
            synthesizer=self.synthesizer,
            tools=[],  # No tools for now
            ai_service=None,  # No AI service
            gating_policy=self.gating_policy,
            embedding_service=self.embedding_service,
            config=OrchestratorConfig(
                max_experts=4,
                parallel_execution=True
            )
        )

        # Feedback generator
        self.feedback_generator = SyntheticFeedbackGenerator(
            noise_std=0.1,
            seed=self.config.seed
        )

        log.info("Components initialized successfully")

    async def run_baseline(self, num_queries: int = 20) -> float:
        """
        Esegue baseline phase (no training).

        Args:
            num_queries: Numero di query per baseline

        Returns:
            Average reward baseline
        """
        log.info(f"Running baseline phase with {num_queries} queries")

        rewards = []
        queries = random.sample(SAMPLE_QUERIES, min(num_queries, len(SAMPLE_QUERIES)))

        for query_text, query_type, gold_expert in queries:
            try:
                # Process senza training
                result, trace = await self.orchestrator.process(
                    query=query_text,
                    return_trace=True
                )

                # Get expert weights from trace
                expert_weights = self._extract_expert_weights(trace)

                # Generate feedback
                feedback = self.feedback_generator.generate(
                    query_type=query_type,
                    expert_weights=expert_weights,
                    trace_id=trace.query_id
                )

                rewards.append(feedback.overall_score())

            except Exception as e:
                log.warning(f"Baseline query failed: {e}")
                continue

        avg_reward = np.mean(rewards) if rewards else 0.0
        log.info(f"Baseline completed", avg_reward=f"{avg_reward:.4f}", num_queries=len(rewards))

        return avg_reward

    def _extract_expert_weights(self, trace: ExecutionTrace) -> Dict[str, float]:
        """Estrae expert weights dal trace."""
        expert_weights = {"literal": 0.0, "systemic": 0.0, "principles": 0.0, "precedent": 0.0}

        for action in trace.actions:
            if action.action_type == "expert_selection":
                expert_type = action.parameters.get("expert_type", "")
                weight = action.parameters.get("weight", 0.0)
                if expert_type in expert_weights:
                    expert_weights[expert_type] = weight

        # Normalizza se necessario
        total = sum(expert_weights.values())
        if total > 0:
            expert_weights = {k: v/total for k, v in expert_weights.items()}
        else:
            # Default uniform
            expert_weights = {k: 0.25 for k in expert_weights}

        return expert_weights

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        Esegue una singola epoca di training.

        Args:
            epoch: Numero epoca

        Returns:
            EpochMetrics
        """
        log.info(f"Starting epoch {epoch + 1}/{self.config.epochs}")

        # Sample queries
        queries = random.choices(SAMPLE_QUERIES, k=self.config.queries_per_epoch)

        rewards = []
        losses = []
        feedbacks_collected = 0
        epoch_expert_usage = {"literal": 0.0, "systemic": 0.0, "principles": 0.0, "precedent": 0.0}

        for i, (query_text, query_type, gold_expert) in enumerate(queries):
            try:
                # Process con trace
                result, trace = await self.orchestrator.process(
                    query=query_text,
                    return_trace=True
                )

                # Extract weights
                expert_weights = self._extract_expert_weights(trace)

                # Accumula usage
                for exp, weight in expert_weights.items():
                    epoch_expert_usage[exp] += weight

                # Decide se generare feedback
                if random.random() < self.config.feedback_rate:
                    # Generate feedback
                    feedback = self.feedback_generator.generate(
                        query_type=query_type,
                        expert_weights=expert_weights,
                        trace_id=trace.query_id
                    )

                    # Update policy
                    metrics = self.trainer.update_from_feedback(trace, feedback)

                    rewards.append(feedback.overall_score())
                    losses.append(metrics.get("loss", 0.0))
                    feedbacks_collected += 1

                # Log progress
                if (i + 1) % self.config.log_every_n_queries == 0:
                    log.info(
                        f"Epoch {epoch + 1} progress",
                        query=i + 1,
                        total=self.config.queries_per_epoch,
                        avg_reward=f"{np.mean(rewards):.4f}" if rewards else "N/A"
                    )

            except Exception as e:
                log.warning(f"Query processing failed: {e}")
                continue

        # Normalize expert usage
        total_usage = sum(epoch_expert_usage.values())
        if total_usage > 0:
            epoch_expert_usage = {k: v/total_usage for k, v in epoch_expert_usage.items()}

        # Track history
        for exp, usage in epoch_expert_usage.items():
            self.expert_usage_history[exp].append(usage)

        # Create metrics
        metrics = EpochMetrics(
            epoch=epoch,
            num_queries=len(queries),
            num_feedbacks=feedbacks_collected,
            avg_reward=float(np.mean(rewards)) if rewards else 0.0,
            avg_loss=float(np.mean(losses)) if losses else 0.0,
            expert_usage=epoch_expert_usage,
            baseline=self.trainer.baseline
        )

        self.epoch_metrics.append(metrics)

        log.info(
            f"Epoch {epoch + 1} completed",
            avg_reward=f"{metrics.avg_reward:.4f}",
            avg_loss=f"{metrics.avg_loss:.4f}",
            feedbacks=feedbacks_collected,
            baseline=f"{metrics.baseline:.4f}"
        )

        return metrics

    async def train(self) -> TrainingResults:
        """
        Esegue training completo.

        Returns:
            TrainingResults
        """
        start_time = datetime.now()

        log.info("Starting E2E RLCF Training")
        log.info(f"Config: {asdict(self.config)}")

        # Baseline
        baseline_reward = await self.run_baseline(num_queries=20)

        # Training epochs
        for epoch in range(self.config.epochs):
            metrics = await self.train_epoch(epoch)

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
                self.trainer.save_checkpoint(
                    str(checkpoint_path),
                    metadata={
                        "epoch": epoch + 1,
                        "avg_reward": metrics.avg_reward,
                        "baseline_reward": baseline_reward
                    }
                )

        # Final evaluation
        final_reward = await self.run_baseline(num_queries=20)

        # Save final checkpoint
        final_checkpoint = Path(self.config.checkpoint_dir) / "final.pt"
        self.trainer.save_checkpoint(
            str(final_checkpoint),
            metadata={
                "final": True,
                "avg_reward": final_reward,
                "baseline_reward": baseline_reward
            }
        )

        # Calculate improvement
        improvement_pct = 0.0
        if baseline_reward > 0:
            improvement_pct = ((final_reward - baseline_reward) / baseline_reward) * 100

        duration = (datetime.now() - start_time).total_seconds()

        # Results
        results = TrainingResults(
            total_epochs=self.config.epochs,
            total_queries=sum(m.num_queries for m in self.epoch_metrics),
            total_feedbacks=sum(m.num_feedbacks for m in self.epoch_metrics),
            final_avg_reward=final_reward,
            baseline_avg_reward=baseline_reward,
            improvement_pct=improvement_pct,
            epoch_history=[asdict(m) for m in self.epoch_metrics],
            config=asdict(self.config),
            duration_seconds=duration
        )

        # Save results
        results_path = Path(self.config.results_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)

        log.info(
            "Training completed",
            baseline_reward=f"{baseline_reward:.4f}",
            final_reward=f"{final_reward:.4f}",
            improvement=f"{improvement_pct:.1f}%",
            duration=f"{duration:.1f}s",
            results_path=str(results_path)
        )

        return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E2E RLCF Training Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--queries-per-epoch", type=int, default=20,
        help="Queries per epoch"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--feedback-rate", type=float, default=0.8,
        help="Probability of feedback per query"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for training"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    # Config
    config = TrainingConfig(
        epochs=args.epochs,
        queries_per_epoch=args.queries_per_epoch,
        learning_rate=args.learning_rate,
        feedback_rate=args.feedback_rate,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    # Run training
    trainer = E2ERLCFTrainer(config)
    results = await trainer.train()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs:           {results.total_epochs}")
    print(f"Total queries:    {results.total_queries}")
    print(f"Total feedbacks:  {results.total_feedbacks}")
    print(f"Baseline reward:  {results.baseline_avg_reward:.4f}")
    print(f"Final reward:     {results.final_avg_reward:.4f}")
    print(f"Improvement:      {results.improvement_pct:+.1f}%")
    print(f"Duration:         {results.duration_seconds:.1f}s")
    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())
