#!/usr/bin/env python3
"""
EXP-024: Real Expert System Integration
=======================================

Esperimento RLCF con Expert System reale:
- LegalKnowledgeGraph.interpret()
- FalkorDB + Qdrant
- Community simulata (da EXP-023)

Usage:
    python scripts/exp024_real_expert.py
    python scripts/exp024_real_expert.py --no-llm
    python scripts/exp024_real_expert.py --queries 10
"""

import asyncio
import json
import random
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Disable structlog noise
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)


# ============================================================================
# Data Classes (riutilizzate da EXP-023)
# ============================================================================

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

    def generate_feedback(self, quality_score: float, rng: random.Random) -> float:
        """Genera feedback con bias e noise."""
        biased = quality_score + self.feedback_bias
        noisy = biased + rng.gauss(0, self.noise_std)
        return max(0.0, min(1.0, noisy))

    def update_authority(self, feedback_accuracy: float, lambda_factor: float = 0.15):
        """Aggiorna authority."""
        self.track_record = (1 - lambda_factor) * self.track_record + lambda_factor * feedback_accuracy
        self.current_authority = 0.4 * self.baseline_authority + 0.35 * self.track_record + 0.25 * feedback_accuracy
        self.feedbacks_given += 1


@dataclass
class QueryResult:
    """Risultato di una query."""
    query_id: str
    query_text: str
    synthesis: str
    confidence: float
    expert_contributions: Dict[str, Any]
    legal_basis: List[str]
    execution_time_ms: float
    routing_decision: Optional[Dict[str, Any]] = None
    feedback: Optional[float] = None
    reward: Optional[float] = None
    user_id: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Query Set per Libro IV (Obbligazioni)
# ============================================================================

LIBRO_IV_QUERIES = [
    # Risoluzione del contratto (Art. 1453-1462)
    "Cos'è la risoluzione del contratto per inadempimento?",
    "Quali sono i presupposti della risoluzione ex art. 1453 c.c.?",
    "Come funziona la diffida ad adempiere secondo l'art. 1454?",
    "Cos'è il termine essenziale nell'art. 1457 c.c.?",
    "Quali sono gli effetti della risoluzione del contratto?",

    # Eccezione di inadempimento (Art. 1460-1462)
    "Cos'è l'eccezione di inadempimento ex art. 1460?",
    "Quando si può invocare l'eccezione inadimplenti non est adimplendum?",
    "Come si applica l'art. 1461 sulla mutatio creditoria?",

    # Clausola penale (Art. 1382-1384)
    "Cos'è la clausola penale secondo il codice civile?",
    "Quando il giudice può ridurre la penale ex art. 1384?",

    # Caparra (Art. 1385-1386)
    "Qual è la differenza tra caparra confirmatoria e penitenziale?",
    "Come funziona la caparra confirmatoria art. 1385?",

    # Responsabilità contrattuale
    "Cos'è la responsabilità del debitore ex art. 1218?",
    "Come si calcola il risarcimento del danno contrattuale?",
    "Cos'è il danno emergente e il lucro cessante?",

    # Query interpretive
    "Come si interpreta l'art. 1453 in combinato disposto con l'art. 1455?",
    "Qual è il rapporto tra risoluzione e risarcimento del danno?",

    # Query procedurali
    "Quali sono i termini per la risoluzione giudiziale?",
    "Come si esercita il diritto di recesso?",

    # Query giurisprudenziali (se disponibili)
    "Come è stata applicata la gravità dell'inadempimento?",
]


# ============================================================================
# Community Manager (da EXP-023)
# ============================================================================

class CommunityManager:
    """Gestisce la community di utenti sintetici."""

    PROFILES = {
        "senior_magistrate": {"count": 2, "baseline": 0.90, "bias": -0.05, "noise": 0.05},
        "strict_expert": {"count": 4, "baseline": 0.85, "bias": 0.0, "noise": 0.08},
        "domain_specialist": {"count": 6, "baseline": 0.70, "bias": 0.05, "noise": 0.10},
        "lenient_student": {"count": 6, "baseline": 0.25, "bias": 0.15, "noise": 0.15},
        "random_noise": {"count": 2, "baseline": 0.10, "bias": 0.0, "noise": 0.30},
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.users: Dict[str, SyntheticUser] = {}
        self._create_community()

    def _create_community(self):
        user_id = 0
        for profile_name, config in self.PROFILES.items():
            for _ in range(config["count"]):
                user = SyntheticUser(
                    user_id=f"user_{user_id:03d}",
                    profile=profile_name,
                    baseline_authority=config["baseline"],
                    current_authority=config["baseline"],
                    feedback_bias=config["bias"],
                    noise_std=config["noise"],
                )
                self.users[user.user_id] = user
                user_id += 1
        log.info(f"Created community with {len(self.users)} users")

    def select_user(self) -> SyntheticUser:
        return self.rng.choice(list(self.users.values()))

    def get_authority_snapshot(self) -> Dict[str, float]:
        return {uid: u.current_authority for uid, u in self.users.items()}

    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        stats = defaultdict(lambda: {"count": 0, "avg_authority": 0.0})
        for user in self.users.values():
            stats[user.profile]["count"] += 1
            stats[user.profile]["avg_authority"] += user.current_authority
        for profile in stats:
            if stats[profile]["count"] > 0:
                stats[profile]["avg_authority"] /= stats[profile]["count"]
        return dict(stats)


# ============================================================================
# Experiment Runner
# ============================================================================

class EXP024Runner:
    """Runner per EXP-024 con Expert System reale."""

    def __init__(
        self,
        num_queries: int = 20,
        use_llm: bool = True,
        seed: int = 42
    ):
        self.num_queries = num_queries
        self.use_llm = use_llm
        self.seed = seed
        self.rng = random.Random(seed)

        # Community
        self.community = CommunityManager(seed=seed + 1)

        # Results
        self.results: Dict[str, Any] = {
            "experiment": {
                "name": "EXP-024",
                "description": "Real Expert System Integration",
                "num_queries": num_queries,
                "use_llm": use_llm,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
            },
            "queries": [],
            "metrics": {},
            "authority_evolution": [],
            "errors": [],
        }

        # Output directory
        self.output_dir = PROJECT_ROOT / "docs/experiments/EXP-024_real_expert_system/results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Knowledge graph (lazy init)
        self._kg = None

    async def _get_kg(self):
        """Lazy init del KnowledgeGraph."""
        if self._kg is None:
            from merlt import LegalKnowledgeGraph, MerltConfig

            config = MerltConfig(
                graph_name="merl_t_dev",
                falkordb_host="localhost",
                falkordb_port=6380,
                qdrant_host="localhost",
                qdrant_port=6333,
            )

            self._kg = LegalKnowledgeGraph(config=config)
            await self._kg.connect()
            log.info("Connected to LegalKnowledgeGraph")

        return self._kg

    async def process_query(self, query_id: str, query_text: str) -> QueryResult:
        """Processa una singola query con Expert System reale."""
        start_time = datetime.now()

        try:
            kg = await self._get_kg()

            # Usa interpret() per query completa
            result = await kg.interpret(
                query=query_text,
                include_search=True,
                max_experts=4,
                timeout_seconds=30.0,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Estrai legal basis
            legal_basis = []
            for item in result.combined_legal_basis:
                if isinstance(item, dict):
                    urn = item.get("urn") or item.get("article_urn", "")
                    if urn:
                        legal_basis.append(urn)
                elif isinstance(item, str):
                    legal_basis.append(item)

            return QueryResult(
                query_id=query_id,
                query_text=query_text,
                synthesis=result.synthesis[:500] if result.synthesis else "",
                confidence=result.confidence,
                expert_contributions=result.expert_contributions,
                legal_basis=legal_basis[:10],  # Limit
                execution_time_ms=execution_time,
                routing_decision=result.routing_decision,
            )

        except Exception as e:
            log.error(f"Error processing query {query_id}: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return QueryResult(
                query_id=query_id,
                query_text=query_text,
                synthesis="",
                confidence=0.0,
                expert_contributions={},
                legal_basis=[],
                execution_time_ms=execution_time,
                error=str(e),
            )

    def compute_quality_score(self, result: QueryResult) -> float:
        """Calcola quality score per feedback generation."""
        if result.error:
            return 0.2

        # Componenti del quality score
        confidence_score = result.confidence

        # Bonus per legal basis trovate
        basis_score = min(1.0, len(result.legal_basis) / 3)

        # Bonus per expert contributions
        expert_score = min(1.0, len(result.expert_contributions) / 2)

        # Penalità per latenza alta
        latency_penalty = max(0, (result.execution_time_ms - 2000) / 5000)

        quality = (
            0.4 * confidence_score +
            0.3 * basis_score +
            0.2 * expert_score -
            0.1 * latency_penalty
        )

        return max(0.1, min(1.0, quality))

    async def run(self) -> Dict[str, Any]:
        """Esegue l'esperimento completo."""
        log.info("=" * 60)
        log.info("EXP-024: REAL EXPERT SYSTEM INTEGRATION")
        log.info("=" * 60)
        log.info(f"Queries: {self.num_queries}")
        log.info(f"Use LLM: {self.use_llm}")
        log.info(f"Community size: {len(self.community.users)}")

        # Select queries
        queries = LIBRO_IV_QUERIES[:self.num_queries]
        if len(queries) < self.num_queries:
            # Pad with repeated queries if needed
            queries = queries * (self.num_queries // len(queries) + 1)
            queries = queries[:self.num_queries]

        log.info(f"Processing {len(queries)} queries...")

        # Process queries
        query_results = []
        rewards = []
        expert_counts = defaultdict(int)

        for i, query_text in enumerate(queries):
            query_id = f"q_{i+1:04d}"
            log.info(f"[{i+1}/{len(queries)}] Processing: {query_text[:50]}...")

            # Process with real expert system
            result = await self.process_query(query_id, query_text)

            # Count experts used
            for expert in result.expert_contributions.keys():
                expert_counts[expert] += 1

            # Select user for feedback
            user = self.community.select_user()

            # Generate feedback
            quality = self.compute_quality_score(result)
            feedback = user.generate_feedback(quality, self.rng)

            # Compute reward
            reward = feedback * user.current_authority * quality

            # Update user authority
            feedback_accuracy = 1.0 - abs(feedback - quality)
            user.update_authority(feedback_accuracy)

            # Store results
            result.feedback = feedback
            result.reward = reward
            result.user_id = user.user_id

            query_results.append(result)
            rewards.append(reward)

            log.info(f"  confidence={result.confidence:.3f}, "
                    f"legal_basis={len(result.legal_basis)}, "
                    f"latency={result.execution_time_ms:.0f}ms, "
                    f"reward={reward:.3f}")

            # Store authority snapshot periodically
            if (i + 1) % 5 == 0:
                self.results["authority_evolution"].append({
                    "query_num": i + 1,
                    "snapshot": self.community.get_authority_snapshot()
                })

        # Compute metrics
        self.results["queries"] = [asdict(r) for r in query_results]

        successful_queries = [r for r in query_results if not r.error]

        self.results["metrics"] = {
            "total_queries": len(queries),
            "successful_queries": len(successful_queries),
            "success_rate": len(successful_queries) / len(queries) if queries else 0,
            "avg_reward": float(np.mean(rewards)) if rewards else 0,
            "std_reward": float(np.std(rewards)) if rewards else 0,
            "avg_confidence": float(np.mean([r.confidence for r in successful_queries])) if successful_queries else 0,
            "avg_latency_ms": float(np.mean([r.execution_time_ms for r in successful_queries])) if successful_queries else 0,
            "avg_legal_basis": float(np.mean([len(r.legal_basis) for r in successful_queries])) if successful_queries else 0,
            "expert_usage": dict(expert_counts),
            "community_stats": self.community.get_profile_stats(),
        }

        # Save results
        self._save_results()

        # Print summary
        log.info("=" * 60)
        log.info("EXPERIMENT COMPLETE")
        log.info("=" * 60)
        log.info(f"Success rate: {self.results['metrics']['success_rate']*100:.1f}%")
        log.info(f"Avg reward: {self.results['metrics']['avg_reward']:.4f}")
        log.info(f"Avg confidence: {self.results['metrics']['avg_confidence']:.3f}")
        log.info(f"Avg latency: {self.results['metrics']['avg_latency_ms']:.0f}ms")
        log.info(f"Avg legal basis: {self.results['metrics']['avg_legal_basis']:.1f}")
        log.info(f"Expert usage: {self.results['metrics']['expert_usage']}")

        return self.results

    def _save_results(self):
        """Salva risultati su file."""
        # Main results
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Metrics only
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.results["metrics"], f, indent=2, default=str)

        # Authority evolution
        with open(self.output_dir / "authority_evolution.json", "w") as f:
            json.dump(self.results["authority_evolution"], f, indent=2, default=str)

        # Individual query results
        interpretations_dir = self.output_dir / "interpretations"
        interpretations_dir.mkdir(exist_ok=True)

        for query_data in self.results["queries"]:
            query_id = query_data["query_id"]
            with open(interpretations_dir / f"{query_id}.json", "w") as f:
                json.dump(query_data, f, indent=2, default=str)

        log.info(f"Results saved to {self.output_dir}")

    async def cleanup(self):
        """Cleanup risorse."""
        if self._kg:
            await self._kg.close()


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="EXP-024: Real Expert System")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    runner = EXP024Runner(
        num_queries=args.queries,
        use_llm=not args.no_llm,
        seed=args.seed
    )

    try:
        results = await runner.run()
    finally:
        await runner.cleanup()

    return results


if __name__ == "__main__":
    asyncio.run(main())
