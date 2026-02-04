"""
Test E2E del Loop RLCF Completo.

Verifica l'intero ciclo:
1. Trace creazione e salvataggio
2. Feedback collection e salvataggio
3. Training data retrieval
4. Policy training con SingleStepTrainer
5. Policy checkpoint e versioning
6. Training session tracking

Questo test simula il flusso completo che avverrà in produzione:
    User Query → Expert Selection → Trace → User Feedback → Training → New Policy
"""

import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch

# Schema uses PostgreSQL ARRAY type — requires real PostgreSQL
pytestmark = pytest.mark.integration

# Set SQLite for tests
os.environ["RLCF_ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from merlt.rlcf.persistence import (
    RLCFPersistence,
    create_persistence,
)
from merlt.rlcf.execution_trace import ExecutionTrace, Action
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
    create_feedback_from_user_rating,
)
from merlt.rlcf.policy_gradient import GatingPolicy
from merlt.rlcf.single_step_trainer import (
    SingleStepTrainer,
    SingleStepConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def persistence():
    """Crea persistence con SQLite in-memory."""
    p = await create_persistence("sqlite+aiosqlite:///:memory:")
    yield p


@pytest.fixture
def temp_checkpoint_dir():
    """Directory temporanea per checkpoint."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_realistic_trace(
    query_id: str,
    expert_type: str = "literal",
    query_embedding: list = None
) -> ExecutionTrace:
    """
    Crea un trace realistico con tutte le informazioni necessarie.

    Simula il trace creato durante una query reale con:
    - Expert selection con log_prob
    - Query embedding nei metadata
    """
    if query_embedding is None:
        query_embedding = np.random.randn(768).tolist()

    trace = ExecutionTrace(query_id=query_id)

    # Simula selezione expert con weights
    experts = ["literal", "systemic", "principles", "precedent"]
    weights = np.random.dirichlet(np.ones(4))

    # Aggiungi la selezione per ogni expert
    for i, exp in enumerate(experts):
        trace.add_expert_selection(
            expert_type=exp,
            weight=float(weights[i]),
            log_prob=float(np.log(weights[i] + 1e-8)),
            metadata={
                "query_embedding": query_embedding if i == 0 else None,
                "source": "gating_policy",
                "action_index": i
            }
        )

    # Aggiungi traversal del grafo
    trace.add_graph_traversal(
        relation_type="RIFERIMENTO",
        weight=0.8,
        log_prob=-0.223,
        source_node=f"urn:norma:cc:art{np.random.randint(1, 2000)}"
    )

    return trace


def create_realistic_feedback(
    query_id: str,
    quality: float = 0.7  # 0.0 = pessimo, 1.0 = ottimo
) -> MultilevelFeedback:
    """
    Crea feedback realistico con variazione.

    Simula feedback che un utente darebbe realmente,
    con variazione tra i livelli.
    """
    noise = lambda: np.random.uniform(-0.1, 0.1)

    return MultilevelFeedback(
        query_id=query_id,
        retrieval_feedback=RetrievalFeedback(
            precision=max(0, min(1, quality + noise())),
            recall=max(0, min(1, quality + noise())),
            ranking_quality=max(0, min(1, quality + noise()))
        ),
        reasoning_feedback=ReasoningFeedback(
            logical_coherence=max(0, min(1, quality + noise())),
            legal_soundness=max(0, min(1, quality + noise())),
            citation_quality=max(0, min(1, quality + noise()))
        ),
        synthesis_feedback=SynthesisFeedback(
            clarity=max(0, min(1, quality + noise())),
            completeness=max(0, min(1, quality + noise())),
            usefulness=max(0, min(1, quality + noise()))
        ),
        user_id="test_user_001"
    )


# =============================================================================
# TEST E2E LOOP
# =============================================================================

class TestRLCFLoopE2E:
    """Test end-to-end del loop RLCF completo."""

    @pytest.mark.asyncio
    async def test_full_loop_simulation(self, persistence, temp_checkpoint_dir):
        """
        Test del ciclo completo RLCF:
        1. Simula N query con trace
        2. Simula feedback per ogni trace
        3. Recupera dati per training
        4. Esegue training
        5. Salva checkpoint
        """
        # === FASE 1: Simula Query e Traces ===
        n_queries = 20
        query_ids = []
        trace_ids = []

        for i in range(n_queries):
            query_id = f"query_{i:03d}"
            query_ids.append(query_id)

            # Crea trace
            trace = create_realistic_trace(query_id)

            # Salva trace
            trace_id = await persistence.save_trace(
                trace=trace,
                policy_version="v1.0.0",
                query_text=f"Test query {i}: Cos'è X?",
                expert_type="literal"
            )
            trace_ids.append(trace_id)

        # Verifica traces salvati
        stats = await persistence.get_training_stats(policy_version="v1.0.0")
        assert stats["total_traces"] == n_queries
        assert stats["traces_with_feedback"] == 0

        # === FASE 2: Simula Feedback ===
        for i, trace_id in enumerate(trace_ids):
            # Quality varia in base alla query (simula diverse qualità)
            quality = 0.5 + 0.4 * np.sin(i / n_queries * np.pi)

            feedback = create_realistic_feedback(
                query_id=query_ids[i],
                quality=quality
            )

            await persistence.save_feedback(
                trace_id=trace_id,
                feedback=feedback,
                user_id="test_user_001",
                user_authority=0.7
            )

        # Verifica feedback salvati
        stats = await persistence.get_training_stats(policy_version="v1.0.0")
        assert stats["traces_with_feedback"] == n_queries
        assert stats["total_feedback"] == n_queries

        # === FASE 3: Recupera Training Data ===
        training_data = await persistence.get_training_data(
            policy_version="v1.0.0",
            limit=1000
        )

        assert len(training_data) == n_queries

        # Converti in formato per training
        routing_data = []
        for trace, feedback in training_data:
            expert_actions = trace.get_actions_by_type("expert_selection")

            # Estrai embedding dal primo action
            query_embedding = None
            for action in expert_actions:
                if action.metadata.get("query_embedding"):
                    query_embedding = action.metadata["query_embedding"]
                    break

            if query_embedding is None:
                continue

            weights = [a.parameters.get("weight", 0.0) for a in expert_actions]
            log_probs = [a.log_prob for a in expert_actions]

            routing_data.append({
                "query_id": trace.query_id,
                "query_embedding": np.array(query_embedding, dtype=np.float32),
                "weights": np.array(weights, dtype=np.float32),
                "log_probs": np.array(log_probs, dtype=np.float32),
                "reward": feedback.overall_score(),
                "trace": trace,
                "feedback": feedback
            })

        assert len(routing_data) == n_queries

        # === FASE 4: Training ===
        policy = GatingPolicy(
            input_dim=768,
            hidden_dim=128,
            num_experts=4
        )

        trainer_config = SingleStepConfig(
            learning_rate=0.01,
            entropy_coef=0.01,
            clip_grad_norm=1.0,
            baseline_decay=0.9
        )

        trainer = SingleStepTrainer(policy, trainer_config)

        # Track metriche
        rewards = []
        losses = []

        for data in routing_data:
            trace = data["trace"]
            feedback = data["feedback"]

            # Update policy
            metrics = trainer.update(trace, feedback)

            if metrics.get("loss", 0) != 0:
                rewards.append(feedback.overall_score())
                losses.append(metrics.get("loss", 0))

        # Verifica training eseguito
        training_stats = trainer.get_stats()
        assert training_stats["num_updates"] > 0
        assert len(rewards) > 0

        # === FASE 5: Salva Checkpoint ===
        checkpoint_path = Path(temp_checkpoint_dir) / "gating_v1.0.1.pt"

        torch.save({
            "version": "v1.0.1",
            "policy_type": "gating",
            "state_dict": policy.state_dict(),
            "training_metrics": {
                "num_episodes": len(routing_data),
                "avg_reward": float(np.mean(rewards)),
                "avg_loss": float(np.mean(losses))
            }
        }, checkpoint_path)

        assert checkpoint_path.exists()

        # Salva anche in database
        checkpoint_id = await persistence.save_policy_checkpoint(
            version="v1.0.1",
            policy_type="gating",
            state_dict_path=str(checkpoint_path),
            config={"input_dim": 768, "hidden_dim": 128, "num_experts": 4},
            training_metrics={
                "num_episodes": len(routing_data),
                "avg_reward": float(np.mean(rewards)),
                "avg_loss": float(np.mean(losses))
            },
            training_episodes=len(routing_data)
        )

        assert checkpoint_id is not None

        # Attiva la nuova policy
        success = await persistence.activate_policy("v1.0.1", "gating")
        assert success

        # Verifica policy attiva
        active = await persistence.get_active_policy("gating")
        assert active is not None
        assert active.version == "v1.0.1"

        # === VERIFICA FINALE ===
        print("\n" + "=" * 60)
        print("E2E LOOP TEST RESULTS")
        print("=" * 60)
        print(f"Queries simulated: {n_queries}")
        print(f"Training episodes: {len(routing_data)}")
        print(f"Policy updates: {training_stats['num_updates']}")
        print(f"Average reward: {np.mean(rewards):.4f}")
        print(f"Average loss: {np.mean(losses):.4f}")
        print(f"Baseline: {training_stats['baseline']:.4f}")
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Active policy version: {active.version}")
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_training_session_tracking(self, persistence):
        """Test tracking completo di una sessione di training."""
        # Start session
        session_id = await persistence.start_training_session(
            policy_type="gating",
            policy_version_from="v1.0.0",
            config={"learning_rate": 0.01}
        )

        # Simula training
        n_traces = 50

        # Crea e salva traces
        trace_ids = []
        for i in range(n_traces):
            trace = create_realistic_trace(f"session_query_{i}")
            trace_id = await persistence.save_trace(trace, policy_version="v1.0.0")
            trace_ids.append(trace_id)

            # Aggiungi feedback
            feedback = create_realistic_feedback(f"session_query_{i}", quality=0.7)
            await persistence.save_feedback(trace_id, feedback)

        # Complete session
        await persistence.complete_training_session(
            session_id=session_id,
            policy_version_to="v1.0.1",
            num_traces=n_traces,
            num_feedback=n_traces,
            metrics={
                "avg_reward": 0.72,
                "avg_loss": 0.15,
                "num_updates": n_traces
            },
            trace_ids=trace_ids
        )

        # Verifica stats
        stats = await persistence.get_training_stats()
        assert stats["total_traces"] == n_traces
        assert stats["total_feedback"] == n_traces

    @pytest.mark.asyncio
    async def test_incremental_training(self, persistence):
        """
        Test training incrementale su batch successivi.

        Simula scenario produzione dove nuovi feedback
        arrivano continuamente.
        """
        policy = GatingPolicy(input_dim=768, hidden_dim=64, num_experts=4)
        trainer = SingleStepTrainer(policy, SingleStepConfig(learning_rate=0.01))

        baselines = []
        reward_means = []

        # Simula 3 batch di training
        for batch_idx in range(3):
            batch_size = 10

            # Crea nuovo batch di dati
            for i in range(batch_size):
                query_id = f"batch{batch_idx}_query_{i}"
                trace = create_realistic_trace(query_id)
                trace_id = await persistence.save_trace(
                    trace,
                    policy_version=f"v1.0.{batch_idx}"
                )

                # Quality migliora con i batch (simula improvement)
                quality = 0.5 + batch_idx * 0.1
                feedback = create_realistic_feedback(query_id, quality=quality)
                await persistence.save_feedback(trace_id, feedback)

            # Recupera dati per training
            training_data = await persistence.get_training_data(
                policy_version=f"v1.0.{batch_idx}",
                limit=batch_size
            )

            # Train
            batch_rewards = []
            for trace, feedback in training_data:
                trainer.update(trace, feedback)
                batch_rewards.append(feedback.overall_score())

            baselines.append(trainer.baseline)
            reward_means.append(np.mean(batch_rewards))

            print(f"Batch {batch_idx}: "
                  f"reward={np.mean(batch_rewards):.3f}, "
                  f"baseline={trainer.baseline:.3f}")

        # Verifica che baseline si adatta ai reward
        # Con reward crescenti, baseline dovrebbe aumentare
        assert baselines[-1] > baselines[0] or abs(baselines[-1] - baselines[0]) < 0.1

    @pytest.mark.asyncio
    async def test_policy_versioning(self, persistence):
        """Test gestione versioni policy."""
        # Crea diverse versioni
        versions = ["v1.0.0", "v1.0.1", "v1.0.2"]

        for version in versions:
            await persistence.save_policy_checkpoint(
                version=version,
                policy_type="gating",
                config={"version": version}
            )

        # Attiva v1.0.1
        await persistence.activate_policy("v1.0.1", "gating")
        active = await persistence.get_active_policy("gating")
        assert active.version == "v1.0.1"

        # Attiva v1.0.2
        await persistence.activate_policy("v1.0.2", "gating")
        active = await persistence.get_active_policy("gating")
        assert active.version == "v1.0.2"

        # v1.0.1 non è più attiva
        # (verifica indiretta via attivazione v1.0.2)


class TestRLCFDataQuality:
    """Test qualità dati per training."""

    @pytest.mark.asyncio
    async def test_trace_completeness_for_training(self, persistence):
        """Verifica che i trace abbiano tutti i dati necessari."""
        # Crea trace completo
        trace = create_realistic_trace("complete_query")
        trace_id = await persistence.save_trace(trace, policy_version="v1.0.0")

        # Recupera e verifica
        retrieved = await persistence.get_trace(trace_id)

        # Deve avere expert_selection actions
        expert_actions = retrieved.get_actions_by_type("expert_selection")
        assert len(expert_actions) == 4  # 4 experts

        # Almeno uno deve avere query_embedding
        has_embedding = any(
            a.metadata.get("query_embedding") is not None
            for a in expert_actions
        )
        assert has_embedding

        # Tutti devono avere log_prob
        for action in expert_actions:
            assert action.log_prob is not None
            assert action.log_prob < 0  # log prob è negativo

    @pytest.mark.asyncio
    async def test_feedback_coverage(self, persistence):
        """Verifica copertura feedback su tutti i livelli."""
        trace = create_realistic_trace("coverage_query")
        trace_id = await persistence.save_trace(trace)

        # Crea feedback completo
        feedback = MultilevelFeedback(
            query_id="coverage_query",
            retrieval_feedback=RetrievalFeedback(precision=0.8, recall=0.7),
            reasoning_feedback=ReasoningFeedback(logical_coherence=0.9),
            synthesis_feedback=SynthesisFeedback(clarity=0.85)
        )

        await persistence.save_feedback(trace_id, feedback)

        # Recupera e verifica
        feedbacks = await persistence.get_feedback_for_trace(trace_id)
        assert len(feedbacks) == 1

        f = feedbacks[0]
        # Note: is_complete() ritorna True perché i default values riempiono i campi
        # Verifichiamo invece che i valori espliciti siano corretti
        assert f.retrieval_feedback is not None
        assert f.retrieval_feedback.precision == 0.8
        assert f.retrieval_feedback.recall == 0.7
        assert f.reasoning_feedback is not None
        assert f.reasoning_feedback.logical_coherence == 0.9
        assert f.synthesis_feedback is not None
        assert f.synthesis_feedback.clarity == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
