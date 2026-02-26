"""
Test RLCF Full E2E Pipeline.

Covers the complete end-to-end RLCF training loop:
1. Create execution traces with 1024-dim embeddings
2. Submit multilevel feedback (retrieval, reasoning, synthesis)
3. Persist to DB via RLCFPersistence
4. Retrieve training data, verify completeness
5. Train with PolicyGradientTrainer.update_from_feedback() (real REINFORCE backprop)
6. Assert policy weights actually changed
7. Save checkpoint to tmp_path
8. Save + activate policy version in DB

Follows patterns from test_rlcf_loop_e2e.py.
"""

import os
import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

pytestmark = pytest.mark.integration

os.environ["RLCF_ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from merlt.rlcf.persistence import RLCFPersistence, create_persistence
from merlt.rlcf.execution_trace import ExecutionTrace
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
)
from merlt.rlcf.policy_gradient import GatingPolicy, PolicyGradientTrainer, TrainerConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest_asyncio.fixture
async def persistence():
    """RLCFPersistence with SQLite in-memory."""
    p = await create_persistence("sqlite+aiosqlite:///:memory:")
    yield p


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# =============================================================================
# HELPERS
# =============================================================================


def create_trace_1024(query_id: str) -> ExecutionTrace:
    """
    Create a realistic execution trace with 1024-dim embeddings.

    Uses E5-large compatible dimensions (1024) instead of 768.
    """
    query_embedding = np.random.randn(1024).tolist()

    trace = ExecutionTrace(query_id=query_id)

    experts = ["literal", "systemic", "principles", "precedent"]
    weights = np.random.dirichlet(np.ones(4))

    for i, exp in enumerate(experts):
        trace.add_expert_selection(
            expert_type=exp,
            weight=float(weights[i]),
            log_prob=float(np.log(weights[i] + 1e-8)),
            metadata={
                "query_embedding": query_embedding if i == 0 else None,
                "source": "gating_policy",
                "action_index": i,
            },
        )

    trace.add_graph_traversal(
        relation_type="RIFERIMENTO",
        weight=0.8,
        log_prob=-0.223,
        source_node=f"urn:norma:cc:art{np.random.randint(1, 2000)}",
    )

    return trace


def create_full_feedback(
    query_id: str,
    quality: float = 0.7,
) -> MultilevelFeedback:
    """
    Create multilevel feedback with all 3 levels populated.

    Args:
        query_id: Associated query ID
        quality: Base quality [0, 1]
    """
    noise = lambda: np.random.uniform(-0.1, 0.1)
    clamp = lambda v: max(0.0, min(1.0, v))

    return MultilevelFeedback(
        query_id=query_id,
        retrieval_feedback=RetrievalFeedback(
            precision=clamp(quality + noise()),
            recall=clamp(quality + noise()),
            ranking_quality=clamp(quality + noise()),
        ),
        reasoning_feedback=ReasoningFeedback(
            logical_coherence=clamp(quality + noise()),
            legal_soundness=clamp(quality + noise()),
            citation_quality=clamp(quality + noise()),
        ),
        synthesis_feedback=SynthesisFeedback(
            clarity=clamp(quality + noise()),
            completeness=clamp(quality + noise()),
            usefulness=clamp(quality + noise()),
        ),
        user_id="e2e_test_user",
    )


# =============================================================================
# TESTS
# =============================================================================


class TestRLCFFullE2E:
    """Full end-to-end test of the RLCF training loop."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, persistence, temp_checkpoint_dir):
        """
        Complete E2E loop:
        1. Create 10 traces with 1024-dim embeddings
        2. Submit multilevel feedback (all 3 levels)
        3. Persist to DB
        4. Retrieve training data, verify completeness
        5. Train with PolicyGradientTrainer.update_from_feedback()
        6. Assert policy weights actually changed
        7. Save checkpoint
        8. Save + activate policy version in DB
        """
        n_traces = 10
        policy_version = "v_e2e_test"

        # === Step 1: Create and save traces ===
        query_ids = []
        trace_ids = []

        for i in range(n_traces):
            query_id = f"e2e_query_{i:03d}"
            query_ids.append(query_id)

            trace = create_trace_1024(query_id)
            trace_id = await persistence.save_trace(
                trace=trace,
                policy_version=policy_version,
                query_text=f"Domanda giuridica numero {i}",
                expert_type="literal",
            )
            trace_ids.append(trace_id)

        # Verify traces persisted
        stats = await persistence.get_training_stats(policy_version=policy_version)
        assert stats["total_traces"] == n_traces

        # === Step 2: Submit multilevel feedback (all 3 levels) ===
        for i, trace_id in enumerate(trace_ids):
            quality = 0.4 + 0.5 * (i / n_traces)  # increasing quality
            feedback = create_full_feedback(query_ids[i], quality=quality)

            await persistence.save_feedback(
                trace_id=trace_id,
                feedback=feedback,
                user_id="e2e_test_user",
                user_authority=0.8,
            )

        # Verify all feedback saved
        stats = await persistence.get_training_stats(policy_version=policy_version)
        assert stats["traces_with_feedback"] == n_traces
        assert stats["total_feedback"] == n_traces

        # === Step 3 + 4: Retrieve training data and verify completeness ===
        training_data = await persistence.get_training_data(
            policy_version=policy_version,
            limit=1000,
        )

        assert len(training_data) == n_traces

        # Verify each trace has proper structure
        for trace, feedback in training_data:
            expert_actions = trace.get_actions_by_type("expert_selection")
            assert len(expert_actions) == 4, "Should have 4 expert selections"

            # At least one action must have query_embedding
            has_embedding = any(
                a.metadata.get("query_embedding") is not None
                for a in expert_actions
            )
            assert has_embedding, "At least one action must have query_embedding"

            # Feedback must have all 3 levels
            assert feedback.retrieval_feedback is not None
            assert feedback.reasoning_feedback is not None
            assert feedback.synthesis_feedback is not None

            # Overall score should be computable
            score = feedback.overall_score()
            assert 0.0 <= score <= 1.0

        # === Step 5: Train with PolicyGradientTrainer.update_from_feedback() ===
        policy = GatingPolicy(
            input_dim=1024,
            hidden_dim=128,
            num_experts=4,
            device="cpu",
        )

        # Capture initial weights for comparison
        initial_params = {
            name: param.clone().detach()
            for name, param in policy.mlp.named_parameters()
        }

        trainer = PolicyGradientTrainer(
            policy,
            config=TrainerConfig(
                learning_rate=0.01,
                clip_grad_norm=1.0,
                baseline_decay=0.9,
                entropy_coef=0.01,
            ),
        )

        rewards = []
        losses = []

        for trace, feedback in training_data:
            metrics = trainer.update_from_feedback(trace, feedback)
            rewards.append(metrics["reward"])
            if metrics["loss"] != 0:
                losses.append(metrics["loss"])

        assert trainer.num_updates > 0, "Should have performed at least one update"
        assert len(rewards) == n_traces

        # === Step 6: Assert policy weights actually changed ===
        weights_changed = False
        for name, param in policy.mlp.named_parameters():
            if not torch.equal(param, initial_params[name]):
                weights_changed = True
                break

        assert weights_changed, "Policy weights should have changed after training"

        # === Step 7: Save checkpoint to tmp_path ===
        checkpoint_path = Path(temp_checkpoint_dir) / "gating_e2e.pt"
        torch.save(
            {
                "version": "v_e2e_trained",
                "policy_type": "gating",
                "state_dict": policy.state_dict(),
                "training_metrics": {
                    "num_episodes": n_traces,
                    "avg_reward": float(np.mean(rewards)),
                    "avg_loss": float(np.mean(losses)) if losses else 0.0,
                    "num_updates": trainer.num_updates,
                },
            },
            checkpoint_path,
        )

        assert checkpoint_path.exists()

        # Verify checkpoint can be loaded
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["version"] == "v_e2e_trained"
        assert loaded["training_metrics"]["num_episodes"] == n_traces

        # === Step 8: Save + activate policy version in DB ===
        checkpoint_id = await persistence.save_policy_checkpoint(
            version="v_e2e_trained",
            policy_type="gating",
            state_dict_path=str(checkpoint_path),
            config={
                "input_dim": 1024,
                "hidden_dim": 128,
                "num_experts": 4,
            },
            training_metrics={
                "num_episodes": n_traces,
                "avg_reward": float(np.mean(rewards)),
                "num_updates": trainer.num_updates,
            },
            training_episodes=n_traces,
        )

        assert checkpoint_id is not None

        # Activate the trained policy
        success = await persistence.activate_policy("v_e2e_trained", "gating")
        assert success

        # Verify active policy
        active = await persistence.get_active_policy("gating")
        assert active is not None
        assert active.version == "v_e2e_trained"

    @pytest.mark.asyncio
    async def test_feedback_completeness_all_levels(self, persistence):
        """Verify that all 3 feedback levels are round-tripped correctly."""
        trace = create_trace_1024("completeness_test")
        trace_id = await persistence.save_trace(trace)

        feedback = MultilevelFeedback(
            query_id="completeness_test",
            retrieval_feedback=RetrievalFeedback(
                precision=0.85,
                recall=0.78,
                ranking_quality=0.90,
            ),
            reasoning_feedback=ReasoningFeedback(
                logical_coherence=0.92,
                legal_soundness=0.88,
                citation_quality=0.75,
            ),
            synthesis_feedback=SynthesisFeedback(
                clarity=0.95,
                completeness=0.80,
                usefulness=0.87,
            ),
            user_id="test_user",
        )

        await persistence.save_feedback(trace_id, feedback, user_id="test_user")

        # Retrieve and verify
        feedbacks = await persistence.get_feedback_for_trace(trace_id)
        assert len(feedbacks) == 1

        f = feedbacks[0]
        assert f.retrieval_feedback.precision == 0.85
        assert f.retrieval_feedback.recall == 0.78
        assert f.reasoning_feedback.logical_coherence == 0.92
        assert f.reasoning_feedback.legal_soundness == 0.88
        assert f.synthesis_feedback.clarity == 0.95
        assert f.synthesis_feedback.usefulness == 0.87


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
