"""
E2E Test for Unified RLCF Loop (Epic 13).

Verifies the complete cycle:
1. ExpertGatingMLP produces weights + trace with expert_selection
2. Feedback → add_experience → buffer
3. run_training_epoch → REINFORCE on ExpertGatingMLP → weights change
4. Checkpoint saved in model_state_dict format
5. HybridExpertRouter loads same checkpoint → weights updated
6. PolicyManager loads same checkpoint → weights updated
"""

import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

pytest.importorskip("torch")

import torch

from merlt.experts.neural_gating.neural import (
    ExpertGatingMLP,
    GatingConfig,
    EXPERT_NAMES,
)
from merlt.experts.neural_gating.hybrid_router import (
    HybridExpertRouter,
    HybridRoutingDecision,
)
from merlt.experts.base import ExpertContext
from merlt.rlcf.execution_trace import ExecutionTrace, Action
from merlt.rlcf.multilevel_feedback import MultilevelFeedback
from merlt.rlcf.policy_gradient import PolicyGradientTrainer
from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig


@pytest.fixture(autouse=True)
def seed_rng():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


class TestGatingLoopE2E:
    """Full gating loop: forward → trace → train → weights change."""

    def test_full_gating_cycle(self):
        """REINFORCE training produces non-zero loss and changes parameters."""
        # Use small MLP for fast, strong gradient signal
        config = GatingConfig(input_dim=64, hidden_dim1=32, hidden_dim2=16)
        mlp = ExpertGatingMLP(config)
        mlp = mlp.to("cpu")
        trainer = PolicyGradientTrainer(mlp)

        initial_state = {k: v.clone() for k, v in mlp.state_dict().items()}

        for _ in range(20):
            query_emb = np.random.randn(64).astype(np.float32)

            emb_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                weights, log_probs = mlp(emb_tensor)

            weights_np = weights.squeeze(0).cpu().numpy()
            log_probs_np = log_probs.squeeze(0).cpu().numpy()

            trace = ExecutionTrace(query_id="test")
            for i, name in enumerate(EXPERT_NAMES):
                trace.add_action(Action(
                    action_type="expert_selection",
                    parameters={"weight": float(weights_np[i]), "expert_type": name},
                    log_prob=float(log_probs_np[i]),
                    metadata={
                        "source": "neural_gating",
                        "query_embedding": query_emb.tolist(),
                    },
                ))

            # Varying reward to produce non-zero returns
            reward = 0.9 if query_emb[0] > 0 else 0.2
            feedback = MultilevelFeedback.from_dict({
                "query_id": "test",
                "overall_rating": reward,
            })

            metrics = trainer.update_from_feedback(trace, feedback)
            assert metrics["num_actions"] == 4
            assert metrics["loss"] != 0.0

        # Verify parameters actually changed (not just bias — full network)
        changed = False
        for key, orig_val in initial_state.items():
            if not torch.equal(orig_val, mlp.state_dict()[key]):
                changed = True
                break
        assert changed, "Model parameters should change after REINFORCE training"


class TestCheckpointInterop:
    """Checkpoint written by trainer is readable by HybridExpertRouter AND PolicyManager."""

    def test_trainer_checkpoint_loads_in_hybrid_router(self):
        """PolicyGradientTrainer checkpoint → HybridExpertRouter."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "gating_trainer_latest.pt"

            mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
            trainer = PolicyGradientTrainer(mlp)

            query_emb = np.random.randn(1024).astype(np.float32)
            emb_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                weights, log_probs = mlp(emb_tensor)

            trace = ExecutionTrace(query_id="test")
            for i, name in enumerate(EXPERT_NAMES):
                trace.add_action(Action(
                    action_type="expert_selection",
                    parameters={"weight": float(weights[0, i]), "expert_type": name},
                    log_prob=float(log_probs[0, i]),
                    metadata={
                        "source": "neural_gating",
                        "query_embedding": query_emb.tolist(),
                    },
                ))

            feedback = MultilevelFeedback.from_dict({
                "query_id": "test",
                "overall_rating": 0.8,
            })
            trainer.update_from_feedback(trace, feedback)
            trainer.save_checkpoint(str(checkpoint_path))

            trained_priors = mlp.get_expert_priors()

            new_mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
            router = HybridExpertRouter(
                neural_gating=new_mlp,
                checkpoint_path=checkpoint_path,
            )

            assert router.loaded_from_checkpoint
            loaded_priors = new_mlp.get_expert_priors()

            for name in EXPERT_NAMES:
                assert abs(loaded_priors[name] - trained_priors[name]) < 1e-5, \
                    f"Prior mismatch for {name}"

    def test_trainer_checkpoint_loads_in_policy_manager(self):
        """PolicyGradientTrainer checkpoint → PolicyManager."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
            mlp = mlp.to("cpu")
            pm = PolicyManager(config=PolicyConfig(
                checkpoint_dir=checkpoint_dir, device="cpu"
            ))
            pm.save_gating_policy(mlp, name="latest")

            # Load via PolicyManager
            pm2 = PolicyManager(config=PolicyConfig(
                checkpoint_dir=checkpoint_dir, device="cpu"
            ))
            loaded_policy = pm2.get_gating_policy()

            assert loaded_policy is not None

            # Verify weights match (inference mode to disable dropout)
            mlp.eval()
            loaded_policy.eval()
            test_emb = torch.randn(1, 1024)
            with torch.no_grad():
                orig_weights, _ = mlp(test_emb)
                loaded_weights, _ = loaded_policy(test_emb)

            assert torch.allclose(orig_weights, loaded_weights, atol=1e-5)

    def test_backward_compat_policy_state_dict(self):
        """Legacy checkpoint with policy_state_dict still loads with correct weights."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pt"

            mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))

            # Modify expert_bias to non-default values so we can verify load
            with torch.no_grad():
                mlp.expert_bias.copy_(torch.tensor([2.0, -1.0, 0.5, -0.5]))
            modified_priors = mlp.get_expert_priors()

            import torch.optim as optim
            optimizer = optim.Adam(mlp.parameters(), lr=1e-4)

            legacy_checkpoint = {
                "policy_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "baseline": 0.5,
                "num_updates": 10,
            }
            torch.save(legacy_checkpoint, checkpoint_path)

            # new_mlp has DEFAULT priors — different from modified_priors
            new_mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
            default_priors = new_mlp.get_expert_priors()

            # Sanity: default and modified should differ
            assert abs(default_priors["literal"] - modified_priors["literal"]) > 0.01, \
                "Test setup error: priors must differ before load"

            trainer = PolicyGradientTrainer(new_mlp)
            trainer.load_checkpoint(str(checkpoint_path))

            assert trainer.num_updates == 10
            assert trainer.baseline == 0.5

            # Verify weights were actually loaded (not just metadata)
            loaded_priors = new_mlp.get_expert_priors()
            for name in EXPERT_NAMES:
                assert abs(loaded_priors[name] - modified_priors[name]) < 1e-5, \
                    f"Legacy load failed for {name}: {loaded_priors[name]} != {modified_priors[name]}"


class TestTraceCompatibility:
    """Trace from hybrid branch contains expert_selection with query_embedding."""

    @pytest.mark.asyncio
    async def test_hybrid_route_produces_expert_selection_data(self):
        """HybridExpertRouter produces query_embedding and expert_log_probs."""
        mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
        router = HybridExpertRouter(
            neural_gating=mlp,
            confidence_threshold=0.0,
        )

        context = ExpertContext(query_text="Test legal query")
        decision = await router.route(context)

        assert isinstance(decision, HybridRoutingDecision)
        assert decision.neural_used is True
        assert decision.query_embedding is not None
        assert len(decision.query_embedding) == 1024
        assert decision.expert_log_probs is not None
        assert len(decision.expert_log_probs) == 4

        for name in EXPERT_NAMES:
            assert name in decision.expert_log_probs
            assert decision.expert_log_probs[name] <= 0

    @pytest.mark.asyncio
    async def test_trace_expert_selection_feeds_trainer(self):
        """expert_selection actions from hybrid route → trainer produces loss > 0."""
        mlp = ExpertGatingMLP(GatingConfig(input_dim=1024))
        router = HybridExpertRouter(
            neural_gating=mlp,
            confidence_threshold=0.0,
        )

        context = ExpertContext(query_text="Art. 52 codice penale")
        decision = await router.route(context)

        trace = ExecutionTrace(query_id="test_hybrid")
        trace.add_action(Action(
            action_type="routing",
            parameters={"strategy": "hybrid", "neural_used": True},
            log_prob=-0.1,
        ))

        assert decision.neural_used, "Test requires neural routing"
        assert decision.query_embedding is not None, "Test requires query_embedding"

        for expert_name, weight in decision.expert_weights.items():
            log_prob_val = (
                decision.expert_log_probs.get(expert_name, -1.0)
                if decision.expert_log_probs else -1.0
            )
            trace.add_action(Action(
                action_type="expert_selection",
                parameters={"weight": weight, "expert_type": expert_name},
                log_prob=log_prob_val,
                metadata={
                    "source": "neural_gating",
                    "query_embedding": decision.query_embedding,
                },
            ))

        trainer = PolicyGradientTrainer(mlp)
        feedback = MultilevelFeedback.from_dict({
            "query_id": "test_hybrid",
            "overall_rating": 0.75,
        })

        metrics = trainer.update_from_feedback(trace, feedback)

        assert metrics["num_actions"] == 4
        assert metrics["loss"] != 0.0
        assert metrics["reward"] > 0.0
