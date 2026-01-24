"""
Test integrazione GatingPolicy con MultiExpertOrchestrator.

Verifica che:
1. Orchestrator accetti GatingPolicy come parametro
2. Policy venga usata per routing invece di ExpertRouter
3. ExecutionTrace venga generato correttamente
4. Trace contenga azioni di expert_selection con log_prob
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from merlt.experts.orchestrator import MultiExpertOrchestrator
from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig, SynthesisMode
from merlt.rlcf.policy_gradient import GatingPolicy
from merlt.rlcf.execution_trace import ExecutionTrace


@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService che ritorna embedding fake."""
    service = MagicMock()

    async def mock_encode(query: str):
        # Ritorna embedding 768-dim deterministico
        np.random.seed(42)
        return np.random.randn(768).astype(np.float32)

    service.encode_query_async = mock_encode
    return service


@pytest.fixture
def mock_synthesizer():
    """Mock AdaptiveSynthesizer."""
    synthesizer = MagicMock(spec=AdaptiveSynthesizer)

    # Mock config.mode for orchestrator initialization
    mock_config = MagicMock()
    mock_config.mode = SynthesisMode.CONVERGENT
    synthesizer.config = mock_config

    # Mock synthesize method to return proper SynthesisResult
    mock_result = MagicMock()
    mock_result.synthesis = "Mock synthesis"
    mock_result.mode = SynthesisMode.CONVERGENT
    mock_result.expert_contributions = {}
    mock_result.combined_sources = []
    mock_result.confidence = 0.85
    mock_result.disagreement_analysis = None
    mock_result.alternatives = []
    mock_result.explanation = None
    mock_result.trace_id = "test_trace"
    mock_result.execution_time_ms = 100.0
    synthesizer.synthesize = AsyncMock(return_value=mock_result)
    return synthesizer


@pytest.fixture
def gating_policy():
    """Crea GatingPolicy per test."""
    # Policy con 4 expert (literal, systemic, principles, precedent)
    policy = GatingPolicy(
        input_dim=768,
        hidden_dim=128,
        num_experts=4,
        device="cpu"
    )
    return policy


@pytest.mark.asyncio
async def test_orchestrator_with_gating_policy(mock_embedding_service, gating_policy, mock_synthesizer):
    """Test che orchestrator usi GatingPolicy per routing."""

    # Setup orchestrator con GatingPolicy
    orchestrator = MultiExpertOrchestrator(
        synthesizer=mock_synthesizer,
        gating_policy=gating_policy,
        embedding_service=mock_embedding_service,
        ai_service=None  # Mock per ora
    )

    # Verifica inizializzazione
    assert orchestrator.gating_policy is not None
    assert orchestrator.embedding_service is not None

    # Process query con return_trace=True
    query = "Cos'è la legittima difesa?"

    # Mock degli expert per evitare chiamate reali
    for expert_type in orchestrator._experts.keys():
        expert = orchestrator._experts[expert_type]
        mock_response = MagicMock()
        mock_response.expert_type = expert_type
        mock_response.interpretation = f"Mock interpretation from {expert_type}"
        mock_response.confidence = 0.8
        mock_response.limitations = ""
        mock_response.trace_id = "test_trace"
        mock_response.legal_basis = []
        expert.analyze = AsyncMock(return_value=mock_response)

    # Process with trace
    result = await orchestrator.process(query, return_trace=True)

    # Verifica return type
    assert isinstance(result, tuple)
    assert len(result) == 2

    response, trace = result

    # Verifica trace
    assert isinstance(trace, ExecutionTrace)
    assert trace.query_id is not None
    assert trace.num_actions > 0

    # Verifica azioni di expert_selection
    expert_selections = trace.get_actions_by_type("expert_selection")
    assert len(expert_selections) == 4  # 4 expert

    # Verifica log_prob
    for action in expert_selections:
        assert "log_prob" in action.to_dict()
        assert isinstance(action.log_prob, float)
        assert action.parameters["weight"] >= 0.0
        assert action.parameters["weight"] <= 1.0

    # Verifica somma pesi = 1.0 (softmax)
    total_weight = sum(a.parameters["weight"] for a in expert_selections)
    assert abs(total_weight - 1.0) < 0.01


@pytest.mark.asyncio
async def test_orchestrator_backward_compatibility(mock_synthesizer):
    """Test che orchestrator senza GatingPolicy usi routing tradizionale."""

    # Setup senza GatingPolicy
    orchestrator = MultiExpertOrchestrator(
        synthesizer=mock_synthesizer,
        ai_service=None
    )

    # Verifica che usi router tradizionale (no gating_policy)
    assert orchestrator.gating_policy is None
    assert orchestrator.embedding_service is None

    # Mock expert
    for expert_type in orchestrator._experts.keys():
        expert = orchestrator._experts[expert_type]
        mock_response = MagicMock()
        mock_response.expert_type = expert_type
        mock_response.interpretation = f"Mock interpretation from {expert_type}"
        mock_response.confidence = 0.8
        mock_response.limitations = ""
        mock_response.trace_id = "test_trace"
        mock_response.legal_basis = []
        expert.analyze = AsyncMock(return_value=mock_response)

    # Process senza return_trace (backward compatibility)
    response = await orchestrator.process("Test query")

    # Verifica che non ritorna trace (backward compatibility)
    assert not isinstance(response, tuple)


@pytest.mark.asyncio
async def test_apply_gating_policy_method(mock_embedding_service, gating_policy, mock_synthesizer):
    """Test diretto del metodo _apply_gating_policy."""

    from merlt.experts.base import ExpertContext

    orchestrator = MultiExpertOrchestrator(
        synthesizer=mock_synthesizer,
        gating_policy=gating_policy,
        embedding_service=mock_embedding_service
    )

    # Context mock
    context = ExpertContext(
        query_text="Test query",
        entities={},
        retrieved_chunks=[],
        trace_id="test_trace"
    )

    # Trace per registrare azioni
    trace = ExecutionTrace(query_id="test_trace")

    # Apply policy
    weights = await orchestrator._apply_gating_policy(context, trace)

    # Verifica weights
    assert isinstance(weights, dict)
    assert len(weights) == 4  # 4 expert

    expert_types = ["literal", "systemic", "principles", "precedent"]
    for expert_type in expert_types:
        assert expert_type in weights
        assert 0.0 <= weights[expert_type] <= 1.0

    # Verifica somma = 1.0 (softmax)
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01

    # Verifica trace
    assert trace.num_actions == 4
    expert_selections = trace.get_actions_by_type("expert_selection")
    assert len(expert_selections) == 4

    # Verifica metadata
    for action in expert_selections:
        assert action.metadata["source"] == "gating_policy"
        assert "query_embedding_dim" in action.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
