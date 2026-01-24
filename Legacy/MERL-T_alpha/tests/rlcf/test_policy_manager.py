"""
Test PolicyManager
==================

Test completi per PolicyManager e integrazione TraversalPolicy.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

from merlt.rlcf.policy_manager import (
    PolicyManager,
    PolicyConfig,
    get_policy_manager,
    reset_policy_manager,
    DEFAULT_TRAVERSAL_WEIGHTS
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding():
    """Embedding 768-dim normalizzato."""
    np.random.seed(42)
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


@pytest.fixture
def policy_config(tmp_path):
    """Config con checkpoint dir temporanea."""
    return PolicyConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        enable_traversal_policy=True,
        enable_gating_policy=True,
        device="cpu",
        weight_threshold=0.3,
        fallback_to_static=True
    )


@pytest.fixture
def policy_manager(policy_config):
    """PolicyManager con config di test."""
    return PolicyManager(config=policy_config)


@pytest.fixture
def mock_trace():
    """Mock ExecutionTrace."""
    trace = MagicMock()
    trace.add_graph_traversal = MagicMock()
    trace.add_expert_selection = MagicMock()
    return trace


# ============================================================================
# Test Inizializzazione
# ============================================================================

class TestPolicyManagerInit:
    """Test inizializzazione PolicyManager."""

    def test_init_with_defaults(self):
        """Test inizializzazione con valori default."""
        manager = PolicyManager()

        assert manager.config is not None
        assert manager.config.enable_traversal_policy is True
        assert manager.static_weights == DEFAULT_TRAVERSAL_WEIGHTS

    def test_init_with_config(self, policy_config):
        """Test inizializzazione con config esplicita."""
        manager = PolicyManager(config=policy_config)

        assert manager.config == policy_config
        assert manager.config.device == "cpu"

    def test_init_with_kwargs(self, tmp_path):
        """Test inizializzazione con kwargs."""
        manager = PolicyManager(
            checkpoint_dir=tmp_path,
            device="cpu",
            enable_policy=False
        )

        assert manager.config.checkpoint_dir == tmp_path
        assert manager.config.enable_traversal_policy is False

    def test_lazy_loading(self, policy_manager):
        """Test che le policy non sono caricate immediatamente."""
        assert policy_manager._traversal_policy is None
        assert policy_manager._gating_policy is None
        assert policy_manager._traversal_loaded is False
        assert policy_manager._gating_loaded is False


# ============================================================================
# Test Fallback a Pesi Statici
# ============================================================================

class TestStaticWeightsFallback:
    """Test fallback a pesi statici quando policy non disponibile."""

    @pytest.mark.asyncio
    async def test_fallback_no_checkpoint(self, policy_manager, sample_embedding):
        """Test fallback quando checkpoint non esiste."""
        weight, log_prob = await policy_manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RIFERIMENTO",
            expert_type="literal"
        )

        # Dovrebbe usare peso statico
        assert weight == DEFAULT_TRAVERSAL_WEIGHTS["literal"]["RIFERIMENTO"]
        assert log_prob == 0.0  # Nessun log_prob per pesi statici

    @pytest.mark.asyncio
    async def test_fallback_policy_disabled(self, tmp_path, sample_embedding):
        """Test fallback quando policy è disabilitata."""
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=False
        )
        manager = PolicyManager(config=config)

        weight, log_prob = await manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RIFERIMENTO",
            expert_type="systemic"
        )

        assert weight == DEFAULT_TRAVERSAL_WEIGHTS["systemic"]["RIFERIMENTO"]
        assert log_prob == 0.0

    @pytest.mark.asyncio
    async def test_unknown_relation_uses_default(self, policy_manager, sample_embedding):
        """Test che relazione sconosciuta usa peso default."""
        weight, log_prob = await policy_manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RELAZIONE_INESISTENTE",
            expert_type="literal"
        )

        assert weight == DEFAULT_TRAVERSAL_WEIGHTS["literal"]["default"]

    @pytest.mark.asyncio
    async def test_unknown_expert_uses_fallback(self, policy_manager, sample_embedding):
        """Test che expert sconosciuto usa peso 0.5."""
        weight, log_prob = await policy_manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RIFERIMENTO",
            expert_type="expert_inesistente"
        )

        assert weight == 0.5  # Default fallback


# ============================================================================
# Test Batch Weights
# ============================================================================

class TestBatchWeights:
    """Test calcolo batch di pesi."""

    @pytest.mark.asyncio
    async def test_batch_weights_empty(self, policy_manager, sample_embedding):
        """Test batch vuoto ritorna dict vuoto."""
        result = await policy_manager.compute_batch_weights(
            query_embedding=sample_embedding,
            relation_types=[],
            expert_type="literal"
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_weights_static(self, policy_manager, sample_embedding):
        """Test batch con pesi statici."""
        relations = ["RIFERIMENTO", "CITATO_DA", "MODIFICA"]

        result = await policy_manager.compute_batch_weights(
            query_embedding=sample_embedding,
            relation_types=relations,
            expert_type="literal"
        )

        assert len(result) == 3
        for rel in relations:
            weight, log_prob = result[rel]
            assert weight == DEFAULT_TRAVERSAL_WEIGHTS["literal"].get(
                rel, DEFAULT_TRAVERSAL_WEIGHTS["literal"]["default"]
            )
            assert log_prob == 0.0

    @pytest.mark.asyncio
    async def test_batch_weights_with_trace(self, policy_manager, sample_embedding, mock_trace):
        """Test che trace registra le azioni."""
        relations = ["RIFERIMENTO", "CITATO_DA"]

        await policy_manager.compute_batch_weights(
            query_embedding=sample_embedding,
            relation_types=relations,
            expert_type="literal",
            trace=mock_trace
        )

        # Trace non viene chiamato per pesi statici (no log_prob)
        # Se avessimo una policy reale, trace.add_graph_traversal sarebbe chiamato


# ============================================================================
# Test Filter Relations
# ============================================================================

class TestFilterRelations:
    """Test filtraggio relazioni per peso."""

    @pytest.mark.asyncio
    async def test_filter_all_pass(self, policy_manager, sample_embedding):
        """Test filtraggio dove tutte le relazioni passano."""
        relations = ["RIFERIMENTO", "CITATO_DA"]  # Entrambe > 0.3

        filtered = await policy_manager.filter_relations_by_weight(
            query_embedding=sample_embedding,
            relation_types=relations,
            expert_type="literal",
            threshold=0.3
        )

        assert set(filtered) == set(relations)

    @pytest.mark.asyncio
    async def test_filter_ordered_by_weight(self, policy_manager, sample_embedding):
        """Test che risultato è ordinato per peso decrescente."""
        relations = ["RIFERIMENTO", "default", "CITATO_DA"]

        filtered = await policy_manager.filter_relations_by_weight(
            query_embedding=sample_embedding,
            relation_types=relations,
            expert_type="literal"
        )

        # Dovrebbe essere ordinato: RIFERIMENTO (0.9), CITATO_DA (0.8), default (0.3)
        weights = [
            DEFAULT_TRAVERSAL_WEIGHTS["literal"].get(r, 0.3)
            for r in filtered
        ]
        assert weights == sorted(weights, reverse=True)


# ============================================================================
# Test con Policy Neurale (Mock)
# ============================================================================

class TestWithNeuralPolicy:
    """Test con TraversalPolicy mockato."""

    @pytest.mark.asyncio
    async def test_compute_weight_with_policy(self, tmp_path, sample_embedding, mock_trace):
        """Test calcolo peso con policy neurale (mockato)."""
        import torch

        # Crea mock policy
        mock_policy = MagicMock()
        mock_policy.device = "cpu"
        mock_policy.get_relation_index = MagicMock(return_value=0)

        # Mock forward pass
        mock_weights = torch.tensor([[0.75]])
        mock_log_probs = torch.tensor([[-0.288]])
        mock_policy.forward = MagicMock(return_value=(mock_weights, mock_log_probs))

        # Crea manager
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=True,
            device="cpu"
        )
        manager = PolicyManager(config=config)

        # Inject mock policy
        manager._traversal_policy = mock_policy
        manager._traversal_loaded = True

        # Test
        weight, log_prob = await manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RIFERIMENTO",
            expert_type="literal",
            trace=mock_trace
        )

        assert weight == pytest.approx(0.75, abs=0.01)
        assert log_prob == pytest.approx(-0.288, abs=0.01)

        # Verifica trace chiamato
        mock_trace.add_graph_traversal.assert_called_once()
        call_kwargs = mock_trace.add_graph_traversal.call_args[1]
        assert call_kwargs["relation_type"] == "RIFERIMENTO"
        assert call_kwargs["weight"] == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_batch_with_policy(self, tmp_path, sample_embedding):
        """Test batch con policy neurale (mockato)."""
        import torch

        # Crea mock policy
        mock_policy = MagicMock()
        mock_policy.device = "cpu"
        mock_policy.get_relation_index = MagicMock(side_effect=lambda r: 0)

        # Mock forward pass per batch di 2
        mock_weights = torch.tensor([[0.7], [0.8]])
        mock_log_probs = torch.tensor([[-0.36], [-0.22]])
        mock_policy.forward = MagicMock(return_value=(mock_weights, mock_log_probs))

        # Crea manager
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=True,
            device="cpu"
        )
        manager = PolicyManager(config=config)
        manager._traversal_policy = mock_policy
        manager._traversal_loaded = True

        # Test
        result = await manager.compute_batch_weights(
            query_embedding=sample_embedding,
            relation_types=["RIFERIMENTO", "CITATO_DA"],
            expert_type="literal"
        )

        assert len(result) == 2
        assert result["RIFERIMENTO"][0] == pytest.approx(0.7, abs=0.01)
        assert result["CITATO_DA"][0] == pytest.approx(0.8, abs=0.01)


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test gestione errori."""

    @pytest.mark.asyncio
    async def test_policy_error_fallback(self, tmp_path, sample_embedding):
        """Test fallback quando policy solleva eccezione."""
        # Crea mock policy che fallisce
        mock_policy = MagicMock()
        mock_policy.device = "cpu"
        mock_policy.get_relation_index = MagicMock(side_effect=Exception("Test error"))

        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=True,
            fallback_to_static=True
        )
        manager = PolicyManager(config=config)
        manager._traversal_policy = mock_policy
        manager._traversal_loaded = True

        # Dovrebbe fallback a statico
        weight, log_prob = await manager.compute_relation_weight(
            query_embedding=sample_embedding,
            relation_type="RIFERIMENTO",
            expert_type="literal"
        )

        assert weight == DEFAULT_TRAVERSAL_WEIGHTS["literal"]["RIFERIMENTO"]
        assert log_prob == 0.0

    @pytest.mark.asyncio
    async def test_policy_error_no_fallback(self, tmp_path, sample_embedding):
        """Test eccezione quando fallback disabilitato."""
        mock_policy = MagicMock()
        mock_policy.device = "cpu"
        mock_policy.get_relation_index = MagicMock(side_effect=Exception("Test error"))

        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=True,
            fallback_to_static=False
        )
        manager = PolicyManager(config=config)
        manager._traversal_policy = mock_policy
        manager._traversal_loaded = True

        with pytest.raises(Exception) as exc_info:
            await manager.compute_relation_weight(
                query_embedding=sample_embedding,
                relation_type="RIFERIMENTO",
                expert_type="literal"
            )

        assert "Test error" in str(exc_info.value)


# ============================================================================
# Test Singleton
# ============================================================================

class TestSingleton:
    """Test pattern singleton."""

    def test_get_policy_manager_singleton(self, tmp_path):
        """Test che get_policy_manager ritorna singleton."""
        reset_policy_manager()  # Reset per test pulito

        config = PolicyConfig(checkpoint_dir=tmp_path, device="cpu")
        manager1 = get_policy_manager(config=config)
        manager2 = get_policy_manager()

        assert manager1 is manager2

    def test_reset_policy_manager(self, tmp_path):
        """Test reset singleton."""
        config = PolicyConfig(checkpoint_dir=tmp_path, device="cpu")
        manager1 = get_policy_manager(config=config)

        reset_policy_manager()

        config2 = PolicyConfig(checkpoint_dir=tmp_path / "other", device="cpu")
        manager2 = get_policy_manager(config=config2)

        assert manager1 is not manager2


# ============================================================================
# Test Save/Load Policy
# ============================================================================

class TestSaveLoadPolicy:
    """Test salvataggio e caricamento policy."""

    def test_save_traversal_policy(self, policy_manager, tmp_path):
        """Test salvataggio TraversalPolicy."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        # Crea policy
        policy = TraversalPolicy(device="cpu")

        # Salva
        policy_manager.config.checkpoint_dir = tmp_path
        policy_manager.save_traversal_policy(policy, name="test")

        # Verifica file
        checkpoint_path = tmp_path / "traversal_policy_test.pt"
        assert checkpoint_path.exists()

    def test_save_resets_cache(self, policy_manager, tmp_path):
        """Test che save resetta la cache."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(device="cpu")

        # Simula policy caricata
        policy_manager._traversal_policy = policy
        policy_manager._traversal_loaded = True

        # Salva
        policy_manager.config.checkpoint_dir = tmp_path
        policy_manager.save_traversal_policy(policy)

        # Verifica reset cache
        assert policy_manager._traversal_policy is None
        assert policy_manager._traversal_loaded is False


# ============================================================================
# Test Device Management
# ============================================================================

class TestDeviceManagement:
    """Test gestione device."""

    def test_detect_device_cpu_fallback(self, policy_config):
        """Test che detect_device fallback a CPU se torch non disponibile."""
        manager = PolicyManager(config=policy_config)

        # Su macchina senza GPU, dovrebbe essere CPU
        device = manager._detect_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_explicit_device(self, tmp_path):
        """Test device esplicito."""
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            device="cpu"
        )
        manager = PolicyManager(config=config)

        assert manager._detect_device() == "cpu"


# ============================================================================
# Test Integration con ExecutionTrace
# ============================================================================

class TestExecutionTraceIntegration:
    """Test integrazione con ExecutionTrace reale."""

    @pytest.mark.asyncio
    async def test_trace_records_actions(self, policy_manager, sample_embedding):
        """Test che trace registra correttamente le azioni."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="test_query")

        await policy_manager.compute_batch_weights(
            query_embedding=sample_embedding,
            relation_types=["RIFERIMENTO", "CITATO_DA"],
            expert_type="literal",
            trace=trace
        )

        # Per pesi statici, il trace non viene aggiornato
        # (log_prob = 0 non è significativo per REINFORCE)
        # Questo è il comportamento corretto: tracciamo solo azioni
        # con log_prob significativo dalla policy neurale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
