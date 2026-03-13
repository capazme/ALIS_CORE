"""
Test singleton thread safety (STORY-12-2)
=========================================

Tests for:
- get_policy_manager() thread-safe singleton
- reset_policy_manager() thread-safe reset
- PolicyManager._load_gating_policy() double-checked locking
- PolicyManager._load_traversal_policy() double-checked locking
- get_orchestrator() async-safe singleton

Example:
    pytest tests/rlcf/test_singleton_thread_safety.py -v
"""

import asyncio
import threading
import time
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from merlt.rlcf.policy_manager import (
    PolicyManager,
    PolicyConfig,
    get_policy_manager,
    reset_policy_manager,
)
from merlt.rlcf.policy_gradient import GatingPolicy


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clean_singleton():
    """Reset singleton before and after each test."""
    reset_policy_manager()
    yield
    reset_policy_manager()


@pytest.fixture
def pm(tmp_path):
    """PolicyManager with tmp checkpoint dir."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    config = PolicyConfig(checkpoint_dir=d)
    return PolicyManager(config=config)


# =============================================================================
# TEST get_policy_manager() THREAD SAFETY
# =============================================================================


class TestGetPolicyManagerThreadSafety:
    """get_policy_manager() returns same instance across threads."""

    def test_concurrent_calls_return_same_instance(self):
        """Multiple threads calling get_policy_manager() get the same object."""
        results = []
        errors = []

        def get_pm():
            try:
                pm = get_policy_manager()
                results.append(id(pm))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_pm) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert len(set(results)) == 1, f"Got {len(set(results))} distinct instances"

    def test_threadpool_concurrent_calls(self):
        """ThreadPoolExecutor concurrent calls return same instance."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(get_policy_manager) for _ in range(20)]
            instances = [f.result() for f in futures]

        ids = {id(inst) for inst in instances}
        assert len(ids) == 1

    def test_reset_clears_instance(self):
        """reset_policy_manager() clears the singleton."""
        pm1 = get_policy_manager()
        reset_policy_manager()
        pm2 = get_policy_manager()
        assert id(pm1) != id(pm2)

    def test_reset_is_thread_safe(self):
        """Concurrent reset + get doesn't crash."""
        errors = []

        def reset_and_get():
            try:
                reset_policy_manager()
                get_policy_manager()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reset_and_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"


# =============================================================================
# TEST PolicyManager LAZY LOADING THREAD SAFETY
# =============================================================================


class TestLazyLoadingThreadSafety:
    """_load_gating_policy and _load_traversal_policy are thread-safe."""

    def test_gating_policy_double_checked_locking(self, pm):
        """Concurrent _load_gating_policy() calls don't race."""
        results = []
        errors = []

        def load():
            try:
                result = pm._load_gating_policy()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert all(r is None for r in results)
        assert pm._gating_loaded is True

    def test_traversal_policy_double_checked_locking(self, pm):
        """Concurrent _load_traversal_policy() calls don't race."""
        results = []
        errors = []

        def load():
            try:
                result = pm._load_traversal_policy()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert all(r is None for r in results)
        assert pm._traversal_loaded is True

    def test_load_lock_exists(self, pm):
        """PolicyManager has _load_lock attribute."""
        assert hasattr(pm, '_load_lock')
        assert isinstance(pm._load_lock, type(threading.Lock()))

    def test_concurrent_load_with_checkpoint_returns_same_policy(self, pm):
        """Concurrent loads with a real checkpoint all return the same instance."""
        # Save a valid gating checkpoint
        policy = GatingPolicy(input_dim=1024, hidden_dim=256, num_experts=4)
        ckpt = {
            "input_dim": 1024,
            "hidden_dim": 256,
            "num_experts": 4,
            "mlp_state_dict": policy.mlp.state_dict(),
        }
        ckpt_path = pm.config.checkpoint_dir / "gating_policy_latest.pt"
        torch.save(ckpt, ckpt_path)

        results = []
        errors = []
        barrier = threading.Barrier(8)

        def load():
            try:
                barrier.wait()  # Maximize concurrency
                result = pm._load_gating_policy()
                results.append(id(result))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        # All threads must get the same policy object (not None)
        assert all(r is not None for r in results)
        assert len(set(results)) == 1, f"Got {len(set(results))} distinct instances"

    def test_reset_policies_is_thread_safe(self, pm):
        """reset_policies() acquires _load_lock."""
        errors = []

        def load_and_reset():
            try:
                pm._load_gating_policy()
                pm.reset_policies()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_and_reset) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"


# =============================================================================
# TEST get_orchestrator() ASYNC SAFETY
# =============================================================================


class TestGetOrchestratorAsyncSafety:
    """get_orchestrator() uses asyncio.Lock for async safety."""

    @pytest.mark.asyncio
    async def test_orchestrator_lock_is_asyncio_lock(self):
        """Module-level _orchestrator_lock is an asyncio.Lock."""
        from merlt.rlcf.orchestrator import _orchestrator_lock
        assert isinstance(_orchestrator_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_get_orchestrator(self):
        """Concurrent async calls to get_orchestrator() return same instance."""
        from merlt.rlcf.orchestrator import get_orchestrator
        import merlt.rlcf.orchestrator as orch_module

        # Reset state
        orch_module._orchestrator_instance = None

        mock_session = AsyncMock()

        try:
            tasks = [get_orchestrator(db_session=mock_session) for _ in range(10)]
            instances = await asyncio.gather(*tasks)

            ids = {id(inst) for inst in instances}
            assert len(ids) == 1
        finally:
            orch_module._orchestrator_instance = None
