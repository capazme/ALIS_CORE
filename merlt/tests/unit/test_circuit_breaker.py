"""
[P0] Unit tests for CircuitBreaker state machine.

Tests state transitions:
- closed -> open when failure threshold exceeded
- open rejects calls immediately
- open -> half-open after recovery timeout
- half-open -> closed on success
- half-open -> open on failure
- Thread-safe registry
- State change callbacks
"""

import time

import pytest

from merlt.experts.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


@pytest.fixture
def config():
    return CircuitBreakerConfig(
        failure_threshold=3,
        failure_window_seconds=60.0,
        recovery_timeout_seconds=0.1,  # fast for tests
        half_open_max_calls=1,
        success_threshold=1,
    )


@pytest.fixture
def breaker(config):
    return CircuitBreaker(name="test_expert", config=config)


# --- State transitions ---

class TestClosedToOpen:
    def test_p0_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True

    def test_p0_stays_closed_below_threshold(self, breaker):
        breaker.record_failure(ValueError, ValueError("err"))
        breaker.record_failure(ValueError, ValueError("err"))
        assert breaker.is_closed

    def test_p0_opens_at_threshold(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_p0_open_rejects_calls(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        assert breaker.is_open
        with pytest.raises(CircuitOpenError):
            await breaker._before_call()


class TestOpenToHalfOpen:
    def test_p0_transitions_to_half_open_after_timeout(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        assert breaker.is_open

        time.sleep(0.15)  # exceed recovery_timeout_seconds
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN


class TestHalfOpenTransitions:
    def test_p0_half_open_to_closed_on_success(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        time.sleep(0.15)
        breaker.can_execute()  # triggers transition to HALF_OPEN
        assert breaker.is_half_open

        breaker.record_success()
        assert breaker.is_closed

    def test_p0_half_open_to_open_on_failure(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        time.sleep(0.15)
        breaker.can_execute()
        assert breaker.is_half_open

        breaker.record_failure(RuntimeError, RuntimeError("fail again"))
        assert breaker.is_open


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_p0_success_records_via_context_manager(self, breaker):
        async with breaker:
            pass  # success
        assert breaker._total_successes == 1

    @pytest.mark.asyncio
    async def test_p0_failure_records_via_context_manager(self, breaker):
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("boom")
        assert breaker._total_failures == 1


# --- Reset ---

class TestReset:
    def test_p0_manual_reset_to_closed(self, breaker):
        for _ in range(3):
            breaker.record_failure(ValueError, ValueError("err"))
        assert breaker.is_open
        breaker.reset()
        assert breaker.is_closed
        assert breaker._failures == []


# --- Registry ---

class TestRegistry:
    def test_p0_registry_get_or_create(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("expert_a")
        cb2 = registry.get_or_create("expert_a")
        assert cb1 is cb2

    def test_p0_registry_multiple_breakers(self):
        registry = CircuitBreakerRegistry()
        a = registry.get_or_create("expert_a")
        b = registry.get_or_create("expert_b")
        assert a is not b
        stats = registry.get_all_stats()
        assert "expert_a" in stats
        assert "expert_b" in stats

    def test_p0_registry_open_circuits(self):
        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create(
            "failing",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(ValueError, ValueError("err"))
        assert "failing" in registry.get_open_circuits()

    def test_p0_registry_reset_all(self):
        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create(
            "reset_test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(ValueError, ValueError("err"))
        assert cb.is_open
        registry.reset_all()
        assert cb.is_closed


# --- State callback ---

class TestStateCallback:
    def test_p0_callback_invoked_on_state_change(self):
        transitions = []

        def on_change(name, old, new):
            transitions.append((name, old, new))

        cb = CircuitBreaker(
            name="cb_callback",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=on_change,
        )
        cb.record_failure(ValueError, ValueError("err"))
        assert len(transitions) == 1
        assert transitions[0] == ("cb_callback", CircuitState.CLOSED, CircuitState.OPEN)

    def test_p0_callback_error_does_not_crash(self):
        def bad_callback(name, old, new):
            raise RuntimeError("callback failed")

        cb = CircuitBreaker(
            name="cb_bad",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=bad_callback,
        )
        # Should not raise
        cb.record_failure(ValueError, ValueError("err"))
        assert cb.is_open


# --- Stats ---

class TestStats:
    def test_p0_stats_reflect_state(self, breaker):
        stats = breaker.get_stats()
        assert stats.name == "test_expert"
        assert stats.state == CircuitState.CLOSED
        assert stats.total_failures == 0

        breaker.record_failure(ValueError, ValueError("err"))
        stats = breaker.get_stats()
        assert stats.total_failures == 1
        assert stats.failure_count == 1
