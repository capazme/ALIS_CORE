"""
Tests for Circuit Breaker.

Tests cover:
- AC1: Expert timeout/failure triggers circuit, pipeline continues, "unavailable" note
- AC2: Repeated failures (>3 in 5 min) opens circuit, requests skip until health check
- AC3: Half-open state after recovery timeout, success closes, failure reopens
- AC4: Circuit breaker events are logged for admin review
"""

import asyncio
import pytest
import time
from typing import List
from unittest.mock import MagicMock

from visualex.experts import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    CircuitState,
    CircuitOpenError,
    EXPERT_CIRCUIT_BREAKERS,
    get_expert_circuit_breaker,
    create_unavailable_response,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default circuit breaker config."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        failure_window_seconds=300.0,
        recovery_timeout_seconds=60.0,
    )


@pytest.fixture
def fast_config():
    """Fast config for testing (short timeouts)."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        failure_window_seconds=10.0,
        recovery_timeout_seconds=0.1,  # 100ms for fast testing
    )


@pytest.fixture
def circuit_breaker(default_config):
    """Create a fresh circuit breaker."""
    return CircuitBreaker(name="test_breaker", config=default_config)


@pytest.fixture
def fast_breaker(fast_config):
    """Create circuit breaker with fast timeouts."""
    return CircuitBreaker(name="fast_breaker", config=fast_config)


@pytest.fixture
def registry():
    """Fresh circuit breaker registry."""
    # Create new instance instead of singleton for isolation
    return CircuitBreakerRegistry()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestCircuitBreakerConfiguration:
    """Tests for circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 3
        assert config.failure_window_seconds == 300.0
        assert config.recovery_timeout_seconds == 60.0
        assert config.half_open_max_calls == 1
        assert config.success_threshold == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=120.0,
        )

        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 120.0

    def test_expert_configs_exist(self):
        """Test that expert-specific configs are defined."""
        assert "literal" in EXPERT_CIRCUIT_BREAKERS
        assert "systemic" in EXPERT_CIRCUIT_BREAKERS
        assert "principles" in EXPERT_CIRCUIT_BREAKERS
        assert "precedent" in EXPERT_CIRCUIT_BREAKERS
        assert "gating" in EXPERT_CIRCUIT_BREAKERS
        assert "llm_provider" in EXPERT_CIRCUIT_BREAKERS


# =============================================================================
# State Transition Tests (AC1, AC2, AC3)
# =============================================================================


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Test that circuit starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed
        assert not circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self, circuit_breaker):
        """Test that successful calls keep circuit closed."""
        async with circuit_breaker:
            pass  # Simulates successful call

        assert circuit_breaker.is_closed
        stats = circuit_breaker.get_stats()
        assert stats.total_successes == 1

    @pytest.mark.asyncio
    async def test_single_failure_keeps_circuit_closed(self, circuit_breaker):
        """Test that single failure doesn't open circuit."""
        with pytest.raises(ValueError):
            async with circuit_breaker:
                raise ValueError("Test error")

        assert circuit_breaker.is_closed
        stats = circuit_breaker.get_stats()
        assert stats.total_failures == 1

    @pytest.mark.asyncio
    async def test_threshold_failures_opens_circuit(self, fast_breaker):
        """Test that exceeding threshold opens circuit (AC2)."""
        # Trigger failures equal to threshold
        for i in range(2):  # fast_breaker has threshold=2
            with pytest.raises(ValueError):
                async with fast_breaker:
                    raise ValueError(f"Error {i}")

        assert fast_breaker.is_open
        stats = fast_breaker.get_stats()
        assert stats.times_opened == 1

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, fast_breaker):
        """Test that open circuit raises CircuitOpenError (AC1)."""
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                async with fast_breaker:
                    raise ValueError(f"Error {i}")

        assert fast_breaker.is_open

        # Next call should be blocked
        with pytest.raises(CircuitOpenError) as exc_info:
            async with fast_breaker:
                pass

        assert "fast_breaker" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recovery_timeout_enables_half_open(self, fast_breaker):
        """Test that circuit transitions to half-open after timeout (AC3)."""
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                async with fast_breaker:
                    raise ValueError(f"Error {i}")

        assert fast_breaker.is_open

        # Wait for recovery timeout (100ms)
        await asyncio.sleep(0.15)

        # Next call should be allowed (half-open)
        async with fast_breaker:
            pass  # Success

        # Should now be closed
        assert fast_breaker.is_closed

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, fast_breaker):
        """Test that failure in half-open reopens circuit (AC3)."""
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                async with fast_breaker:
                    raise ValueError(f"Error {i}")

        assert fast_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Fail in half-open state
        with pytest.raises(RuntimeError):
            async with fast_breaker:
                raise RuntimeError("Recovery failed")

        assert fast_breaker.is_open
        stats = fast_breaker.get_stats()
        assert stats.times_opened == 1  # Still same opening


# =============================================================================
# Graceful Degradation Tests (AC1)
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation when circuit opens."""

    @pytest.mark.asyncio
    async def test_pipeline_continues_after_failure(self, fast_breaker):
        """Test that pipeline can continue with unavailable response."""
        results = []

        # Simulate multiple expert calls
        experts = ["literal", "systemic", "precedent"]

        for expert in experts:
            if expert == "systemic":
                # Simulate systemic expert failing
                for i in range(2):
                    try:
                        async with fast_breaker:
                            raise TimeoutError("Expert timeout")
                    except TimeoutError:
                        pass

                # Circuit is now open
                try:
                    async with fast_breaker:
                        results.append({"expert": expert, "status": "ok"})
                except CircuitOpenError:
                    results.append(create_unavailable_response(expert))
            else:
                results.append({"expert": expert, "status": "ok"})

        # Pipeline completed with partial results
        assert len(results) == 3
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "unavailable"
        assert results[2]["status"] == "ok"

    def test_unavailable_response_format(self):
        """Test create_unavailable_response returns correct format."""
        response = create_unavailable_response("literal")

        assert response["expert_type"] == "literal"
        assert response["status"] == "unavailable"
        assert "temporaneamente non disponibile" in response["message"]
        assert response["is_degraded"] is True
        assert response["confidence"] == 0.0


# =============================================================================
# Context Manager and Decorator Tests
# =============================================================================


class TestCircuitBreakerUsage:
    """Tests for circuit breaker usage patterns."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, circuit_breaker):
        """Test context manager with successful call."""
        result = None

        async with circuit_breaker:
            result = "success"

        assert result == "success"
        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, circuit_breaker):
        """Test context manager propagates exceptions."""
        with pytest.raises(ValueError):
            async with circuit_breaker:
                raise ValueError("Test error")

        stats = circuit_breaker.get_stats()
        assert stats.total_failures == 1

    @pytest.mark.asyncio
    async def test_decorator_usage(self, circuit_breaker):
        """Test decorator pattern."""
        @circuit_breaker
        async def protected_function():
            return "result"

        result = await protected_function()

        assert result == "result"
        stats = circuit_breaker.get_stats()
        assert stats.total_successes == 1

    @pytest.mark.asyncio
    async def test_call_method(self, circuit_breaker):
        """Test call() method."""
        async def my_function(x, y):
            return x + y

        result = await circuit_breaker.call(my_function, 2, 3)

        assert result == 5


# =============================================================================
# Registry Tests
# =============================================================================


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_or_create(self, registry):
        """Test get_or_create creates new breaker."""
        breaker = registry.get_or_create("test_expert")

        assert breaker is not None
        assert breaker.name == "test_expert"

    def test_get_or_create_returns_same_instance(self, registry):
        """Test get_or_create returns same instance."""
        breaker1 = registry.get_or_create("test_expert")
        breaker2 = registry.get_or_create("test_expert")

        assert breaker1 is breaker2

    def test_get_returns_none_for_missing(self, registry):
        """Test get returns None for missing breaker."""
        assert registry.get("nonexistent") is None

    def test_get_all_stats(self, registry):
        """Test get_all_stats returns all breaker stats."""
        registry.get_or_create("expert_a")
        registry.get_or_create("expert_b")

        stats = registry.get_all_stats()

        assert "expert_a" in stats
        assert "expert_b" in stats
        assert isinstance(stats["expert_a"], CircuitBreakerStats)

    @pytest.mark.asyncio
    async def test_get_open_circuits(self, registry):
        """Test get_open_circuits returns open circuits."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = registry.get_or_create("failing_expert", config=config)

        # Open the circuit
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error")

        open_circuits = registry.get_open_circuits()
        assert "failing_expert" in open_circuits

    def test_reset_all(self, registry):
        """Test reset_all resets all breakers."""
        breaker = registry.get_or_create("test")
        # Manually set some state
        breaker._state = CircuitState.OPEN

        registry.reset_all()

        assert breaker.is_closed


# =============================================================================
# Stats Tests
# =============================================================================


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    @pytest.mark.asyncio
    async def test_stats_track_successes(self, circuit_breaker):
        """Test stats track successful calls."""
        for _ in range(3):
            async with circuit_breaker:
                pass

        stats = circuit_breaker.get_stats()
        assert stats.total_successes == 3

    @pytest.mark.asyncio
    async def test_stats_track_failures(self, circuit_breaker):
        """Test stats track failed calls."""
        for _ in range(2):
            try:
                async with circuit_breaker:
                    raise ValueError("Error")
            except ValueError:
                pass

        stats = circuit_breaker.get_stats()
        assert stats.total_failures == 2

    def test_stats_to_dict(self, circuit_breaker):
        """Test stats serialization."""
        stats = circuit_breaker.get_stats()
        d = stats.to_dict()

        assert "name" in d
        assert "state" in d
        assert "failure_count" in d
        assert "times_opened" in d


# =============================================================================
# State Change Callback Tests (AC4)
# =============================================================================


class TestStateChangeCallback:
    """Tests for state change callbacks (for logging/monitoring)."""

    @pytest.mark.asyncio
    async def test_callback_on_open(self):
        """Test callback is called when circuit opens."""
        state_changes: List[tuple] = []

        def on_change(name, old_state, new_state):
            state_changes.append((name, old_state, new_state))

        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(
            name="callback_test",
            config=config,
            on_state_change=on_change,
        )

        # Trigger opening
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error")

        assert len(state_changes) == 1
        assert state_changes[0][0] == "callback_test"
        assert state_changes[0][1] == CircuitState.CLOSED
        assert state_changes[0][2] == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_callback_on_recovery(self):
        """Test callback is called when circuit recovers."""
        state_changes: List[tuple] = []

        def on_change(name, old_state, new_state):
            state_changes.append((name, old_state, new_state))

        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,
        )
        breaker = CircuitBreaker(
            name="recovery_test",
            config=config,
            on_state_change=on_change,
        )

        # Open circuit
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error")

        # Wait and recover
        await asyncio.sleep(0.1)
        async with breaker:
            pass

        # Should have: CLOSED->OPEN, OPEN->HALF_OPEN, HALF_OPEN->CLOSED
        assert len(state_changes) == 3
        assert state_changes[-1][2] == CircuitState.CLOSED


# =============================================================================
# Expert Circuit Breaker Helper Tests
# =============================================================================


class TestExpertCircuitBreaker:
    """Tests for expert-specific circuit breaker helper."""

    def test_get_expert_circuit_breaker(self):
        """Test get_expert_circuit_breaker returns breaker."""
        breaker = get_expert_circuit_breaker("literal")

        assert breaker is not None
        assert "literal" in breaker.name

    def test_different_experts_get_different_breakers(self):
        """Test each expert gets its own breaker."""
        literal_breaker = get_expert_circuit_breaker("literal")
        systemic_breaker = get_expert_circuit_breaker("systemic")

        assert literal_breaker is not systemic_breaker
        assert literal_breaker.name != systemic_breaker.name


# =============================================================================
# Excluded Exceptions Tests
# =============================================================================


class TestExcludedExceptions:
    """Tests for exception exclusion."""

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_count(self):
        """Test that excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker(name="excluded_test", config=config)

        # These shouldn't count
        for _ in range(5):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("Excluded")

        assert breaker.is_closed  # Still closed

        # But this should count
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("Not excluded")

        stats = breaker.get_stats()
        assert stats.total_failures == 1


# =============================================================================
# Reset Tests
# =============================================================================


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset."""

    @pytest.mark.asyncio
    async def test_manual_reset(self, fast_breaker):
        """Test manual reset closes circuit."""
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                async with fast_breaker:
                    raise ValueError(f"Error {i}")

        assert fast_breaker.is_open

        # Manual reset
        fast_breaker.reset()

        assert fast_breaker.is_closed
        stats = fast_breaker.get_stats()
        assert stats.failure_count == 0


# =============================================================================
# Failure Window Tests (M2)
# =============================================================================


class TestFailureWindowExpiration:
    """Tests for failure window expiration pruning."""

    @pytest.mark.asyncio
    async def test_old_failures_pruned_from_window(self):
        """Test that failures outside the window are pruned."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window_seconds=0.1,  # 100ms window
        )
        breaker = CircuitBreaker(name="window_test", config=config)

        # First failure
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error 1")

        assert breaker._count_recent_failures() == 1

        # Wait for failure to expire
        await asyncio.sleep(0.15)

        # Second failure - old one should be pruned
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error 2")

        # Should only count the recent failure
        assert breaker._count_recent_failures() == 1
        # Circuit should still be closed (only 1 failure in window)
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_failures_within_window_accumulate(self):
        """Test that failures within window accumulate properly."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window_seconds=10.0,  # Long window
        )
        breaker = CircuitBreaker(name="accumulate_test", config=config)

        # Three failures in quick succession
        for i in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError(f"Error {i}")

        # Should have 3 failures and circuit open
        assert breaker._count_recent_failures() == 3
        assert breaker.is_open


# =============================================================================
# Multiple Open Cycles Tests (M3)
# =============================================================================


class TestMultipleOpenCycles:
    """Tests for multiple circuit open/close cycles."""

    @pytest.mark.asyncio
    async def test_times_opened_increments_across_cycles(self):
        """Test that times_opened increments on each open event."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,  # Fast recovery
        )
        breaker = CircuitBreaker(name="cycle_test", config=config)

        # First cycle: open
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Error 1")

        assert breaker.is_open
        assert breaker.get_stats().times_opened == 1

        # Wait and recover
        await asyncio.sleep(0.1)
        async with breaker:
            pass  # Success closes circuit

        assert breaker.is_closed

        # Second cycle: open again
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("Error 2")

        assert breaker.is_open
        assert breaker.get_stats().times_opened == 2

        # Wait and recover again
        await asyncio.sleep(0.1)
        async with breaker:
            pass

        assert breaker.is_closed

        # Third cycle
        with pytest.raises(TimeoutError):
            async with breaker:
                raise TimeoutError("Error 3")

        assert breaker.is_open
        assert breaker.get_stats().times_opened == 3

    @pytest.mark.asyncio
    async def test_stats_track_across_multiple_cycles(self):
        """Test that total_failures and total_successes accumulate across cycles."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,
        )
        breaker = CircuitBreaker(name="stats_cycle_test", config=config)

        # Cycle 1
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Fail 1")

        await asyncio.sleep(0.1)
        async with breaker:
            pass  # Success 1

        # Cycle 2
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Fail 2")

        await asyncio.sleep(0.1)
        async with breaker:
            pass  # Success 2

        stats = breaker.get_stats()
        assert stats.total_failures == 2
        assert stats.total_successes == 2
        assert stats.times_opened == 2
