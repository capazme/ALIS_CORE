"""
Tests for LLM Provider Abstraction.

Tests cover:
- AC1: Multiple providers configured (Primary, Backup, Local)
- AC2: Common interface (generate, embed)
- AC3: Failover to backup provider transparently
- AC4: Configuration-driven, no code deployment needed
- AC5: Per-Expert model preferences with fallback chains
"""

import pytest
from datetime import datetime
from typing import List, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from visualex.experts.llm import (
    # Config
    LLMConfig,
    ProviderConfig,
    ModelVersionInfo,
    DEFAULT_PROVIDER_CONFIGS,
    # Base
    BaseLLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    # Providers
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    # Factory
    LLMProviderFactory,
    # Failover
    FailoverLLMService,
    FailoverConfig,
    FailoverEvent,
)


# =============================================================================
# Mock Providers for Testing
# =============================================================================


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    provider_name = "mock"

    def __init__(
        self,
        should_fail: bool = False,
        fail_error: Optional[Exception] = None,
        response_content: str = "Mock response",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if name:
            self.provider_name = name
        self.should_fail = should_fail
        self.fail_error = fail_error
        self.response_content = response_content
        self.generate_calls = []

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        # Validate temperature
        temperature = self._validate_temperature(temperature)

        self.generate_calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
        })

        if self.should_fail:
            raise self.fail_error or ProviderError(self.provider_name, "Mock error")

        usage = LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.response_content.split()),
            total_tokens=len(prompt.split()) + len(self.response_content.split()),
        )

        # Track usage for stats
        self._track_usage(usage)

        return LLMResponse(
            content=self.response_content,
            model=model or self.default_model or "mock-model",
            provider=self.provider_name,
            usage=usage,
            latency_ms=50.0,
        )


# =============================================================================
# Configuration Tests (AC1, AC4)
# =============================================================================


class TestLLMConfiguration:
    """Tests for LLM configuration."""

    def test_default_provider_configs_exist(self):
        """Test that default configs exist for all providers (AC1)."""
        assert "openai" in DEFAULT_PROVIDER_CONFIGS
        assert "anthropic" in DEFAULT_PROVIDER_CONFIGS
        assert "ollama" in DEFAULT_PROVIDER_CONFIGS

    def test_provider_config_structure(self):
        """Test provider config has required fields."""
        config = DEFAULT_PROVIDER_CONFIGS["openai"]

        assert config.name == "openai"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.default_model is not None
        assert config.timeout_seconds > 0

    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()

        assert config.primary_provider == "openai"
        assert "anthropic" in config.backup_providers
        assert config.enable_cost_tracking is True
        assert config.enable_model_versioning is True

    def test_llm_config_get_model_for_expert(self):
        """Test getting model preference for expert (AC5)."""
        config = LLMConfig(
            expert_model_preferences={
                "literal": "gpt-4",
                "gating": "gpt-3.5-turbo",
            }
        )

        assert config.get_model_for_expert("literal", "default") == "gpt-4"
        assert config.get_model_for_expert("gating", "default") == "gpt-3.5-turbo"
        assert config.get_model_for_expert("unknown", "default") == "default"

    def test_model_version_info_serialization(self):
        """Test ModelVersionInfo serialization (NFR-R6)."""
        info = ModelVersionInfo(
            provider="openai",
            model_id="gpt-4-0125-preview",
            version="0125",
            timestamp=datetime(2024, 1, 25),
        )

        d = info.to_dict()
        assert d["provider"] == "openai"
        assert d["model_id"] == "gpt-4-0125-preview"
        assert d["version"] == "0125"


# =============================================================================
# Common Interface Tests (AC2)
# =============================================================================


class TestCommonInterface:
    """Tests for common LLM interface (AC2)."""

    @pytest.mark.asyncio
    async def test_generate_interface(self):
        """Test generate() common interface."""
        provider = MockProvider(response_content="Test response")

        response = await provider.generate(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

        assert response.content == "Test response"
        assert response.provider == "mock"
        assert isinstance(response.usage, LLMUsage)

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self):
        """Test that generate returns LLMResponse with all fields."""
        provider = MockProvider()

        response = await provider.generate("Test")

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.model is not None
        assert response.provider == "mock"
        assert response.latency_ms >= 0
        assert isinstance(response.timestamp, datetime)

    def test_llm_usage_tracking(self):
        """Test LLMUsage tracks token counts."""
        usage = LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.015,
        )

        assert usage.total_tokens == 150
        d = usage.to_dict()
        assert "prompt_tokens" in d
        assert "estimated_cost" in d

    @pytest.mark.asyncio
    async def test_embed_not_implemented(self):
        """Test embed raises NotImplementedError by default."""
        provider = MockProvider()

        with pytest.raises(NotImplementedError):
            await provider.embed(["text"])

    @pytest.mark.asyncio
    async def test_temperature_validation(self):
        """Test temperature is validated (L1 fix)."""
        provider = MockProvider()

        # Test clamping high temperature
        response = await provider.generate("Test", temperature=1.5)
        assert response is not None  # Should succeed with clamped value

        # Test clamping low temperature
        response = await provider.generate("Test", temperature=-0.5)
        assert response is not None  # Should succeed with clamped value

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check method."""
        provider = MockProvider()

        result = await provider.health_check()
        assert result is True

        failing_provider = MockProvider(
            should_fail=True,
            fail_error=ProviderError("mock", "fail"),
        )
        result = await failing_provider.health_check()
        assert result is False


# =============================================================================
# Failover Tests (AC3)
# =============================================================================


class TestFailoverService:
    """Tests for failover service (AC3)."""

    @pytest.mark.asyncio
    async def test_uses_primary_provider_first(self):
        """Test that primary provider is used first."""
        primary = MockProvider(response_content="Primary response")
        backup = MockProvider(response_content="Backup response")

        service = FailoverLLMService(providers=[primary, backup])
        result = await service.generate("Test")

        assert result == "Primary response"
        assert len(primary.generate_calls) == 1
        assert len(backup.generate_calls) == 0

    @pytest.mark.asyncio
    async def test_failover_to_backup_on_error(self):
        """Test failover to backup when primary fails (AC3)."""
        primary = MockProvider(
            should_fail=True,
            fail_error=ProviderError("primary", "Connection error"),
        )
        backup = MockProvider(response_content="Backup response")

        service = FailoverLLMService(providers=[primary, backup])
        result = await service.generate("Test")

        assert result == "Backup response"
        assert len(primary.generate_calls) == 1
        assert len(backup.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_failover_on_rate_limit(self):
        """Test failover on rate limit error."""
        primary = MockProvider(
            should_fail=True,
            fail_error=RateLimitError("primary", 60),
        )
        backup = MockProvider(response_content="Backup")

        service = FailoverLLMService(providers=[primary, backup])
        result = await service.generate("Test")

        assert result == "Backup"

    @pytest.mark.asyncio
    async def test_failover_on_auth_error(self):
        """Test failover on authentication error."""
        primary = MockProvider(
            should_fail=True,
            fail_error=AuthenticationError("primary"),
        )
        backup = MockProvider(response_content="Backup")

        service = FailoverLLMService(providers=[primary, backup])
        result = await service.generate("Test")

        assert result == "Backup"

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_error(self):
        """Test that error is raised when all providers fail."""
        primary = MockProvider(should_fail=True, fail_error=ProviderError("p1", "fail"))
        backup = MockProvider(should_fail=True, fail_error=ProviderError("p2", "fail"))

        service = FailoverLLMService(providers=[primary, backup])

        with pytest.raises(ProviderError) as exc_info:
            await service.generate("Test")

        assert "All providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_failover_records_event(self):
        """Test that failover events are recorded."""
        primary = MockProvider(
            should_fail=True,
            fail_error=ProviderError("primary", "Error"),
        )
        primary.provider_name = "primary"
        backup = MockProvider(response_content="Backup")
        backup.provider_name = "backup"

        service = FailoverLLMService(providers=[primary, backup])
        await service.generate("Test")

        stats = service.get_stats()
        assert stats["total_failovers"] == 1

    @pytest.mark.asyncio
    async def test_provider_cooldown(self):
        """Test that failed providers have cooldown."""
        # First call fails - use different provider names
        primary = MockProvider(
            should_fail=True,
            fail_error=ProviderError("primary", "Error"),
            name="primary",
        )
        backup = MockProvider(response_content="Backup", name="backup")

        config = FailoverConfig(
            cooldown_seconds=300,
            max_consecutive_failures=1,
        )
        service = FailoverLLMService(providers=[primary, backup], config=config)

        # First call triggers failover
        await service.generate("Test 1")
        assert len(primary.generate_calls) == 1

        # Second call should skip primary due to cooldown
        await service.generate("Test 2")
        # Primary should not be called again
        assert len(primary.generate_calls) == 1

    def test_get_healthy_providers(self):
        """Test getting list of healthy providers."""
        primary = MockProvider()
        backup = MockProvider()

        service = FailoverLLMService(providers=[primary, backup])

        healthy = service.get_healthy_providers()
        assert len(healthy) == 2

    def test_reset_provider(self):
        """Test resetting a provider's state."""
        primary = MockProvider()
        service = FailoverLLMService(providers=[primary])

        # Manually set failure state
        service._provider_states["mock"].consecutive_failures = 5
        service._provider_states["mock"].is_healthy = False

        service.reset_provider("mock")

        state = service._provider_states["mock"]
        assert state.consecutive_failures == 0
        assert state.is_healthy is True


# =============================================================================
# Factory Tests (AC4)
# =============================================================================


class TestProviderFactory:
    """Tests for provider factory (AC4)."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        factory = LLMProviderFactory()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = factory.create("openai")

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        factory = LLMProviderFactory()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = factory.create("anthropic")

        assert isinstance(provider, AnthropicProvider)
        assert provider.provider_name == "anthropic"

    def test_create_ollama_provider(self):
        """Test creating Ollama provider (no API key needed)."""
        factory = LLMProviderFactory()
        provider = factory.create("ollama")

        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"

    def test_create_with_custom_model(self):
        """Test creating provider with custom model."""
        factory = LLMProviderFactory()
        provider = factory.create("ollama", model="mistral")

        assert provider.default_model == "mistral"

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises error."""
        factory = LLMProviderFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create("unknown_provider")

        assert "Unknown provider" in str(exc_info.value)

    def test_get_or_create_returns_same_instance(self):
        """Test get_or_create returns singleton."""
        factory = LLMProviderFactory()

        p1 = factory.get_or_create("ollama")
        p2 = factory.get_or_create("ollama")

        assert p1 is p2

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = LLMProviderFactory.get_available_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers

    def test_get_provider_for_expert(self):
        """Test getting provider for specific expert (AC5)."""
        config = LLMConfig(
            expert_model_preferences={"literal": "gpt-4"},
        )
        factory = LLMProviderFactory(config=config)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            provider = factory.get_provider_for_expert("literal")

        assert provider is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_provider_error_includes_provider_name(self):
        """Test ProviderError includes provider name."""
        error = ProviderError("openai", "Connection failed")

        assert "openai" in str(error)
        assert error.provider == "openai"

    def test_rate_limit_error_includes_retry_after(self):
        """Test RateLimitError includes retry info."""
        error = RateLimitError("openai", retry_after=60)

        assert error.retry_after == 60
        assert "60" in str(error)

    def test_model_not_found_error_includes_model(self):
        """Test ModelNotFoundError includes model name."""
        error = ModelNotFoundError("openai", "gpt-5")

        assert error.model == "gpt-5"
        assert "gpt-5" in str(error)


# =============================================================================
# Provider Stats Tests
# =============================================================================


class TestProviderStats:
    """Tests for provider statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_requests(self):
        """Test that stats track request count."""
        provider = MockProvider()

        await provider.generate("Test 1")
        await provider.generate("Test 2")

        stats = provider.get_stats()
        assert stats["request_count"] == 2

    @pytest.mark.asyncio
    async def test_stats_track_tokens(self):
        """Test that stats track token usage."""
        provider = MockProvider(response_content="Short response")

        await provider.generate("Test prompt")

        stats = provider.get_stats()
        assert stats["total_tokens"] > 0

    def test_llm_response_to_dict(self):
        """Test LLMResponse serialization."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="openai",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=100,
        )

        d = response.to_dict()
        assert d["content"] == "Test"
        assert d["model"] == "gpt-4"
        assert "usage" in d
        assert "timestamp" in d


# =============================================================================
# Failover Event Tests
# =============================================================================


class TestFailoverEvents:
    """Tests for failover event tracking."""

    def test_failover_event_serialization(self):
        """Test FailoverEvent serialization."""
        event = FailoverEvent(
            timestamp=datetime(2024, 1, 25, 10, 30),
            from_provider="openai",
            to_provider="anthropic",
            error="Connection timeout",
            latency_penalty_ms=500.0,
        )

        d = event.to_dict()
        assert d["from_provider"] == "openai"
        assert d["to_provider"] == "anthropic"
        assert d["error"] == "Connection timeout"

    @pytest.mark.asyncio
    async def test_failover_stats(self):
        """Test failover service stats."""
        primary = MockProvider(
            should_fail=True,
            fail_error=ProviderError("primary", "Error"),
        )
        backup = MockProvider()

        service = FailoverLLMService(providers=[primary, backup])
        await service.generate("Test")

        stats = service.get_stats()
        assert "total_failovers" in stats
        assert "provider_states" in stats
        assert "recent_failovers" in stats


# =============================================================================
# Integration with LLMService Protocol Tests
# =============================================================================


class TestLLMServiceProtocol:
    """Tests for LLMService protocol compatibility."""

    @pytest.mark.asyncio
    async def test_failover_service_implements_protocol(self):
        """Test FailoverLLMService implements LLMService protocol."""
        provider = MockProvider(response_content="Response text")
        service = FailoverLLMService(providers=[provider])

        # LLMService protocol requires: generate(prompt, temperature, max_tokens) -> str
        result = await service.generate(
            prompt="Test prompt",
            temperature=0.3,
            max_tokens=2000,
        )

        # Protocol returns string only
        assert isinstance(result, str)
        assert result == "Response text"
