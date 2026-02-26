"""
Authentication Middleware Unit Tests
=====================================

Tests for API key authentication (auth.py) and rate limiting (rate_limit.py).

Covers:
- hash_api_key: SHA-256 hashing
- verify_api_key: Valid, invalid, inactive, expired keys
- require_role: Admin/user/guest role checking
- optional_api_key: Unauthenticated fallback
- check_rate_limit: Redis sliding window + graceful degradation

Total: ~30 test cases

Run:
    pytest tests/api/test_auth_middleware.py -v
"""

import hashlib
import time
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from merlt.api.auth import hash_api_key, verify_api_key, require_role, optional_api_key
from merlt.api.rate_limit import RATE_LIMIT_QUOTAS, RATE_LIMIT_WINDOW, check_rate_limit
from merlt.experts.models import ApiKey


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_api_key():
    """Create a valid, active, non-expired API key."""
    return ApiKey(
        key_id="test-key-001",
        user_id="user_123",
        api_key_hash=hash_api_key("valid-test-key"),
        role="user",
        rate_limit_tier="standard",
        is_active=True,
        description="Test user key",
        created_at=datetime.now(UTC).replace(tzinfo=None),
        expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=30),
        last_used_at=None,
    )


@pytest.fixture
def admin_api_key():
    """Create admin API key with no expiration."""
    return ApiKey(
        key_id="admin-key-001",
        user_id="admin_123",
        api_key_hash=hash_api_key("admin-test-key"),
        role="admin",
        rate_limit_tier="unlimited",
        is_active=True,
        description="Test admin key",
        created_at=datetime.now(UTC).replace(tzinfo=None),
        expires_at=None,
        last_used_at=None,
    )


@pytest.fixture
def inactive_api_key():
    """Create inactive API key."""
    return ApiKey(
        key_id="inactive-key-001",
        user_id="user_456",
        api_key_hash=hash_api_key("inactive-test-key"),
        role="user",
        rate_limit_tier="standard",
        is_active=False,
        description="Inactive test key",
        created_at=datetime.now(UTC).replace(tzinfo=None),
        expires_at=None,
        last_used_at=None,
    )


@pytest.fixture
def expired_api_key():
    """Create expired API key."""
    return ApiKey(
        key_id="expired-key-001",
        user_id="user_789",
        api_key_hash=hash_api_key("expired-test-key"),
        role="user",
        rate_limit_tier="standard",
        is_active=True,
        description="Expired test key",
        created_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=60),
        expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1),
        last_used_at=None,
    )


@pytest.fixture
def mock_session():
    """Create mock async DB session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


def _mock_session_with_key(session, api_key):
    """Configure mock session to return a specific ApiKey."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = api_key
    session.execute = AsyncMock(return_value=mock_result)


# ============================================================================
# hash_api_key tests
# ============================================================================


class TestHashApiKey:
    """Tests for hash_api_key function."""

    def test_consistency(self):
        """Same key produces same hash."""
        key = "test-key-12345"
        assert hash_api_key(key) == hash_api_key(key)

    def test_sha256_format(self):
        """Hash is 64-character SHA-256 hex digest."""
        h = hash_api_key("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_manual_sha256(self):
        """Matches Python hashlib SHA-256."""
        key = "test-key-12345"
        expected = hashlib.sha256(key.encode()).hexdigest()
        assert hash_api_key(key) == expected

    def test_different_keys_different_hashes(self):
        """Different keys produce different hashes."""
        assert hash_api_key("key-1") != hash_api_key("key-2")

    def test_case_sensitive(self):
        """API key hashing is case-sensitive."""
        assert hash_api_key("testkey") != hash_api_key("TESTKEY")

    def test_special_characters(self):
        """Keys with special characters produce valid hashes."""
        h = hash_api_key("test-key_123!@#$%^&*()")
        assert len(h) == 64

    def test_empty_string(self):
        """Empty string produces valid hash."""
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_api_key("") == expected


# ============================================================================
# verify_api_key tests
# ============================================================================


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_valid_key(self, valid_api_key, mock_session):
        """Valid key authenticates successfully."""
        _mock_session_with_key(mock_session, valid_api_key)

        result = await verify_api_key(
            x_api_key="valid-test-key",
            session=mock_session,
        )

        assert result == valid_api_key
        assert result.role == "user"
        assert result.is_active is True
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_admin_key(self, admin_api_key, mock_session):
        """Admin key authenticates with correct role and tier."""
        _mock_session_with_key(mock_session, admin_api_key)

        result = await verify_api_key(
            x_api_key="admin-test-key",
            session=mock_session,
        )

        assert result.role == "admin"
        assert result.rate_limit_tier == "unlimited"

    @pytest.mark.asyncio
    async def test_no_expiration(self, admin_api_key, mock_session):
        """Key with expires_at=None is valid."""
        _mock_session_with_key(mock_session, admin_api_key)

        result = await verify_api_key(
            x_api_key="admin-test-key",
            session=mock_session,
        )

        assert result.expires_at is None
        assert not result.is_expired()

    @pytest.mark.asyncio
    async def test_invalid_key_401(self, mock_session):
        """Key not in database raises 401."""
        _mock_session_with_key(mock_session, None)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="invalid-key", session=mock_session)

        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_inactive_key_401(self, inactive_api_key, mock_session):
        """Inactive key raises 401."""
        _mock_session_with_key(mock_session, inactive_api_key)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="inactive-test-key", session=mock_session)

        assert exc_info.value.status_code == 401
        assert "inactive" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_expired_key_401(self, expired_api_key, mock_session):
        """Expired key raises 401."""
        _mock_session_with_key(mock_session, expired_api_key)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="expired-test-key", session=mock_session)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_updates_last_used_at(self, valid_api_key, mock_session):
        """Successful auth updates last_used_at."""
        _mock_session_with_key(mock_session, valid_api_key)
        assert valid_api_key.last_used_at is None

        await verify_api_key(x_api_key="valid-test-key", session=mock_session)

        assert valid_api_key.last_used_at is not None

    @pytest.mark.asyncio
    async def test_sql_injection_attempt(self, mock_session):
        """SQL injection in key value is safely hashed."""
        _mock_session_with_key(mock_session, None)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(
                x_api_key="test'; DROP TABLE api_keys; --",
                session=mock_session,
            )

        assert exc_info.value.status_code == 401


# ============================================================================
# require_role tests
# ============================================================================


class TestRequireRole:
    """Tests for require_role factory."""

    @pytest.mark.asyncio
    async def test_admin_allowed(self, admin_api_key):
        """Admin passes admin-only check."""
        check_admin = require_role("admin")
        result = await check_admin(api_key=admin_api_key)
        assert result == admin_api_key

    @pytest.mark.asyncio
    async def test_user_denied_admin_endpoint(self, valid_api_key):
        """User role denied on admin-only endpoint."""
        check_admin = require_role("admin")

        with pytest.raises(HTTPException) as exc_info:
            await check_admin(api_key=valid_api_key)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_multiple_roles_allowed(self, admin_api_key, valid_api_key):
        """Multiple roles accepted."""
        check_auth = require_role(["admin", "user"])

        result_admin = await check_auth(api_key=admin_api_key)
        assert result_admin == admin_api_key

        result_user = await check_auth(api_key=valid_api_key)
        assert result_user == valid_api_key

    @pytest.mark.asyncio
    async def test_guest_denied(self):
        """Guest role denied on admin-only endpoint."""
        check_admin = require_role("admin")

        guest_key = ApiKey(
            key_id="guest-001",
            user_id="guest",
            api_key_hash=hash_api_key("guest-key"),
            role="guest",
            rate_limit_tier="limited",
            is_active=True,
        )

        with pytest.raises(HTTPException) as exc_info:
            await check_admin(api_key=guest_key)

        assert exc_info.value.status_code == 403


# ============================================================================
# optional_api_key tests
# ============================================================================


class TestOptionalApiKey:
    """Tests for optional_api_key dependency."""

    @pytest.mark.asyncio
    async def test_no_key_returns_none(self, mock_session):
        """No key provided returns None (anonymous access)."""
        result = await optional_api_key(x_api_key=None, session=mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_key_returns_api_key(self, valid_api_key, mock_session):
        """Valid key returns the ApiKey object."""
        _mock_session_with_key(mock_session, valid_api_key)

        result = await optional_api_key(
            x_api_key="valid-test-key",
            session=mock_session,
        )

        assert result == valid_api_key

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self, mock_session):
        """Invalid key returns None (graceful degradation)."""
        _mock_session_with_key(mock_session, None)

        result = await optional_api_key(
            x_api_key="invalid-key",
            session=mock_session,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_inactive_key_returns_none(self, inactive_api_key, mock_session):
        """Inactive key returns None."""
        _mock_session_with_key(mock_session, inactive_api_key)

        result = await optional_api_key(
            x_api_key="inactive-test-key",
            session=mock_session,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_expired_key_returns_none(self, expired_api_key, mock_session):
        """Expired key returns None."""
        _mock_session_with_key(mock_session, expired_api_key)

        result = await optional_api_key(
            x_api_key="expired-test-key",
            session=mock_session,
        )

        assert result is None


# ============================================================================
# Rate limit constants tests
# ============================================================================


class TestRateLimitConstants:
    """Tests for rate limit tier quotas."""

    def test_tier_values(self):
        """Verify tier quotas are correct."""
        assert RATE_LIMIT_QUOTAS["unlimited"] == 999999
        assert RATE_LIMIT_QUOTAS["premium"] == 1000
        assert RATE_LIMIT_QUOTAS["standard"] == 100
        assert RATE_LIMIT_QUOTAS["limited"] == 10

    def test_premium_greater_than_standard(self):
        """Premium allows more requests than standard."""
        assert RATE_LIMIT_QUOTAS["premium"] > RATE_LIMIT_QUOTAS["standard"]

    def test_window_is_one_hour(self):
        """Rate limit window is 1 hour."""
        assert RATE_LIMIT_WINDOW == 3600


# ============================================================================
# check_rate_limit tests
# ============================================================================


class TestCheckRateLimit:
    """Tests for check_rate_limit dependency."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_redis(self, valid_api_key):
        """When Redis is unavailable, request is allowed."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {}

        with patch("merlt.api.rate_limit._get_redis", return_value=None):
            await check_rate_limit(
                request=mock_request,
                response=mock_response,
                api_key=valid_api_key,
            )

        # Should set headers even without Redis
        assert "X-RateLimit-Limit" in mock_response.headers
        assert mock_response.headers["X-RateLimit-Limit"] == "100"

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_429(self, valid_api_key):
        """Exceeding rate limit returns 429."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {}

        # Mock Redis pipeline — pipeline() is sync, execute() is async
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_pipe.zadd = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[
            None,   # zremrangebyscore
            100,    # zcard = at limit for standard tier
            None,   # zadd
            None,   # expire
        ])

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch("merlt.api.rate_limit._get_redis", AsyncMock(return_value=mock_redis)):
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(
                    request=mock_request,
                    response=mock_response,
                    api_key=valid_api_key,
                )

            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_headers_set(self, valid_api_key):
        """Rate limit headers are set on successful request."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {}

        # Mock Redis pipeline — pipeline() is sync, execute() is async
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_pipe.zadd = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[
            None,  # zremrangebyscore
            5,     # zcard = 5 used
            None,  # zadd
            None,  # expire
        ])

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch("merlt.api.rate_limit._get_redis", AsyncMock(return_value=mock_redis)):
            await check_rate_limit(
                request=mock_request,
                response=mock_response,
                api_key=valid_api_key,
            )

        assert mock_response.headers["X-RateLimit-Limit"] == "100"
        assert mock_response.headers["X-RateLimit-Used"] == "6"  # 5 + 1
        assert int(mock_response.headers["X-RateLimit-Remaining"]) == 94  # 100 - 5 - 1

    @pytest.mark.asyncio
    async def test_redis_error_allows_request(self, valid_api_key):
        """Redis error during check gracefully allows request."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {}

        # Mock Redis pipeline with execute() raising
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_pipe.zadd = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(side_effect=Exception("Redis timeout"))

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch("merlt.api.rate_limit._get_redis", AsyncMock(return_value=mock_redis)):
            await check_rate_limit(
                request=mock_request,
                response=mock_response,
                api_key=valid_api_key,
            )

        # Should still set headers (graceful degradation)
        assert "X-RateLimit-Limit" in mock_response.headers
