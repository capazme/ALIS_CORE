"""
WebSocket Authentication Tests
================================

Tests for JWT authentication on the WebSocket endpoint.

Covers:
- Token extraction (empty, valid, expired, invalid, missing claims, dev mode)
- WebSocket endpoint behavior (auth enforced vs not enforced)
- Close codes (4001 auth failure, 4003 server misconfig)
"""

import os
import time
from unittest.mock import patch, AsyncMock, MagicMock

import jwt as pyjwt
import pytest
import pytest_asyncio

from merlt.api.ws_router import _extract_user_id_from_token


# =============================================================================
# FIXTURES
# =============================================================================

JWT_SECRET = "test-secret-key-for-ws-auth"
JWT_ALGORITHM = "HS256"


def _make_token(payload: dict, secret: str = JWT_SECRET, algorithm: str = JWT_ALGORITHM) -> str:
    """Create a JWT token with the given payload."""
    return pyjwt.encode(payload, secret, algorithm=algorithm)


def _make_valid_token(user_id: str = "user-123", **extra) -> str:
    """Create a valid JWT token with standard claims."""
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + 3600,
        **extra,
    }
    return _make_token(payload)


def _make_expired_token(user_id: str = "user-123") -> str:
    """Create an expired JWT token."""
    payload = {
        "sub": user_id,
        "exp": int(time.time()) - 3600,
    }
    return _make_token(payload)


# =============================================================================
# TOKEN EXTRACTION TESTS
# =============================================================================


class TestExtractUserIdFromToken:
    """Tests for _extract_user_id_from_token function."""

    def test_empty_token_returns_anonymous_with_error(self):
        """Empty token returns anonymous with error reason."""
        user_id, error = _extract_user_id_from_token("")
        assert user_id == "anonymous"
        assert error == "empty_token"

    def test_whitespace_token_returns_anonymous_with_error(self):
        """Whitespace-only token returns anonymous with error reason."""
        user_id, error = _extract_user_id_from_token("   ")
        assert user_id == "anonymous"
        assert error == "empty_token"

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_valid_token_sub_claim(self):
        """Valid token with 'sub' claim extracts user_id."""
        token = _make_valid_token(user_id="user-456")
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "user-456"
        assert error is None

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_valid_token_userId_claim(self):
        """Valid token with 'userId' claim extracts user_id (preferred)."""
        payload = {
            "userId": "uid-from-userId",
            "sub": "uid-from-sub",
            "exp": int(time.time()) + 3600,
        }
        token = _make_token(payload)
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "uid-from-userId"
        assert error is None

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_valid_token_user_id_claim(self):
        """Valid token with 'user_id' claim (snake_case)."""
        payload = {
            "user_id": "uid-snake",
            "exp": int(time.time()) + 3600,
        }
        token = _make_token(payload)
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "uid-snake"
        assert error is None

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_expired_token_returns_anonymous_with_error(self):
        """Expired token returns anonymous with error reason."""
        token = _make_expired_token()
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "anonymous"
        assert error == "token_expired"

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_invalid_token_returns_anonymous_with_error(self):
        """Malformed token returns anonymous with error reason."""
        user_id, error = _extract_user_id_from_token("not.a.valid.jwt")
        assert user_id == "anonymous"
        assert error == "token_invalid"

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_wrong_secret_returns_anonymous_with_error(self):
        """Token signed with wrong secret is rejected."""
        token = pyjwt.encode(
            {"sub": "user-123", "exp": int(time.time()) + 3600},
            "wrong-secret",
            algorithm=JWT_ALGORITHM,
        )
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "anonymous"
        assert error == "token_invalid"

    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    def test_missing_user_claims_returns_anonymous_with_error(self):
        """Token without any user ID claim returns anonymous with error."""
        payload = {
            "exp": int(time.time()) + 3600,
            "iss": "test",
        }
        token = _make_token(payload)
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "anonymous"
        assert error == "missing_user_id"

    @patch("merlt.api.ws_router._WS_JWT_SECRET", None)
    def test_dev_mode_valid_token_no_verification(self):
        """Dev mode: token decoded without signature verification."""
        # Token signed with any secret should work in dev mode
        token = pyjwt.encode(
            {"sub": "dev-user", "exp": int(time.time()) + 3600},
            "any-secret-works",
            algorithm=JWT_ALGORITHM,
        )
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "dev-user"
        assert error is None

    @patch("merlt.api.ws_router._WS_JWT_SECRET", None)
    def test_dev_mode_expired_token_still_rejected(self):
        """Dev mode: expired tokens are still rejected (verify_exp=True)."""
        token = pyjwt.encode(
            {"sub": "dev-user", "exp": int(time.time()) - 3600},
            "any-secret",
            algorithm=JWT_ALGORITHM,
        )
        user_id, error = _extract_user_id_from_token(token)
        assert user_id == "anonymous"
        assert error == "token_expired"


# =============================================================================
# WEBSOCKET ENDPOINT BEHAVIOR TESTS
# =============================================================================


class TestWebSocketEndpointAuthEnforcement:
    """Tests for WebSocket endpoint auth enforcement behavior."""

    @pytest.mark.asyncio
    @patch("merlt.api.ws_router._WS_REQUIRE_AUTH", False)
    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    async def test_auth_not_enforced_allows_anonymous(self):
        """When auth not enforced, invalid tokens connect as anonymous."""
        from merlt.api.ws_router import merlt_websocket

        ws = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        # Should connect (not close immediately)
        try:
            await merlt_websocket(websocket=ws, token="invalid-token")
        except Exception:
            pass

        # accept() should have been called (via manager.connect)
        # The ws is managed by MerltWebSocketManager, not directly
        # So we verify it wasn't closed with 4001
        ws.close.assert_not_called()

    @pytest.mark.asyncio
    @patch("merlt.api.ws_router._WS_REQUIRE_AUTH", True)
    @patch("merlt.api.ws_router._WS_JWT_SECRET", JWT_SECRET)
    async def test_auth_enforced_rejects_invalid_token(self):
        """When auth enforced, invalid token gets close code 4001."""
        from merlt.api.ws_router import merlt_websocket

        ws = AsyncMock()

        await merlt_websocket(websocket=ws, token="invalid-token")

        ws.accept.assert_called_once()
        ws.close.assert_called_once()
        close_kwargs = ws.close.call_args
        assert close_kwargs[1].get("code") == 4001 or close_kwargs[0][0] == 4001 if close_kwargs[0] else close_kwargs[1].get("code") == 4001

    @pytest.mark.asyncio
    @patch("merlt.api.ws_router._WS_REQUIRE_AUTH", True)
    @patch("merlt.api.ws_router._WS_JWT_SECRET", None)
    async def test_auth_enforced_no_secret_returns_4003(self):
        """When auth enforced but no secret, close code 4003 (server misconfig)."""
        from merlt.api.ws_router import merlt_websocket

        ws = AsyncMock()

        await merlt_websocket(websocket=ws, token="any-token")

        ws.accept.assert_called_once()
        ws.close.assert_called_once()
        close_kwargs = ws.close.call_args
        code = close_kwargs[1].get("code", None) if close_kwargs[1] else (close_kwargs[0][0] if close_kwargs[0] else None)
        assert code == 4003
