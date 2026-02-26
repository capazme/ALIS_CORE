"""Tests for PipelineWebSocketManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from merlt.pipeline.websocket_manager import PipelineWebSocketManager, ws_manager


def _make_ws() -> AsyncMock:
    """Create a mock WebSocket with accept/send_json."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_accepts_and_adds():
    """connect() accepts the websocket and adds it to the connections dict."""
    mgr = PipelineWebSocketManager()
    ws = _make_ws()

    await mgr.connect("run-1", ws)

    ws.accept.assert_awaited_once()
    assert mgr.get_connected_count("run-1") == 1


@pytest.mark.asyncio
async def test_connect_multiple_clients_same_run():
    """Multiple clients can connect to the same run_id."""
    mgr = PipelineWebSocketManager()
    ws1 = _make_ws()
    ws2 = _make_ws()
    ws3 = _make_ws()

    await mgr.connect("run-1", ws1)
    await mgr.connect("run-1", ws2)
    await mgr.connect("run-1", ws3)

    assert mgr.get_connected_count("run-1") == 3


@pytest.mark.asyncio
async def test_connect_clients_different_runs_isolated():
    """Clients on different run_ids are isolated."""
    mgr = PipelineWebSocketManager()
    ws_a = _make_ws()
    ws_b = _make_ws()

    await mgr.connect("run-a", ws_a)
    await mgr.connect("run-b", ws_b)

    assert mgr.get_connected_count("run-a") == 1
    assert mgr.get_connected_count("run-b") == 1


@pytest.mark.asyncio
async def test_disconnect_removes_and_cleans_empty_run():
    """disconnect() removes client and deletes the run_id entry when empty."""
    mgr = PipelineWebSocketManager()
    ws = _make_ws()

    await mgr.connect("run-1", ws)
    assert mgr.get_connected_count("run-1") == 1

    await mgr.disconnect("run-1", ws)
    assert mgr.get_connected_count("run-1") == 0
    assert "run-1" not in mgr._connections


@pytest.mark.asyncio
async def test_disconnect_unknown_run_no_error():
    """disconnect() with unknown run_id does not raise."""
    mgr = PipelineWebSocketManager()
    ws = _make_ws()

    await mgr.disconnect("nonexistent", ws)
    # Should not raise


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_clients():
    """broadcast() sends the message to all clients of a run."""
    mgr = PipelineWebSocketManager()
    ws1 = _make_ws()
    ws2 = _make_ws()

    await mgr.connect("run-1", ws1)
    await mgr.connect("run-1", ws2)

    msg = {"event": "progress", "data": {"pct": 50}}
    await mgr.broadcast("run-1", msg)

    ws1.send_json.assert_awaited_once_with(msg)
    ws2.send_json.assert_awaited_once_with(msg)


@pytest.mark.asyncio
async def test_broadcast_unknown_run_noop():
    """broadcast() to unknown run_id is a no-op."""
    mgr = PipelineWebSocketManager()

    await mgr.broadcast("nonexistent", {"event": "test"})
    # Should not raise


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections():
    """broadcast() removes connections that raise on send."""
    mgr = PipelineWebSocketManager()
    ws_alive = _make_ws()
    ws_dead = _make_ws()
    ws_dead.send_json.side_effect = Exception("Connection closed")

    await mgr.connect("run-1", ws_alive)
    await mgr.connect("run-1", ws_dead)
    assert mgr.get_connected_count("run-1") == 2

    await mgr.broadcast("run-1", {"event": "test"})

    assert mgr.get_connected_count("run-1") == 1


@pytest.mark.asyncio
async def test_broadcast_partial_failure():
    """When one client is dead and one alive, alive still receives."""
    mgr = PipelineWebSocketManager()
    ws_alive = _make_ws()
    ws_dead = _make_ws()
    ws_dead.send_json.side_effect = RuntimeError("broken pipe")

    await mgr.connect("run-1", ws_alive)
    await mgr.connect("run-1", ws_dead)

    msg = {"event": "update", "data": {}}
    await mgr.broadcast("run-1", msg)

    ws_alive.send_json.assert_awaited_once_with(msg)
    assert mgr.get_connected_count("run-1") == 1


def test_get_connected_count_empty():
    """get_connected_count returns 0 for unknown run_id."""
    mgr = PipelineWebSocketManager()
    assert mgr.get_connected_count("nonexistent") == 0


@pytest.mark.asyncio
async def test_get_connected_count_after_connect_disconnect():
    """get_connected_count tracks connect and disconnect correctly."""
    mgr = PipelineWebSocketManager()
    ws1 = _make_ws()
    ws2 = _make_ws()

    assert mgr.get_connected_count("run-1") == 0

    await mgr.connect("run-1", ws1)
    assert mgr.get_connected_count("run-1") == 1

    await mgr.connect("run-1", ws2)
    assert mgr.get_connected_count("run-1") == 2

    await mgr.disconnect("run-1", ws1)
    assert mgr.get_connected_count("run-1") == 1

    await mgr.disconnect("run-1", ws2)
    assert mgr.get_connected_count("run-1") == 0


def test_singleton_ws_manager_exists():
    """Module-level ws_manager singleton exists and is a PipelineWebSocketManager."""
    assert ws_manager is not None
    assert isinstance(ws_manager, PipelineWebSocketManager)
