"""
MERL-T General WebSocket Router
================================

WebSocket hub per notifiche real-time verso il frontend:
- enrichment:started / enrichment:completed
- validation:assigned
- training:epoch_complete
- keepalive (ping/pong)

Il frontend si connette a WS /api/v1/ws?token=JWT e riceve tutti gli eventi
relativi all'utente corrente.
"""

import json
import asyncio
from typing import Dict, Set
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

log = structlog.get_logger()

router = APIRouter(tags=["websocket"])


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class MerltWebSocketManager:
    """
    Manager globale per connessioni WebSocket MERL-T.

    Gestisce connessioni per utente (via token/user_id).
    Permette broadcast a tutti o a specifici utenti.
    """

    def __init__(self) -> None:
        # user_id -> set of websocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # All connections (for global broadcast)
        self._all_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, user_id: str = "anonymous") -> None:
        await websocket.accept()
        self._all_connections.add(websocket)

        if user_id not in self._connections:
            self._connections[user_id] = set()
        self._connections[user_id].add(websocket)

        log.info(
            "MERLT WS connected",
            user_id=user_id,
            total_connections=len(self._all_connections),
        )

    async def disconnect(self, websocket: WebSocket, user_id: str = "anonymous") -> None:
        self._all_connections.discard(websocket)

        if user_id in self._connections:
            self._connections[user_id].discard(websocket)
            if not self._connections[user_id]:
                del self._connections[user_id]

        log.info(
            "MERLT WS disconnected",
            user_id=user_id,
            remaining=len(self._all_connections),
        )

    async def send_to_user(self, user_id: str, message: dict) -> None:
        """Send message to all connections of a specific user."""
        if user_id not in self._connections:
            return

        dead: Set[WebSocket] = set()
        for ws in self._connections[user_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)

        # Cleanup dead connections
        for ws in dead:
            self._connections[user_id].discard(ws)
            self._all_connections.discard(ws)

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        dead: Set[WebSocket] = set()
        for ws in self._all_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)

        # Cleanup dead connections
        self._all_connections -= dead

    @property
    def connection_count(self) -> int:
        return len(self._all_connections)


# Singleton globale
merlt_ws_manager = MerltWebSocketManager()


# =============================================================================
# Helper: extract user_id from JWT token (lightweight, no full verification)
# =============================================================================

def _extract_user_id_from_token(token: str) -> str:
    """
    Extract user_id from JWT without full verification.
    In production, use proper JWT validation.
    """
    try:
        import base64
        # JWT: header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            return "anonymous"

        # Decode payload (add padding)
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("userId") or payload.get("user_id") or payload.get("sub") or "anonymous"
    except Exception:
        return "anonymous"


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/ws")
async def merlt_websocket(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    """
    General-purpose MERL-T WebSocket endpoint.

    Frontend connects to: ws://host/api/merlt/ws?token=JWT

    Protocol:
    - Server sends JSON messages: { "type": "event_name", "payload": {...} }
    - Client sends "ping" -> Server responds "pong"
    - Server sends keepalive every 30s

    Events emitted:
    - enrichment:started   { article_urn }
    - enrichment:completed { article_urn, entities_count, relations_count }
    - validation:assigned  { entity_id, entity_type }
    - system:connected     { user_id, timestamp }
    """
    user_id = _extract_user_id_from_token(token) if token else "anonymous"

    await merlt_ws_manager.connect(websocket, user_id)

    # Send initial connected event
    try:
        await websocket.send_json({
            "type": "system:connected",
            "payload": {
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_version": "1.0.0",
            },
        })
    except Exception:
        pass

    # Keepalive task
    async def keepalive():
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_json({
                    "type": "keepalive",
                    "payload": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                })
            except Exception:
                break

    keepalive_task = asyncio.create_task(keepalive())

    try:
        while True:
            data = await websocket.receive_text()

            if data == "ping":
                await websocket.send_text("pong")
            else:
                log.debug("MERLT WS received", user_id=user_id, data=data[:100])

    except WebSocketDisconnect:
        log.info("MERLT WS client disconnected", user_id=user_id)

    except Exception as e:
        log.warning("MERLT WS error", user_id=user_id, error=str(e))

    finally:
        keepalive_task.cancel()
        await merlt_ws_manager.disconnect(websocket, user_id)


# =============================================================================
# Public API for other modules to emit events
# =============================================================================

async def emit_enrichment_started(article_urn: str, user_id: str | None = None) -> None:
    """Notify clients that enrichment has started for an article."""
    msg = {
        "type": "enrichment:started",
        "payload": {"article_urn": article_urn},
    }
    if user_id:
        await merlt_ws_manager.send_to_user(user_id, msg)
    else:
        await merlt_ws_manager.broadcast(msg)


async def emit_enrichment_completed(
    article_urn: str,
    entities_count: int = 0,
    relations_count: int = 0,
    user_id: str | None = None,
) -> None:
    """Notify clients that enrichment has completed."""
    msg = {
        "type": "enrichment:completed",
        "payload": {
            "article_urn": article_urn,
            "entities_count": entities_count,
            "relations_count": relations_count,
        },
    }
    if user_id:
        await merlt_ws_manager.send_to_user(user_id, msg)
    else:
        await merlt_ws_manager.broadcast(msg)


async def emit_validation_assigned(
    entity_id: str,
    entity_type: str,
    user_id: str | None = None,
) -> None:
    """Notify a user that a new validation has been assigned."""
    msg = {
        "type": "validation:assigned",
        "payload": {
            "entity_id": entity_id,
            "entity_type": entity_type,
        },
    }
    if user_id:
        await merlt_ws_manager.send_to_user(user_id, msg)
    else:
        await merlt_ws_manager.broadcast(msg)


__all__ = [
    "router",
    "merlt_ws_manager",
    "emit_enrichment_started",
    "emit_enrichment_completed",
    "emit_validation_assigned",
]
