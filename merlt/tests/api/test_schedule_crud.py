"""
Test Schedule CRUD Endpoints
==============================
P1-SCHED-1: Verify ingestion schedule CRUD via REST API.

Tests:
- POST /ingestion/schedules — create schedule
- GET /ingestion/schedules — list schedules
- PUT /ingestion/schedules/{id} — update schedule
- DELETE /ingestion/schedules/{id} — delete schedule
- POST /ingestion/schedules/{id}/toggle — toggle enabled

All DB and auth dependencies are mocked; no live DB required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.schedule_router import router, _get_scheduler
from merlt.api.auth import verify_api_key, require_role
from merlt.experts.models import ApiKey


# =============================================================================
# FIXTURES
# =============================================================================

def _make_fake_api_key(role: str = "admin") -> ApiKey:
    """Build a minimal ApiKey object for dependency overrides."""
    key = MagicMock(spec=ApiKey)
    key.key_id = "test-key-id"
    key.role = role
    key.is_active = True
    key.user_id = "test-user"
    key.is_expired = MagicMock(return_value=False)
    return key


@pytest.fixture
def mock_scheduler():
    """Mock IngestionScheduler with CRUD helpers."""
    scheduler = MagicMock()
    # In-memory store for schedules
    _store: dict[int, dict] = {}
    _next_id = [1]

    async def _add(session, tipo_atto, cron_expr, enabled=True, description=None):
        sid = _next_id[0]
        _next_id[0] += 1
        entry = {
            "id": sid,
            "tipo_atto": tipo_atto,
            "cron_expr": cron_expr,
            "enabled": enabled,
            "description": description,
            "last_run_at": None,
            "last_run_status": None,
            "next_run_at": None,
            "created_at": "2026-02-16T00:00:00",
        }
        _store[sid] = entry
        return entry

    async def _list(session):
        return list(_store.values())

    async def _update(session, schedule_id, **kwargs):
        if schedule_id not in _store:
            return None
        _store[schedule_id].update(kwargs)
        return _store[schedule_id]

    async def _remove(session, schedule_id):
        if schedule_id not in _store:
            return False
        del _store[schedule_id]
        return True

    async def _toggle(session, schedule_id):
        if schedule_id not in _store:
            return None
        _store[schedule_id]["enabled"] = not _store[schedule_id]["enabled"]
        return _store[schedule_id]

    scheduler.add_schedule = AsyncMock(side_effect=_add)
    scheduler.list_schedules = AsyncMock(side_effect=_list)
    scheduler.update_schedule = AsyncMock(side_effect=_update)
    scheduler.remove_schedule = AsyncMock(side_effect=_remove)
    scheduler.toggle_schedule = AsyncMock(side_effect=_toggle)

    return scheduler


@pytest.fixture
def app(mock_scheduler):
    """FastAPI test app with mocked auth and scheduler."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")

    # Override auth dependencies
    admin_key = _make_fake_api_key("admin")
    test_app.dependency_overrides[verify_api_key] = lambda: admin_key
    test_app.dependency_overrides[require_role("admin")] = lambda: admin_key

    return test_app


@pytest.fixture
def client(app, mock_scheduler):
    """TestClient with scheduler patched."""
    with patch("merlt.api.schedule_router._get_scheduler", return_value=mock_scheduler):
        yield TestClient(app)


# =============================================================================
# MOCK DB SESSION
# =============================================================================

@pytest.fixture(autouse=True)
def _mock_async_session():
    """Patch get_async_session so router never touches a real DB."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield MagicMock()

    with patch("merlt.api.schedule_router.get_async_session", _fake_session):
        yield


# =============================================================================
# CREATE TESTS
# =============================================================================

class TestCreateSchedule:
    """Test POST /ingestion/schedules."""

    def test_create_schedule_success(self, client):
        """Creating a schedule returns 200 with the new schedule data."""
        response = client.post(
            "/api/v1/ingestion/schedules",
            json={
                "tipo_atto": "codice civile",
                "cron_expr": "0 3 * * *",
                "enabled": True,
                "description": "Nightly civil code sync",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["tipo_atto"] == "codice civile"
        assert data["cron_expr"] == "0 3 * * *"
        assert data["enabled"] is True
        assert data["description"] == "Nightly civil code sync"

    def test_create_schedule_minimal(self, client):
        """Only required fields (tipo_atto, cron_expr) are needed."""
        response = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "costituzione", "cron_expr": "0 0 * * 1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tipo_atto"] == "costituzione"
        assert data["enabled"] is True  # default

    def test_create_schedule_missing_tipo_atto(self, client):
        """Missing required field returns 422."""
        response = client.post(
            "/api/v1/ingestion/schedules",
            json={"cron_expr": "0 3 * * *"},
        )
        assert response.status_code == 422

    def test_create_schedule_empty_tipo_atto(self, client):
        """Empty tipo_atto violates min_length=1 and returns 422."""
        response = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "", "cron_expr": "0 3 * * *"},
        )
        assert response.status_code == 422


# =============================================================================
# LIST TESTS
# =============================================================================

class TestListSchedules:
    """Test GET /ingestion/schedules."""

    def test_list_empty(self, client):
        """Empty list returns count 0."""
        response = client.get("/api/v1/ingestion/schedules")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["schedules"] == []

    def test_list_after_create(self, client):
        """After creating schedules, list returns them."""
        client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *"},
        )
        client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice penale", "cron_expr": "0 4 * * *"},
        )

        response = client.get("/api/v1/ingestion/schedules")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["schedules"]) == 2

        tipo_atti = [s["tipo_atto"] for s in data["schedules"]]
        assert "codice civile" in tipo_atti
        assert "codice penale" in tipo_atti


# =============================================================================
# UPDATE TESTS
# =============================================================================

class TestUpdateSchedule:
    """Test PUT /ingestion/schedules/{id}."""

    def test_update_schedule_success(self, client):
        """Update existing schedule fields."""
        # Create
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *"},
        )
        schedule_id = create_resp.json()["id"]

        # Update
        response = client.put(
            f"/api/v1/ingestion/schedules/{schedule_id}",
            json={"cron_expr": "0 6 * * *", "description": "Updated to 6am"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cron_expr"] == "0 6 * * *"
        assert data["description"] == "Updated to 6am"

    def test_update_schedule_not_found(self, client):
        """Updating nonexistent schedule returns 404."""
        response = client.put(
            "/api/v1/ingestion/schedules/9999",
            json={"cron_expr": "0 6 * * *"},
        )
        assert response.status_code == 404

    def test_update_schedule_no_fields(self, client):
        """Sending empty update body returns 400."""
        # Create
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *"},
        )
        schedule_id = create_resp.json()["id"]

        response = client.put(
            f"/api/v1/ingestion/schedules/{schedule_id}",
            json={},
        )
        assert response.status_code == 400
        assert "No fields" in response.json()["detail"]


# =============================================================================
# DELETE TESTS
# =============================================================================

class TestDeleteSchedule:
    """Test DELETE /ingestion/schedules/{id}."""

    def test_delete_schedule_success(self, client):
        """Delete existing schedule."""
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *"},
        )
        schedule_id = create_resp.json()["id"]

        response = client.delete(f"/api/v1/ingestion/schedules/{schedule_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Schedule deleted"
        assert data["schedule_id"] == schedule_id

        # Verify gone from list
        list_resp = client.get("/api/v1/ingestion/schedules")
        assert list_resp.json()["count"] == 0

    def test_delete_schedule_not_found(self, client):
        """Deleting nonexistent schedule returns 404."""
        response = client.delete("/api/v1/ingestion/schedules/9999")
        assert response.status_code == 404


# =============================================================================
# TOGGLE TESTS
# =============================================================================

class TestToggleSchedule:
    """Test POST /ingestion/schedules/{id}/toggle."""

    def test_toggle_schedule_disables(self, client):
        """Toggle enabled=True -> enabled=False."""
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *", "enabled": True},
        )
        schedule_id = create_resp.json()["id"]

        response = client.post(f"/api/v1/ingestion/schedules/{schedule_id}/toggle")
        assert response.status_code == 200
        assert response.json()["enabled"] is False

    def test_toggle_schedule_enables(self, client):
        """Toggle enabled=False -> enabled=True."""
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *", "enabled": False},
        )
        schedule_id = create_resp.json()["id"]

        response = client.post(f"/api/v1/ingestion/schedules/{schedule_id}/toggle")
        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_toggle_schedule_not_found(self, client):
        """Toggling nonexistent schedule returns 404."""
        response = client.post("/api/v1/ingestion/schedules/9999/toggle")
        assert response.status_code == 404


# =============================================================================
# EDGE CASES
# =============================================================================

class TestScheduleEdgeCases:
    """Edge-case and validation tests."""

    def test_create_two_same_tipo_atto(self, client):
        """Multiple schedules for the same tipo_atto are allowed."""
        r1 = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 3 * * *"},
        )
        r2 = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "codice civile", "cron_expr": "0 6 * * *"},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["id"] != r2.json()["id"]

    def test_crud_lifecycle(self, client):
        """Full lifecycle: create -> read -> update -> toggle -> delete."""
        # Create
        create_resp = client.post(
            "/api/v1/ingestion/schedules",
            json={"tipo_atto": "decreto legislativo", "cron_expr": "30 2 * * 0"},
        )
        assert create_resp.status_code == 200
        sid = create_resp.json()["id"]

        # Read (via list)
        list_resp = client.get("/api/v1/ingestion/schedules")
        assert list_resp.json()["count"] >= 1

        # Update
        update_resp = client.put(
            f"/api/v1/ingestion/schedules/{sid}",
            json={"description": "Weekly DLgs sync"},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["description"] == "Weekly DLgs sync"

        # Toggle off
        toggle_resp = client.post(f"/api/v1/ingestion/schedules/{sid}/toggle")
        assert toggle_resp.status_code == 200
        assert toggle_resp.json()["enabled"] is False

        # Delete
        del_resp = client.delete(f"/api/v1/ingestion/schedules/{sid}")
        assert del_resp.status_code == 200
