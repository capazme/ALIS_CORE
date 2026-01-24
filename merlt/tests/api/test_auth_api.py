"""
Test suite per Auth API
=======================

Test per:
- Endpoint /auth/sync
- Endpoint /auth/authority/{user_id}
- Endpoint /auth/delta
- Endpoint /auth/estimate
- Endpoint /auth/qualifications

Esempio:
    pytest tests/api/test_auth_api.py -v
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def app():
    """Create FastAPI app with auth router."""
    from merlt.api.auth_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# TEST SYNC ENDPOINT
# =============================================================================

class TestSyncEndpoint:
    """Test per /auth/sync."""

    def test_sync_basic_user(self, client):
        """Test sync utente base."""
        response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "visualex-123",
                "merlt_user_id": "merl-t-456",
                "qualification": "avvocato",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["user_id"] == "merl-t-456"
        assert "authority" in data
        assert "breakdown" in data
        assert "synced_at" in data

        # Authority = 0.4 * 0.6 = 0.24 (solo baseline)
        assert data["authority"] == pytest.approx(0.24, abs=0.01)

    def test_sync_experienced_user(self, client):
        """Test sync utente con esperienza."""
        response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "visualex-456",
                "merlt_user_id": "merl-t-789",
                "qualification": "avvocato",
                "years_experience": 10,
                "specializations": ["civile"],
                "total_feedback": 20,
                "validated_feedback": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        # Higher authority due to experience and track record
        assert data["authority"] > 0.24

        breakdown = data["breakdown"]
        assert breakdown["baseline"] > 0.6  # Has years bonus
        assert breakdown["track_record"] > 0  # Has feedback

    def test_sync_expert_user(self, client):
        """Test sync utente esperto."""
        response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "visualex-expert",
                "merlt_user_id": "merl-t-expert",
                "qualification": "magistrato",
                "years_experience": 15,
                "specializations": ["civile", "penale"],
                "institution": "Corte di Cassazione",
                "total_feedback": 50,
                "validated_feedback": 30,
                "ingestions": 10,
                "validations": 100,
                "domain_activity": {"civile": 80, "penale": 40},
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        # High authority
        assert data["authority"] > 0.6

        breakdown = data["breakdown"]
        assert "civile" in breakdown["domain_scores"]
        assert "penale" in breakdown["domain_scores"]

    def test_sync_missing_required_fields(self, client):
        """Test sync con campi mancanti."""
        response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "test",
                # Missing merlt_user_id and qualification
            },
        )

        assert response.status_code == 422  # Validation error

    def test_sync_invalid_qualification(self, client):
        """Test sync con qualifica sconosciuta (usa default)."""
        response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "test",
                "merlt_user_id": "test",
                "qualification": "unknown_qualification",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Uses default baseline 0.3
        # 0.3 * 0.4 = 0.12
        assert data["authority"] == pytest.approx(0.12, abs=0.01)


# =============================================================================
# TEST AUTHORITY ENDPOINT
# =============================================================================

class TestAuthorityEndpoint:
    """Test per /auth/authority/{user_id}."""

    def test_get_authority_not_found(self, client):
        """Test get authority utente non trovato."""
        response = client.get("/api/v1/auth/authority/nonexistent-user")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["found"] is False
        assert data["authority"] is None

    def test_get_authority_after_sync(self, client):
        """Test get authority dopo sync."""
        # Prima sync
        client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "vis-test",
                "merlt_user_id": "cached-user-123",
                "qualification": "avvocato",
            },
        )

        # Poi get
        response = client.get("/api/v1/auth/authority/cached-user-123")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["found"] is True
        assert data["authority"] is not None
        assert data["authority"] == pytest.approx(0.24, abs=0.01)


# =============================================================================
# TEST DELTA ENDPOINT
# =============================================================================

class TestDeltaEndpoint:
    """Test per /auth/delta."""

    def test_apply_delta_feedback_simple(self, client):
        """Test delta per feedback semplice."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "feedback_simple",
                "current_authority": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["delta"] == 0.001
        assert data["new_authority"] == 0.501

    def test_apply_delta_feedback_detailed(self, client):
        """Test delta per feedback dettagliato."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "feedback_detailed",
                "current_authority": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["delta"] == 0.005
        assert data["new_authority"] == 0.505

    def test_apply_delta_ingestion_approved(self, client):
        """Test delta per ingestion approvata."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "ingestion_approved",
                "current_authority": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["delta"] == 0.01
        assert data["new_authority"] == 0.51

    def test_apply_delta_validation_incorrect(self, client):
        """Test delta negativo per validazione incorretta."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "validation_incorrect",
                "current_authority": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["delta"] == -0.002
        assert data["new_authority"] == 0.498

    def test_apply_delta_unknown_action(self, client):
        """Test delta per azione sconosciuta."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "unknown_action",
                "current_authority": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["delta"] == 0
        assert data["new_authority"] == 0.5

    def test_apply_delta_diminishing_returns(self, client):
        """Test diminishing returns per authority alta."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "feedback_detailed",
                "current_authority": 0.85,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 0.005 * 0.5 = 0.0025
        assert data["delta"] == 0.0025

    def test_apply_delta_very_high_authority(self, client):
        """Test diminishing returns per authority molto alta."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "feedback_detailed",
                "current_authority": 0.95,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 0.005 * 0.25 = 0.00125
        assert data["delta"] == 0.00125

    def test_apply_delta_clamped(self, client):
        """Test che new_authority sia clamped a 1.0."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "ingestion_approved",
                "current_authority": 0.995,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Clamped to 1.0
        assert data["new_authority"] <= 1.0

    def test_apply_delta_invalid_authority(self, client):
        """Test con authority fuori range."""
        response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "test-user",
                "action": "feedback_simple",
                "current_authority": 1.5,  # Invalid
            },
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# TEST ESTIMATE ENDPOINT
# =============================================================================

class TestEstimateEndpoint:
    """Test per /auth/estimate."""

    def test_estimate_studente(self, client):
        """Test stima per studente."""
        response = client.post(
            "/api/v1/auth/estimate",
            json={
                "qualification": "studente",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 0.2 * 0.4 = 0.08
        assert data["estimated_authority"] == pytest.approx(0.08, abs=0.01)
        assert data["qualification"] == "studente"

    def test_estimate_avvocato(self, client):
        """Test stima per avvocato."""
        response = client.post(
            "/api/v1/auth/estimate",
            json={
                "qualification": "avvocato",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 0.6 * 0.4 = 0.24
        assert data["estimated_authority"] == pytest.approx(0.24, abs=0.01)

    def test_estimate_with_experience(self, client):
        """Test stima con esperienza."""
        response = client.post(
            "/api/v1/auth/estimate",
            json={
                "qualification": "avvocato",
                "years_experience": 10,
                "specializations": ["civile"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # (0.6 + 0.1 + 0.025) * 0.4 = 0.29
        assert data["estimated_authority"] == pytest.approx(0.29, abs=0.01)

    def test_estimate_magistrato(self, client):
        """Test stima per magistrato."""
        response = client.post(
            "/api/v1/auth/estimate",
            json={
                "qualification": "magistrato",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 0.8 * 0.4 = 0.32
        assert data["estimated_authority"] == pytest.approx(0.32, abs=0.01)


# =============================================================================
# TEST QUALIFICATIONS ENDPOINT
# =============================================================================

class TestQualificationsEndpoint:
    """Test per /auth/qualifications."""

    def test_list_qualifications(self, client):
        """Test lista qualifiche."""
        response = client.get("/api/v1/auth/qualifications")

        assert response.status_code == 200
        data = response.json()

        assert "qualifications" in data
        assert "weights" in data
        assert "formula" in data

        # Verifica qualifiche presenti
        qualifications = data["qualifications"]
        assert "studente" in qualifications
        assert "avvocato" in qualifications
        assert "magistrato" in qualifications

        # Verifica weights
        weights = data["weights"]
        assert weights["baseline"] == 0.4
        assert weights["track_record"] == 0.4
        assert weights["level_authority"] == 0.2

        # Somma pesi = 1.0
        assert sum(weights.values()) == 1.0

    def test_qualifications_values(self, client):
        """Test valori qualifiche."""
        response = client.get("/api/v1/auth/qualifications")
        data = response.json()

        qualifications = data["qualifications"]

        # Verifica ordinamento (studente < avvocato < magistrato)
        assert qualifications["studente"] < qualifications["avvocato"]
        assert qualifications["avvocato"] < qualifications["magistrato"]


# =============================================================================
# TEST INTEGRATION
# =============================================================================

class TestIntegration:
    """Test integrazione workflow completo."""

    def test_full_workflow(self, client):
        """Test workflow completo: sync -> delta -> get."""
        # 1. Sync nuovo utente
        sync_response = client.post(
            "/api/v1/auth/sync",
            json={
                "visualex_user_id": "workflow-test-vis",
                "merlt_user_id": "workflow-test-user",
                "qualification": "avvocato",
                "total_feedback": 10,
            },
        )

        assert sync_response.status_code == 200
        initial_authority = sync_response.json()["authority"]

        # 2. Applica delta
        delta_response = client.post(
            "/api/v1/auth/delta",
            json={
                "user_id": "workflow-test-user",
                "action": "feedback_detailed",
                "current_authority": initial_authority,
            },
        )

        assert delta_response.status_code == 200
        new_authority = delta_response.json()["new_authority"]
        assert new_authority > initial_authority

        # 3. Verifica cache aggiornato
        get_response = client.get("/api/v1/auth/authority/workflow-test-user")

        assert get_response.status_code == 200
        # Cache was updated by delta
        assert get_response.json()["authority"] == new_authority

    def test_multiple_deltas(self, client):
        """Test applicazione multipli delta."""
        current = 0.5

        actions = [
            ("feedback_simple", 0.001),
            ("feedback_detailed", 0.005),
            ("validation_correct", 0.003),
            ("ingestion_approved", 0.01),
        ]

        for action, expected_delta in actions:
            response = client.post(
                "/api/v1/auth/delta",
                json={
                    "user_id": "multi-delta-user",
                    "action": action,
                    "current_authority": current,
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["delta"] == expected_delta
            current = data["new_authority"]

        # Final authority should be higher
        assert current > 0.5
