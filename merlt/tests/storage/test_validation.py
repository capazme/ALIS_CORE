"""
Test suite per KG Validation Module
====================================

Test per:
- PendingValidation dataclass
- ValidationVote dataclass
- ValidationService CRUD
- Voting workflow

Esempio:
    pytest tests/storage/test_validation.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any


# =============================================================================
# TEST DATACLASS
# =============================================================================

class TestPendingValidation:
    """Test per PendingValidation dataclass."""

    def test_pending_validation_creation(self):
        """Test creazione PendingValidation."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()
        expires = now + timedelta(days=7)

        pending = PendingValidation(
            id="test-id",
            type=ValidationType.INGESTION,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.6,
            source="visualex",
            trigger="search_not_found",
            proposed_data={"key": "value"},
            created_at=now,
            expires_at=expires,
            approvals=0.0,
            rejections=0.0,
            required_approvals=2.0,
            status=ValidationStatus.PENDING,
        )

        assert pending.id == "test-id"
        assert pending.type == ValidationType.INGESTION
        assert pending.contributor_authority == 0.6
        assert pending.is_approved is False
        assert pending.is_rejected is False

    def test_pending_is_approved(self):
        """Test property is_approved."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()
        expires = now + timedelta(days=7)

        # Not approved
        pending = PendingValidation(
            id="test-id",
            type=ValidationType.INGESTION,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.6,
            source="visualex",
            trigger="search_not_found",
            proposed_data={},
            created_at=now,
            expires_at=expires,
            approvals=1.5,
            required_approvals=2.0,
        )
        assert pending.is_approved is False

        # Approved
        pending.approvals = 2.0
        assert pending.is_approved is True

        # More than required
        pending.approvals = 3.0
        assert pending.is_approved is True

    def test_pending_is_rejected(self):
        """Test property is_rejected."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()
        expires = now + timedelta(days=7)

        pending = PendingValidation(
            id="test-id",
            type=ValidationType.RELATION,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.5,
            source="manual",
            trigger="annotation",
            proposed_data={},
            created_at=now,
            expires_at=expires,
            rejections=2.5,
            required_approvals=2.0,
        )

        assert pending.is_rejected is True

    def test_pending_is_expired(self):
        """Test property is_expired."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()

        # Not expired
        pending_valid = PendingValidation(
            id="test-id",
            type=ValidationType.INGESTION,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.5,
            source="visualex",
            trigger="search_not_found",
            proposed_data={},
            created_at=now,
            expires_at=now + timedelta(days=7),
        )
        assert pending_valid.is_expired is False

        # Expired
        pending_expired = PendingValidation(
            id="test-id",
            type=ValidationType.INGESTION,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.5,
            source="visualex",
            trigger="search_not_found",
            proposed_data={},
            created_at=now - timedelta(days=10),
            expires_at=now - timedelta(days=3),
        )
        assert pending_expired.is_expired is True

    def test_pending_to_dict(self):
        """Test serializzazione to_dict."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()
        expires = now + timedelta(days=7)

        pending = PendingValidation(
            id="test-id",
            type=ValidationType.ENRICHMENT,
            target_urn="urn:test",
            contributor_id="user-123",
            contributor_authority=0.7,
            source="manual",
            trigger="annotation",
            proposed_data={"concept": "test concept"},
            created_at=now,
            expires_at=expires,
        )

        result = pending.to_dict()

        assert result["id"] == "test-id"
        assert result["type"] == "enrichment"
        assert result["proposed_data"]["concept"] == "test concept"
        assert result["status"] == "pending"

    def test_pending_from_dict(self):
        """Test deserializzazione from_dict."""
        from merlt.storage.graph.validation import (
            PendingValidation,
            ValidationType,
            ValidationStatus,
        )

        now = datetime.now()
        expires = now + timedelta(days=7)

        data = {
            "id": "test-id",
            "type": "concept",
            "target_urn": "urn:test",
            "contributor_id": "user-123",
            "contributor_authority": 0.5,
            "source": "visualex",
            "trigger": "manual",
            "proposed_data": {"key": "value"},
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "approvals": 1.0,
            "rejections": 0.5,
            "required_approvals": 2.0,
            "status": "pending",
        }

        pending = PendingValidation.from_dict(data)

        assert pending.id == "test-id"
        assert pending.type == ValidationType.CONCEPT
        assert pending.approvals == 1.0
        assert pending.rejections == 0.5


class TestValidationVote:
    """Test per ValidationVote dataclass."""

    def test_vote_creation(self):
        """Test creazione ValidationVote."""
        from merlt.storage.graph.validation import ValidationVote

        now = datetime.now()

        vote = ValidationVote(
            id="vote-id",
            pending_id="pending-id",
            voter_id="voter-123",
            voter_authority=0.8,
            vote=True,
            reason="Looks correct",
            created_at=now,
        )

        assert vote.voter_authority == 0.8
        assert vote.vote is True
        assert vote.weight == 0.8

    def test_vote_to_dict(self):
        """Test serializzazione vote."""
        from merlt.storage.graph.validation import ValidationVote

        now = datetime.now()

        vote = ValidationVote(
            id="vote-id",
            pending_id="pending-id",
            voter_id="voter-123",
            voter_authority=0.6,
            vote=False,
            reason="Incorrect relation",
            created_at=now,
        )

        result = vote.to_dict()

        assert result["id"] == "vote-id"
        assert result["vote"] is False
        assert result["reason"] == "Incorrect relation"


# =============================================================================
# TEST VALIDATION SERVICE
# =============================================================================

class TestValidationService:
    """Test per ValidationService."""

    @pytest.fixture
    def mock_client(self):
        """Fixture per mock FalkorDB client."""
        client = AsyncMock()
        client.query = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Fixture per ValidationService."""
        from merlt.storage.graph.validation import ValidationService
        return ValidationService(mock_client)

    @pytest.mark.asyncio
    async def test_create_pending(self, service, mock_client):
        """Test creazione pending validation."""
        from merlt.storage.graph.validation import ValidationType

        pending_id = await service.create_pending(
            type=ValidationType.INGESTION,
            target_urn="urn:test:article",
            contributor_id="user-123",
            contributor_authority=0.6,
            source="visualex",
            trigger="search_not_found",
            proposed_data={"key": "value"},
        )

        assert pending_id is not None
        assert len(pending_id) == 36  # UUID length with dashes

        # Verify query was called with CREATE
        assert mock_client.query.called
        call_args = mock_client.query.call_args
        assert "CREATE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_pending_not_found(self, service, mock_client):
        """Test get_pending quando non esiste."""
        mock_client.query = AsyncMock(return_value=[])

        result = await service.get_pending("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_found(self, service, mock_client):
        """Test get_pending quando esiste."""
        from merlt.storage.graph.validation import ValidationType, ValidationStatus

        now = datetime.now()
        expires = now + timedelta(days=7)

        mock_client.query = AsyncMock(return_value=[{
            "p": {
                "properties": {
                    "id": "test-id",
                    "type": "ingestion",
                    "target_urn": "urn:test",
                    "contributor_id": "user-123",
                    "contributor_authority": 0.6,
                    "source": "visualex",
                    "trigger": "search_not_found",
                    "proposed_data": "{}",
                    "created_at": now.isoformat(),
                    "expires_at": expires.isoformat(),
                    "approvals": 1.0,
                    "rejections": 0.0,
                    "required_approvals": 2.0,
                    "status": "pending",
                }
            }
        }])

        result = await service.get_pending("test-id")

        assert result is not None
        assert result.id == "test-id"
        assert result.type == ValidationType.INGESTION
        assert result.approvals == 1.0

    @pytest.mark.asyncio
    async def test_list_pending_empty(self, service, mock_client):
        """Test list_pending senza risultati."""
        mock_client.query = AsyncMock(return_value=[])

        result = await service.list_pending()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_pending_with_results(self, service, mock_client):
        """Test list_pending con risultati."""
        now = datetime.now()
        expires = now + timedelta(days=7)

        mock_client.query = AsyncMock(return_value=[
            {
                "p": {
                    "properties": {
                        "id": "pending-1",
                        "type": "ingestion",
                        "target_urn": "urn:test1",
                        "contributor_id": "user-1",
                        "contributor_authority": 0.5,
                        "source": "visualex",
                        "trigger": "search_not_found",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            },
            {
                "p": {
                    "properties": {
                        "id": "pending-2",
                        "type": "relation",
                        "target_urn": "urn:test2",
                        "contributor_id": "user-2",
                        "contributor_authority": 0.7,
                        "source": "manual",
                        "trigger": "annotation",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            },
        ])

        result = await service.list_pending()

        assert len(result) == 2
        assert result[0].id == "pending-1"
        assert result[1].id == "pending-2"


class TestValidationVoting:
    """Test per voting workflow."""

    @pytest.fixture
    def mock_client(self):
        """Fixture per mock FalkorDB client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Fixture per ValidationService."""
        from merlt.storage.graph.validation import ValidationService
        return ValidationService(mock_client)

    @pytest.mark.asyncio
    async def test_add_vote_pending_not_found(self, service, mock_client):
        """Test voto su pending inesistente."""
        mock_client.query = AsyncMock(return_value=[])

        success, message, new_status = await service.add_vote(
            pending_id="nonexistent",
            voter_id="voter-123",
            voter_authority=0.8,
            vote=True,
        )

        assert success is False
        assert "not found" in message

    @pytest.mark.asyncio
    async def test_add_vote_already_voted(self, service, mock_client):
        """Test voto duplicato."""
        from merlt.storage.graph.validation import ValidationStatus

        now = datetime.now()
        expires = now + timedelta(days=7)

        # Prima query: get_pending restituisce pending valido
        # Seconda query: check voto esistente
        mock_client.query = AsyncMock(side_effect=[
            # get_pending
            [{
                "p": {
                    "properties": {
                        "id": "test-id",
                        "type": "ingestion",
                        "target_urn": "urn:test",
                        "contributor_id": "user-123",
                        "contributor_authority": 0.6,
                        "source": "visualex",
                        "trigger": "search_not_found",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            }],
            # check existing vote - vote exists
            [{"v.id": "existing-vote"}],
        ])

        success, message, new_status = await service.add_vote(
            pending_id="test-id",
            voter_id="voter-123",
            voter_authority=0.8,
            vote=True,
        )

        assert success is False
        assert "already voted" in message

    @pytest.mark.asyncio
    async def test_add_vote_success_no_threshold(self, service, mock_client):
        """Test voto valido che non raggiunge threshold."""
        now = datetime.now()
        expires = now + timedelta(days=7)

        mock_client.query = AsyncMock(side_effect=[
            # get_pending
            [{
                "p": {
                    "properties": {
                        "id": "test-id",
                        "type": "ingestion",
                        "target_urn": "urn:test",
                        "contributor_id": "user-123",
                        "contributor_authority": 0.6,
                        "source": "visualex",
                        "trigger": "search_not_found",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            }],
            # check existing vote - no existing vote
            [],
            # create vote
            [],
            # update approvals
            [],
            # check threshold
            [{"approvals": 0.8, "rejections": 0.0, "required": 2.0}],
        ])

        success, message, new_status = await service.add_vote(
            pending_id="test-id",
            voter_id="voter-123",
            voter_authority=0.8,
            vote=True,
        )

        assert success is True
        assert new_status is None  # Threshold not reached

    @pytest.mark.asyncio
    async def test_add_vote_triggers_approval(self, service, mock_client):
        """Test voto che raggiunge threshold approvazione."""
        from merlt.storage.graph.validation import ValidationStatus

        now = datetime.now()
        expires = now + timedelta(days=7)

        mock_client.query = AsyncMock(side_effect=[
            # get_pending
            [{
                "p": {
                    "properties": {
                        "id": "test-id",
                        "type": "ingestion",
                        "target_urn": "urn:test",
                        "contributor_id": "user-123",
                        "contributor_authority": 0.6,
                        "source": "visualex",
                        "trigger": "search_not_found",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            }],
            # check existing vote
            [],
            # create vote
            [],
            # update approvals
            [],
            # check threshold - APPROVED
            [{"approvals": 2.5, "rejections": 0.0, "required": 2.0}],
            # update status
            [],
        ])

        success, message, new_status = await service.add_vote(
            pending_id="test-id",
            voter_id="voter-123",
            voter_authority=0.9,
            vote=True,
        )

        assert success is True
        assert new_status == ValidationStatus.APPROVED

    @pytest.mark.asyncio
    async def test_add_vote_triggers_rejection(self, service, mock_client):
        """Test voto che raggiunge threshold rifiuto."""
        from merlt.storage.graph.validation import ValidationStatus

        now = datetime.now()
        expires = now + timedelta(days=7)

        mock_client.query = AsyncMock(side_effect=[
            # get_pending
            [{
                "p": {
                    "properties": {
                        "id": "test-id",
                        "type": "relation",
                        "target_urn": "urn:test",
                        "contributor_id": "user-123",
                        "contributor_authority": 0.5,
                        "source": "manual",
                        "trigger": "annotation",
                        "proposed_data": "{}",
                        "created_at": now.isoformat(),
                        "expires_at": expires.isoformat(),
                        "status": "pending",
                    }
                }
            }],
            # check existing vote
            [],
            # create vote
            [],
            # update rejections
            [],
            # check threshold - REJECTED
            [{"approvals": 0.3, "rejections": 2.1, "required": 2.0}],
            # update status
            [],
        ])

        success, message, new_status = await service.add_vote(
            pending_id="test-id",
            voter_id="voter-456",
            voter_authority=0.7,
            vote=False,  # Rejection
        )

        assert success is True
        assert new_status == ValidationStatus.REJECTED


class TestValidationMaintenance:
    """Test per maintenance operations."""

    @pytest.fixture
    def mock_client(self):
        """Fixture per mock FalkorDB client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Fixture per ValidationService."""
        from merlt.storage.graph.validation import ValidationService
        return ValidationService(mock_client)

    @pytest.mark.asyncio
    async def test_expire_old_validations(self, service, mock_client):
        """Test scadenza validazioni vecchie."""
        mock_client.query = AsyncMock(return_value=[{"count": 5}])

        count = await service.expire_old_validations()

        assert count == 5
        assert mock_client.query.called

    @pytest.mark.asyncio
    async def test_cleanup_old_votes(self, service, mock_client):
        """Test pulizia voti vecchi."""
        mock_client.query = AsyncMock(return_value=[{"count": 10}])

        count = await service.cleanup_old_votes(days=90)

        assert count == 10

    @pytest.mark.asyncio
    async def test_get_stats(self, service, mock_client):
        """Test statistiche."""
        mock_client.query = AsyncMock(side_effect=[
            # by_status query
            [{
                "by_status": [
                    {"status": "pending", "count": 10},
                    {"status": "approved", "count": 25},
                    {"status": "rejected", "count": 5},
                ]
            }],
            # total_votes query
            [{"total_votes": 100}],
        ])

        stats = await service.get_stats()

        assert stats["total_pending"] == 10
        assert stats["total_approved"] == 25
        assert stats["total_rejected"] == 5
        assert stats["total_votes"] == 100
