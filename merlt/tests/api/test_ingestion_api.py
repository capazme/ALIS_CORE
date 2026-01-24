"""
Test suite per Ingestion API
=============================

Test per:
- Modelli dataclass (ExternalIngestionRequest, IngestionResponse)
- Logica auto-approvazione
- Preview generation
- Endpoint FastAPI

Esempio:
    pytest tests/api/test_ingestion_api.py -v
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# TEST MODELS
# =============================================================================

class TestIngestionModels:
    """Test per dataclass di ingestion."""

    def test_suggested_relation_validation(self):
        """Test validazione SuggestedRelation."""
        from merlt.api.models.ingestion import SuggestedRelation, RelationType

        # Valid relation
        rel = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type=RelationType.RIFERIMENTO,
            evidence="cross_ref",
            confidence=0.8,
        )
        assert rel.confidence == 0.8
        assert rel.relation_type == RelationType.RIFERIMENTO

    def test_suggested_relation_confidence_clamping(self):
        """Test che confidence sia tra 0 e 1."""
        from merlt.api.models.ingestion import SuggestedRelation, RelationType

        # Confidence fuori range deve fallire
        with pytest.raises(ValueError):
            SuggestedRelation(
                source_urn="urn:source",
                target_urn="urn:target",
                relation_type=RelationType.RIFERIMENTO,
                evidence="cross_ref",
                confidence=1.5,  # Invalid
            )

    def test_suggested_relation_string_to_enum(self):
        """Test conversione stringa a enum."""
        from merlt.api.models.ingestion import SuggestedRelation, RelationType

        rel = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type="RIFERIMENTO",  # String instead of enum
            evidence="cross_ref",
            confidence=0.7,
        )
        assert rel.relation_type == RelationType.RIFERIMENTO

    def test_external_ingestion_request_creation(self):
        """Test creazione ExternalIngestionRequest."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
        )

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-123",
            user_authority=0.65,
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.SEARCH_NOT_FOUND,
        )

        assert request.source == "visualex"
        assert request.user_authority == 0.65
        assert request.is_high_authority is False  # < 0.7

    def test_external_ingestion_request_high_authority(self):
        """Test property is_high_authority."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
        )

        # Low authority
        request_low = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-123",
            user_authority=0.5,
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.SEARCH_NOT_FOUND,
        )
        assert request_low.is_high_authority is False

        # High authority
        request_high = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-456",
            user_authority=0.8,
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.SEARCH_NOT_FOUND,
        )
        assert request_high.is_high_authority is True

    def test_ingestion_response_to_dict(self):
        """Test serializzazione IngestionResponse."""
        from merlt.api.models.ingestion import (
            IngestionResponse,
            IngestionStatus,
            GraphPreview,
        )

        response = IngestionResponse(
            success=True,
            status=IngestionStatus.COMPLETED,
            reason="test",
            preview=GraphPreview(),
            article_urn="urn:test",
            nodes_created=["node1", "node2"],
        )

        result = response.to_dict()
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["article_urn"] == "urn:test"
        assert len(result["nodes_created"]) == 2

    def test_graph_preview_properties(self):
        """Test proprietà GraphPreview."""
        from merlt.api.models.ingestion import (
            GraphPreview,
            GraphNodePreview,
            GraphRelationPreview,
        )

        preview = GraphPreview()
        assert preview.total_new_nodes == 0
        assert preview.total_new_relations == 0
        assert preview.has_pending_validations is False

        # Aggiungi nodi e relazioni
        preview.nodes_new.append(
            GraphNodePreview(urn="urn:1", tipo="Norma", label="Art. 1", exists=False)
        )
        preview.relations_pending.append(
            GraphRelationPreview(
                source_urn="urn:1",
                target_urn="urn:2",
                relation_type="RELATED_TO",
                requires_validation=True,
            )
        )

        assert preview.total_new_nodes == 1
        assert preview.has_pending_validations is True


# =============================================================================
# TEST AUTO-APPROVAZIONE LOGIC
# =============================================================================

class TestAutoApprovazione:
    """Test per logica di auto-approvazione."""

    def test_high_authority_auto_approved(self):
        """Utente con authority >= 0.7 è auto-approvato."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-expert",
            user_authority=0.85,  # High authority
            tipo_atto="decreto legislativo",  # Non-standard code
            articolo="100",
            trigger=IngestionTrigger.MANUAL,  # Manual trigger
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "high_authority_user"

    def test_standard_code_search_not_found_auto_approved(self):
        """Articolo da codice standard + search_not_found è auto-approvato."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-student",
            user_authority=0.3,  # Low authority
            tipo_atto="codice civile",  # Standard code
            articolo="1453",
            trigger=IngestionTrigger.SEARCH_NOT_FOUND,
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "official_source_standard_code"

    def test_cross_ref_click_auto_approved(self):
        """Cross-reference click è auto-approvato."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-student",
            user_authority=0.3,  # Low authority
            tipo_atto="legge speciale",  # Non-standard
            articolo="10",
            trigger=IngestionTrigger.CROSS_REF_CLICK,
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "explicit_textual_reference"

    def test_dossier_grouping_requires_validation(self):
        """Dossier grouping richiede validazione community."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-student",
            user_authority=0.6,  # Medium authority
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.DOSSIER_GROUPING,
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.PENDING_VALIDATION
        assert reason == "community_validation_required"

    def test_annotation_requires_validation(self):
        """Annotation richiede validazione community."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-student",
            user_authority=0.5,
            tipo_atto="codice penale",
            articolo="52",
            trigger=IngestionTrigger.ANNOTATION,
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.PENDING_VALIDATION
        assert reason == "community_validation_required"

    def test_default_requires_validation(self):
        """Caso default richiede validazione."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_ingestion_request

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-student",
            user_authority=0.4,  # Low authority
            tipo_atto="decreto ministeriale",  # Non-standard
            articolo="5",
            trigger=IngestionTrigger.MANUAL,
        )

        status, reason = evaluate_ingestion_request(request)
        assert status == IngestionStatus.PENDING_VALIDATION
        assert reason == "default_community_review"


class TestRelationAutoApprovazione:
    """Test per logica auto-approvazione relazioni."""

    def test_high_authority_relation_auto_approved(self):
        """Relazione da utente high authority è auto-approvata."""
        from merlt.api.models.ingestion import (
            SuggestedRelation,
            RelationType,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_relation_suggestion

        relation = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type=RelationType.RELATED_TO,
            evidence="user_annotation",
            confidence=0.6,
        )

        status, reason = evaluate_relation_suggestion(relation, user_authority=0.8)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "high_authority_user"

    def test_cross_ref_evidence_auto_approved(self):
        """Cross-ref esplicito è auto-approvato."""
        from merlt.api.models.ingestion import (
            SuggestedRelation,
            RelationType,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_relation_suggestion

        relation = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type=RelationType.RIFERIMENTO,
            evidence="cross_ref",  # Explicit cross-reference
            confidence=0.9,
        )

        status, reason = evaluate_relation_suggestion(relation, user_authority=0.3)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "explicit_cross_reference"

    def test_high_confidence_extraction_auto_approved(self):
        """Text extraction con alta confidenza è auto-approvata."""
        from merlt.api.models.ingestion import (
            SuggestedRelation,
            RelationType,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_relation_suggestion

        relation = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type=RelationType.CITATO_DA,
            evidence="text_extraction",
            confidence=0.95,  # High confidence
        )

        status, reason = evaluate_relation_suggestion(relation, user_authority=0.4)
        assert status == IngestionStatus.AUTO_APPROVED
        assert reason == "high_confidence_extraction"

    def test_low_confidence_requires_validation(self):
        """Low confidence richiede validazione."""
        from merlt.api.models.ingestion import (
            SuggestedRelation,
            RelationType,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import evaluate_relation_suggestion

        relation = SuggestedRelation(
            source_urn="urn:source",
            target_urn="urn:target",
            relation_type=RelationType.RELATED_TO,
            evidence="dossier_grouping",
            confidence=0.6,
        )

        status, reason = evaluate_relation_suggestion(relation, user_authority=0.5)
        assert status == IngestionStatus.PENDING_VALIDATION
        assert reason == "community_validation_required"


# =============================================================================
# TEST PIPELINE
# =============================================================================

class TestExternalIngestionPipeline:
    """Test per ExternalIngestionPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_dry_run(self):
        """Test dry-run genera solo preview."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import ExternalIngestionPipeline

        # Mock FalkorDB client
        mock_falkordb = AsyncMock()
        mock_falkordb.query = AsyncMock(return_value=[])

        pipeline = ExternalIngestionPipeline(
            falkordb_client=mock_falkordb,
        )

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-123",
            user_authority=0.8,
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.SEARCH_NOT_FOUND,
        )

        response = await pipeline.process(request, dry_run=True)

        assert response.success is True
        assert "[DRY RUN]" in response.reason
        assert response.preview is not None

    @pytest.mark.asyncio
    async def test_pipeline_creates_pending_validation(self):
        """Test che pipeline crei pending validation per trigger appropriati."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
            IngestionStatus,
        )
        from merlt.pipeline.external_ingestion import ExternalIngestionPipeline

        # Mock FalkorDB client
        mock_falkordb = AsyncMock()
        mock_falkordb.query = AsyncMock(return_value=[])

        pipeline = ExternalIngestionPipeline(
            falkordb_client=mock_falkordb,
        )

        request = ExternalIngestionRequest(
            source="visualex",
            user_id="uuid-123",
            user_authority=0.5,  # Low authority
            tipo_atto="codice civile",
            articolo="1337",
            trigger=IngestionTrigger.ANNOTATION,  # Requires validation
        )

        response = await pipeline.process(request)

        assert response.success is True
        assert response.status == IngestionStatus.PENDING_VALIDATION
        assert response.pending_id is not None
        assert response.required_approvals > 0

        # Verifica che sia stata chiamata query per creare PendingValidation
        assert mock_falkordb.query.called


# =============================================================================
# TEST API ENDPOINTS
# =============================================================================

class TestIngestionEndpoints:
    """Test per endpoint FastAPI."""

    @pytest.fixture
    def mock_pipeline(self):
        """Fixture per mock pipeline."""
        from merlt.pipeline.external_ingestion import ExternalIngestionPipeline

        pipeline = MagicMock(spec=ExternalIngestionPipeline)
        pipeline.falkordb = AsyncMock()
        pipeline.falkordb.query = AsyncMock(return_value=[])
        return pipeline

    @pytest.mark.asyncio
    async def test_preview_endpoint(self, mock_pipeline):
        """Test endpoint preview."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from merlt.api.ingestion_api import router, get_pipeline
        from merlt.api.models.ingestion import (
            IngestionResponse,
            IngestionStatus,
            GraphPreview,
        )

        # Setup mock response
        mock_response = IngestionResponse(
            success=True,
            status=IngestionStatus.AUTO_APPROVED,
            reason="[DRY RUN] test",
            preview=GraphPreview(),
        )
        mock_pipeline.process = AsyncMock(return_value=mock_response)

        # Override dependency
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

        client = TestClient(app)

        response = client.post(
            "/ingestion/preview",
            json={
                "source": "visualex",
                "user_id": "uuid-test",
                "user_authority": 0.7,
                "tipo_atto": "codice civile",
                "articolo": "1337",
                "trigger": "search_not_found",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "[DRY RUN]" in data["reason"]

    @pytest.mark.asyncio
    async def test_process_endpoint_validation_error(self, mock_pipeline):
        """Test validazione input endpoint."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from merlt.api.ingestion_api import router, get_pipeline

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

        client = TestClient(app)

        # Invalid authority (> 1.0)
        response = client.post(
            "/ingestion/process",
            json={
                "source": "visualex",
                "user_id": "uuid-test",
                "user_authority": 1.5,  # Invalid
                "tipo_atto": "codice civile",
                "articolo": "1337",
                "trigger": "search_not_found",
            }
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_article_exists_endpoint(self, mock_pipeline):
        """Test endpoint verifica esistenza articolo."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from merlt.api.ingestion_api import router, get_pipeline

        # Mock: articolo non esiste
        mock_pipeline.falkordb.query = AsyncMock(return_value=[])

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

        client = TestClient(app)

        response = client.get("/ingestion/article-exists/codice civile/1337")

        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is False


# =============================================================================
# TEST CONSTANTS
# =============================================================================

class TestConstants:
    """Test per costanti."""

    def test_codici_principali_list(self):
        """Verifica lista codici principali."""
        from merlt.api.models.ingestion import CODICI_PRINCIPALI

        assert "codice civile" in CODICI_PRINCIPALI
        assert "codice penale" in CODICI_PRINCIPALI
        assert "costituzione" in CODICI_PRINCIPALI
        assert "codice di procedura civile" in CODICI_PRINCIPALI

    def test_authority_threshold(self):
        """Verifica threshold authority."""
        from merlt.api.models.ingestion import AUTHORITY_AUTO_APPROVE_THRESHOLD

        assert AUTHORITY_AUTO_APPROVE_THRESHOLD == 0.7

    def test_default_approvals(self):
        """Verifica default approvazioni richieste."""
        from merlt.api.models.ingestion import DEFAULT_REQUIRED_APPROVALS

        assert DEFAULT_REQUIRED_APPROVALS == 3


# =============================================================================
# TEST INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Test di integrazione (richiedono FalkorDB running)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Richiede FalkorDB running")
    async def test_full_ingestion_flow(self):
        """Test flusso completo ingestion."""
        from merlt.api.models.ingestion import (
            ExternalIngestionRequest,
            IngestionTrigger,
        )
        from merlt.pipeline.external_ingestion import ExternalIngestionPipeline
        from merlt.storage.graph.client import FalkorDBClient
        from merlt.storage.graph.config import FalkorDBConfig

        # Connect to real FalkorDB
        config = FalkorDBConfig()
        client = FalkorDBClient(config)
        await client.connect()

        try:
            pipeline = ExternalIngestionPipeline(
                falkordb_client=client,
            )

            request = ExternalIngestionRequest(
                source="test",
                user_id="test-user",
                user_authority=0.8,
                tipo_atto="codice civile",
                articolo="9999",  # Test article
                trigger=IngestionTrigger.MANUAL,
            )

            # Dry run first
            preview_response = await pipeline.process(request, dry_run=True)
            assert preview_response.success is True

        finally:
            await client.close()
