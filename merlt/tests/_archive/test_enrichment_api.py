"""
Test per Enrichment API
=======================

Test degli endpoint di live enrichment e validazione granulare.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.api.models.enrichment_models import (
    EntityType,
    LiveEnrichmentRequest,
    LiveEnrichmentResponse,
    EntityValidationRequest,
    EntityValidationResponse,
    EntityProposalRequest,
    PendingEntityData,
    PendingQueueRequest,
    ValidationStatus,
    VoteType,
)
from merlt.rlcf.entity_feedback import (
    EntityValidationFeedback,
    EntityValidationAggregator,
    RelationValidationFeedback,
    RelationValidationAggregator,
    AuthorityImpactCalculator,
)
from merlt.pipeline.enrichment.models import RelationType


# =============================================================================
# TEST ENTITY VALIDATION FEEDBACK
# =============================================================================

class TestEntityValidationFeedback:
    """Test per EntityValidationFeedback."""

    def test_weighted_vote_approve(self):
        """Test voto approve pesato per authority."""
        feedback = EntityValidationFeedback(
            entity_id="test-entity",
            entity_type=EntityType.CONCETTO,
            vote="approve",
            user_id="user-1",
            user_authority=0.8,
        )
        assert feedback.weighted_vote == 0.8
        assert feedback.is_positive is True

    def test_weighted_vote_reject(self):
        """Test voto reject pesato per authority."""
        feedback = EntityValidationFeedback(
            entity_id="test-entity",
            entity_type=EntityType.PRINCIPIO,
            vote="reject",
            user_id="user-1",
            user_authority=0.6,
        )
        assert feedback.weighted_vote == -0.6
        assert feedback.is_positive is False

    def test_weighted_vote_edit(self):
        """Test voto edit (approvazione parziale)."""
        feedback = EntityValidationFeedback(
            entity_id="test-entity",
            entity_type=EntityType.DEFINIZIONE,
            vote="edit",
            user_id="user-1",
            user_authority=0.8,
            suggested_edits={"descrizione": "nuova descrizione"},
        )
        assert feedback.weighted_vote == 0.4  # 0.8 * 0.5
        assert feedback.is_positive is True

    def test_to_dict(self):
        """Test serializzazione in dict."""
        feedback = EntityValidationFeedback(
            entity_id="test-entity",
            entity_type=EntityType.CONCETTO,
            vote="approve",
            user_id="user-1",
            user_authority=0.8,
        )
        data = feedback.to_dict()
        assert data["entity_id"] == "test-entity"
        assert data["entity_type"] == "concetto"
        assert data["vote"] == "approve"
        assert data["weighted_vote"] == 0.8


# =============================================================================
# TEST ENTITY VALIDATION AGGREGATOR
# =============================================================================

class TestEntityValidationAggregator:
    """Test per EntityValidationAggregator."""

    def test_aggregate_approved(self):
        """Test aggregazione con threshold raggiunto."""
        aggregator = EntityValidationAggregator(approval_threshold=2.0)

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.9,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.8,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.5,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        # 0.9 + 0.8 + 0.5 = 2.2 >= 2.0
        assert result.status == ValidationStatus.APPROVED
        assert result.score == pytest.approx(2.2)

    def test_aggregate_rejected(self):
        """Test aggregazione con rifiuto."""
        aggregator = EntityValidationAggregator(rejection_threshold=-2.0)

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="reject",
                user_authority=0.9,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="reject",
                user_authority=0.8,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="reject",
                user_authority=0.5,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        # -0.9 - 0.8 - 0.5 = -2.2 <= -2.0
        assert result.status == ValidationStatus.REJECTED
        assert result.score == pytest.approx(-2.2)

    def test_aggregate_pending(self):
        """Test aggregazione con voti insufficienti."""
        aggregator = EntityValidationAggregator(approval_threshold=2.0)

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.5,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        # 0.5 < 2.0
        assert result.status == ValidationStatus.PENDING
        assert result.score == 0.5

    def test_aggregate_mixed_votes(self):
        """Test aggregazione con voti misti."""
        aggregator = EntityValidationAggregator(
            approval_threshold=2.0,
            rejection_threshold=-2.0,
        )

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.9,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="reject",
                user_authority=0.5,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.8,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        # 0.9 - 0.5 + 0.8 = 1.2 (pending)
        assert result.status == ValidationStatus.PENDING
        assert result.score == pytest.approx(1.2)

    def test_aggregate_with_edits(self):
        """Test aggregazione con voti edit."""
        aggregator = EntityValidationAggregator(approval_threshold=2.0)

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.9,
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="edit",
                user_authority=0.8,
                suggested_edits={"descrizione": "nuova desc"},
            ),
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.9,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        # 0.9 + 0.4 (edit) + 0.9 = 2.2 >= 2.0
        assert result.status == ValidationStatus.APPROVED
        assert result.merged_edits == {"descrizione": "nuova desc"}

    def test_aggregate_empty(self):
        """Test aggregazione lista vuota."""
        aggregator = EntityValidationAggregator()
        result = aggregator.aggregate([])
        assert result.status == ValidationStatus.PENDING
        assert result.score == 0.0

    def test_calculate_required_votes(self):
        """Test calcolo voti necessari."""
        aggregator = EntityValidationAggregator(approval_threshold=2.0)

        # Authority media 0.5 -> servono ~5 voti
        votes = aggregator.calculate_required_votes(average_authority=0.5)
        assert votes == 5

        # Authority media 0.8 -> servono ~3 voti
        votes = aggregator.calculate_required_votes(average_authority=0.8)
        assert votes == 3

    def test_get_progress(self):
        """Test calcolo progresso."""
        aggregator = EntityValidationAggregator(approval_threshold=2.0)

        feedbacks = [
            EntityValidationFeedback(
                entity_id="e1",
                entity_type=EntityType.CONCETTO,
                vote="approve",
                user_authority=0.5,
            ),
        ]

        progress = aggregator.get_progress(feedbacks)
        assert progress["total_score"] == 0.5
        assert progress["positive_votes"] == 1
        assert progress["negative_votes"] == 0
        assert progress["progress"] == 0.25  # 0.5/2.0
        assert progress["direction"] == "approval"


# =============================================================================
# TEST AUTHORITY IMPACT CALCULATOR
# =============================================================================

class TestAuthorityImpactCalculator:
    """Test per AuthorityImpactCalculator."""

    def test_contributor_delta_approved_entity(self):
        """Test delta contributor per entita' approvata."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_contributor_delta("entity", ValidationStatus.APPROVED)
        assert delta == 0.02

    def test_contributor_delta_approved_relation(self):
        """Test delta contributor per relazione approvata."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_contributor_delta("relation", ValidationStatus.APPROVED)
        assert delta == 0.01

    def test_contributor_delta_rejected(self):
        """Test nessuna penalita' per rifiuto."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_contributor_delta("entity", ValidationStatus.REJECTED)
        assert delta == 0.0

    def test_voter_delta_correct_approve(self):
        """Test reward per voto corretto (approve -> approved)."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_voter_delta("approve", ValidationStatus.APPROVED)
        assert delta == 0.005

    def test_voter_delta_wrong_approve(self):
        """Test penalita' per voto sbagliato (approve -> rejected)."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_voter_delta("approve", ValidationStatus.REJECTED)
        assert delta == -0.002

    def test_voter_delta_correct_reject(self):
        """Test reward per voto corretto (reject -> rejected)."""
        calc = AuthorityImpactCalculator()
        delta = calc.calculate_voter_delta("reject", ValidationStatus.REJECTED)
        assert delta == 0.005


# =============================================================================
# TEST PENDING ENTITY DATA
# =============================================================================

class TestPendingEntityData:
    """Test per PendingEntityData."""

    def test_create_pending_entity(self):
        """Test creazione PendingEntityData."""
        entity = PendingEntityData(
            id="pending:abc123",
            nome="Buona fede",
            tipo=EntityType.CONCETTO,
            descrizione="Principio di lealtà nelle trattative",
            articoli_correlati=["urn:nir:stato:codice.civile:1942:art:1337"],
            ambito="obbligazioni",
            fonte="brocardi",
            llm_confidence=0.85,
            raw_context="Art. 1337 c.c.",
            contributed_by="user-123",
            contributor_authority=0.7,
        )

        assert entity.id == "pending:abc123"
        assert entity.tipo == EntityType.CONCETTO
        assert entity.validation_status == ValidationStatus.PENDING
        assert entity.approval_score == 0.0
        assert entity.llm_confidence == 0.85


# =============================================================================
# TEST RELATION VALIDATION
# =============================================================================

class TestRelationValidation:
    """Test per validazione relazioni."""

    def test_relation_validation_feedback(self):
        """Test RelationValidationFeedback."""
        feedback = RelationValidationFeedback(
            relation_id="rel-1",
            relation_type=RelationType.DISCIPLINA,
            vote="approve",
            user_id="user-1",
            user_authority=0.7,
        )

        assert feedback.weighted_vote == 0.7
        assert feedback.is_positive is True

    def test_relation_aggregator(self):
        """Test RelationValidationAggregator."""
        aggregator = RelationValidationAggregator(approval_threshold=2.0)

        feedbacks = [
            RelationValidationFeedback(
                relation_id="rel-1",
                relation_type=RelationType.DISCIPLINA,
                vote="approve",
                user_authority=0.9,
            ),
            RelationValidationFeedback(
                relation_id="rel-1",
                relation_type=RelationType.DISCIPLINA,
                vote="approve",
                user_authority=0.8,
            ),
            RelationValidationFeedback(
                relation_id="rel-1",
                relation_type=RelationType.DISCIPLINA,
                vote="approve",
                user_authority=0.5,
            ),
        ]

        result = aggregator.aggregate(feedbacks)
        assert result.status == ValidationStatus.APPROVED


# =============================================================================
# TEST LIVE ENRICHMENT SERVICE
# =============================================================================

class TestLiveEnrichmentService:
    """Test per LiveEnrichmentService."""

    @pytest.mark.asyncio
    async def test_generate_preview(self):
        """Test generazione preview grafo."""
        from merlt.pipeline.live_enrichment import LiveEnrichmentService
        from merlt.api.models.enrichment_models import ArticleData

        service = LiveEnrichmentService()

        article = ArticleData(
            urn="urn:nir:stato:codice.civile:1942:art:1337",
            tipo_atto="codice civile",
            numero_articolo="1337",
            rubrica="",
            testo_vigente="Test",
            estremi="",
            url="",
        )

        entities = [
            PendingEntityData(
                id="pending:1",
                nome="Buona fede",
                tipo=EntityType.CONCETTO,
                fonte="llm",
                llm_confidence=0.8,
                contributed_by="user-1",
                contributor_authority=0.5,
            ),
        ]

        preview = service._generate_preview(article, entities, [])

        assert len(preview.nodes) == 2  # Articolo + 1 entita'
        assert len(preview.links) == 1  # Articolo -> Entita'
        assert preview.nodes[0].type == "ARTICOLO"
        assert preview.nodes[1].type == "concetto"


# =============================================================================
# TEST ENRICHMENT ROUTER
# =============================================================================

class TestEnrichmentRouter:
    """Test per enrichment router (integration-style)."""

    @pytest.mark.asyncio
    async def test_propose_entity(self):
        """Test endpoint propose-entity."""
        from merlt.api.enrichment_router import propose_entity, _pending_entities

        # Clear state
        _pending_entities.clear()

        request = EntityProposalRequest(
            article_urn="urn:nir:stato:codice.civile:1942:art:1337",
            nome="Responsabilita' precontrattuale",
            tipo=EntityType.CONCETTO,
            descrizione="Obbligo di buona fede nelle trattative",
            articoli_correlati=["urn:nir:stato:codice.civile:1942:art:1338"],
            ambito="obbligazioni",
            evidence="L'art. 1337 c.c. stabilisce...",
            user_id="user-123",
            user_authority=0.6,
        )

        response = await propose_entity(request)

        assert response.success is True
        assert response.pending_entity.nome == "Responsabilita' precontrattuale"
        assert response.pending_entity.fonte == "user_proposal"
        assert response.pending_entity.id in _pending_entities

    @pytest.mark.asyncio
    async def test_get_pending_queue(self):
        """Test endpoint pending queue."""
        from merlt.api.enrichment_router import (
            get_pending_queue,
            _pending_entities,
            _entity_votes,
        )

        # Setup state
        _pending_entities.clear()
        _entity_votes.clear()

        entity = PendingEntityData(
            id="pending:test1",
            nome="Test Entity",
            tipo=EntityType.CONCETTO,
            fonte="llm",
            llm_confidence=0.8,
            contributed_by="user-1",
            contributor_authority=0.5,
        )
        _pending_entities["pending:test1"] = entity
        _entity_votes["pending:test1"] = []

        request = PendingQueueRequest(
            user_id="user-2",  # Different user
            include_own=False,
        )

        response = await get_pending_queue(request)

        assert response.total_entities == 1
        assert len(response.pending_entities) == 1
        assert response.pending_entities[0].id == "pending:test1"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_feedbacks():
    """Sample feedbacks per test."""
    return [
        EntityValidationFeedback(
            entity_id="e1",
            entity_type=EntityType.CONCETTO,
            vote="approve",
            user_id="user-1",
            user_authority=0.8,
        ),
        EntityValidationFeedback(
            entity_id="e1",
            entity_type=EntityType.CONCETTO,
            vote="approve",
            user_id="user-2",
            user_authority=0.7,
        ),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
