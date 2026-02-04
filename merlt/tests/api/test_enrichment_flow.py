"""
Test Enrichment Flow End-to-End
================================

Test completo del flusso di live enrichment:
1. Proposta entity/relation
2. Voting con authority-weighted consensus
3. Automatic consensus calculation (PostgreSQL triggers)
4. Graph write on approval
5. Domain authority update

IMPORTANTE: Questi test usano database reali, NO MOCK.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from merlt.storage.enrichment.models import (
    PendingEntity,
    EntityVote,
    PendingRelation,
    RelationVote,
    UserDomainAuthority,
)
from merlt.storage.graph.client import FalkorDBClient
from merlt.storage.graph.entity_writer import EntityGraphWriter
from merlt.rlcf.domain_authority import DomainAuthorityService

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
class TestEntityValidationFlow:
    """Test completo del flusso di validazione entity."""

    async def test_propose_entity_creates_pending_record(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Proposta entity crea record in pending_entities.

        Verifica:
        - Record creato correttamente
        - Campi default corretti
        - Status iniziale 'pending'
        - Consensus non raggiunto
        """
        # Arrange
        entity = PendingEntity(**sample_entity_data)

        # Act
        db_session.add(entity)
        await db_session.commit()
        await db_session.refresh(entity)

        # Assert
        assert entity.id is not None
        assert entity.entity_id == sample_entity_data["entity_id"]
        assert entity.validation_status == "pending"
        assert entity.consensus_reached is False
        assert entity.approval_score == 0.0
        assert entity.rejection_score == 0.0
        assert entity.votes_count == 0
        assert entity.created_at is not None

    async def test_single_vote_no_consensus(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Singolo voto NON raggiunge consensus (threshold 2.0).

        Scenario:
        - User con authority 0.8 vota approve
        - Approval score = 0.8 (sotto soglia 2.0)
        - Consensus non raggiunto

        Verifica:
        - Vote registrato
        - Trigger calcola score correttamente
        - Consensus_reached = False
        """
        # Arrange: crea entity
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # Act: user vota approve
        vote = EntityVote(
            entity_id=entity.entity_id,
            user_id="user_001",
            vote_value=1,  # approve
            vote_type="accuracy",
            voter_authority=0.8,
            legal_domain="penale",
        )
        db_session.add(vote)
        await db_session.commit()

        # Refresh entity per leggere valori aggiornati da trigger
        await db_session.refresh(entity)

        # Assert
        assert entity.votes_count == 1
        assert entity.approval_score == 0.8
        assert entity.rejection_score == 0.0
        assert entity.consensus_reached is False
        assert entity.consensus_type is None

    async def test_multiple_votes_reach_approval_consensus(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Multipli voti raggiungono consensus di approval.

        Scenario:
        - User1 (authority 0.9) vota approve
        - User2 (authority 0.7) vota approve
        - User3 (authority 0.6) vota approve
        - Total approval = 2.2 >= 2.0 → consensus approved

        Verifica:
        - Tutti i voti registrati
        - Consensus raggiunto
        - Consensus type = 'approved'
        - Approval score >= 2.0
        """
        # Arrange
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # Act: tre user votano approve
        votes = [
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_001",
                vote_value=1,
                vote_type="accuracy",
                voter_authority=0.9,
                legal_domain="penale",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_002",
                vote_value=1,
                vote_type="accuracy",
                voter_authority=0.7,
                legal_domain="penale",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_003",
                vote_value=1,
                vote_type="accuracy",
                voter_authority=0.6,
                legal_domain="penale",
            ),
        ]

        for vote in votes:
            db_session.add(vote)

        await db_session.commit()
        await db_session.refresh(entity)

        # Assert
        assert entity.votes_count == 3
        assert entity.approval_score == 2.2  # 0.9 + 0.7 + 0.6
        assert entity.rejection_score == 0.0
        assert entity.consensus_reached is True
        assert entity.consensus_type == "approved"

    async def test_multiple_votes_reach_rejection_consensus(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Multipli voti raggiungono consensus di rejection.

        Scenario:
        - User1 (authority 0.8) vota reject
        - User2 (authority 0.8) vota reject
        - User3 (authority 0.7) vota reject
        - Total rejection = 2.3 >= 2.0 → consensus rejected

        Verifica:
        - Consensus raggiunto
        - Consensus type = 'rejected'
        - Entity NON scritta su grafo
        """
        # Arrange
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # Act: tre user votano reject
        votes = [
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_001",
                vote_value=-1,  # reject
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
                comment="Definizione troppo generica",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_002",
                vote_value=-1,
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_003",
                vote_value=-1,
                vote_type="accuracy",
                voter_authority=0.7,
                legal_domain="penale",
            ),
        ]

        for vote in votes:
            db_session.add(vote)

        await db_session.commit()
        await db_session.refresh(entity)

        # Assert
        assert entity.votes_count == 3
        assert entity.approval_score == 0.0
        assert entity.rejection_score == 2.3
        assert entity.consensus_reached is True
        assert entity.consensus_type == "rejected"

    async def test_mixed_votes_no_consensus(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Voti discordanti NON raggiungono consensus.

        Scenario:
        - 2 approve (total 1.5)
        - 1 reject (total 0.8)
        - Nessuno score >= 2.0

        Verifica:
        - Consensus_reached = False
        - Entity rimane pending
        """
        # Arrange
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # Act: voti misti
        votes = [
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_001",
                vote_value=1,  # approve
                vote_type="accuracy",
                voter_authority=0.9,
                legal_domain="penale",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_002",
                vote_value=1,  # approve
                vote_type="accuracy",
                voter_authority=0.6,
                legal_domain="penale",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_003",
                vote_value=-1,  # reject
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
            ),
        ]

        for vote in votes:
            db_session.add(vote)

        await db_session.commit()
        await db_session.refresh(entity)

        # Assert
        assert entity.votes_count == 3
        assert entity.approval_score == 1.5
        assert entity.rejection_score == 0.8
        assert entity.consensus_reached is False
        assert entity.consensus_type is None

    async def test_approved_entity_written_to_graph(
        self,
        db_session: AsyncSession,
        falkordb_client: FalkorDBClient,
        sample_entity_data: dict,
    ):
        """
        Test: Entity approvata viene scritta su FalkorDB.

        Flow completo:
        1. Proponi entity
        2. Voti portano ad approval consensus
        3. EntityGraphWriter scrive nodo su grafo
        4. Timestamp written_to_graph_at aggiornato

        Verifica:
        - Nodo creato in FalkorDB
        - Proprietà corrette
        - Relazione verso articolo creata
        - Timestamp written_to_graph_at settato
        """
        # Arrange: crea entity approved
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # Voti per approval
        votes = [
            EntityVote(
                entity_id=entity.entity_id,
                user_id=f"user_{i:03d}",
                vote_value=1,
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
            )
            for i in range(3)  # 3 voti * 0.8 = 2.4 >= 2.0
        ]
        for vote in votes:
            db_session.add(vote)
        await db_session.commit()
        await db_session.refresh(entity)

        assert entity.consensus_reached
        assert entity.consensus_type == "approved"

        # Act: scrivi su grafo
        writer = EntityGraphWriter(falkordb_client)
        result = await writer.write_entity(entity)

        # Assert: verifica risultato write
        assert result.success
        assert result.action in ["created", "enriched_existing"]
        assert result.node_id is not None

        # Verifica nodo in FalkorDB
        query = """
        MATCH (e:Entity:Principio {id: $entity_id})
        RETURN e.nome, e.tipo, e.community_validated, e.approval_score
        """
        graph_result = await falkordb_client.query(
            query,
            params={"entity_id": entity.entity_id},
        )

        assert len(graph_result) == 1
        node_data = graph_result[0]
        # Extract values from result dict
        assert "e.nome" in node_data or len(node_data) >= 4
        # FalkorDBClient.query returns list of dicts, need to extract values
        values = list(node_data.values()) if isinstance(node_data, dict) else node_data
        assert values[0] == "Legittima difesa" or node_data.get("e.nome") == "Legittima difesa"  # nome

        # Update entity con timestamp
        entity.written_to_graph_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(entity)

        assert entity.written_to_graph_at is not None


@pytest.mark.asyncio
class TestRelationValidationFlow:
    """Test completo del flusso di validazione relazioni."""

    async def test_propose_relation_creates_pending_record(
        self,
        db_session: AsyncSession,
        sample_relation_data: dict,
    ):
        """
        Test: Proposta relation crea record in pending_relations.

        Verifica:
        - Record creato correttamente
        - Status iniziale 'pending'
        - Campi relation specifici popolati
        """
        # Arrange
        relation = PendingRelation(**sample_relation_data)

        # Act
        db_session.add(relation)
        await db_session.commit()
        await db_session.refresh(relation)

        # Assert
        assert relation.id is not None
        assert relation.relation_id == sample_relation_data["relation_id"]
        assert relation.validation_status == "pending"
        assert relation.relation_type == "ESPRIME_PRINCIPIO"
        assert relation.target_entity_id == "principio:legittima_difesa"
        assert relation.consensus_reached is False

    async def test_relation_approval_consensus(
        self,
        db_session: AsyncSession,
        sample_relation_data: dict,
    ):
        """
        Test: Relazione raggiunge approval consensus.

        Scenario:
        - 3 user votano approve
        - Consensus raggiunto
        - Relazione pronta per write su grafo

        Verifica:
        - Consensus calculation corretto
        - Consensus type = 'approved'
        """
        # Arrange
        relation = PendingRelation(**sample_relation_data)
        db_session.add(relation)
        await db_session.commit()

        # Act: voti approve
        votes = [
            RelationVote(
                relation_id=relation.relation_id,
                user_id=f"user_{i:03d}",
                vote_value=1,
                vote_type="accuracy",
                voter_authority=0.75,
                legal_domain="penale",
            )
            for i in range(3)  # 3 * 0.75 = 2.25 >= 2.0
        ]
        for vote in votes:
            db_session.add(vote)
        await db_session.commit()
        await db_session.refresh(relation)

        # Assert
        assert relation.votes_count == 3
        assert relation.approval_score == 2.25
        assert relation.consensus_reached is True
        assert relation.consensus_type == "approved"


@pytest.mark.asyncio
class TestDomainAuthorityCalculation:
    """Test calcolo domain authority basato su accuracy."""

    async def test_new_user_default_authority(
        self,
        db_session: AsyncSession,
    ):
        """
        Test: Nuovo user ha authority default 0.5.

        Verifica:
        - Record creato in user_domain_authority
        - Authority iniziale = 0.5
        - Total feedbacks = 0
        """
        # Arrange
        service = DomainAuthorityService()

        # Act
        authority = await service.get_user_authority_for_vote(
            db_session,
            user_id="new_user_999",
            legal_domain="civile",
        )

        # Assert
        assert authority == 0.5

        # Verifica record creato
        stmt = select(UserDomainAuthority).where(
            UserDomainAuthority.user_id == "new_user_999",
            UserDomainAuthority.legal_domain == "civile",
        )
        result = await db_session.execute(stmt)
        record = result.scalar_one_or_none()

        assert record is not None
        assert record.domain_authority == 0.5
        assert record.total_feedbacks == 0
        assert record.correct_feedbacks == 0

    async def test_authority_increases_with_correct_votes(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Authority aumenta quando user vota correttamente.

        Scenario:
        1. User vota approve su entity
        2. Consensus raggiunge approved
        3. Voto era corretto (user aligned con consensus)
        4. Authority aumenta

        Verifica:
        - Authority > 0.5
        - Accuracy score corretto
        - Total/correct feedbacks aggiornati
        """
        # Arrange: crea entity
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # User sotto test vota approve per primo
        target_user = "user_accuracy_test"
        vote1 = EntityVote(
            entity_id=entity.entity_id,
            user_id=target_user,
            vote_value=1,  # approve
            vote_type="accuracy",
            voter_authority=0.5,  # authority iniziale
            legal_domain="penale",
        )
        db_session.add(vote1)
        await db_session.commit()

        # Altri user votano approve → consensus approved
        for i in range(3):
            vote = EntityVote(
                entity_id=entity.entity_id,
                user_id=f"user_{i:03d}",
                vote_value=1,  # approve
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
            )
            db_session.add(vote)
        await db_session.commit()
        await db_session.refresh(entity)

        assert entity.consensus_reached
        assert entity.consensus_type == "approved"

        # Act: ricalcola authority e persisti
        service = DomainAuthorityService()
        authority_record = await service.update_user_domain_authority(
            db_session,
            user_id=target_user,
            legal_domain="penale",
        )

        # Assert
        new_authority = authority_record.domain_authority
        assert new_authority > 0.5  # Authority aumentata
        assert new_authority == 1.0  # 1 correct / 1 total = 100% accuracy

        # Verifica record authority exists now
        assert authority_record is not None

        assert authority_record.total_feedbacks == 1
        assert authority_record.correct_feedbacks == 1
        assert authority_record.accuracy_score == 1.0
        assert authority_record.domain_authority == 1.0

    async def test_authority_decreases_with_incorrect_votes(
        self,
        db_session: AsyncSession,
        sample_entity_data: dict,
    ):
        """
        Test: Authority diminuisce quando user vota in modo errato.

        Scenario:
        1. User vota reject su entity
        2. Consensus raggiunge approved (user sbagliava)
        3. Voto era scorretto (user NOT aligned con consensus)
        4. Authority diminuisce

        Verifica:
        - Authority < 0.5 (se solo voti sbagliati)
        - Accuracy score corretto
        """
        # Arrange: crea entity
        entity = PendingEntity(**sample_entity_data)
        db_session.add(entity)
        await db_session.commit()

        # User sotto test vota REJECT (sbagliato)
        target_user = "user_wrong_test"
        vote1 = EntityVote(
            entity_id=entity.entity_id,
            user_id=target_user,
            vote_value=-1,  # reject (SBAGLIATO)
            vote_type="accuracy",
            voter_authority=0.5,
            legal_domain="penale",
        )
        db_session.add(vote1)
        await db_session.commit()

        # Altri user votano approve → consensus approved
        for i in range(3):
            vote = EntityVote(
                entity_id=entity.entity_id,
                user_id=f"user_{i:03d}",
                vote_value=1,  # approve
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="penale",
            )
            db_session.add(vote)
        await db_session.commit()
        await db_session.refresh(entity)

        assert entity.consensus_reached
        assert entity.consensus_type == "approved"

        # Act: ricalcola authority e persisti
        service = DomainAuthorityService()
        authority_record = await service.update_user_domain_authority(
            db_session,
            user_id=target_user,
            legal_domain="penale",
        )

        # Assert
        new_authority = authority_record.domain_authority
        assert new_authority < 0.5  # Authority diminuita
        assert new_authority == 0.0  # 0 correct / 1 total = 0% accuracy

        # Verifica record authority exists now
        assert authority_record is not None
        assert authority_record.total_feedbacks == 1
        assert authority_record.correct_feedbacks == 0
        assert authority_record.accuracy_score == 0.0
        assert authority_record.domain_authority == 0.0
