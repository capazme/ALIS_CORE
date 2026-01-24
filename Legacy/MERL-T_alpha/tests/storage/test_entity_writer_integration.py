"""
Test EntityGraphWriter Integration
===================================

Test completo EntityGraphWriter con FalkorDB reale:
1. 3-layer deduplication (mechanical, LLM, peer-reviewed)
2. Entity node creation con schema corretto
3. Relation creation verso articoli
4. Enrichment nodi esistenti

IMPORTANTE: Test con FalkorDB reale, NO MOCK.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from merlt.storage.enrichment.models import PendingEntity
from merlt.storage.graph.client import FalkorDBClient
from merlt.storage.graph.entity_writer import EntityGraphWriter, WriteResult
from merlt.storage.vectors.embeddings import EmbeddingService


@pytest.mark.asyncio
class TestEntityGraphWriter:
    """Test EntityGraphWriter con FalkorDB."""

    async def test_write_entity_creates_node(
        self,
        falkordb_client: FalkorDBClient,
        sample_entity_data: dict,
    ):
        """
        Test: Write entity crea nodo in FalkorDB.

        Verifica:
        - Nodo creato con tutte le proprietà
        - Label corrette (:Entity:Principio)
        - community_validated = true
        - approval_score salvato
        """
        # Arrange: crea approved entity
        entity = PendingEntity(**sample_entity_data)
        entity.consensus_reached = True
        entity.consensus_type = "approved"
        entity.approval_score = 2.5
        entity.votes_count = 3

        writer = EntityGraphWriter(falkordb_client)

        # Act
        result = await writer.write_entity(entity)

        # Assert
        assert result.success
        assert result.action == "created"
        assert result.node_id is not None

        # Verifica nodo in graph
        query = """
        MATCH (e:Entity:Principio {id: $entity_id})
        RETURN e.id, e.nome, e.tipo, e.descrizione, e.community_validated,
               e.approval_score, e.votes_count, e.sources
        """
        graph_result = await falkordb_client.query(
            query,
            params={"entity_id": entity.entity_id},
        )

        assert len(graph_result) == 1
        node = graph_result[0]

        assert node["e.id"] == "principio:legittima_difesa"  # id
        assert node["e.nome"] == "Legittima difesa"  # nome
        assert node["e.tipo"] == "principio"  # tipo
        assert "difendere" in node["e.descrizione"].lower()  # descrizione
        assert node["e.community_validated"] is True  # community_validated
        assert node["e.approval_score"] == 2.5  # approval_score
        assert node["e.votes_count"] == 3  # votes_count
        assert len(node["e.sources"]) > 0  # sources (array)

    async def test_write_entity_creates_relation_to_article(
        self,
        falkordb_client: FalkorDBClient,
        sample_entity_data: dict,
    ):
        """
        Test: Write entity crea relazione verso articolo sorgente.

        Verifica:
        - Relazione creata
        - Tipo relazione corretto (ESPRIME_PRINCIPIO per principio)
        - Nodo articolo creato se non esiste
        """
        # Arrange
        entity = PendingEntity(**sample_entity_data)
        entity.consensus_reached = True
        entity.consensus_type = "approved"
        entity.approval_score = 2.2

        writer = EntityGraphWriter(falkordb_client)

        # Act
        result = await writer.write_entity(entity)

        # Assert: verifica relazione
        query = """
        MATCH (a:Norma)-[r:ESPRIME_PRINCIPIO]->(e:Entity:Principio {id: $entity_id})
        RETURN a.URN, type(r), e.id
        """
        graph_result = await falkordb_client.query(
            query,
            params={"entity_id": entity.entity_id},
        )

        assert len(graph_result) >= 1
        rel = graph_result[0]

        assert entity.article_urn in rel["a.URN"]  # article URN
        assert rel["type(r)"] == "ESPRIME_PRINCIPIO"  # relation type
        assert rel["e.id"] == entity.entity_id  # entity id

    async def test_mechanical_deduplication_exact_match(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Mechanical deduplication riconosce duplicato esatto.

        Scenario:
        - Entity1: "Legittima difesa" tipo principio
        - Entity2: "Legittima difesa" tipo principio (stesso nome, stesso tipo)
        - Mechanical check DEVE riconoscere duplicato

        Verifica:
        - Writer NON crea secondo nodo
        - Ritorna action='enriched_existing'
        - Existing node arricchito con nuova source
        """
        # Arrange: crea prima entity
        entity1_data = {
            "entity_id": "principio:legittima_difesa_v1",
            "article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52",
            "entity_type": "principio",
            "entity_text": "Legittima difesa",
            "descrizione": "Prima versione",
            "ambito": "penale",
            "contributed_by": "user_001",
        }
        entity1 = PendingEntity(**entity1_data)
        entity1.consensus_reached = True
        entity1.consensus_type = "approved"
        entity1.approval_score = 2.0

        writer = EntityGraphWriter(falkordb_client)
        result1 = await writer.write_entity(entity1)
        assert result1.success
        assert result1.action == "created"

        # Act: proponi entity duplicata (stesso testo normalizzato)
        entity2_data = {
            "entity_id": "principio:legittima_difesa_v2",
            "article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art54",  # articolo diverso
            "entity_type": "principio",
            "entity_text": "Legittima Difesa",  # Stesso testo, case diverso
            "descrizione": "Seconda versione (duplicato)",
            "ambito": "penale",
            "contributed_by": "user_002",
        }
        entity2 = PendingEntity(**entity2_data)
        entity2.consensus_reached = True
        entity2.consensus_type = "approved"
        entity2.approval_score = 2.1

        result2 = await writer.write_entity(entity2)

        # Assert
        assert result2.success
        assert result2.action == "enriched_existing"  # NON creato nuovo
        assert result2.node_id == result1.node_id  # Stesso nodo

        # Verifica: solo 1 nodo nel graph
        count_query = """
        MATCH (e:Entity:Principio)
        WHERE e.nome CONTAINS 'Legittima difesa' OR e.nome CONTAINS 'legittima difesa'
        RETURN count(e) as count
        """
        count_result = await falkordb_client.query(count_query)
        assert count_result[0]["count"] == 1  # Solo 1 nodo

        # Verifica: nodo ha multipli sources
        sources_query = """
        MATCH (e:Entity:Principio {id: $node_id})
        RETURN e.sources
        """
        sources_result = await falkordb_client.query(
            sources_query,
            params={"node_id": result1.node_id},
        )
        sources = sources_result[0]["e.sources"]
        assert len(sources) >= 2  # Due articoli sorgenti

    async def test_mechanical_deduplication_normalization(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Mechanical normalization ignora articoli e punteggiatura.

        Verifica che:
        - "La legittima difesa" == "Legittima difesa"
        - "Legittima-difesa" == "Legittima difesa"
        - Case insensitive
        """
        # Arrange: crea entity con articolo
        entity1_data = {
            "entity_id": "principio:test_norm_1",
            "article_urn": "urn:nir:test:art1",
            "entity_type": "principio",
            "entity_text": "La legittima difesa",  # Con articolo
            "ambito": "penale",
        }
        entity1 = PendingEntity(**entity1_data)
        entity1.consensus_reached = True
        entity1.consensus_type = "approved"
        entity1.approval_score = 2.0

        writer = EntityGraphWriter(falkordb_client)
        result1 = await writer.write_entity(entity1)

        # Act: proponi senza articolo, con punteggiatura
        entity2_data = {
            "entity_id": "principio:test_norm_2",
            "article_urn": "urn:nir:test:art2",
            "entity_type": "principio",
            "entity_text": "Legittima-Difesa!",  # Senza articolo, con punteggiatura
            "ambito": "penale",
        }
        entity2 = PendingEntity(**entity2_data)
        entity2.consensus_reached = True
        entity2.consensus_type = "approved"
        entity2.approval_score = 2.0

        result2 = await writer.write_entity(entity2)

        # Assert: riconosciuto come duplicato
        assert result2.action == "enriched_existing"
        assert result2.node_id == result1.node_id

    async def test_different_type_not_duplicate(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Stesso nome ma tipo diverso → NON duplicato.

        Scenario:
        - "Responsabilità" tipo CONCETTO
        - "Responsabilità" tipo PRINCIPIO
        - Sono entity diverse

        Verifica:
        - Due nodi creati
        - Labels diverse (:Concetto vs :Principio)
        """
        # Arrange
        entity_concetto = PendingEntity(
            entity_id="concetto:responsabilita",
            article_urn="urn:nir:test:art1",
            entity_type="concetto",
            entity_text="Responsabilità",
            ambito="civile",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        entity_principio = PendingEntity(
            entity_id="principio:responsabilita",
            article_urn="urn:nir:test:art2",
            entity_type="principio",
            entity_text="Responsabilità",
            ambito="civile",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)

        # Act
        result1 = await writer.write_entity(entity_concetto)
        result2 = await writer.write_entity(entity_principio)

        # Assert: due nodi creati
        assert result1.action == "created"
        assert result2.action == "created"
        assert result1.node_id != result2.node_id

        # Verifica labels
        query = """
        MATCH (e:Entity)
        WHERE e.nome = 'Responsabilità'
        RETURN labels(e) as labels, e.tipo
        ORDER BY e.tipo
        """
        graph_result = await falkordb_client.query(query)
        assert len(graph_result) == 2

        # Primo: Concetto
        assert "Concetto" in graph_result[0]["labels"]
        assert graph_result[0]["e.tipo"] == "concetto"

        # Secondo: Principio
        assert "Principio" in graph_result[1]["labels"]
        assert graph_result[1]["e.tipo"] == "principio"

    async def test_enrichment_adds_sources(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Enrichment di nodo esistente aggiunge sources.

        Scenario:
        1. Entity scritta da art.52 CP
        2. Stessa entity trovata anche in art.54 CP
        3. Enrichment aggiunge art.54 alle sources

        Verifica:
        - sources array contiene entrambi URN
        - approval_score aggiornato (massimo tra i due)
        - votes_count aggiornato (somma)
        """
        # Arrange: crea prima entity
        entity1 = PendingEntity(
            entity_id="principio:test_enrich",
            article_urn="urn:nir:stato:cp:art52",
            entity_type="principio",
            entity_text="Test enrichment",
            ambito="penale",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
            votes_count=3,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity1)

        # Act: enrichment da secondo articolo
        entity2 = PendingEntity(
            entity_id="principio:test_enrich_v2",
            article_urn="urn:nir:stato:cp:art54",  # Articolo diverso
            entity_type="principio",
            entity_text="Test enrichment",  # Stesso nome
            ambito="penale",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.8,  # Score più alto
            votes_count=5,
        )

        result2 = await writer.write_entity(entity2)

        # Assert
        assert result2.action == "enriched_existing"

        # Verifica sources
        query = """
        MATCH (e:Entity:Principio)
        WHERE e.nome = 'Test enrichment'
        RETURN e.sources, e.approval_score, e.votes_count
        """
        graph_result = await falkordb_client.query(query)
        node = graph_result[0]

        sources = node["e.sources"]
        assert len(sources) == 2
        assert "art52" in str(sources)
        assert "art54" in str(sources)

        # Score aggiornato al massimo
        assert node["e.approval_score"] == 2.8  # Max tra 2.0 e 2.8

        # Votes aggiornato a somma
        assert node["e.votes_count"] == 8  # 3 + 5


@pytest.mark.asyncio
class TestEntityTypeLabelMapping:
    """Test mapping EntityType → Graph Labels."""

    async def test_principio_has_principio_label(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """Verifica: entity tipo 'principio' → label :Principio."""
        entity = PendingEntity(
            entity_id="principio:test",
            article_urn="urn:test",
            entity_type="principio",
            entity_text="Test",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity)

        query = "MATCH (e:Principio {id: 'principio:test'}) RETURN labels(e)"
        result = await falkordb_client.query(query)

        labels = result[0]["labels(e)"]
        assert "Entity" in labels
        assert "Principio" in labels

    async def test_concetto_has_concetto_label(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """Verifica: entity tipo 'concetto' → label :Concetto."""
        entity = PendingEntity(
            entity_id="concetto:test",
            article_urn="urn:test",
            entity_type="concetto",
            entity_text="Test",  # Normalizes to "test"
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity)

        query = "MATCH (e:Concetto {id: 'concetto:test'}) RETURN labels(e)"
        result = await falkordb_client.query(query)

        labels = result[0]["labels(e)"]
        assert "Entity" in labels
        assert "Concetto" in labels

    async def test_definizione_has_definizione_label(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """Verifica: entity tipo 'definizione' → label :Definizione."""
        entity = PendingEntity(
            entity_id="definizione:test",
            article_urn="urn:test",
            entity_type="definizione",
            entity_text="Test",  # Normalizes to "test"
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity)

        query = "MATCH (e:Definizione {id: 'definizione:test'}) RETURN labels(e)"
        result = await falkordb_client.query(query)

        labels = result[0]["labels(e)"]
        assert "Entity" in labels
        assert "Definizione" in labels


@pytest.mark.asyncio
class TestRelationTypeMapping:
    """Test mapping EntityType → Relation Type."""

    async def test_principio_creates_esprime_principio_relation(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """Verifica: principio → ESPRIME_PRINCIPIO relation."""
        entity = PendingEntity(
            entity_id="principio:test_rel",
            article_urn="urn:nir:test:art1",
            entity_type="principio",
            entity_text="Test Rel",  # Normalizes to "test_rel"
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity)

        query = """
        MATCH (a:Norma)-[r]->(e:Principio {id: 'principio:test_rel'})
        RETURN type(r)
        """
        result = await falkordb_client.query(query)

        assert len(result) >= 1
        assert result[0]["type(r)"] == "ESPRIME_PRINCIPIO"

    async def test_definizione_creates_definisce_relation(
        self,
        falkordb_client: FalkorDBClient,
    ):
        """Verifica: definizione → DEFINISCE relation."""
        entity = PendingEntity(
            entity_id="definizione:test_def",
            article_urn="urn:nir:test:art1",
            entity_type="definizione",
            entity_text="Test Def",  # Normalizes to "test_def"
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.0,
        )

        writer = EntityGraphWriter(falkordb_client)
        await writer.write_entity(entity)

        query = """
        MATCH (a:Norma)-[r]->(e:Definizione {id: 'definizione:test_def'})
        RETURN type(r)
        """
        result = await falkordb_client.query(query)

        assert len(result) >= 1
        assert result[0]["type(r)"] == "DEFINISCE"
