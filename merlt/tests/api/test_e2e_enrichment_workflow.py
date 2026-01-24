"""
Test End-to-End Enrichment Workflow
====================================

Test completo end-to-end del flusso di live enrichment:

Flow completo:
1. User propone entity da articolo
2. Multipli user votano (weighted voting)
3. Consensus raggiunto automaticamente (PostgreSQL trigger)
4. Entity scritta su FalkorDB (3-layer deduplication)
5. Domain authority aggiornata per voters
6. Relazione creata verso articolo
7. Entity disponibile per query multi-expert

Questo test simula scenario reale con multipli user e database reali.

IMPORTANTE: NO MOCK - Tutti database reali (PostgreSQL, FalkorDB).
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from merlt.storage.enrichment.models import (
    PendingEntity,
    EntityVote,
    UserDomainAuthority,
)
from merlt.storage.graph.client import FalkorDBClient
from merlt.storage.graph.entity_writer import EntityGraphWriter
from merlt.rlcf.domain_authority import DomainAuthorityService


@pytest.mark.asyncio
class TestEndToEndEnrichmentWorkflow:
    """Test E2E workflow completo."""

    async def test_complete_entity_lifecycle(
        self,
        db_session: AsyncSession,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Ciclo di vita completo di una entity.

        Scenario Realistico:
        - Giurista A (esperto, authority 0.9) propone "Legittima difesa"
        - Giurista B (authority 0.7) vota approve
        - Giurista C (authority 0.6) vota approve
        - Giurista D (novizio, authority 0.4) vota reject
        - Consensus: 0.9 + 0.7 + 0.6 = 2.2 approve vs 0.4 reject → APPROVED
        - Entity scritta su grafo
        - Authority aggiornate per tutti

        Verifica:
        - Consensus corretto
        - Graph write successful
        - Authority A, B, C aumentate (voto corretto)
        - Authority D diminuita (voto sbagliato)
        - Nodo accessibile nel grafo
        """
        # === PHASE 1: Proposta Entity ===
        entity_data = {
            "entity_id": "principio:legittima_difesa_e2e",
            "article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52",
            "source_type": "article",
            "entity_type": "principio",
            "entity_text": "Legittima difesa",
            "descrizione": "Diritto di difendere sé stessi o altri da aggressione ingiusta",
            "ambito": "penale",
            "fonte": "llm_extraction",
            "llm_confidence": 0.95,
            "llm_model": "claude-sonnet-4",
            "contributed_by": "giurista_a",
            "contributor_authority": 0.9,
        }

        entity = PendingEntity(**entity_data)
        db_session.add(entity)
        await db_session.commit()
        await db_session.refresh(entity)

        print(f"\n✅ Phase 1: Entity proposta - {entity.entity_id}")
        assert entity.validation_status == "pending"

        # === PHASE 2: Voting ===
        # Giurista A (proponente, già votato implicitamente? No, vota esplicitamente)
        # Giurista B, C votano approve
        # Giurista D vota reject

        voters = [
            {"user_id": "giurista_a", "vote": 1, "authority": 0.9},  # approve
            {"user_id": "giurista_b", "vote": 1, "authority": 0.7},  # approve
            {"user_id": "giurista_c", "vote": 1, "authority": 0.6},  # approve
            {"user_id": "giurista_d", "vote": -1, "authority": 0.4},  # reject
        ]

        for voter in voters:
            vote = EntityVote(
                entity_id=entity.entity_id,
                user_id=voter["user_id"],
                vote_value=voter["vote"],
                vote_type="accuracy",
                voter_authority=voter["authority"],
                legal_domain="penale",
                comment=f"Voto da {voter['user_id']}",
            )
            db_session.add(vote)

        await db_session.commit()
        await db_session.refresh(entity)

        print(f"✅ Phase 2: Voting completato - {entity.votes_count} voti")

        # === PHASE 3: Verifica Consensus ===
        assert entity.consensus_reached is True
        assert entity.consensus_type == "approved"
        assert entity.approval_score == 2.2  # 0.9 + 0.7 + 0.6
        assert entity.rejection_score == 0.4
        assert entity.votes_count == 4

        print(f"✅ Phase 3: Consensus raggiunto - APPROVED (score: {entity.approval_score})")

        # === PHASE 4: Write to Graph ===
        writer = EntityGraphWriter(falkordb_client)
        write_result = await writer.write_entity(entity)

        assert write_result.success is True
        assert write_result.action == "created"
        assert write_result.node_id is not None

        # Update entity con timestamp
        entity.written_to_graph_at = datetime.now()
        await db_session.commit()

        print(f"✅ Phase 4: Scritto su FalkorDB - node_id: {write_result.node_id}")

        # === PHASE 5: Verifica Graph ===
        # Verifica nodo esiste con proprietà corrette
        query = """
        MATCH (e:Entity:Principio {id: $entity_id})
        RETURN e.nome, e.tipo, e.community_validated, e.approval_score,
               e.votes_count, e.sources
        """
        graph_result = await falkordb_client.query(
            query,
            params={"entity_id": entity.entity_id},
        )

        assert len(graph_result) == 1
        node = graph_result[0]

        assert node["e.nome"] == "Legittima difesa"
        assert node["e.tipo"] == "principio"
        assert node["e.community_validated"] is True  # community_validated
        assert node["e.approval_score"] == 2.2  # approval_score
        assert node["e.votes_count"] == 4  # votes_count
        assert len(node["e.sources"]) > 0  # sources

        print(f"✅ Phase 5: Nodo verificato in FalkorDB")

        # Verifica relazione verso articolo
        rel_query = """
        MATCH (a:Norma)-[r:ESPRIME_PRINCIPIO]->(e:Entity:Principio {id: $entity_id})
        RETURN a.urn, type(r)
        """
        rel_result = await falkordb_client.query(
            rel_query,
            params={"entity_id": entity.entity_id},
        )

        assert len(rel_result) >= 1
        assert "art52" in rel_result[0]["a.urn"]
        assert rel_result[0]["type(r)"] == "ESPRIME_PRINCIPIO"

        print(f"✅ Phase 5b: Relazione verso articolo verificata")

        # === PHASE 6: Domain Authority Update ===
        service = DomainAuthorityService()

        # Update authority per tutti i voters
        authorities_after = {}
        for voter in voters:
            auth = await service.calculate_user_authority(
                db_session,
                user_id=voter["user_id"],
                legal_domain="penale",
            )
            authorities_after[voter["user_id"]] = auth

            # Persist update
            await service.update_or_create_authority(
                db_session,
                user_id=voter["user_id"],
                legal_domain="penale",
                domain_authority=auth,
            )

        await db_session.commit()

        print(f"✅ Phase 6: Authority aggiornate")

        # Verifica: giuristi A, B, C hanno authority aumentata (votarono corretto)
        # giurista_d ha authority diminuita (votò sbagliato)
        assert authorities_after["giurista_a"] == 1.0  # 1 correct / 1 total
        assert authorities_after["giurista_b"] == 1.0
        assert authorities_after["giurista_c"] == 1.0
        assert authorities_after["giurista_d"] == 0.0  # 0 correct / 1 total (votò reject)

        print(f"   - giurista_a: {authorities_after['giurista_a']:.2f} ✅")
        print(f"   - giurista_b: {authorities_after['giurista_b']:.2f} ✅")
        print(f"   - giurista_c: {authorities_after['giurista_c']:.2f} ✅")
        print(f"   - giurista_d: {authorities_after['giurista_d']:.2f} ❌ (voto sbagliato)")

        # === PHASE 7: Verifica Authority Records ===
        for voter_id in ["giurista_a", "giurista_b", "giurista_c", "giurista_d"]:
            stmt = select(UserDomainAuthority).where(
                UserDomainAuthority.user_id == voter_id,
                UserDomainAuthority.legal_domain == "penale",
            )
            result = await db_session.execute(stmt)
            auth_record = result.scalar_one()

            assert auth_record.total_feedbacks == 1
            if voter_id == "giurista_d":
                assert auth_record.correct_feedbacks == 0  # Votò sbagliato
            else:
                assert auth_record.correct_feedbacks == 1  # Votarono corretto

        print(f"✅ Phase 7: Authority records verificati")

        print(f"\n🎉 E2E Test Completato con Successo!")

    async def test_multi_entity_workflow_with_deduplication(
        self,
        db_session: AsyncSession,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Workflow con multipli entity e deduplication.

        Scenario:
        1. Entity A proposta e approvata
        2. Entity B (duplicato di A) proposta → dedup riconosce
        3. Entity C (diversa) proposta e approvata
        4. Grafo ha solo 2 nodi (A merged con B, C separato)

        Verifica:
        - Mechanical deduplication funziona
        - Entity diverse create separatamente
        - Authority calcolate correttamente
        """
        # === Entity A: "Buona fede" ===
        entity_a = PendingEntity(
            entity_id="principio:buona_fede_a",
            article_urn="urn:nir:stato:cc:art1375",
            entity_type="principio",
            entity_text="Buona fede",
            descrizione="Principio di correttezza e buona fede",
            ambito="civile",
            contributed_by="user_001",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.5,
            votes_count=3,
        )
        db_session.add(entity_a)
        await db_session.commit()

        writer = EntityGraphWriter(falkordb_client)
        result_a = await writer.write_entity(entity_a)
        assert result_a.action == "created"

        print(f"✅ Entity A creata: {entity_a.entity_id}")

        # === Entity B: "Buona Fede" (duplicato case-insensitive) ===
        entity_b = PendingEntity(
            entity_id="principio:buona_fede_b",
            article_urn="urn:nir:stato:cc:art1337",  # Articolo diverso
            entity_type="principio",
            entity_text="Buona Fede",  # Case diverso
            descrizione="Principio applicato in fase precontrattuale",
            ambito="civile",
            contributed_by="user_002",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.3,
            votes_count=3,
        )
        db_session.add(entity_b)
        await db_session.commit()

        result_b = await writer.write_entity(entity_b)
        assert result_b.action == "enriched_existing"  # Dedup!
        assert result_b.node_id == result_a.node_id  # Stesso nodo

        print(f"✅ Entity B riconosciuta come duplicato → enrichment di A")

        # === Entity C: "Correttezza" (diversa) ===
        entity_c = PendingEntity(
            entity_id="principio:correttezza",
            article_urn="urn:nir:stato:cc:art1175",
            entity_type="principio",
            entity_text="Correttezza",
            descrizione="Principio di correttezza nell'esecuzione",
            ambito="civile",
            contributed_by="user_003",
            consensus_reached=True,
            consensus_type="approved",
            approval_score=2.1,
            votes_count=3,
        )
        db_session.add(entity_c)
        await db_session.commit()

        result_c = await writer.write_entity(entity_c)
        assert result_c.action == "created"  # Nuovo nodo

        print(f"✅ Entity C creata: {entity_c.entity_id}")

        # === Verifica Grafo ===
        # Dovrebbero esserci solo 2 nodi: Buona fede (merged A+B) e Correttezza
        count_query = """
        MATCH (e:Entity:Principio)
        WHERE e.nome = 'Buona fede' OR e.nome = 'Buona Fede' OR e.nome = 'Correttezza'
        RETURN e.nome, e.sources
        ORDER BY e.nome
        """
        graph_result = await falkordb_client.query(count_query)

        # 2 nodi totali
        assert len(graph_result) == 2

        # Primo nodo: Buona fede con 2 sources (art1375 + art1337)
        buona_fede_node = [r for r in graph_result if "Buona" in r["e.nome"]][0]
        assert len(buona_fede_node["e.sources"]) == 2  # 2 sources

        # Secondo nodo: Correttezza con 1 source
        correttezza_node = [r for r in graph_result if "Correttezza" in r["e.nome"]][0]
        assert len(correttezza_node["e.sources"]) == 1  # 1 source

        print(f"✅ Grafo verificato: 2 nodi (Buona fede merged, Correttezza separato)")
        print(f"   - Buona fede: {len(buona_fede_node['e.sources'])} sources")
        print(f"   - Correttezza: {len(correttezza_node['e.sources'])} sources")

    async def test_workflow_with_rejection(
        self,
        db_session: AsyncSession,
        falkordb_client: FalkorDBClient,
    ):
        """
        Test: Workflow con entity REJECTED.

        Scenario:
        - Entity proposta
        - Maggioranza vota reject
        - Consensus: rejected
        - Entity NON scritta su grafo
        - Status rimane rejected

        Verifica:
        - Consensus rejection corretto
        - Nessun nodo creato in FalkorDB
        - Voters authority aggiornate
        """
        # === Proposta entity dubbiosa ===
        entity = PendingEntity(
            entity_id="principio:entity_dubbiosa",
            article_urn="urn:nir:test:art999",
            entity_type="principio",
            entity_text="Principio dubbioso",
            descrizione="Definizione troppo vaga",
            ambito="civile",
            contributed_by="user_newbie",
        )
        db_session.add(entity)
        await db_session.commit()

        # === Voting: maggioranza reject ===
        voters_reject = [
            EntityVote(
                entity_id=entity.entity_id,
                user_id="expert_001",
                vote_value=-1,  # reject
                vote_type="accuracy",
                voter_authority=0.9,
                legal_domain="civile",
                comment="Troppo vago",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="expert_002",
                vote_value=-1,  # reject
                vote_type="accuracy",
                voter_authority=0.8,
                legal_domain="civile",
                comment="Non è un principio",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="expert_003",
                vote_value=-1,  # reject
                vote_type="accuracy",
                voter_authority=0.7,
                legal_domain="civile",
            ),
            EntityVote(
                entity_id=entity.entity_id,
                user_id="user_newbie",
                vote_value=1,  # approve (minority)
                vote_type="accuracy",
                voter_authority=0.3,
                legal_domain="civile",
            ),
        ]

        for vote in voters_reject:
            db_session.add(vote)

        await db_session.commit()
        await db_session.refresh(entity)

        # === Verifica Consensus Rejection ===
        assert entity.consensus_reached is True
        assert entity.consensus_type == "rejected"
        assert entity.rejection_score == pytest.approx(2.4)  # 0.9 + 0.8 + 0.7
        assert entity.approval_score == pytest.approx(0.3)

        print(f"✅ Consensus REJECTED - rejection score: {entity.rejection_score}")

        # === Verifica: NON scrivibile su grafo ===
        writer = EntityGraphWriter(falkordb_client)

        # write_entity dovrebbe fallire per entity rejected
        with pytest.raises(ValueError, match="not approved"):
            await writer.write_entity(entity)

        print(f"✅ Entity rejected NON scritta su grafo (come previsto)")

        # === Verifica: Nessun nodo creato ===
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN count(e)
        """
        result = await falkordb_client.query(
            query,
            params={"entity_id": entity.entity_id},
        )

        assert len(result) == 1  # One result row
        assert list(result[0].values())[0] == 0  # Nessun nodo

        print(f"✅ Verificato: 0 nodi nel grafo per entity rejected")

        print(f"\n🎉 Workflow rejection completato correttamente!")
