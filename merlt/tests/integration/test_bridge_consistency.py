"""
[P0] Bridge Table Cross-Store Consistency Tests
=================================================

Integration tests verifying data consistency across the three storage layers:
- PostgreSQL (bridge_table) -- chunk_id <-> graph_node_urn mappings
- Qdrant (vector store)    -- chunk embeddings
- FalkorDB (graph store)   -- knowledge graph nodes

Requirements:
    Docker services running: docker-compose -f docker-compose.dev.yml up -d
    PostgreSQL on port 5433, FalkorDB on 6380, Qdrant on 6333, Redis on 6379

Run:
    pytest tests/integration/test_bridge_consistency.py -v --timeout=30
"""

import os
import uuid
from typing import List, Dict, Any

import pytest
import pytest_asyncio
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from merlt.storage.bridge import BridgeTable, BridgeTableConfig
from merlt.storage.graph.client import FalkorDBClient
from merlt.storage.graph.config import FalkorDBConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def bridge_table():
    """Bridge table connected to the dev database."""
    bt = BridgeTable(BridgeTableConfig())
    await bt.connect()
    yield bt
    await bt.close()


@pytest_asyncio.fixture(scope="function")
async def falkordb():
    """FalkorDB client connected to the dev graph."""
    client = FalkorDBClient(FalkorDBConfig())
    await client.connect()
    yield client
    await client.close()


def _qdrant_client():
    """Create a synchronous Qdrant client for the dev instance."""
    from qdrant_client import QdrantClient
    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        timeout=5,
    )


@pytest.fixture(scope="function")
def qdrant():
    """Qdrant client connected to the dev instance."""
    client = _qdrant_client()
    yield client
    client.close()


@pytest_asyncio.fixture(scope="function")
async def bridge_session():
    """Raw async session for bridge_table queries."""
    config = BridgeTableConfig()
    engine = create_async_engine(config.get_connection_string(), echo=False, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.connect() as conn:
        session = factory(bind=conn)
        try:
            yield session
        finally:
            await session.close()

    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "merl_t_dev_chunks")
BRIDGE_TABLE_NAME = "bridge_table"
SAMPLE_LIMIT = 50  # Number of rows to sample for cross-store checks


async def _fetch_bridge_sample(session: AsyncSession, limit: int = SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    """Fetch a sample of bridge_table entries."""
    result = await session.execute(
        text(
            f"SELECT chunk_id, graph_node_urn, node_type "
            f"FROM {BRIDGE_TABLE_NAME} "
            f"ORDER BY id "
            f"LIMIT :limit"
        ),
        {"limit": limit},
    )
    rows = result.fetchall()
    return [{"chunk_id": str(r[0]), "graph_node_urn": r[1], "node_type": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# [P0] Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_p0_health_check_reports_all_dependencies():
    """GET /health returns status for PostgreSQL, FalkorDB, Qdrant, Redis."""
    from merlt.app import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Top-level structure
    assert "status" in data
    assert "dependencies" in data
    assert data["status"] in ("healthy", "degraded")

    deps = data["dependencies"]
    expected_deps = {"postgresql", "falkordb", "qdrant", "redis"}
    assert expected_deps.issubset(set(deps.keys())), (
        f"Missing dependency checks: {expected_deps - set(deps.keys())}"
    )

    # Each dependency must report a known status
    for name, status in deps.items():
        assert status in ("healthy", "unhealthy"), (
            f"Dependency {name} has unexpected status: {status}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_p0_bridge_table_maps_to_valid_qdrant_vectors(bridge_session, qdrant):
    """For entries in bridge_table, verify corresponding vectors exist in Qdrant."""
    rows = await _fetch_bridge_sample(bridge_session)
    if not rows:
        pytest.skip("bridge_table is empty -- nothing to validate")

    # Collect unique chunk_ids as strings for Qdrant lookup
    chunk_ids = list({r["chunk_id"] for r in rows})

    # Qdrant retrieve accepts point IDs -- bridge stores UUIDs
    from qdrant_client.models import PointIdsList

    found_ids: set = set()
    # Batch retrieve to avoid hitting limits
    batch_size = 50
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i : i + batch_size]
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        found_ids.update(str(p.id) for p in points)

    missing = set(chunk_ids) - found_ids
    # Allow a small tolerance for recently-deleted vectors
    tolerance = max(1, int(len(chunk_ids) * 0.05))
    assert len(missing) <= tolerance, (
        f"{len(missing)}/{len(chunk_ids)} bridge entries have no Qdrant vector. "
        f"First missing: {list(missing)[:5]}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_p0_bridge_table_maps_to_valid_graph_nodes(bridge_session, falkordb):
    """For entries in bridge_table, verify corresponding nodes exist in FalkorDB."""
    rows = await _fetch_bridge_sample(bridge_session)
    if not rows:
        pytest.skip("bridge_table is empty -- nothing to validate")

    # Collect unique URNs
    urns = list({r["graph_node_urn"] for r in rows})

    missing_urns: list = []
    # Check in batches to avoid oversized Cypher queries
    batch_size = 25
    for i in range(0, len(urns), batch_size):
        batch = urns[i : i + batch_size]
        # MATCH nodes whose URN is in the batch list
        result = await falkordb.query(
            "UNWIND $urns AS u MATCH (n {URN: u}) RETURN n.URN AS urn",
            {"urns": batch},
        )
        found = {r["urn"] for r in result if r.get("urn")}
        missing_urns.extend(u for u in batch if u not in found)

    tolerance = max(1, int(len(urns) * 0.05))
    assert len(missing_urns) <= tolerance, (
        f"{len(missing_urns)}/{len(urns)} bridge entries have no FalkorDB node. "
        f"First missing: {missing_urns[:5]}"
    )


# ---------------------------------------------------------------------------
# [P1] Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_p1_orphan_detection(bridge_session, qdrant, falkordb):
    """Detect orphaned entries: bridge rows without Qdrant vectors or FalkorDB nodes."""
    rows = await _fetch_bridge_sample(bridge_session, limit=100)
    if not rows:
        pytest.skip("bridge_table is empty -- nothing to validate")

    chunk_ids = list({r["chunk_id"] for r in rows})
    urns = list({r["graph_node_urn"] for r in rows})

    # --- Qdrant orphan check ---
    qdrant_found: set = set()
    batch_size = 50
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i : i + batch_size]
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        qdrant_found.update(str(p.id) for p in points)

    qdrant_orphans = set(chunk_ids) - qdrant_found

    # --- FalkorDB orphan check ---
    graph_found: set = set()
    batch_size = 25
    for i in range(0, len(urns), batch_size):
        batch = urns[i : i + batch_size]
        result = await falkordb.query(
            "UNWIND $urns AS u MATCH (n {URN: u}) RETURN n.URN AS urn",
            {"urns": batch},
        )
        graph_found.update(r["urn"] for r in result if r.get("urn"))

    graph_orphans = set(urns) - graph_found

    total_orphans = len(qdrant_orphans) + len(graph_orphans)

    # Report orphans (informational -- not necessarily a hard failure)
    if total_orphans > 0:
        msg = (
            f"Orphaned bridge entries detected (sample of {len(rows)}):\n"
            f"  Qdrant orphans: {len(qdrant_orphans)} chunk_ids without vectors\n"
            f"  Graph orphans:  {len(graph_orphans)} URNs without FalkorDB nodes\n"
        )
        if qdrant_orphans:
            msg += f"  Sample Qdrant orphans: {list(qdrant_orphans)[:3]}\n"
        if graph_orphans:
            msg += f"  Sample graph orphans:  {list(graph_orphans)[:3]}\n"

        # Warn but allow up to 10% orphan rate before failing
        orphan_rate = total_orphans / (len(chunk_ids) + len(urns))
        if orphan_rate > 0.10:
            pytest.fail(
                f"Orphan rate {orphan_rate:.1%} exceeds 10% threshold.\n{msg}"
            )
        else:
            import warnings
            warnings.warn(msg, stacklevel=1)
