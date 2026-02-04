"""
Tests for Bridge Table Manager
==============================

Integration tests using real PostgreSQL instance.
Requires: docker container 'merl-t-postgres-dev' running on localhost:5433

Tests cover:
- AC1: Mapping schema with source_type, source_authority, mapping_type, expert_affinity
- AC2: Multiple mappings per chunk (PRIMARY vs REFERENCE)
- AC3: Expert-specific queries with affinity filtering
- AC4: F8 feedback hook (update_expert_affinity stub)
"""

import pytest
import uuid
from datetime import datetime

from visualex.graph.bridge import (
    BridgeConfig,
    BridgeTableManager,
    BridgeMapping,
    MappingType,
    DEFAULT_EXPERT_AFFINITIES,
)
from visualex.graph.chunking import SourceType


# =============================================================================
# Test Configuration
# =============================================================================


# Use a test-specific table to avoid conflicts
TEST_TABLE = f"bridge_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def bridge_config():
    """Create test configuration."""
    return BridgeConfig(
        host="localhost",
        port=5433,
        database="rlcf_dev",
        user="dev",
        password="devpassword",
        table_name=TEST_TABLE,
    )


@pytest.fixture
async def manager(bridge_config):
    """Create manager and clean up after tests."""
    mgr = BridgeTableManager(bridge_config)

    await mgr.connect()
    await mgr.ensure_table_exists()

    yield mgr

    # Cleanup: drop test table
    try:
        await mgr.drop_table()
    except Exception:
        pass

    await mgr.close()


@pytest.fixture
def sample_chunk_id():
    """Generate a sample chunk UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_graph_urn():
    """Sample graph node URN."""
    return f"urn:nir:stato:legge:2020-12-30;178~art{uuid.uuid4().hex[:4]}"


# =============================================================================
# BridgeConfig Tests
# =============================================================================


class TestBridgeConfig:
    """Test suite for BridgeConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = BridgeConfig()

        assert config.host == "localhost"
        assert config.port == 5433
        assert config.database == "rlcf_dev"
        assert config.table_name == "bridge_table_enhanced"

    def test_connection_string(self):
        """Test connection string generation."""
        config = BridgeConfig(
            host="testhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
        )

        cs = config.get_connection_string()
        assert "postgresql+asyncpg://" in cs
        assert "testuser:testpass" in cs
        assert "testhost:5432" in cs
        assert "testdb" in cs

    def test_from_env(self, monkeypatch):
        """Test configuration from environment."""
        monkeypatch.setenv("BRIDGE_DB_HOST", "pg-server")
        monkeypatch.setenv("BRIDGE_DB_PORT", "5434")
        monkeypatch.setenv("BRIDGE_DB_NAME", "my_db")
        monkeypatch.setenv("BRIDGE_TABLE_NAME", "my_bridge")

        config = BridgeConfig.from_env()

        assert config.host == "pg-server"
        assert config.port == 5434
        assert config.database == "my_db"
        assert config.table_name == "my_bridge"

    def test_from_env_invalid_port(self, monkeypatch):
        """Test invalid port falls back to default."""
        monkeypatch.setenv("BRIDGE_DB_PORT", "invalid")

        config = BridgeConfig.from_env()

        assert config.port == 5433  # Default


# =============================================================================
# MappingType Tests
# =============================================================================


class TestMappingType:
    """Test suite for MappingType enum."""

    def test_values(self):
        """Test MappingType enum values."""
        assert MappingType.PRIMARY.value == "PRIMARY"
        assert MappingType.REFERENCE.value == "REFERENCE"
        assert MappingType.CONCEPT.value == "CONCEPT"
        assert MappingType.DOCTRINE.value == "DOCTRINE"


# =============================================================================
# BridgeMapping Tests
# =============================================================================


class TestBridgeMapping:
    """Test suite for BridgeMapping dataclass."""

    def test_creation(self):
        """Test BridgeMapping creation."""
        mapping = BridgeMapping(
            chunk_id="test-chunk-123",
            graph_node_urn="urn:test:node",
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
            expert_affinity={"literal": 0.9, "precedent": 0.3},
        )

        assert mapping.chunk_id == "test-chunk-123"
        assert mapping.source_type == "norm"
        assert mapping.expert_affinity["literal"] == 0.9

    def test_to_dict(self):
        """Test BridgeMapping serialization."""
        mapping = BridgeMapping(
            chunk_id="test-chunk",
            graph_node_urn="urn:test",
            source_type="jurisprudence",
            source_authority=0.8,
            mapping_type="REFERENCE",
            expert_affinity={"precedent": 0.9},
            metadata={"key": "value"},
        )

        d = mapping.to_dict()

        assert d["chunk_id"] == "test-chunk"
        assert d["source_type"] == "jurisprudence"
        assert d["metadata"]["key"] == "value"


# =============================================================================
# DEFAULT_EXPERT_AFFINITIES Tests
# =============================================================================


class TestDefaultExpertAffinities:
    """Test suite for default expert affinities."""

    def test_norm_affinities(self):
        """Test norm source type affinities."""
        affinities = DEFAULT_EXPERT_AFFINITIES[SourceType.NORM]

        assert affinities["literal"] == 0.9
        assert affinities["systemic"] == 0.8
        assert affinities["principles"] == 0.5
        assert affinities["precedent"] == 0.3

    def test_jurisprudence_affinities(self):
        """Test jurisprudence source type affinities."""
        affinities = DEFAULT_EXPERT_AFFINITIES[SourceType.JURISPRUDENCE]

        assert affinities["precedent"] == 0.9
        assert affinities["literal"] == 0.3

    def test_doctrine_affinities(self):
        """Test doctrine source type affinities."""
        affinities = DEFAULT_EXPERT_AFFINITIES[SourceType.DOCTRINE]

        assert affinities["principles"] == 0.9


# =============================================================================
# BridgeTableManager Tests (Integration)
# =============================================================================


class TestBridgeTableManager:
    """Integration tests for BridgeTableManager with real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """Test health check passes."""
        healthy = await manager.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_create_mapping_auto_affinity(self, manager):
        """Test create_mapping auto-computes expert affinity (AC1)."""
        mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn="urn:test:auto-affinity",
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
        )

        # Should have auto-computed norm affinities
        assert mapping.expert_affinity["literal"] == 0.9
        assert mapping.expert_affinity["precedent"] == 0.3

    @pytest.mark.asyncio
    async def test_create_mapping_jurisprudence_affinity(self, manager):
        """Test create_mapping for jurisprudence source type."""
        mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn="urn:test:jurisprudence",
            source_type="jurisprudence",
            source_authority=0.8,
            mapping_type="PRIMARY",
        )

        # Jurisprudence should favor precedent
        assert mapping.expert_affinity["precedent"] == 0.9
        assert mapping.expert_affinity["literal"] == 0.3

    @pytest.mark.asyncio
    async def test_add_mapping(self, manager, sample_chunk_id, sample_graph_urn):
        """Test adding a single mapping (AC1)."""
        mapping = manager.create_mapping(
            chunk_id=sample_chunk_id,
            graph_node_urn=sample_graph_urn,
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
        )

        entry_id = await manager.add_mapping(mapping)

        assert entry_id is not None
        assert entry_id > 0

    @pytest.mark.asyncio
    async def test_add_mappings_batch(self, manager):
        """Test batch mapping insertion (AC2)."""
        base_urn = f"urn:test:batch:{uuid.uuid4().hex[:8]}"

        # Create multiple mappings for same chunk (PRIMARY and REFERENCE)
        chunk_id = str(uuid.uuid4())
        mappings = [
            manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:primary",
                source_type="norm",
                source_authority=1.0,
                mapping_type="PRIMARY",
            ),
            manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:ref1",
                source_type="norm",
                source_authority=0.8,
                mapping_type="REFERENCE",
            ),
            manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:ref2",
                source_type="norm",
                source_authority=0.7,
                mapping_type="REFERENCE",
            ),
        ]

        count = await manager.add_mappings_batch(mappings)

        assert count == 3

        # Verify all mappings were created
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_multiple_mappings_types(self, manager):
        """Test chunk with multiple mapping types (AC2)."""
        chunk_id = str(uuid.uuid4())
        base_urn = f"urn:test:multi:{uuid.uuid4().hex[:8]}"

        mappings = [
            manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:primary",
                source_type="norm",
                source_authority=1.0,
                mapping_type="PRIMARY",
            ),
            manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:reference",
                source_type="norm",
                source_authority=0.5,
                mapping_type="REFERENCE",
            ),
        ]

        await manager.add_mappings_batch(mappings)

        # Query only PRIMARY
        primary_results = await manager.get_mappings_for_chunk(
            chunk_id, mapping_type="PRIMARY"
        )
        assert len(primary_results) == 1
        assert primary_results[0].mapping_type == "PRIMARY"

        # Query only REFERENCE
        ref_results = await manager.get_mappings_for_chunk(
            chunk_id, mapping_type="REFERENCE"
        )
        assert len(ref_results) == 1
        assert ref_results[0].mapping_type == "REFERENCE"

    @pytest.mark.asyncio
    async def test_get_chunks_for_node(self, manager):
        """Test reverse lookup from graph node to chunks."""
        graph_urn = f"urn:test:reverse:{uuid.uuid4().hex[:8]}"

        # Add multiple chunks pointing to same node
        for i in range(3):
            mapping = manager.create_mapping(
                chunk_id=str(uuid.uuid4()),
                graph_node_urn=graph_urn,
                source_type="norm",
                source_authority=0.9 - i * 0.1,
                mapping_type="PRIMARY",
            )
            await manager.add_mapping(mapping)

        results = await manager.get_chunks_for_node(graph_urn)

        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_get_chunks_for_node_with_min_authority(self, manager):
        """Test filtering by minimum authority."""
        graph_urn = f"urn:test:authority:{uuid.uuid4().hex[:8]}"

        # Add chunks with different authorities
        for authority in [1.0, 0.8, 0.5, 0.3]:
            mapping = manager.create_mapping(
                chunk_id=str(uuid.uuid4()),
                graph_node_urn=graph_urn,
                source_type="norm",
                source_authority=authority,
                mapping_type="PRIMARY",
            )
            await manager.add_mapping(mapping)

        # Query with min_authority=0.7
        results = await manager.get_chunks_for_node(
            graph_urn, min_authority=0.7
        )

        # Should only get authority >= 0.7
        assert len(results) >= 2
        for r in results:
            assert r.source_authority >= 0.7

    @pytest.mark.asyncio
    async def test_get_chunks_for_expert(self, manager):
        """Test expert-specific query with affinity sorting (AC3)."""
        graph_urn = f"urn:test:expert:{uuid.uuid4().hex[:8]}"

        # Add norm (high literal affinity)
        norm_mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn=graph_urn,
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(norm_mapping)

        # Add jurisprudence (high precedent affinity)
        juris_mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn=graph_urn,
            source_type="jurisprudence",
            source_authority=0.8,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(juris_mapping)

        # Query for literal expert - norm should be first
        literal_results = await manager.get_chunks_for_expert(
            graph_urn, expert_type="literal", min_affinity=0.0
        )

        assert len(literal_results) >= 2
        # First result should have higher literal affinity
        assert literal_results[0].expert_affinity["literal"] >= literal_results[1].expert_affinity["literal"]

        # Query for precedent expert - jurisprudence should be first
        precedent_results = await manager.get_chunks_for_expert(
            graph_urn, expert_type="precedent", min_affinity=0.0
        )

        assert len(precedent_results) >= 2
        # First result should have higher precedent affinity
        assert precedent_results[0].expert_affinity["precedent"] >= precedent_results[1].expert_affinity["precedent"]

    @pytest.mark.asyncio
    async def test_get_chunks_for_expert_with_min_affinity(self, manager):
        """Test expert query with minimum affinity threshold (AC3)."""
        graph_urn = f"urn:test:expert-min:{uuid.uuid4().hex[:8]}"

        # Add norm (literal=0.9)
        norm_mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn=graph_urn,
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(norm_mapping)

        # Add jurisprudence (literal=0.3)
        juris_mapping = manager.create_mapping(
            chunk_id=str(uuid.uuid4()),
            graph_node_urn=graph_urn,
            source_type="jurisprudence",
            source_authority=0.8,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(juris_mapping)

        # Query for literal expert with min_affinity=0.7
        results = await manager.get_chunks_for_expert(
            graph_urn, expert_type="literal", min_affinity=0.7
        )

        # Should only get norm (literal=0.9), not jurisprudence (literal=0.3)
        assert len(results) >= 1
        for r in results:
            assert r.expert_affinity["literal"] >= 0.7

    @pytest.mark.asyncio
    async def test_update_expert_affinity(self, manager):
        """Test F8 feedback hook for updating expert affinity (AC4)."""
        chunk_id = str(uuid.uuid4())
        graph_urn = f"urn:test:feedback:{uuid.uuid4().hex[:8]}"

        # Add initial mapping
        mapping = manager.create_mapping(
            chunk_id=chunk_id,
            graph_node_urn=graph_urn,
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(mapping)

        # Initial literal affinity should be 0.9
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert results[0].expert_affinity["literal"] == 0.9

        # Update via F8 feedback hook
        updated = await manager.update_expert_affinity(
            chunk_id=chunk_id,
            graph_node_urn=graph_urn,
            expert_type="literal",
            new_affinity=0.95,
            feedback_source="test_f8_feedback",
        )

        assert updated is True

        # Verify update
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert results[0].expert_affinity["literal"] == 0.95

        # Check feedback metadata was logged
        assert "last_feedback" in results[0].metadata
        assert results[0].metadata["last_feedback"]["source"] == "test_f8_feedback"

    @pytest.mark.asyncio
    async def test_update_expert_affinity_not_found(self, manager):
        """Test update_expert_affinity returns False when mapping not found."""
        updated = await manager.update_expert_affinity(
            chunk_id="non-existent-chunk",
            graph_node_urn="urn:non:existent",
            expert_type="literal",
            new_affinity=0.5,
        )

        assert updated is False

    @pytest.mark.asyncio
    async def test_delete_mappings_for_chunk(self, manager):
        """Test deleting all mappings for a chunk."""
        chunk_id = str(uuid.uuid4())
        base_urn = f"urn:test:delete:{uuid.uuid4().hex[:8]}"

        # Add multiple mappings
        for i in range(3):
            mapping = manager.create_mapping(
                chunk_id=chunk_id,
                graph_node_urn=f"{base_urn}:{i}",
                source_type="norm",
                source_authority=1.0,
                mapping_type="PRIMARY",
            )
            await manager.add_mapping(mapping)

        # Verify they exist
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert len(results) == 3

        # Delete
        deleted = await manager.delete_mappings_for_chunk(chunk_id)
        assert deleted == 3

        # Verify deletion
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, manager):
        """Test that add_mapping upserts (updates if exists)."""
        chunk_id = str(uuid.uuid4())
        graph_urn = f"urn:test:upsert:{uuid.uuid4().hex[:8]}"

        # Add initial mapping
        mapping1 = manager.create_mapping(
            chunk_id=chunk_id,
            graph_node_urn=graph_urn,
            source_type="norm",
            source_authority=0.5,
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(mapping1)

        # Upsert with different authority
        mapping2 = manager.create_mapping(
            chunk_id=chunk_id,
            graph_node_urn=graph_urn,
            source_type="norm",
            source_authority=0.9,  # Different
            mapping_type="PRIMARY",
        )
        await manager.add_mapping(mapping2)

        # Should only have one mapping with updated authority
        results = await manager.get_mappings_for_chunk(chunk_id)
        assert len(results) == 1
        assert results[0].source_authority == 0.9

    @pytest.mark.asyncio
    async def test_count(self, manager):
        """Test count returns total mappings."""
        count = await manager.count()
        assert count >= 0

    @pytest.mark.asyncio
    async def test_expert_query_performance_under_50ms(self, manager):
        """Test AC3 performance requirement: <50ms for single URN lookup."""
        import time

        graph_urn = f"urn:test:perf:{uuid.uuid4().hex[:8]}"

        # Add some mappings
        for i in range(10):
            mapping = manager.create_mapping(
                chunk_id=str(uuid.uuid4()),
                graph_node_urn=graph_urn,
                source_type="norm",
                source_authority=0.9,
                mapping_type="PRIMARY",
            )
            await manager.add_mapping(mapping)

        # Measure query time
        start = time.perf_counter()
        results = await manager.get_chunks_for_expert(
            graph_urn, expert_type="literal", min_affinity=0.0
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) >= 10
        assert elapsed_ms < 50, f"Query took {elapsed_ms:.2f}ms, expected <50ms"

    @pytest.mark.asyncio
    async def test_create_mapping_invalid_authority(self, manager):
        """Test source_authority validation rejects out-of-range values."""
        with pytest.raises(ValueError, match="source_authority must be between"):
            manager.create_mapping(
                chunk_id=str(uuid.uuid4()),
                graph_node_urn="urn:test:invalid",
                source_type="norm",
                source_authority=1.5,  # Invalid - > 1.0
                mapping_type="PRIMARY",
            )

        with pytest.raises(ValueError, match="source_authority must be between"):
            manager.create_mapping(
                chunk_id=str(uuid.uuid4()),
                graph_node_urn="urn:test:invalid",
                source_type="norm",
                source_authority=-0.1,  # Invalid - < 0.0
                mapping_type="PRIMARY",
            )

    @pytest.mark.asyncio
    async def test_get_chunks_for_expert_invalid_expert_type(self, manager):
        """Test expert_type validation rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid expert_type"):
            await manager.get_chunks_for_expert(
                graph_node_urn="urn:test:any",
                expert_type="invalid_expert",
                min_affinity=0.0,
            )

    @pytest.mark.asyncio
    async def test_update_expert_affinity_invalid_expert_type(self, manager):
        """Test update_expert_affinity validates expert_type."""
        with pytest.raises(ValueError, match="Invalid expert_type"):
            await manager.update_expert_affinity(
                chunk_id="any-chunk",
                graph_node_urn="urn:any",
                expert_type="invalid_expert",
                new_affinity=0.5,
            )

    @pytest.mark.asyncio
    async def test_update_expert_affinity_invalid_value(self, manager):
        """Test update_expert_affinity validates affinity range."""
        with pytest.raises(ValueError, match="new_affinity must be between"):
            await manager.update_expert_affinity(
                chunk_id="any-chunk",
                graph_node_urn="urn:any",
                expert_type="literal",
                new_affinity=1.5,  # Invalid
            )


class TestBridgeConfigValidation:
    """Test suite for BridgeConfig validation."""

    def test_valid_table_name(self):
        """Test valid table names are accepted."""
        config = BridgeConfig(table_name="my_bridge_table")
        assert config.table_name == "my_bridge_table"

        config = BridgeConfig(table_name="_private_table")
        assert config.table_name == "_private_table"

    def test_invalid_table_name_special_chars(self):
        """Test table names with special characters are rejected."""
        with pytest.raises(ValueError, match="Invalid table name"):
            BridgeConfig(table_name="table; DROP TABLE users;--")

    def test_invalid_table_name_starts_with_number(self):
        """Test table names starting with number are rejected."""
        with pytest.raises(ValueError, match="Invalid table name"):
            BridgeConfig(table_name="123_table")

    def test_table_name_too_long(self):
        """Test table names exceeding 63 chars are rejected."""
        long_name = "a" * 64
        with pytest.raises(ValueError, match="exceeds 63 character limit"):
            BridgeConfig(table_name=long_name)
