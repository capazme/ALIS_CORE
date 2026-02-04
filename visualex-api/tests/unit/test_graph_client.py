"""
Tests for Graph Client Module
=============================

Unit tests for FalkorDB client wrapper.
Uses mocking to avoid requiring a running FalkorDB instance.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from visualex.graph.client import FalkorDBClient
from visualex.graph.config import FalkorDBConfig
from visualex.graph.schema import NodeType, EdgeType, GraphSchema, Direction


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestFalkorDBClientInit:
    """Tests for client initialization."""

    def test_init_with_default_config(self):
        """Client initializes with default config."""
        client = FalkorDBClient()
        assert client.config.host == "localhost"
        assert client.config.port == 6379
        assert client.config.graph_name == "visualex_dev"

    def test_init_with_custom_config(self):
        """Client initializes with custom config."""
        config = FalkorDBConfig(
            host="custom.host.com",
            port=6380,
            graph_name="custom_graph",
        )
        client = FalkorDBClient(config)
        assert client.config.host == "custom.host.com"
        assert client.config.port == 6380

    def test_init_with_graph_name_override(self):
        """Graph name can be overridden in constructor."""
        config = FalkorDBConfig(graph_name="original")
        client = FalkorDBClient(config, graph_name="overridden")
        assert client.config.graph_name == "overridden"

    def test_init_not_connected(self):
        """Client is not connected after init."""
        client = FalkorDBClient()
        assert client.is_connected is False

    def test_init_has_schema(self):
        """Client has schema manager."""
        client = FalkorDBClient()
        assert isinstance(client.schema, GraphSchema)


# =============================================================================
# Connection Tests
# =============================================================================


class TestFalkorDBClientConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_sets_connected(self):
        """Connect sets connected flag."""
        with patch("visualex.graph.client.FalkorDBClient._connect_sync"):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock()
                client = FalkorDBClient()

                # Mock the import
                with patch.dict("sys.modules", {"falkordb": MagicMock()}):
                    client._connected = True  # Simulate successful connection
                    assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(self):
        """Connecting when already connected does nothing."""
        client = FalkorDBClient()
        client._connected = True

        # Should not raise and should not try to connect again
        await client.connect()  # This should be a no-op
        assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """Close resets connection state."""
        client = FalkorDBClient()
        client._connected = True
        client._db = Mock()
        client._graph = Mock()

        await client.close()

        assert client.is_connected is False
        assert client._db is None
        assert client._graph is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        """Close when not connected is safe."""
        client = FalkorDBClient()
        assert client.is_connected is False
        await client.close()  # Should not raise
        assert client.is_connected is False


# =============================================================================
# Query Tests
# =============================================================================


class TestFalkorDBClientQuery:
    """Tests for query execution."""

    @pytest.mark.asyncio
    async def test_query_when_not_connected_raises(self):
        """Query raises when not connected."""
        client = FalkorDBClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.query("RETURN 1")

    @pytest.mark.asyncio
    async def test_query_returns_list(self):
        """Query returns list of dicts."""
        client = FalkorDBClient()
        client._connected = True

        # Mock the query execution
        with patch.object(client, "_query_sync", return_value=[{"count": 42}]):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=[{"count": 42}]
                )
                results = await client.query("MATCH (n) RETURN count(n) as count")
                assert isinstance(results, list)


# =============================================================================
# Schema Initialization Tests
# =============================================================================


class TestFalkorDBClientSchema:
    """Tests for schema operations."""

    @pytest.mark.asyncio
    async def test_initialize_schema_creates_indexes(self):
        """Initialize schema creates all indexes."""
        client = FalkorDBClient()
        client._connected = True

        # Count expected queries
        expected_count = (
            len(client.schema.get_create_index_queries()) +
            len(client.schema.get_create_fulltext_index_queries())
        )

        query_count = 0
        async def mock_query(cypher, params=None):
            nonlocal query_count
            query_count += 1
            return []

        client.query = mock_query

        executed = await client.initialize_schema()

        assert query_count == expected_count
        assert client.schema._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_schema_handles_existing_indexes(self):
        """Initialize schema handles 'already exists' errors gracefully."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query_with_error(cypher, params=None):
            raise Exception("Index already exists")

        client.query = mock_query_with_error

        # Should not raise
        executed = await client.initialize_schema()
        assert executed == []


# =============================================================================
# Node Operations Tests
# =============================================================================


class TestFalkorDBClientNodes:
    """Tests for node operations."""

    @pytest.mark.asyncio
    async def test_create_node_validates_data(self):
        """Create node validates data before execution."""
        client = FalkorDBClient()
        client._connected = True

        # Missing required URN
        result = await client.create_node(
            NodeType.NORMA,
            {"tipo_atto": "legge"}  # Missing urn
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_node_returns_created_node(self):
        """Create node returns the created node."""
        client = FalkorDBClient()
        client._connected = True

        created_node = {"n": {"properties": {"urn": "test:urn"}}}

        async def mock_query(cypher, params):
            return [created_node]

        client.query = mock_query

        result = await client.create_node(
            NodeType.NORMA,
            {"urn": "test:urn", "tipo_atto": "legge"}
        )

        assert result == created_node

    @pytest.mark.asyncio
    async def test_merge_node_builds_correct_query(self):
        """Merge node builds MERGE query."""
        client = FalkorDBClient()
        client._connected = True

        captured_query = None

        async def mock_query(cypher, params):
            nonlocal captured_query
            captured_query = cypher
            return [{"n": {}}]

        client.query = mock_query

        await client.merge_node(
            NodeType.COMMA,
            "urn",
            "urn:nir:stato:legge:2020;1~art1-com1",
            {"posizione": 1, "testo": "Comma text"}
        )

        assert "MERGE" in captured_query
        assert "Comma" in captured_query

    @pytest.mark.asyncio
    async def test_get_node_by_urn(self):
        """Get node by URN returns matching node."""
        client = FalkorDBClient()
        client._connected = True

        expected_node = {"n": {"properties": {"urn": "test:urn"}}}

        async def mock_query(cypher, params):
            assert params["urn"] == "test:urn"
            return [expected_node]

        client.query = mock_query

        result = await client.get_node_by_urn(NodeType.NORMA, "test:urn")
        assert result == expected_node

    @pytest.mark.asyncio
    async def test_get_node_by_urn_not_found(self):
        """Get node by URN returns None when not found."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params):
            return []

        client.query = mock_query

        result = await client.get_node_by_urn(NodeType.NORMA, "nonexistent")
        assert result is None


# =============================================================================
# Edge Operations Tests
# =============================================================================


class TestFalkorDBClientEdges:
    """Tests for edge operations."""

    @pytest.mark.asyncio
    async def test_create_edge(self):
        """Create edge creates relationship."""
        client = FalkorDBClient()
        client._connected = True

        captured_query = None

        async def mock_query(cypher, params):
            nonlocal captured_query
            captured_query = cypher
            return [{"r": {}}]

        client.query = mock_query

        result = await client.create_edge(
            EdgeType.CONTIENE,
            NodeType.NORMA, "urn:norma",
            NodeType.COMMA, "urn:comma",
        )

        assert "contiene" in captured_query  # Edge values are lowercase
        assert result is not None

    @pytest.mark.asyncio
    async def test_create_edge_with_properties(self):
        """Create edge with properties."""
        client = FalkorDBClient()
        client._connected = True

        captured_params = None

        async def mock_query(cypher, params):
            nonlocal captured_params
            captured_params = params
            return [{"r": {}}]

        client.query = mock_query

        await client.create_edge(
            EdgeType.CITA,
            NodeType.NORMA, "urn1",
            NodeType.NORMA, "urn2",
            properties={"tipo_citazione": "rinvio"}
        )

        assert captured_params["tipo_citazione"] == "rinvio"

    @pytest.mark.asyncio
    async def test_create_edge_handles_error(self):
        """Create edge handles errors gracefully."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params):
            raise Exception("Node not found")

        client.query = mock_query

        result = await client.create_edge(
            EdgeType.CONTIENE,
            NodeType.NORMA, "nonexistent",
            NodeType.COMMA, "also_nonexistent",
        )

        assert result is None


# =============================================================================
# Traversal Tests
# =============================================================================


class TestFalkorDBClientTraversal:
    """Tests for graph traversal."""

    @pytest.mark.asyncio
    async def test_find_related_nodes_outgoing(self):
        """Find related nodes with outgoing direction."""
        client = FalkorDBClient()
        client._connected = True

        captured_query = None

        async def mock_query(cypher, params):
            nonlocal captured_query
            captured_query = cypher
            return []

        client.query = mock_query

        await client.find_related_nodes(
            NodeType.NORMA,
            "urn:test",
            direction=Direction.OUT,
        )

        assert "->" in captured_query

    @pytest.mark.asyncio
    async def test_find_related_nodes_incoming(self):
        """Find related nodes with incoming direction."""
        client = FalkorDBClient()
        client._connected = True

        captured_query = None

        async def mock_query(cypher, params):
            nonlocal captured_query
            captured_query = cypher
            return []

        client.query = mock_query

        await client.find_related_nodes(
            NodeType.NORMA,
            "urn:test",
            direction=Direction.IN,
        )

        assert "<-" in captured_query

    @pytest.mark.asyncio
    async def test_find_related_nodes_string_direction(self):
        """Find related nodes accepts string direction for backwards compat."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params):
            return []

        client.query = mock_query

        # Should not raise with string direction
        await client.find_related_nodes(
            NodeType.NORMA,
            "urn:test",
            direction="out",
        )

    @pytest.mark.asyncio
    async def test_find_related_nodes_with_edge_filter(self):
        """Find related nodes filtered by edge type."""
        client = FalkorDBClient()
        client._connected = True

        captured_query = None

        async def mock_query(cypher, params):
            nonlocal captured_query
            captured_query = cypher
            return []

        client.query = mock_query

        await client.find_related_nodes(
            NodeType.NORMA,
            "urn:test",
            edge_types=[EdgeType.CITA, EdgeType.SOSTITUISCE],
        )

        assert "cita" in captured_query
        assert "sostituisce" in captured_query


# =============================================================================
# Utility Tests
# =============================================================================


class TestFalkorDBClientUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_health_check_when_connected(self):
        """Health check returns True when healthy."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params=None):
            return [{"col_0": 1}]

        client.query = mock_query

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_when_query_fails(self):
        """Health check returns False when query fails."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params=None):
            raise Exception("Connection lost")

        client.query = mock_query

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Get stats returns node and edge counts."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params=None):
            return [{"count": 10}]

        client.query = mock_query

        stats = await client.get_stats()

        # Should have counts for all node and edge types
        assert "nodes_norma" in stats
        assert "nodes_comma" in stats  # Comma instead of Articolo
        assert "edges_contiene" in stats

    @pytest.mark.asyncio
    async def test_delete_all(self):
        """Delete all removes all nodes when confirmed."""
        client = FalkorDBClient()
        client._connected = True

        async def mock_query(cypher, params=None):
            return [{"deleted": 100}]

        client.query = mock_query

        deleted = await client.delete_all(confirm=True)
        assert deleted == 100

    @pytest.mark.asyncio
    async def test_delete_all_requires_confirm(self):
        """Delete all raises without confirm=True."""
        client = FalkorDBClient()
        client._connected = True

        with pytest.raises(ValueError, match="confirm=True"):
            await client.delete_all()

    @pytest.mark.asyncio
    async def test_delete_all_confirm_false_raises(self):
        """Delete all raises when confirm=False."""
        client = FalkorDBClient()
        client._connected = True

        with pytest.raises(ValueError, match="confirm=True"):
            await client.delete_all(confirm=False)


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestFalkorDBClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects(self):
        """Context manager calls connect on entry."""
        connect_called = False

        async def mock_connect():
            nonlocal connect_called
            connect_called = True

        client = FalkorDBClient()
        client.connect = mock_connect
        client.close = AsyncMock()

        async with client:
            assert connect_called

    @pytest.mark.asyncio
    async def test_context_manager_closes(self):
        """Context manager calls close on exit."""
        close_called = False

        async def mock_close():
            nonlocal close_called
            close_called = True

        client = FalkorDBClient()
        client.connect = AsyncMock()
        client.close = mock_close

        async with client:
            pass

        assert close_called

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self):
        """Context manager closes even on exception."""
        close_called = False

        async def mock_close():
            nonlocal close_called
            close_called = True

        client = FalkorDBClient()
        client.connect = AsyncMock()
        client.close = mock_close

        with pytest.raises(ValueError):
            async with client:
                raise ValueError("test error")

        assert close_called
