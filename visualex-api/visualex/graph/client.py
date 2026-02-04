"""
FalkorDB Client
===============

Async client wrapper for FalkorDB graph database.

FalkorDB runs on Redis protocol and supports Cypher queries.
This client provides async operations via executor for the sync falkordb-py library.

Example:
    from visualex.graph import FalkorDBClient, FalkorDBConfig

    config = FalkorDBConfig()
    client = FalkorDBClient(config)
    await client.connect()

    # Execute Cypher query
    results = await client.query('''
        MATCH (n:Norma {urn: $urn})
        RETURN n.titolo, n.tipo_atto
    ''', {"urn": "urn:nir:stato:legge:2020-12-30;178"})

    await client.close()
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Union

from visualex.graph.config import FalkorDBConfig
from visualex.graph.schema import GraphSchema, NodeType, EdgeType, Direction

if TYPE_CHECKING:
    from falkordb import FalkorDB, Graph

logger = logging.getLogger(__name__)

__all__ = ["FalkorDBClient"]


class FalkorDBClient:
    """
    Async client for FalkorDB graph database.

    Wraps the synchronous falkordb-py library with async execution.
    """

    def __init__(
        self,
        config: Optional[FalkorDBConfig] = None,
        graph_name: Optional[str] = None,
    ):
        """
        Initialize FalkorDB client.

        Args:
            config: FalkorDB configuration (optional, uses defaults)
            graph_name: Override graph name (useful for test isolation)
        """
        self.config = config or FalkorDBConfig()

        # Allow graph_name override
        if graph_name:
            self.config.graph_name = graph_name

        self._db: Optional["FalkorDB"] = None
        self._graph: Optional["Graph"] = None
        self._connected = False
        self._schema = GraphSchema()

        logger.info(
            "FalkorDBClient initialized - host=%s:%d, graph=%s",
            self.config.host,
            self.config.port,
            self.config.graph_name,
        )

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    @property
    def schema(self) -> GraphSchema:
        """Get the graph schema manager."""
        return self._schema

    async def connect(self) -> None:
        """
        Establish connection to FalkorDB.

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.debug("Already connected to FalkorDB")
            return

        try:
            # Import here to allow optional dependency
            from falkordb import FalkorDB

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._connect_sync, FalkorDB)

            logger.info(
                "Connected to FalkorDB at %s:%d",
                self.config.host,
                self.config.port,
            )

        except ImportError as e:
            raise ImportError(
                "falkordb package not installed. "
                "Install with: pip install falkordb"
            ) from e
        except Exception as e:
            logger.error("Failed to connect to FalkorDB: %s", e)
            raise ConnectionError(f"FalkorDB connection failed: {e}") from e

    def _connect_sync(self, falkordb_class: type) -> None:
        """Synchronous connection (called in executor)."""
        self._db = falkordb_class(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
        )
        self._graph = self._db.select_graph(self.config.graph_name)
        self._connected = True

    async def close(self) -> None:
        """Close connection."""
        if not self._connected:
            return

        # FalkorDB connection is managed by redis connection pool
        self._connected = False
        self._db = None
        self._graph = None
        logger.info("Disconnected from FalkorDB")

    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dicts

        Raises:
            RuntimeError: If not connected
            Exception: If query fails

        Example:
            results = await client.query(
                "MATCH (n:Norma {urn: $urn}) RETURN n.titolo, n.testo_vigente",
                {"urn": "urn:nir:stato:legge:2020-12-30;178"}
            )
        """
        if not self._connected:
            raise RuntimeError("Not connected to FalkorDB. Call connect() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._query_sync,
            cypher,
            params or {},
        )

    def _query_sync(
        self, cypher: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute query synchronously (called in executor)."""
        try:
            result = self._graph.query(cypher, params)

            # Convert result set to list of dicts
            records = []
            if result.result_set:
                headers = result.header

                for row in result.result_set:
                    record = {}
                    for i, header in enumerate(headers):
                        # Extract column name (format is [[type, alias]])
                        col_name = header[1] if len(header) > 1 else f"col_{i}"
                        value = row[i]

                        # Handle FalkorDB Node/Edge objects
                        if hasattr(value, "properties"):
                            record[col_name] = {
                                "properties": value.properties,
                                "labels": getattr(value, "labels", []),
                                "id": getattr(value, "id", None),
                            }
                        else:
                            record[col_name] = value

                    records.append(record)

            logger.debug(
                "Query executed: %s... (params=%s) -> %d records",
                cypher[:80],
                list(params.keys()),
                len(records),
            )
            return records

        except Exception as e:
            logger.error("Query failed: %s... Error: %s", cypher[:80], e)
            raise

    async def initialize_schema(self) -> List[str]:
        """
        Initialize graph schema with indexes.

        Creates all standard and full-text indexes.

        Returns:
            List of executed queries

        Example:
            queries = await client.initialize_schema()
            print(f"Created {len(queries)} indexes")
        """
        executed = []

        # Create standard indexes
        for query in self._schema.get_create_index_queries():
            try:
                await self.query(query)
                executed.append(query)
                logger.debug("Created index: %s", query)
            except Exception as e:
                # Index may already exist
                if "already exists" not in str(e).lower():
                    logger.warning("Failed to create index: %s - %s", query, e)

        # Create full-text indexes
        for query in self._schema.get_create_fulltext_index_queries():
            try:
                await self.query(query)
                executed.append(query)
                logger.debug("Created full-text index: %s", query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Failed to create full-text index: %s - %s", query, e)

        self._schema._initialized = True
        self._schema._indexes_created = executed
        logger.info("Schema initialized with %d indexes", len(executed))

        return executed

    async def create_node(
        self,
        node_type: NodeType,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new node.

        Args:
            node_type: The node type
            data: Node properties

        Returns:
            Created node data or None if failed

        Example:
            node = await client.create_node(
                NodeType.NORMA,
                {
                    "urn": "urn:nir:stato:legge:2020-12-30;178",
                    "tipo_atto": "legge",
                    "numero": "178",
                    "titolo": "Bilancio di previsione 2021",
                    "data_emanazione": "2020-12-30",
                }
            )
        """
        # Validate data
        errors = self._schema.validate_node_data(node_type, data)
        if errors:
            logger.error("Node validation failed: %s", errors)
            return None

        query, params = self._schema.build_create_node_query(node_type, data, "n")
        query += " RETURN n"

        results = await self.query(query, params)
        return results[0] if results else None

    async def merge_node(
        self,
        node_type: NodeType,
        match_key: str,
        match_value: Any,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Merge (upsert) a node.

        Args:
            node_type: The node type
            match_key: Property to match on (usually 'urn' or 'id')
            match_value: Value to match
            data: Node properties

        Returns:
            Merged node data or None if failed

        Example:
            node = await client.merge_node(
                NodeType.COMMA,
                "urn",
                "urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1",
                {
                    "posizione": 1,
                    "testo": "Nei contratti con prestazioni corrispettive...",
                }
            )
        """
        query, params = self._schema.build_merge_node_query(
            node_type, match_key, match_value, data, "n"
        )
        query += " RETURN n"

        results = await self.query(query, params)
        return results[0] if results else None

    async def create_edge(
        self,
        edge_type: EdgeType,
        from_type: NodeType,
        from_value: str,
        to_type: NodeType,
        to_value: str,
        properties: Optional[Dict[str, Any]] = None,
        from_key: str = "urn",
        to_key: str = "urn",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a relationship between nodes.

        Args:
            edge_type: The edge type
            from_type: Source node type
            from_value: Source node identifier value
            to_type: Target node type
            to_value: Target node identifier value
            properties: Optional edge properties
            from_key: Source match key (default: "urn", can be "node_id")
            to_key: Target match key (default: "urn", can be "node_id")

        Returns:
            Created edge data or None if failed

        Example:
            # Match by URN (default)
            edge = await client.create_edge(
                EdgeType.CONTIENE,
                NodeType.NORMA, "urn:nir:stato:regio.decreto:1942-03-16;262",
                NodeType.COMMA, "urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1",
            )

            # Match by node_id
            edge = await client.create_edge(
                EdgeType.DISCIPLINA,
                NodeType.NORMA, "norma_001",
                NodeType.CONCETTO, "concetto_contratto",
                from_key="node_id",
                to_key="node_id",
            )
        """
        query, params = self._schema.build_create_edge_query(
            edge_type,
            from_type, from_key, from_value,
            to_type, to_key, to_value,
            properties,
        )

        try:
            results = await self.query(query, params)
            return results[0] if results else None
        except Exception as e:
            logger.error(
                "Failed to create edge %s: %s -> %s: %s",
                edge_type.value, from_value, to_value, e
            )
            return None

    async def get_node_by_urn(
        self,
        node_type: NodeType,
        urn: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a node by its URN.

        Args:
            node_type: The node type
            urn: The node URN

        Returns:
            Node data or None if not found

        Example:
            node = await client.get_node_by_urn(
                NodeType.COMMA,
                "urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1"
            )
        """
        cypher = f"MATCH (n:{node_type.value} {{urn: $urn}}) RETURN n"
        results = await self.query(cypher, {"urn": urn})
        return results[0] if results else None

    async def find_related_nodes(
        self,
        node_type: NodeType,
        urn: str,
        edge_types: Optional[List[EdgeType]] = None,
        direction: Union[Direction, str] = Direction.BOTH,
        max_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Find nodes related to a given node.

        Args:
            node_type: The source node type
            urn: Source node URN
            edge_types: Filter by edge types (None = all)
            direction: Direction.IN, Direction.OUT, or Direction.BOTH
            max_depth: Maximum traversal depth

        Returns:
            List of related nodes with relationship info

        Example:
            related = await client.find_related_nodes(
                NodeType.NORMA,
                "urn:nir:stato:regio.decreto:1942-03-16;262",
                edge_types=[EdgeType.CITA],
                direction=Direction.OUT,
                max_depth=2,
            )
        """
        # Normalize direction to enum
        if isinstance(direction, str):
            direction = Direction(direction)

        # Build edge pattern
        edge_pattern = ""
        if edge_types:
            edge_names = "|".join(e.value for e in edge_types)
            edge_pattern = f":{edge_names}"

        # Build direction pattern
        if direction == Direction.OUT:
            pattern = f"-[r{edge_pattern}*1..{max_depth}]->"
        elif direction == Direction.IN:
            pattern = f"<-[r{edge_pattern}*1..{max_depth}]-"
        else:
            pattern = f"-[r{edge_pattern}*1..{max_depth}]-"

        cypher = f"""
            MATCH (start:{node_type.value} {{urn: $urn}}){pattern}(related)
            RETURN DISTINCT related, labels(related) AS labels
        """

        results = await self.query(cypher, {"urn": urn})
        return results

    async def health_check(self) -> bool:
        """
        Check if FalkorDB is healthy and reachable.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._connected:
                await self.connect()

            await self.query("RETURN 1")
            return True

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    async def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.

        Returns:
            Dict with node and edge counts per type
        """
        stats = {}

        # Count nodes by type
        for node_type in NodeType:
            cypher = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
            results = await self.query(cypher)
            stats[f"nodes_{node_type.value.lower()}"] = (
                results[0]["count"] if results else 0
            )

        # Count edges by type
        for edge_type in EdgeType:
            cypher = f"MATCH ()-[r:{edge_type.value}]->() RETURN count(r) as count"
            results = await self.query(cypher)
            stats[f"edges_{edge_type.value.lower()}"] = (
                results[0]["count"] if results else 0
            )

        return stats

    async def delete_all(self, confirm: bool = False) -> int:
        """
        Delete all nodes and edges in the graph.

        WARNING: This is destructive and cannot be undone.

        Args:
            confirm: Must be True to execute deletion (safety check)

        Returns:
            Number of deleted nodes

        Raises:
            ValueError: If confirm is not True

        Example:
            # Must explicitly confirm deletion
            deleted = await client.delete_all(confirm=True)
        """
        if not confirm:
            raise ValueError(
                "delete_all() requires confirm=True to prevent accidental data loss"
            )

        cypher = "MATCH (n) DETACH DELETE n RETURN count(n) as deleted"
        results = await self.query(cypher)
        deleted = results[0]["deleted"] if results else 0
        logger.warning("Deleted %d nodes from graph", deleted)
        return deleted

    async def __aenter__(self) -> "FalkorDBClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
