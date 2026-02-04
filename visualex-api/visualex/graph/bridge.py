"""
Bridge Table Manager
====================

Manages chunk-to-graph mappings with expert affinity support for MERL-T.

Features:
- Extended schema with source_type, source_authority, mapping_type
- Expert affinity auto-computation from source_type
- Query with expert affinity filtering
- Batch operations for ingestion
- F8 feedback hook stub for future RLCF integration

The Bridge Table connects:
- Qdrant chunks (vector embeddings) -> FalkorDB nodes (knowledge graph)
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

from visualex.graph.chunking import SourceType
from visualex.graph.qdrant import ExpertType

__all__ = [
    "BridgeConfig",
    "BridgeTableManager",
    "BridgeMapping",
    "MappingType",
    "DEFAULT_EXPERT_AFFINITIES",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class MappingType(str, Enum):
    """Type of chunk-to-graph mapping."""

    PRIMARY = "PRIMARY"  # Main topic of the chunk
    REFERENCE = "REFERENCE"  # Citation/reference in the chunk
    CONCEPT = "CONCEPT"  # Related legal concept
    DOCTRINE = "DOCTRINE"  # Doctrinal interpretation


# Default expert affinities by source type (same as qdrant.py for consistency)
DEFAULT_EXPERT_AFFINITIES: Dict[SourceType, Dict[str, float]] = {
    SourceType.NORM: {
        "literal": 0.9,
        "systemic": 0.8,
        "principles": 0.5,
        "precedent": 0.3,
    },
    SourceType.JURISPRUDENCE: {
        "literal": 0.3,
        "systemic": 0.5,
        "principles": 0.6,
        "precedent": 0.9,
    },
    SourceType.COMMENTARY: {
        "literal": 0.5,
        "systemic": 0.6,
        "principles": 0.7,
        "precedent": 0.6,
    },
    SourceType.DOCTRINE: {
        "literal": 0.4,
        "systemic": 0.5,
        "principles": 0.9,
        "precedent": 0.4,
    },
}


# =============================================================================
# Data Structures
# =============================================================================


def _validate_table_name(name: str) -> str:
    """
    Validate table name to prevent SQL injection.

    Args:
        name: Table name to validate

    Returns:
        Validated table name

    Raises:
        ValueError: If table name is invalid
    """
    # Only allow alphanumeric, underscores, and must start with letter/underscore
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(
            f"Invalid table name '{name}'. Must start with letter/underscore "
            "and contain only alphanumeric characters and underscores."
        )
    if len(name) > 63:  # PostgreSQL identifier limit
        raise ValueError(f"Table name '{name}' exceeds 63 character limit.")
    return name


@dataclass
class BridgeConfig:
    """Configuration for Bridge Table connection."""

    host: str = "localhost"
    port: int = 5433  # Dev container port
    database: str = "rlcf_dev"
    user: str = "dev"
    password: str = "devpassword"
    table_name: str = "bridge_table_enhanced"
    pool_size: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.table_name = _validate_table_name(self.table_name)

    def get_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """Create configuration from environment variables."""
        try:
            port = int(os.getenv("BRIDGE_DB_PORT", "5433"))
        except ValueError:
            logger.warning("Invalid BRIDGE_DB_PORT, using default 5433")
            port = 5433

        return cls(
            host=os.getenv("BRIDGE_DB_HOST", "localhost"),
            port=port,
            database=os.getenv("BRIDGE_DB_NAME", "rlcf_dev"),
            user=os.getenv("BRIDGE_DB_USER", "dev"),
            password=os.getenv("BRIDGE_DB_PASSWORD", "devpassword"),
            table_name=os.getenv("BRIDGE_TABLE_NAME", "bridge_table_enhanced"),
        )


@dataclass
class BridgeMapping:
    """A chunk-to-graph mapping with expert affinity."""

    chunk_id: str  # UUID as string
    graph_node_urn: str
    source_type: str  # norm|jurisprudence|commentary|doctrine
    source_authority: float  # 0.0-1.0
    mapping_type: str  # PRIMARY|REFERENCE|CONCEPT|DOCTRINE
    expert_affinity: Dict[str, float]  # Per-expert weights
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for insertion."""
        return {
            "chunk_id": self.chunk_id,
            "graph_node_urn": self.graph_node_urn,
            "source_type": self.source_type,
            "source_authority": self.source_authority,
            "mapping_type": self.mapping_type,
            "expert_affinity": self.expert_affinity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }


# =============================================================================
# Bridge Table Manager
# =============================================================================


class BridgeTableManager:
    """
    Manages chunk-to-graph mappings with expert affinity support.

    Provides:
    - Extended schema with source_type, source_authority, mapping_type
    - Expert affinity auto-computation from source_type
    - Query with expert affinity filtering for expert-specific retrieval
    - Batch operations for efficient ingestion
    - F8 feedback hook for future RLCF integration

    Example:
        manager = BridgeTableManager(config)
        await manager.connect()
        await manager.ensure_table_exists()

        # Add mapping
        mapping = manager.create_mapping(
            chunk_id="uuid-string",
            graph_node_urn="urn:nir:stato:legge:2020;178~art1",
            source_type="norm",
            source_authority=1.0,
            mapping_type="PRIMARY"
        )
        await manager.add_mapping(mapping)

        # Query for expert
        results = await manager.get_chunks_for_expert(
            graph_node_urn="urn:...",
            expert_type="literal",
            min_affinity=0.7
        )

        await manager.close()
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        """
        Initialize BridgeTableManager.

        Args:
            config: Bridge table configuration. If None, uses environment variables.
        """
        self.config = config or BridgeConfig.from_env()
        self._engine = None
        self._session_maker = None
        self._connected = False

        logger.info(
            "BridgeTableManager initialized: %s:%d/%s (table=%s)",
            self.config.host,
            self.config.port,
            self.config.database,
            self.config.table_name,
        )

    async def connect(self):
        """Establish connection pool to PostgreSQL."""
        if self._connected:
            logger.debug("Already connected to PostgreSQL")
            return

        try:
            from sqlalchemy.ext.asyncio import (
                create_async_engine,
                async_sessionmaker,
                AsyncSession,
            )
        except ImportError as e:
            raise ImportError(
                "sqlalchemy[asyncio] and asyncpg are required. "
                "Install with: pip install sqlalchemy[asyncio] asyncpg"
            ) from e

        connection_string = self.config.get_connection_string()
        self._engine = create_async_engine(
            connection_string,
            pool_size=self.config.pool_size,
            echo=False,
        )

        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._connected = True
        logger.info(
            "Connected to PostgreSQL at %s:%d",
            self.config.host,
            self.config.port,
        )

    async def close(self):
        """Close connection pool."""
        if not self._connected:
            return

        await self._engine.dispose()
        self._connected = False
        logger.info("Disconnected from PostgreSQL")

    async def ensure_table_exists(self):
        """Create the enhanced bridge table if it doesn't exist."""
        if not self._connected:
            await self.connect()

        from sqlalchemy import text

        table_name = self.config.table_name

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(36) NOT NULL,
            graph_node_urn VARCHAR(500) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            source_authority FLOAT NOT NULL DEFAULT 0.5,
            mapping_type VARCHAR(50) NOT NULL DEFAULT 'PRIMARY',
            expert_affinity JSONB NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB,
            CONSTRAINT {table_name}_chunk_urn_key UNIQUE (chunk_id, graph_node_urn),
            CONSTRAINT {table_name}_authority_check CHECK (source_authority >= 0 AND source_authority <= 1)
        );

        -- Indexes for fast queries
        CREATE INDEX IF NOT EXISTS {table_name}_chunk_id_idx ON {table_name}(chunk_id);
        CREATE INDEX IF NOT EXISTS {table_name}_graph_node_urn_idx ON {table_name}(graph_node_urn);
        CREATE INDEX IF NOT EXISTS {table_name}_source_type_idx ON {table_name}(source_type);
        CREATE INDEX IF NOT EXISTS {table_name}_mapping_type_idx ON {table_name}(mapping_type);

        -- GIN index for expert_affinity JSONB queries
        CREATE INDEX IF NOT EXISTS {table_name}_expert_affinity_idx ON {table_name} USING GIN (expert_affinity);
        """

        async with self._engine.begin() as conn:
            for statement in create_sql.strip().split(";"):
                if statement.strip():
                    await conn.execute(text(statement))

        logger.info("Table %s ensured to exist", table_name)

    async def drop_table(self):
        """Drop the bridge table (WARNING: destructive operation)."""
        if not self._connected:
            await self.connect()

        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DROP TABLE IF EXISTS {self.config.table_name} CASCADE")
            )

        logger.warning("Table %s dropped", self.config.table_name)

    def create_mapping(
        self,
        chunk_id: str,
        graph_node_urn: str,
        source_type: str,
        source_authority: float = 0.5,
        mapping_type: str = "PRIMARY",
        expert_affinity: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BridgeMapping:
        """
        Create a BridgeMapping with auto-computed expert affinity (AC1).

        Args:
            chunk_id: UUID of the chunk in Qdrant
            graph_node_urn: URN of the node in FalkorDB
            source_type: Type of source (norm, jurisprudence, commentary, doctrine)
            source_authority: Authority score 0.0-1.0
            mapping_type: Type of mapping (PRIMARY, REFERENCE, CONCEPT, DOCTRINE)
            expert_affinity: Custom expert affinity (auto-computed if None)
            metadata: Additional metadata

        Returns:
            BridgeMapping with expert_affinity computed from source_type

        Raises:
            ValueError: If source_authority is not in range [0.0, 1.0]
        """
        # Validate source_authority range (M1)
        if not 0.0 <= source_authority <= 1.0:
            raise ValueError(
                f"source_authority must be between 0.0 and 1.0, got {source_authority}"
            )

        # Auto-compute expert affinity if not provided
        if expert_affinity is None:
            try:
                source_type_enum = SourceType(source_type.lower())
                expert_affinity = DEFAULT_EXPERT_AFFINITIES.get(
                    source_type_enum,
                    {
                        ExpertType.LITERAL.value: 0.5,
                        ExpertType.SYSTEMIC.value: 0.5,
                        ExpertType.PRINCIPLES.value: 0.5,
                        ExpertType.PRECEDENT.value: 0.5,
                    },
                )
            except ValueError:
                # Unknown source type - use neutral affinity
                expert_affinity = {
                    ExpertType.LITERAL.value: 0.5,
                    ExpertType.SYSTEMIC.value: 0.5,
                    ExpertType.PRINCIPLES.value: 0.5,
                    ExpertType.PRECEDENT.value: 0.5,
                }

        # Ensure metadata is always a dict (M3 - consistent handling)
        final_metadata = metadata if metadata is not None else {}

        return BridgeMapping(
            chunk_id=chunk_id,
            graph_node_urn=graph_node_urn,
            source_type=source_type.lower(),
            source_authority=source_authority,
            mapping_type=mapping_type.upper(),
            expert_affinity=expert_affinity,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=final_metadata,
        )

    async def add_mapping(self, mapping: BridgeMapping) -> int:
        """
        Add a single mapping to the bridge table.

        Args:
            mapping: BridgeMapping to insert

        Returns:
            ID of the created entry
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text
        import json

        insert_sql = text(f"""
            INSERT INTO {self.config.table_name}
            (chunk_id, graph_node_urn, source_type, source_authority, mapping_type,
             expert_affinity, metadata)
            VALUES (:chunk_id, :graph_node_urn, :source_type, :source_authority,
                    :mapping_type, CAST(:expert_affinity AS jsonb), CAST(:metadata AS jsonb))
            ON CONFLICT (chunk_id, graph_node_urn) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                source_authority = EXCLUDED.source_authority,
                mapping_type = EXCLUDED.mapping_type,
                expert_affinity = EXCLUDED.expert_affinity,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id
        """)

        async with self._session_maker() as session:
            result = await session.execute(
                insert_sql,
                {
                    "chunk_id": mapping.chunk_id,
                    "graph_node_urn": mapping.graph_node_urn,
                    "source_type": mapping.source_type,
                    "source_authority": mapping.source_authority,
                    "mapping_type": mapping.mapping_type,
                    "expert_affinity": json.dumps(mapping.expert_affinity),
                    "metadata": json.dumps(mapping.metadata) if mapping.metadata else None,
                },
            )
            await session.commit()
            entry_id = result.scalar()

            logger.debug(
                "Added mapping: chunk_id=%s -> node_urn=%s",
                mapping.chunk_id,
                mapping.graph_node_urn[:50],
            )

            return entry_id

    async def add_mappings_batch(
        self,
        mappings: List[BridgeMapping],
        batch_size: int = 100,
    ) -> int:
        """
        Add multiple mappings in batches (AC2).

        Uses a single transaction for atomicity - all mappings succeed or none.

        Args:
            mappings: List of BridgeMapping to insert
            batch_size: Number of mappings per batch (for progress logging)

        Returns:
            Total number of mappings inserted/updated

        Raises:
            RuntimeError: If not connected or transaction fails
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        if not mappings:
            return 0

        from sqlalchemy import text
        import json

        insert_sql = text(f"""
            INSERT INTO {self.config.table_name}
            (chunk_id, graph_node_urn, source_type, source_authority, mapping_type,
             expert_affinity, metadata)
            VALUES (:chunk_id, :graph_node_urn, :source_type, :source_authority,
                    :mapping_type, CAST(:expert_affinity AS jsonb), CAST(:metadata AS jsonb))
            ON CONFLICT (chunk_id, graph_node_urn) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                source_authority = EXCLUDED.source_authority,
                mapping_type = EXCLUDED.mapping_type,
                expert_affinity = EXCLUDED.expert_affinity,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """)

        total_inserted = 0

        # Single transaction for atomicity - all or nothing
        async with self._session_maker() as session:
            try:
                for i in range(0, len(mappings), batch_size):
                    batch = mappings[i : i + batch_size]

                    for mapping in batch:
                        await session.execute(
                            insert_sql,
                            {
                                "chunk_id": mapping.chunk_id,
                                "graph_node_urn": mapping.graph_node_urn,
                                "source_type": mapping.source_type,
                                "source_authority": mapping.source_authority,
                                "mapping_type": mapping.mapping_type,
                                "expert_affinity": json.dumps(mapping.expert_affinity),
                                "metadata": json.dumps(mapping.metadata) if mapping.metadata else None,
                            },
                        )

                    total_inserted += len(batch)

                    if total_inserted % 500 == 0:
                        logger.info("Processing %d/%d mappings", total_inserted, len(mappings))

                # Commit only after all mappings processed successfully
                await session.commit()
                logger.info("Batch inserted %d mappings", total_inserted)

            except Exception as e:
                await session.rollback()
                logger.error("Batch insert failed, rolled back: %s", e)
                raise RuntimeError(f"Batch insert failed: {e}") from e

        return total_inserted

    async def get_mappings_for_chunk(
        self,
        chunk_id: str,
        mapping_type: Optional[str] = None,
    ) -> List[BridgeMapping]:
        """
        Get all graph node mappings for a chunk.

        Args:
            chunk_id: UUID of the chunk
            mapping_type: Optional filter by mapping type

        Returns:
            List of BridgeMapping
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text

        query = f"""
            SELECT chunk_id, graph_node_urn, source_type, source_authority,
                   mapping_type, expert_affinity, created_at, updated_at, metadata
            FROM {self.config.table_name}
            WHERE chunk_id = :chunk_id
        """
        params = {"chunk_id": chunk_id}

        if mapping_type:
            query += " AND mapping_type = :mapping_type"
            params["mapping_type"] = mapping_type.upper()

        async with self._session_maker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

            return [
                BridgeMapping(
                    chunk_id=row[0],
                    graph_node_urn=row[1],
                    source_type=row[2],
                    source_authority=row[3],
                    mapping_type=row[4],
                    expert_affinity=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=row[8] or {},
                )
                for row in rows
            ]

    async def get_chunks_for_node(
        self,
        graph_node_urn: str,
        source_type: Optional[str] = None,
        min_authority: Optional[float] = None,
    ) -> List[BridgeMapping]:
        """
        Get all chunks linked to a graph node.

        Args:
            graph_node_urn: URN of the graph node
            source_type: Optional filter by source type
            min_authority: Optional minimum authority threshold

        Returns:
            List of BridgeMapping
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text

        query = f"""
            SELECT chunk_id, graph_node_urn, source_type, source_authority,
                   mapping_type, expert_affinity, created_at, updated_at, metadata
            FROM {self.config.table_name}
            WHERE graph_node_urn = :graph_node_urn
        """
        params = {"graph_node_urn": graph_node_urn}

        if source_type:
            query += " AND source_type = :source_type"
            params["source_type"] = source_type.lower()

        if min_authority is not None:
            query += " AND source_authority >= :min_authority"
            params["min_authority"] = min_authority

        async with self._session_maker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

            return [
                BridgeMapping(
                    chunk_id=row[0],
                    graph_node_urn=row[1],
                    source_type=row[2],
                    source_authority=row[3],
                    mapping_type=row[4],
                    expert_affinity=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=row[8] or {},
                )
                for row in rows
            ]

    async def get_chunks_for_expert(
        self,
        graph_node_urn: str,
        expert_type: str,
        min_affinity: float = 0.0,
        limit: int = 100,
    ) -> List[BridgeMapping]:
        """
        Get chunks for a specific expert, sorted by affinity (AC3).

        This is the key query for expert-specific retrieval.

        Args:
            graph_node_urn: URN of the graph node
            expert_type: Expert type (literal, systemic, principles, precedent)
            min_affinity: Minimum affinity threshold
            limit: Maximum results to return

        Returns:
            List of BridgeMapping sorted by expert affinity descending

        Raises:
            ValueError: If expert_type is not a valid ExpertType
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text

        # Validate expert_type using enum (M4)
        expert_key = expert_type.lower()
        valid_experts = {e.value for e in ExpertType}
        if expert_key not in valid_experts:
            raise ValueError(
                f"Invalid expert_type '{expert_type}'. "
                f"Must be one of: {', '.join(valid_experts)}"
            )

        # Query with JSONB extraction and filtering
        query = text(f"""
            SELECT chunk_id, graph_node_urn, source_type, source_authority,
                   mapping_type, expert_affinity, created_at, updated_at, metadata,
                   (expert_affinity->>:expert_key)::float AS affinity_score
            FROM {self.config.table_name}
            WHERE graph_node_urn = :graph_node_urn
              AND (expert_affinity->>:expert_key)::float >= :min_affinity
            ORDER BY (expert_affinity->>:expert_key)::float DESC
            LIMIT :limit
        """)

        async with self._session_maker() as session:
            result = await session.execute(
                query,
                {
                    "graph_node_urn": graph_node_urn,
                    "expert_key": expert_key,
                    "min_affinity": min_affinity,
                    "limit": limit,
                },
            )
            rows = result.fetchall()

            return [
                BridgeMapping(
                    chunk_id=row[0],
                    graph_node_urn=row[1],
                    source_type=row[2],
                    source_authority=row[3],
                    mapping_type=row[4],
                    expert_affinity=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=row[8] or {},
                )
                for row in rows
            ]

    async def update_expert_affinity(
        self,
        chunk_id: str,
        graph_node_urn: str,
        expert_type: str,
        new_affinity: float,
        feedback_source: str = "f8_feedback",
    ) -> bool:
        """
        Update expert affinity for a mapping (AC4 - F8 feedback hook stub).

        This is the hook for future RLCF integration from Epic 6.

        Args:
            chunk_id: UUID of the chunk
            graph_node_urn: URN of the graph node
            expert_type: Expert type to update (literal, systemic, principles, precedent)
            new_affinity: New affinity value (0.0-1.0)
            feedback_source: Source of the feedback for audit

        Returns:
            True if mapping was updated, False if not found

        Raises:
            ValueError: If expert_type is invalid or new_affinity out of range
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        # Validate expert_type using enum (M4)
        expert_key = expert_type.lower()
        valid_experts = {e.value for e in ExpertType}
        if expert_key not in valid_experts:
            raise ValueError(
                f"Invalid expert_type '{expert_type}'. "
                f"Must be one of: {', '.join(valid_experts)}"
            )

        # Validate new_affinity range
        if not 0.0 <= new_affinity <= 1.0:
            raise ValueError(
                f"new_affinity must be between 0.0 and 1.0, got {new_affinity}"
            )

        from sqlalchemy import text
        import json

        # Update the specific expert affinity in JSONB
        # Note: asyncpg requires CAST() instead of :: for bound params
        update_sql = text(f"""
            UPDATE {self.config.table_name}
            SET expert_affinity = jsonb_set(
                    expert_affinity,
                    ARRAY[:expert_key],
                    CAST(:new_affinity_json AS jsonb)
                ),
                updated_at = NOW(),
                metadata = jsonb_set(
                    COALESCE(metadata, '{{}}'),
                    ARRAY['last_feedback'],
                    CAST(:feedback_info_json AS jsonb)
                )
            WHERE chunk_id = :chunk_id AND graph_node_urn = :graph_node_urn
        """)

        feedback_info = {
            "source": feedback_source,
            "expert": expert_key,
            "value": new_affinity,
            "timestamp": datetime.now().isoformat(),
        }

        async with self._session_maker() as session:
            result = await session.execute(
                update_sql,
                {
                    "chunk_id": chunk_id,
                    "graph_node_urn": graph_node_urn,
                    "expert_key": expert_key,
                    "new_affinity_json": json.dumps(new_affinity),
                    "feedback_info_json": json.dumps(feedback_info),
                },
            )
            await session.commit()

            updated = result.rowcount > 0

            if updated:
                logger.info(
                    "Updated expert_affinity[%s]=%.2f for chunk=%s, source=%s",
                    expert_key,
                    new_affinity,
                    chunk_id,
                    feedback_source,
                )
            else:
                logger.warning(
                    "Mapping not found: chunk_id=%s, graph_node_urn=%s",
                    chunk_id,
                    graph_node_urn,
                )

            return updated

    async def delete_mappings_for_chunk(self, chunk_id: str) -> int:
        """
        Delete all mappings for a chunk.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            Number of mappings deleted
        """
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text

        delete_sql = text(f"""
            DELETE FROM {self.config.table_name}
            WHERE chunk_id = :chunk_id
        """)

        async with self._session_maker() as session:
            result = await session.execute(delete_sql, {"chunk_id": chunk_id})
            await session.commit()

            count = result.rowcount
            logger.debug("Deleted %d mappings for chunk_id=%s", count, chunk_id)
            return count

    async def count(self) -> int:
        """Get total number of mappings in the table."""
        if not self._connected:
            raise RuntimeError("Not connected to PostgreSQL. Call connect() first.")

        from sqlalchemy import text

        async with self._session_maker() as session:
            result = await session.execute(
                text(f"SELECT COUNT(*) FROM {self.config.table_name}")
            )
            return result.scalar()

    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        try:
            if not self._connected:
                await self.connect()

            from sqlalchemy import text

            async with self._session_maker() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()

            return True

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False
