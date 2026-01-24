#!/usr/bin/env python3
"""
Run Live Enrichment Migration
==============================

Executes PostgreSQL schema migration for live enrichment.

Hard Cutover Strategy:
1. Run migration script to create tables
2. Deploy updated enrichment_router.py
3. No data migration needed (fresh start)

Usage:
    python scripts/run_enrichment_migration.py
    python scripts/run_enrichment_migration.py --verify-only
    python scripts/run_enrichment_migration.py --rollback  # Drop all tables (WARNING!)

Requirements:
    - PostgreSQL running (docker-compose.dev.yml)
    - Database: rlcf_dev
    - User/password configured in .env
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from merlt.storage.enrichment.database import get_database_url

log = structlog.get_logger()

# Migration file path
MIGRATION_FILE = Path(__file__).parent.parent / "merlt/storage/migrations/001_live_enrichment_schema.sql"


async def run_migration(engine):
    """Execute migration SQL script."""
    log.info("Reading migration file", path=str(MIGRATION_FILE))

    if not MIGRATION_FILE.exists():
        raise FileNotFoundError(f"Migration file not found: {MIGRATION_FILE}")

    # Read SQL file
    with open(MIGRATION_FILE, "r") as f:
        migration_sql = f.read()

    log.info("Executing migration SQL", size_kb=len(migration_sql) // 1024)

    # Split SQL into statements (handle multi-line functions/triggers)
    statements = _split_sql_statements(migration_sql)
    log.info(f"Parsed {len(statements)} SQL statements")

    # Execute in transaction
    async with engine.begin() as conn:
        for i, stmt in enumerate(statements, 1):
            if stmt.strip():  # Skip empty statements
                try:
                    await conn.execute(text(stmt))
                    log.debug(f"Executed statement {i}/{len(statements)}")
                except Exception as e:
                    log.error(f"Failed at statement {i}", stmt_preview=stmt[:100], error=str(e))
                    raise

    log.info("Migration executed successfully")


def _split_sql_statements(sql: str) -> list[str]:
    """
    Split SQL script into individual statements.

    Handles multi-line constructs like CREATE FUNCTION, CREATE TRIGGER.
    Uses $$ delimiters to detect function bodies.
    """
    statements = []
    current_stmt = []
    in_function = False
    in_comment = False

    for line in sql.split('\n'):
        stripped = line.strip()

        # Skip pure comment lines (but keep inline comments with code)
        if stripped.startswith('--') and not current_stmt:
            continue

        # Track if we're inside a function definition (uses $$)
        if '$$' in line:
            in_function = not in_function

        # Add line to current statement
        if stripped:
            current_stmt.append(line)

        # Check for statement terminator (only outside functions)
        if ';' in stripped and not in_function:
            # Statement complete
            stmt_text = '\n'.join(current_stmt)
            statements.append(stmt_text)
            current_stmt = []

    # Handle any remaining statement
    if current_stmt:
        statements.append('\n'.join(current_stmt))

    return statements


async def verify_migration(engine):
    """Verify that all tables and indexes were created."""
    log.info("Verifying migration...")

    expected_tables = [
        "pending_entities",
        "entity_votes",
        "pending_relations",
        "relation_votes",
        "user_documents",
        "pending_amendments",
        "amendment_votes",
        "user_domain_authority",
    ]

    expected_indexes = [
        "idx_pending_entities_status",
        "idx_pending_entities_type",
        "idx_entity_votes_entity",
        "idx_pending_relations_status",
        "idx_user_documents_uploader",
        "idx_pending_amendments_target",
    ]

    expected_triggers = [
        "trigger_entity_vote_consensus",
        "trigger_relation_vote_consensus",
        "trigger_amendment_vote_consensus",
    ]

    async with engine.connect() as conn:
        # Check tables
        result = await conn.execute(
            text(
                """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
            )
        )
        tables = [row[0] for row in result]

        log.info(f"Found {len(tables)} tables", tables=tables)

        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            log.error("Missing tables", tables=list(missing_tables))
            return False

        # Check indexes
        result = await conn.execute(
            text(
                """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY indexname
        """
            )
        )
        indexes = [row[0] for row in result]

        log.info(f"Found {len(indexes)} indexes")

        missing_indexes = set(expected_indexes) - set(indexes)
        if missing_indexes:
            log.warning("Some expected indexes missing", indexes=list(missing_indexes))
            # Don't fail - indexes might be named differently

        # Check triggers
        result = await conn.execute(
            text(
                """
            SELECT trigger_name FROM information_schema.triggers
            WHERE trigger_schema = 'public'
        """
            )
        )
        triggers = [row[0] for row in result]

        log.info(f"Found {len(triggers)} triggers", triggers=triggers)

        missing_triggers = set(expected_triggers) - set(triggers)
        if missing_triggers:
            log.error("Missing triggers", triggers=list(missing_triggers))
            return False

        # Check foreign keys
        result = await conn.execute(
            text(
                """
            SELECT conname FROM pg_constraint
            WHERE contype = 'f'
        """
            )
        )
        fks = [row[0] for row in result]
        log.info(f"Found {len(fks)} foreign keys")

    log.info("✅ Migration verification passed")
    return True


async def rollback_migration(engine):
    """Drop all tables (destructive!)."""
    log.warning("⚠️  ROLLBACK: This will DROP ALL TABLES. Are you sure?")
    log.warning("⚠️  Type 'yes' to confirm:")

    confirmation = input().strip().lower()
    if confirmation != "yes":
        log.info("Rollback cancelled")
        return

    log.warning("Dropping all tables...")

    drop_sql = """
    DROP TABLE IF EXISTS amendment_votes CASCADE;
    DROP TABLE IF EXISTS pending_amendments CASCADE;
    DROP TABLE IF EXISTS relation_votes CASCADE;
    DROP TABLE IF EXISTS pending_relations CASCADE;
    DROP TABLE IF EXISTS entity_votes CASCADE;
    DROP TABLE IF EXISTS pending_entities CASCADE;
    DROP TABLE IF EXISTS user_documents CASCADE;
    DROP TABLE IF EXISTS user_domain_authority CASCADE;

    DROP FUNCTION IF EXISTS update_entity_consensus() CASCADE;
    DROP FUNCTION IF EXISTS update_relation_consensus() CASCADE;
    DROP FUNCTION IF EXISTS update_amendment_consensus() CASCADE;
    """

    async with engine.begin() as conn:
        await conn.execute(text(drop_sql))

    log.warning("All tables dropped")


async def main():
    parser = argparse.ArgumentParser(description="Run Live Enrichment Migration")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't run migration")
    parser.add_argument("--rollback", action="store_true", help="Drop all tables (WARNING!)")
    args = parser.parse_args()

    # Get database URL
    db_url = get_database_url()
    log.info("Connecting to database", url=db_url.split("@")[1] if "@" in db_url else db_url)

    # Create engine (no pool for migration script)
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)

    try:
        if args.rollback:
            await rollback_migration(engine)
        elif args.verify_only:
            success = await verify_migration(engine)
            sys.exit(0 if success else 1)
        else:
            # Run migration
            await run_migration(engine)

            # Verify
            success = await verify_migration(engine)

            if success:
                log.info("=" * 60)
                log.info("MIGRATION COMPLETE")
                log.info("=" * 60)
                log.info("Next steps:")
                log.info("1. Update enrichment_router.py to use database")
                log.info("2. Deploy updated code")
                log.info("3. Test live enrichment endpoint")
            else:
                log.error("Migration verification failed")
                sys.exit(1)

    except Exception as e:
        log.error("Migration failed", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
