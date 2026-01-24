#!/usr/bin/env python3
"""
Migration Script: Alter entity_id to TEXT in entity_issue_reports
==================================================================

Fixes the issue where relation entity_ids exceed VARCHAR(100) limit.
Example relation entity_id:
    rel_https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262:2~art1337_interpreta_massima_cassazione_civile_7288_2023

Changes:
1. Drop FK constraint on entity_id (if exists) - relations don't exist in pending_entities
2. Alter entity_id column from VARCHAR(100) to TEXT

Usage:
    python scripts/migrate_entity_id_to_text.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from merlt.storage.enrichment import get_db_session, init_db, close_db


MIGRATIONS = [
    # 1. Drop FK constraint if exists (may have different names)
    """
    DO $$
    BEGIN
        -- Try to drop FK constraint with common naming patterns
        IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'entity_issue_reports_entity_id_fkey'
            AND table_name = 'entity_issue_reports'
        ) THEN
            ALTER TABLE entity_issue_reports DROP CONSTRAINT entity_issue_reports_entity_id_fkey;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'fk_entity_id'
            AND table_name = 'entity_issue_reports'
        ) THEN
            ALTER TABLE entity_issue_reports DROP CONSTRAINT fk_entity_id;
        END IF;
    END $$;
    """,

    # 2. Alter column type from VARCHAR(100) to TEXT
    """
    ALTER TABLE entity_issue_reports
    ALTER COLUMN entity_id TYPE TEXT;
    """,
]


async def run_migration():
    """Execute the migration."""
    print("=" * 60)
    print("MERL-T Migration: Alter entity_id to TEXT")
    print("=" * 60)
    print()
    print("This migration:")
    print("1. Drops FK constraint on entity_id (allows relations)")
    print("2. Changes entity_id from VARCHAR(100) to TEXT")
    print()

    # Initialize database
    await init_db(echo=False)
    print("Database connection established\n")

    async with get_db_session() as session:
        try:
            # Check current column type
            check_sql = """
            SELECT data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'entity_issue_reports'
            AND column_name = 'entity_id';
            """
            result = await session.execute(text(check_sql))
            row = result.fetchone()

            if row:
                print(f"Current column type: {row[0]}, max_length: {row[1]}")
            else:
                print("Warning: entity_id column not found. Table may not exist.")
                return

            print()

            for i, sql in enumerate(MIGRATIONS, 1):
                print(f"{i}. Executing migration step...")
                await session.execute(text(sql))
                print(f"   Done")

            await session.commit()

            # Verify the change
            result = await session.execute(text(check_sql))
            row = result.fetchone()
            print()
            print(f"New column type: {row[0]}, max_length: {row[1]}")

            print("\n" + "=" * 60)
            print("Migration completed successfully!")
            print("=" * 60)

        except Exception as e:
            await session.rollback()
            print(f"\nError during migration: {e}")
            raise

    await close_db()


if __name__ == "__main__":
    asyncio.run(run_migration())
