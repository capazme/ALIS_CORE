#!/usr/bin/env python3
"""
Migration: Add target_is_pending column to pending_relations table.

This migration adds support for tracking relations that point to pending entities.
When a pending entity is rejected, relations pointing to it are reset to pending.

Run:
    cd /Users/gpuzio/Desktop/CODE/MERL-T_alpha
    source .venv/bin/activate
    python scripts/migrate_add_target_is_pending.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from merlt.storage.enrichment.database import get_db_session, init_db


async def migrate():
    """Add target_is_pending column to pending_relations table."""
    print("Starting migration: add target_is_pending column...")

    # Initialize database connection
    await init_db()

    async with get_db_session() as session:
        # Check if column already exists
        check_query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'pending_relations'
            AND column_name = 'target_is_pending'
        """)

        result = await session.execute(check_query)
        exists = result.scalar_one_or_none()

        if exists:
            print("✓ Column 'target_is_pending' already exists. Skipping.")
            return

        # Add the column
        alter_query = text("""
            ALTER TABLE pending_relations
            ADD COLUMN target_is_pending BOOLEAN DEFAULT FALSE
        """)

        await session.execute(alter_query)
        await session.commit()

        print("✓ Added column 'target_is_pending' to pending_relations table")

        # Create index for efficient cascade queries
        index_query = text("""
            CREATE INDEX IF NOT EXISTS ix_pending_relations_target_is_pending
            ON pending_relations (target_is_pending)
            WHERE target_is_pending = TRUE
        """)

        await session.execute(index_query)
        await session.commit()

        print("✓ Created index on target_is_pending")

        # Update existing relations: mark those with entity: prefix as pending
        update_query = text("""
            UPDATE pending_relations
            SET target_is_pending = TRUE
            WHERE target_entity_id LIKE 'entity:%'
        """)

        result = await session.execute(update_query)
        await session.commit()

        print(f"✓ Updated {result.rowcount} existing relations with target_is_pending=TRUE")

        # === Add 'fonte' column if not exists ===
        check_fonte_query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'pending_relations'
            AND column_name = 'fonte'
        """)

        result = await session.execute(check_fonte_query)
        fonte_exists = result.scalar_one_or_none()

        if not fonte_exists:
            alter_fonte_query = text("""
                ALTER TABLE pending_relations
                ADD COLUMN fonte VARCHAR(50) DEFAULT 'llm_extraction'
            """)
            await session.execute(alter_fonte_query)
            await session.commit()
            print("✓ Added column 'fonte' to pending_relations table")
        else:
            print("✓ Column 'fonte' already exists. Skipping.")

    print("\n✅ Migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(migrate())
