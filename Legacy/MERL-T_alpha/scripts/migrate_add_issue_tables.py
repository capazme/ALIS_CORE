#!/usr/bin/env python3
"""
Migration Script: Add Issue Reporting Tables
=============================================

Creates the entity_issue_reports and entity_issue_votes tables
for the RLCF feedback loop on approved entities.

Usage:
    python scripts/migrate_add_issue_tables.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from merlt.storage.enrichment import get_db_session, init_db, close_db


# Each statement must be executed separately with asyncpg
MIGRATIONS = [
    # 1. Entity Issue Reports table
    """
    CREATE TABLE IF NOT EXISTS entity_issue_reports (
        id SERIAL PRIMARY KEY,
        issue_id VARCHAR(100) UNIQUE NOT NULL,
        entity_id VARCHAR(100) NOT NULL,
        entity_type VARCHAR(50),
        reported_by VARCHAR(100) NOT NULL,
        reporter_authority FLOAT DEFAULT 0.0,
        issue_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) DEFAULT 'medium',
        description TEXT,
        upvote_score FLOAT DEFAULT 0.0,
        downvote_score FLOAT DEFAULT 0.0,
        votes_count INTEGER DEFAULT 0,
        status VARCHAR(20) DEFAULT 'open',
        resolved_at TIMESTAMP,
        resolved_by VARCHAR(100),
        resolution_notes TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """,

    # 2. Indexes for entity_issue_reports
    "CREATE INDEX IF NOT EXISTS idx_issue_reports_entity ON entity_issue_reports(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_issue_reports_status ON entity_issue_reports(status)",
    "CREATE INDEX IF NOT EXISTS idx_issue_reports_type ON entity_issue_reports(issue_type)",
    "CREATE INDEX IF NOT EXISTS idx_issue_reports_reported_by ON entity_issue_reports(reported_by)",
    "CREATE INDEX IF NOT EXISTS idx_issue_reports_issue_id ON entity_issue_reports(issue_id)",

    # 3. Entity Issue Votes table
    """
    CREATE TABLE IF NOT EXISTS entity_issue_votes (
        id SERIAL PRIMARY KEY,
        issue_id VARCHAR(100) NOT NULL,
        user_id VARCHAR(100) NOT NULL,
        vote_value INTEGER NOT NULL CHECK (vote_value IN (-1, 1)),
        voter_authority FLOAT DEFAULT 0.0,
        comment TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_issue FOREIGN KEY (issue_id)
            REFERENCES entity_issue_reports(issue_id) ON DELETE CASCADE,
        CONSTRAINT unique_issue_user_vote UNIQUE (issue_id, user_id)
    )
    """,

    # 4. Indexes for entity_issue_votes
    "CREATE INDEX IF NOT EXISTS idx_issue_votes_issue ON entity_issue_votes(issue_id)",
    "CREATE INDEX IF NOT EXISTS idx_issue_votes_user ON entity_issue_votes(user_id)",

    # 5. Relation Issue Reports table
    """
    CREATE TABLE IF NOT EXISTS relation_issue_reports (
        id SERIAL PRIMARY KEY,
        issue_id VARCHAR(100) UNIQUE NOT NULL,
        relation_id VARCHAR(100) NOT NULL,
        relation_type VARCHAR(50),
        reported_by VARCHAR(100) NOT NULL,
        reporter_authority FLOAT DEFAULT 0.0,
        issue_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) DEFAULT 'medium',
        description TEXT,
        upvote_score FLOAT DEFAULT 0.0,
        downvote_score FLOAT DEFAULT 0.0,
        votes_count INTEGER DEFAULT 0,
        status VARCHAR(20) DEFAULT 'open',
        resolved_at TIMESTAMP,
        resolved_by VARCHAR(100),
        resolution_notes TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """,

    # 6. Indexes for relation_issue_reports
    "CREATE INDEX IF NOT EXISTS idx_relation_issue_reports_relation ON relation_issue_reports(relation_id)",
    "CREATE INDEX IF NOT EXISTS idx_relation_issue_reports_status ON relation_issue_reports(status)",
    "CREATE INDEX IF NOT EXISTS idx_relation_issue_reports_issue_id ON relation_issue_reports(issue_id)",

    # 7. Relation Issue Votes table
    """
    CREATE TABLE IF NOT EXISTS relation_issue_votes (
        id SERIAL PRIMARY KEY,
        issue_id VARCHAR(100) NOT NULL,
        user_id VARCHAR(100) NOT NULL,
        vote_value INTEGER NOT NULL CHECK (vote_value IN (-1, 1)),
        voter_authority FLOAT DEFAULT 0.0,
        comment TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_relation_issue FOREIGN KEY (issue_id)
            REFERENCES relation_issue_reports(issue_id) ON DELETE CASCADE,
        CONSTRAINT unique_relation_issue_user_vote UNIQUE (issue_id, user_id)
    )
    """,

    # 8. Indexes for relation_issue_votes
    "CREATE INDEX IF NOT EXISTS idx_relation_issue_votes_issue ON relation_issue_votes(issue_id)",
    "CREATE INDEX IF NOT EXISTS idx_relation_issue_votes_user ON relation_issue_votes(user_id)",
]


async def run_migration():
    """Execute the migration."""
    print("=" * 60)
    print("MERL-T Migration: Add Issue Reporting Tables")
    print("=" * 60)

    # Initialize database
    await init_db(echo=False)
    print("Database connection established\n")

    async with get_db_session() as session:
        try:
            for i, sql in enumerate(MIGRATIONS, 1):
                # Extract table/index name for logging
                sql_stripped = sql.strip()
                if "CREATE TABLE" in sql_stripped:
                    name = sql_stripped.split("(")[0].split()[-1]
                    print(f"{i}. Creating table {name}...")
                elif "CREATE INDEX" in sql_stripped:
                    name = sql_stripped.split(" ON ")[0].split()[-1]
                    print(f"{i}. Creating index {name}...")
                else:
                    print(f"{i}. Executing migration step...")

                await session.execute(text(sql))

            await session.commit()
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
