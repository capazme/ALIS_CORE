#!/usr/bin/env python3
"""
Create Q&A Tables Migration Script
===================================

Creates qa_traces and qa_feedback tables in PostgreSQL database.

Usage:
    python scripts/create_qa_tables.py

Environment Variables:
    RLCF_POSTGRES_URL: PostgreSQL connection URL
                      (default: postgresql+asyncpg://postgres:postgres@localhost:5432/merl_t_rlcf)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from merlt.rlcf.database import init_async_db, get_async_session, Base
from merlt.experts.models import QATrace, QAFeedback
from sqlalchemy import text


async def main():
    """Create Q&A tables in database."""
    print("🗄️  MERL-T Q&A Tables Migration")
    print("=" * 60)

    # Initialize database
    print("\n1️⃣  Initializing database connection...")
    await init_async_db()
    print("   ✅ Database initialized")

    # Check if tables already exist
    print("\n2️⃣  Checking existing tables...")
    async with get_async_session() as session:
        result = await session.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('qa_traces', 'qa_feedback')
        """))
        existing_tables = [row[0] for row in result.fetchall()]

    if existing_tables:
        print(f"   ⚠️  Found existing tables: {', '.join(existing_tables)}")
        response = input("   Do you want to DROP and recreate them? [y/N]: ")
        if response.lower() != 'y':
            print("   ❌ Migration cancelled")
            return

        # Drop existing tables
        print("\n3️⃣  Dropping existing tables...")
        async with get_async_session() as session:
            await session.execute(text("DROP TABLE IF EXISTS qa_feedback CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS qa_traces CASCADE"))
            await session.commit()
        print("   ✅ Existing tables dropped")

    # Create tables
    print("\n4️⃣  Creating tables...")
    from merlt.rlcf.database import get_async_engine
    engine = get_async_engine()

    async with engine.begin() as conn:
        # Create only Q&A tables (not all Base tables)
        await conn.run_sync(QATrace.__table__.create, checkfirst=True)
        await conn.run_sync(QAFeedback.__table__.create, checkfirst=True)

    print("   ✅ Tables created successfully")

    # Verify tables
    print("\n5️⃣  Verifying table structure...")
    async with get_async_session() as session:
        # Check qa_traces columns
        result = await session.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'qa_traces'
            ORDER BY ordinal_position
        """))
        qa_traces_cols = result.fetchall()

        # Check qa_feedback columns
        result = await session.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'qa_feedback'
            ORDER BY ordinal_position
        """))
        qa_feedback_cols = result.fetchall()

    print("\n   📋 qa_traces columns:")
    for col_name, col_type in qa_traces_cols:
        print(f"      - {col_name}: {col_type}")

    print("\n   📋 qa_feedback columns:")
    for col_name, col_type in qa_feedback_cols:
        print(f"      - {col_name}: {col_type}")

    # Check constraints
    print("\n6️⃣  Verifying constraints...")
    async with get_async_session() as session:
        result = await session.execute(text("""
            SELECT constraint_name, constraint_type
            FROM information_schema.table_constraints
            WHERE table_name IN ('qa_traces', 'qa_feedback')
            ORDER BY table_name, constraint_type
        """))
        constraints = result.fetchall()

    for constraint_name, constraint_type in constraints:
        print(f"   ✅ {constraint_type}: {constraint_name}")

    print("\n" + "=" * 60)
    print("✨ Migration completed successfully!")
    print("\nNext steps:")
    print("  1. Test Q&A endpoint: POST /api/experts/query")
    print("  2. Test feedback endpoint: POST /api/experts/feedback/inline")
    print("  3. Verify data with: SELECT * FROM qa_traces LIMIT 5;")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
