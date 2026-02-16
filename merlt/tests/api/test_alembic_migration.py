"""
Test Alembic Migration Structure
=================================
P1-MIGR-1: Verify migration scripts have up/down methods.
"""
import os
import pytest
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "alembic" / "versions"


class TestAlembicMigrations:
    """Verify migration scripts structure."""

    def test_migrations_directory_exists(self):
        assert MIGRATIONS_DIR.exists(), f"Migrations dir not found: {MIGRATIONS_DIR}"

    def test_migration_files_exist(self):
        migrations = list(MIGRATIONS_DIR.glob("*.py"))
        assert len(migrations) > 0, "No migration files found"

    def test_all_migrations_have_upgrade_and_downgrade(self):
        """Each migration must have upgrade() and downgrade() functions."""
        migrations = list(MIGRATIONS_DIR.glob("*.py"))
        for migration_file in migrations:
            if migration_file.name == "__pycache__":
                continue
            content = migration_file.read_text()
            assert "def upgrade()" in content, f"{migration_file.name} missing upgrade()"
            assert "def downgrade()" in content, f"{migration_file.name} missing downgrade()"

    def test_migrations_have_revision_ids(self):
        """Each migration must have revision and down_revision."""
        migrations = list(MIGRATIONS_DIR.glob("*.py"))
        for migration_file in migrations:
            if migration_file.name == "__pycache__":
                continue
            content = migration_file.read_text()
            assert "revision" in content, f"{migration_file.name} missing revision"

    def test_migration_chain_is_valid(self):
        """Verify no orphan migrations (all down_revision point to existing revision or None)."""
        migrations = list(MIGRATIONS_DIR.glob("*.py"))
        revisions = set()
        down_revisions = set()

        for migration_file in migrations:
            if migration_file.name == "__pycache__":
                continue
            content = migration_file.read_text()
            # Extract revision
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("revision") and "=" in stripped and "down_revision" not in stripped:
                    rev = stripped.split("=")[1].strip().strip("'\"")
                    if rev and rev != "None":
                        revisions.add(rev)
                if stripped.startswith("down_revision"):
                    down_rev = stripped.split("=")[1].strip().strip("'\"")
                    if down_rev and down_rev != "None":
                        down_revisions.add(down_rev)

        # Every down_revision should point to an existing revision or be None
        orphans = down_revisions - revisions
        assert len(orphans) == 0, f"Orphan down_revisions: {orphans}"
