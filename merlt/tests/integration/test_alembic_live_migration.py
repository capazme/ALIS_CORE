"""
Alembic Live Migration Tests
==============================

Integration tests verifying Alembic migrations work correctly.

Tests:
- upgrade head -> downgrade base -> upgrade head roundtrip
- All expected tables exist after upgrade
- Current revision matches head

Requirements:
- PostgreSQL running on port 5433
- alembic installed

Run: pytest tests/integration/test_alembic_live_migration.py -m integration
"""

import os
import pytest

try:
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, inspect, text

    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="alembic not installed"),
]


def _get_alembic_config() -> "Config":
    """Get Alembic config pointing to merlt/alembic.ini."""
    ini_path = os.path.join(os.path.dirname(__file__), "..", "..", "alembic.ini")
    ini_path = os.path.abspath(ini_path)
    cfg = Config(ini_path)
    # Override with sync driver for test introspection
    db_url = os.environ.get(
        "RLCF_POSTGRES_URL",
        "postgresql://dev:devpassword@localhost:5433/rlcf_dev",
    )
    # Alembic env.py uses asyncpg; we override for sync operations
    cfg.set_main_option("sqlalchemy.url", db_url.replace("+asyncpg", ""))
    return cfg


def _db_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        db_url = os.environ.get(
            "RLCF_POSTGRES_URL",
            "postgresql://dev:devpassword@localhost:5433/rlcf_dev",
        ).replace("+asyncpg", "")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db():
    """Skip all tests if database is not available."""
    if not _db_available():
        pytest.skip("PostgreSQL not available on port 5433")


class TestAlembicRoundtrip:
    """Test upgrade/downgrade roundtrip."""

    def test_upgrade_downgrade_upgrade(self):
        """Upgrade head -> downgrade base -> upgrade head should not error."""
        cfg = _get_alembic_config()

        # Step 1: upgrade to head
        command.upgrade(cfg, "head")

        # Step 2: downgrade to base
        command.downgrade(cfg, "base")

        # Step 3: upgrade to head again
        command.upgrade(cfg, "head")


class TestExpectedTablesExist:
    """Test that all expected tables exist after migration."""

    def test_tables_exist_after_upgrade(self):
        """After upgrade head, key tables must be present."""
        cfg = _get_alembic_config()
        command.upgrade(cfg, "head")

        db_url = cfg.get_main_option("sqlalchemy.url")
        engine = create_engine(db_url)

        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        engine.dispose()

        # Alembic's own tracking table must exist
        assert "alembic_version" in existing_tables

    def test_weight_versions_table_exists(self):
        """After upgrade head, weight_versions table must be present (migration 004)."""
        cfg = _get_alembic_config()
        command.upgrade(cfg, "head")

        db_url = cfg.get_main_option("sqlalchemy.url")
        engine = create_engine(db_url)

        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        engine.dispose()

        assert "weight_versions" in existing_tables, (
            f"weight_versions table not found after upgrade head. "
            f"Existing tables: {existing_tables}"
        )

    def test_weight_versions_downgrade_removes_table(self):
        """Downgrading to pre-004 removes weight_versions table."""
        cfg = _get_alembic_config()

        # First upgrade to head (includes 004)
        command.upgrade(cfg, "head")

        # Downgrade to 003 (pre-004)
        command.downgrade(cfg, "003_add_api_keys_table")

        db_url = cfg.get_main_option("sqlalchemy.url")
        engine = create_engine(db_url)

        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        engine.dispose()

        assert "weight_versions" not in existing_tables, (
            "weight_versions should NOT exist after downgrade to 003"
        )

        # Re-upgrade to head for cleanup
        command.upgrade(cfg, "head")


class TestMigration004Structure:
    """Structural checks for migration 004 file (no DB needed)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_db(self):
        """Override: structural tests don't need PostgreSQL."""
        pass

    def test_migration_004_file_exists(self):
        """Migration 004 file must exist in alembic/versions/."""
        migration_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "alembic", "versions"
        )
        migration_path = os.path.join(
            os.path.abspath(migration_dir), "004_add_weight_versions_table.py"
        )
        assert os.path.exists(migration_path), (
            f"Migration file not found: {migration_path}"
        )

    def test_migration_004_has_upgrade_downgrade(self):
        """Migration 004 must define upgrade() and downgrade() functions."""
        import importlib.util

        migration_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "alembic", "versions"
        )
        migration_path = os.path.join(
            os.path.abspath(migration_dir), "004_add_weight_versions_table.py"
        )

        spec = importlib.util.spec_from_file_location("migration_004", migration_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "upgrade"), "Migration 004 must have upgrade()"
        assert hasattr(mod, "downgrade"), "Migration 004 must have downgrade()"
        assert callable(mod.upgrade)
        assert callable(mod.downgrade)
        assert mod.down_revision == "003_add_api_keys_table"


class TestCurrentRevisionMatchesHead:
    """Test that current DB revision matches head."""

    def test_revision_matches_head(self):
        """After upgrade, current revision must equal head revision."""
        cfg = _get_alembic_config()
        command.upgrade(cfg, "head")

        # Get head revision from script directory
        script = ScriptDirectory.from_config(cfg)
        head_rev = script.get_current_head()

        # Get current DB revision
        db_url = cfg.get_main_option("sqlalchemy.url")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.fetchone()
        engine.dispose()

        assert row is not None, "No alembic_version row found after upgrade"
        assert row[0] == head_rev, f"DB revision {row[0]} != head {head_rev}"
