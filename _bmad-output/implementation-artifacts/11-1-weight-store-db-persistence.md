# STORY-11-1: Weight Store DB Persistence

**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Priority:** Must Have
**Story Points:** 5
**Status:** done
**Assigned To:** gpuzio
**Created:** 2026-03-12
**Sprint:** A (RLCF Persistence)

---

## User Story

As the RLCF training loop,
I want the WeightStore to persist weight configs to the `weight_versions` PostgreSQL table,
So that trained weights survive process restarts and experiments can be tracked across sessions.

---

## Description

### Background

The `WeightStore` class (`merlt-models/weights/store.py`) manages all system weights — retrieval alpha, expert traversal, RLCF authority, gating priors. It supports loading from YAML, runtime overrides, and database persistence. However, the two database methods are **stubs**:

- `_load_from_database()` (line 271-279): returns `None` with a TODO comment
- `save_weights()` (line 281-320): generates a UUID but only logs "in-memory only"

The Alembic migration `004_add_weight_versions_table` already created the PostgreSQL table with columns: `id`, `experiment_id`, `version_tag`, `config_json`, `metrics_json`, `is_active`, `created_at`, `created_by`. The table exists but is never written to.

This means weight changes from RLCF training (authority adjustments, gating priors, traversal weights) are lost on restart.

### Scope

**In scope:**
- Implement `_load_from_database()` to query `weight_versions` table for the active version of a given `experiment_id`
- Implement `save_weights()` to INSERT into `weight_versions` table with JSON-serialized `WeightConfig`
- Add `version_tag` support (auto-generated timestamp-based if not provided)
- Add `set_active()` method to mark a specific version as active (deactivate others for same experiment)
- Add SQLAlchemy model for `weight_versions` table in `merlt/rlcf/database.py` (or co-located models file)
- Integration tests against real PostgreSQL (Docker)

**Out of scope:**
- Weight diffing / comparison between versions
- A/B experiment routing (uses `ExperimentConfig` which is already modeled)
- Weight migration between schema versions
- Frontend UI for weight management

---

## Acceptance Criteria

- [ ] `WeightStore.save_weights(config, experiment_id)` persists a row in `weight_versions` with `config_json` = serialized `WeightConfig`, `is_active=True`, and deactivates previous active versions for same `experiment_id`
- [ ] `WeightStore._load_from_database(experiment_id)` returns a `WeightConfig` parsed from the active row's `config_json`, or `None` if no active version exists
- [ ] `save_weights()` returns the `version_id` (UUID) of the inserted row
- [ ] `save_weights()` stores `metrics_json` when metrics are provided
- [ ] `save_weights()` auto-generates a `version_tag` from timestamp if not explicitly provided
- [ ] Loading a non-existent `experiment_id` returns `None` (falls back to YAML via existing `get_weights` logic)
- [ ] Roundtrip test: `save_weights()` → `_load_from_database()` preserves all `WeightConfig` fields (retrieval, expert_traversal, rlcf, gating)
- [ ] `save_weights()` without database connection logs a warning and returns a version_id (graceful degradation, same as current behavior)
- [ ] Unit tests cover: save, load, roundtrip, missing experiment, active version switching, no-database fallback
- [ ] No regressions in existing `WeightStore` tests (YAML loading, runtime override, singleton)

---

## Technical Notes

### Implementation Approach

**1. SQLAlchemy model for `weight_versions`**

Add to `merlt/merlt/rlcf/models.py` (or create if needed, next to `database.py`):

```python
from merlt.rlcf.database import Base

class WeightVersion(Base):
    __tablename__ = "weight_versions"

    id = Column(String(50), primary_key=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    version_tag = Column(String(50), nullable=True)
    config_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, server_default="false", nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))
    created_by = Column(String(100), nullable=True)
```

**2. `_load_from_database(experiment_id)` implementation**

```python
async def _load_from_database(self, experiment_id: str) -> Optional[WeightConfig]:
    from merlt.rlcf.database import get_async_session
    from merlt.rlcf.models import WeightVersion

    try:
        async with get_async_session() as session:
            result = await session.execute(
                select(WeightVersion)
                .where(WeightVersion.experiment_id == experiment_id)
                .where(WeightVersion.is_active == True)
                .order_by(WeightVersion.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row and row.config_json:
                return self._parse_db_json_to_config(row.config_json)
    except Exception as e:
        log.warning("Database weight load failed, falling back to YAML",
                    experiment_id=experiment_id, error=str(e))
    return None
```

**3. `save_weights()` implementation**

```python
async def save_weights(self, config, experiment_id, metrics=None, version_tag=None, created_by=None) -> str:
    version_id = str(uuid4())
    version_tag = version_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    if not self.database_url:
        log.warning("No database configured, weights saved in-memory only")
        return version_id

    from merlt.rlcf.database import get_async_session
    from merlt.rlcf.models import WeightVersion

    async with get_async_session() as session:
        # Deactivate previous active versions
        await session.execute(
            update(WeightVersion)
            .where(WeightVersion.experiment_id == experiment_id)
            .where(WeightVersion.is_active == True)
            .values(is_active=False)
        )
        # Insert new active version
        row = WeightVersion(
            id=version_id,
            experiment_id=experiment_id,
            version_tag=version_tag,
            config_json=config.model_dump(),
            metrics_json=metrics,
            is_active=True,
            created_by=created_by,
        )
        session.add(row)

    # Invalidate cache
    ...
    return version_id
```

**4. Helper: `_parse_db_json_to_config()`**

Parse the JSON dict back into a `WeightConfig` Pydantic model. Since `WeightConfig` is Pydantic v2, `WeightConfig.model_validate(data)` should work directly.

### Files to Modify

| File | Change |
|------|--------|
| `merlt/merlt/rlcf/models.py` | Add `WeightVersion` SQLAlchemy model (may need to create file or add to existing models) |
| `merlt-models/weights/store.py` | Implement `_load_from_database()`, `save_weights()`, add `_parse_db_json_to_config()` |

### Files to Read (context)

| File | Why |
|------|-----|
| `merlt-models/weights/config.py` | `WeightConfig` Pydantic model structure |
| `merlt/merlt/rlcf/database.py` | `get_async_session()`, `Base`, async session pattern |
| `merlt/alembic/versions/004_add_weight_versions_table.py` | Table schema reference |

### Cross-Package Dependency

`merlt-models/weights/store.py` will import from `merlt.rlcf.database` and `merlt.rlcf.models`. This is an existing pattern — `merlt-models` is a leaf package that depends on `merlt` for database access.

---

## Dependencies

**Prerequisite Stories:**
- None (table already exists via migration 004)

**Blocked Stories:**
- STORY-11-3 benefits from this: checkpoint loading can use DB-persisted weights as initial values

**External Dependencies:**
- PostgreSQL running (Docker: `docker-compose up -d postgres`)
- Alembic migration 004 applied

---

## Definition of Done

- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files):
  - [ ] `save_weights()` inserts row with correct fields
  - [ ] `_load_from_database()` returns valid `WeightConfig`
  - [ ] Save → load roundtrip preserves all config fields
  - [ ] Active version switching (new save deactivates old)
  - [ ] Non-existent experiment_id → `None`
  - [ ] No database URL → graceful fallback
  - [ ] `version_tag` auto-generation
  - [ ] `metrics_json` persistence
- [ ] No regressions in existing test suite (`pytest merlt/tests/`)
- [ ] Acceptance criteria validated

---

## Story Points Breakdown

- **SQLAlchemy model:** 1 point
- **`_load_from_database()` + `save_weights()`:** 2 points
- **Tests:** 2 points
- **Total:** 5 points

**Rationale:** The table already exists, Pydantic models are defined, and the async session pattern is established. Main work is wiring the two stub methods to real SQL queries and writing thorough tests with a real database.
