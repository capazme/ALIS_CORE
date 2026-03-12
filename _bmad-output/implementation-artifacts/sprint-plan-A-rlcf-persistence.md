# Sprint Plan A: RLCF Persistence Cycle

**Date:** 2026-03-12
**Scrum Master:** gpuzio
**Project Level:** 4
**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Total Stories:** 3
**Total Points:** 11
**Sprint Goal:** Make RLCF training incremental — weights, buffer, and policies survive restarts.

---

## Executive Summary

The RLCF training loop runs but is a no-op in practice: the GatingPolicy is recreated from scratch on every `run_training_epoch()`, the replay buffer lives in RAM only, and `WeightStore.save_weights()` is a stub despite the `weight_versions` DB table already existing (migration 004). After this sprint, training becomes truly incremental.

**Key insight:** Most infrastructure already exists — this is wiring work, not greenfield.

---

## Story Inventory

### STORY-11-1: Weight Store DB Persistence

**Priority:** Must Have
**Points:** 5
**Depends on:** None (migration 004 already applied)

**User Story:**
As the RLCF training pipeline,
I want to persist learned weight configurations to the `weight_versions` PostgreSQL table,
So that weights survive process restarts and can be loaded by experiment ID.

**Acceptance Criteria:**
- [ ] `WeightStore._load_from_database()` queries `weight_versions` table by experiment_id, returns most recent active config
- [ ] `WeightStore.save_weights()` inserts into `weight_versions` with config_json, metrics_json, version_tag
- [ ] When `is_active=True` is set on a new version, previous active version for same experiment_id is deactivated
- [ ] Fallback to YAML still works when database_url is None or DB is unreachable
- [ ] Unit tests: save → load roundtrip, deactivation of previous version, YAML fallback
- [ ] Integration test: verify against real PostgreSQL (weight_versions table)

**Technical Notes:**
- `weight_versions` table schema (from migration 004): `id, experiment_id, version_tag, config_json, metrics_json, is_active, created_at, created_by`
- Use async SQLAlchemy (same pattern as `merlt/rlcf/database.py`)
- `config_json` stores serialized `WeightConfig` (Pydantic `.model_dump()`)
- `metrics_json` stores training metrics dict
- Need SQLAlchemy ORM model `WeightVersion` in `merlt-models/weights/models.py` (new file)
- DB URL from env `RLCF_DATABASE_URL` or constructor param

**Files to modify:**
- `merlt-models/weights/store.py` — implement `_load_from_database()`, `save_weights()`
- `merlt-models/weights/models.py` — NEW: SQLAlchemy `WeightVersion` model

**Files to read (context):**
- `merlt/alembic/versions/004_add_weight_versions_table.py` — table schema
- `merlt/merlt/rlcf/database.py` — async session pattern to replicate

---

### STORY-11-2: Replay Buffer Auto-Persistence

**Priority:** Must Have
**Points:** 3
**Depends on:** None

**User Story:**
As the RLCF training scheduler,
I want the replay buffer to auto-save to disk on training completion and auto-load on startup,
So that accumulated experiences survive process restarts.

**Acceptance Criteria:**
- [ ] `TrainingScheduler.__init__()` loads buffer from known path if file exists
- [ ] `TrainingScheduler.run_training_epoch()` saves buffer after successful training
- [ ] Buffer save path is configurable via `SchedulerConfig.buffer_persistence_path` (default: `data/rlcf/replay_buffer.json`)
- [ ] Graceful handling: if buffer file is corrupted/missing, start with empty buffer (log warning)
- [ ] `PrioritizedReplayBuffer` also supports `save()`/`load()` (currently only `ExperienceReplayBuffer` has them)
- [ ] Unit tests: save/load roundtrip for both buffer types, corrupted file handling, auto-save after training

**Technical Notes:**
- `ExperienceReplayBuffer` already has `save(path)` / `load(path)` methods (lines 311-350)
- `PrioritizedReplayBuffer` does NOT have save/load — needs implementation (serialize SumTree data)
- `TrainingScheduler` uses `PrioritizedReplayBuffer` by default (`prioritized_replay: True`)
- Add `buffer_persistence_path` to `SchedulerConfig`
- Call `buffer.save()` at end of `run_training_epoch()` (after checkpoint save)
- Call `buffer.load()` in `__init__()` if path exists

**Files to modify:**
- `merlt/merlt/rlcf/replay_buffer.py` — add `save()`/`load()` to `PrioritizedReplayBuffer`
- `merlt/merlt/rlcf/training_scheduler.py` — wire auto-save/load, add config field

---

### STORY-11-3: Training Scheduler Checkpoint Loading

**Priority:** Must Have
**Points:** 3
**Depends on:** STORY-11-1 (WeightStore must work for full loop, but checkpoint loading is independent)

**User Story:**
As the RLCF training scheduler,
I want to load the latest GatingPolicy checkpoint before training instead of creating a fresh policy,
So that training is incremental and builds on previous learning.

**Acceptance Criteria:**
- [ ] `run_training_epoch()` loads GatingPolicy via `PolicyManager` instead of `GatingPolicy(input_dim=768, hidden_dim=256)`
- [ ] If no checkpoint exists, falls back to fresh GatingPolicy (current behavior)
- [ ] After training, saves checkpoint as both versioned name AND `latest` alias
- [ ] `PolicyManager` is instantiated once (not per-training-run), checkpoint_dir configurable
- [ ] Orchestrator `trained` flag reads actual checkpoint existence instead of hardcoded `False`
- [ ] Unit tests: load from checkpoint, fresh fallback, save latest alias, trained flag

**Technical Notes:**
- Line 427 in `training_scheduler.py`: replace `policy = GatingPolicy(input_dim=768, hidden_dim=256)` with `PolicyManager._load_gating_policy()` + fallback
- `PolicyManager.save_gating_policy()` already saves `.pt` files (line 664-689 of `policy_manager.py`)
- Current save (line 509): `pm.save_gating_policy(policy, name=f"gating_v{version_tag}")` — also needs `pm.save_gating_policy(policy, name="latest")`
- `orchestrator.py` line 509: `"trained": False` → check `PolicyManager.is_gating_policy_available()`
- `PolicyManager` uses `checkpoints/` dir by default — ensure it resolves to a persistent location

**Files to modify:**
- `merlt/merlt/rlcf/training_scheduler.py` — lines 423-428 (policy loading), line 509 (save latest alias)
- `merlt/merlt/experts/orchestrator.py` — line 509 (`trained` flag)

---

## Sprint Allocation

**Sprint A — RLCF Persistence (11 points)**

| Order | Story | Points | Rationale |
|-------|-------|--------|-----------|
| 1 | STORY-11-2: Buffer Auto-Persistence | 3 | No dependencies, enables immediate data survival |
| 2 | STORY-11-1: Weight Store DB | 5 | Core persistence, enables full loop |
| 3 | STORY-11-3: Checkpoint Loading | 3 | Completes the loop, depends on 11-1 conceptually |

**Execution order rationale:** Start with buffer (smallest, independent, immediate value), then weight store (largest, foundational), then checkpoint loading (ties everything together).

---

## Risks and Mitigation

**Medium:**
- `PrioritizedReplayBuffer.save()` needs SumTree serialization — mitigation: serialize experiences list, rebuild tree on load (simpler, same outcome)
- `weight_versions` migration may not be applied on dev DB — mitigation: check with `alembic current`, run `alembic upgrade head` if needed

**Low:**
- `merlt-models` doesn't currently depend on SQLAlchemy async — mitigation: add `sqlalchemy[asyncio]` + `asyncpg` to `merlt-models/pyproject.toml`

---

## Definition of Done

For each story:
- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files)
- [ ] Integration test with real PostgreSQL where applicable
- [ ] No regressions in existing test suite
- [ ] Acceptance criteria validated

---

## Next Steps

Run `/bmad:create-story` for STORY-11-2 (first in execution order), then `/bmad:dev-story` to implement.
