# Sprint Plan B: RLCF Persistence — End-to-End Training Loop

**Date:** 2026-03-12
**Scrum Master:** gpuzio
**Project Level:** 4
**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Total Stories:** 3
**Total Points:** 11
**Sprint Goal:** Make the full RLCF training loop functional — traversal training actually trains, weights persist to DB, checkpoints serialize correctly.

---

## Executive Summary

Sprint A made GatingPolicy training incremental. But two critical subsystems are broken silently: the TraversalTrainingService is a complete no-op (AttributeError + missing optimizer), and WeightStore.save_weights() is implemented but never called. After Sprint B, the entire training pipeline works end-to-end: GatingPolicy, TraversalPolicy, and learned weight configurations all persist across restarts.

**Sprint A velocity:** 11 points in 3 stories — using same capacity for Sprint B.

---

## Story Inventory

### STORY-11-4: Fix TraversalTrainingService (Silent No-Op)

**Priority:** Must Have
**Points:** 5
**Depends on:** 11-3 (checkpoint infrastructure exists)

**User Story:**
As the RLCF training pipeline,
I want TraversalPolicy training to actually update the policy weights and save checkpoints,
So that traversal preferences learned from F8 feedback improve over time.

**Acceptance Criteria:**
- [ ] `TraversalTrainingService.train_traversal_policy()` calls a valid PolicyManager method (not the non-existent `load_traversal_policy()`)
- [ ] REINFORCE training loop creates a local optimizer and calls `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- [ ] After training, TraversalPolicy checkpoint is saved (both versioned + latest alias)
- [ ] Training produces non-zero loss and actual parameter updates (verifiable via checkpoint diff)
- [ ] Graceful fallback: if no traversal checkpoint exists, fresh TraversalPolicy is used
- [ ] Graceful fallback: if traversal training fails, GatingPolicy training still completes
- [ ] Unit tests: optimizer wiring, parameter update verification, checkpoint save/load roundtrip
- [ ] No regressions in existing 743 RLCF tests

**Technical Notes:**

**Bug 1 — Non-existent method call:**
- `traversal_training_service.py:176` and `:255` call `pm.load_traversal_policy()` — this method doesn't exist
- `PolicyManager` exposes `get_traversal_policy()` (public) and `_load_traversal_policy()` (private)
- Fix: replace `pm.load_traversal_policy()` with `pm.get_traversal_policy()`
- The `except Exception` in `training_scheduler.py:528` silently swallows the `AttributeError`

**Bug 2 — No optimizer in REINFORCE loop:**
- `traversal_training_service.py:182-210` does `if hasattr(policy, 'optimizer'): ...`
- `TraversalPolicy` has no `.optimizer` attribute → the `if` is always `False` → `loss.backward()` never runs
- Fix: create `torch.optim.Adam(policy.parameters(), lr=config.learning_rate)` locally in `train_traversal_policy()`
- Store optimizer in service instance for potential future checkpoint persistence

**Files to modify:**
- `merlt/merlt/rlcf/traversal_training_service.py` — fix method call + add optimizer

**Files to create:**
- `merlt/tests/rlcf/test_traversal_training_fix.py` — new tests

**Files to read (context):**
- `merlt/merlt/rlcf/policy_manager.py` — `get_traversal_policy()` API
- `merlt/merlt/rlcf/training_scheduler.py` — how traversal training is called (line ~525)
- `merlt/merlt/rlcf/policy_gradient.py` — reference REINFORCE pattern from GatingPolicy

---

### STORY-11-5: Wire WeightStore Persistence into Training Loop

**Priority:** Must Have
**Points:** 3
**Depends on:** 11-1 (WeightStore.save_weights implemented), 11-4 (traversal training works)

**User Story:**
As the RLCF training pipeline,
I want learned weight configurations (gating priors, retrieval alpha, traversal weights) to be persisted to the `weight_versions` PostgreSQL table after each training epoch,
So that weight optimizations survive process restarts and can be tracked with version history.

**Acceptance Criteria:**
- [ ] `run_training_epoch()` calls `WeightStore.save_weights()` after successful training (after `_save_checkpoint()`)
- [ ] Saved weight config includes current GatingPolicy output (expert weights), retrieval alpha, traversal weights
- [ ] Weight version is tagged with training epoch metadata (epoch number, timestamp, num_updates)
- [ ] Previous active version for same experiment_id is deactivated (single active version)
- [ ] If database is unavailable, training still completes (graceful degradation — log warning, skip DB save)
- [ ] `run_training_epoch()` loads active weight config from DB on startup (if available)
- [ ] Unit tests: save after training, load on startup, DB unavailable fallback
- [ ] No regressions in existing RLCF tests

**Technical Notes:**
- `WeightStore.save_weights()` and `_load_from_database()` are implemented (11-1) but never called from the training loop
- `training_scheduler.py:run_training_epoch()` should call `save_weights()` after `_save_checkpoint()` and buffer save
- Need to extract current weight state from trained GatingPolicy (softmax output → expert priors)
- `WeightStore` uses async SQLAlchemy — training scheduler already runs in async context
- `RLCF_DATABASE_URL` env var controls DB connection; if not set, skip DB persistence

**Files to modify:**
- `merlt/merlt/rlcf/training_scheduler.py` — add weight persistence call

**Files to create:**
- `merlt/tests/rlcf/test_weight_store_wiring.py` — integration tests

**Files to read (context):**
- `merlt-models/weights/store.py` — `save_weights()`, `_load_from_database()` API
- `merlt/merlt/rlcf/persistence.py` — `WeightVersion` ORM model

---

### STORY-11-6: Fix Checkpoint Serialization + Orchestrator Trained Flag

**Priority:** Should Have
**Points:** 3
**Depends on:** 11-3 (checkpoint system works)

**User Story:**
As the RLCF system,
I want checkpoint serialization to use the standard PyTorch `state_dict()` pattern and the orchestrator to correctly report whether a trained policy is loaded,
So that checkpoints are future-proof and pipeline traces accurately reflect training state.

**Acceptance Criteria:**
- [ ] `PolicyGradientTrainer.save_checkpoint()` uses `self.policy.mlp.state_dict()` instead of `named_parameters()` dict comprehension
- [ ] `PolicyGradientTrainer.load_checkpoint()` uses `self.policy.mlp.load_state_dict()` instead of manual `.data` assignment
- [ ] Existing checkpoints saved with old format can still be loaded (backward compatibility)
- [ ] Orchestrator `pipeline_trace["trained"]` reflects actual checkpoint presence (not hardcoded `False`)
- [ ] Unit tests: save/load roundtrip with new format, backward compat with old format, trained flag detection
- [ ] No regressions — all 743 RLCF tests still pass

**Technical Notes:**

**Fix 1 — Checkpoint serialization:**
- `policy_gradient.py:834-838` `save_checkpoint` uses `named_parameters()` — misses buffers (BatchNorm etc.)
- `policy_gradient.py:885-888` `load_checkpoint` manually iterates `named_parameters()` and assigns `.data`
- Fix: use `mlp.state_dict()` / `mlp.load_state_dict()` (standard PyTorch)
- Backward compat: detect old format (tensor values instead of state_dict structure) and fall back to manual assignment

**Fix 2 — Orchestrator trained flag:**
- `orchestrator.py:509` has `"trained": False` with TODO comment
- Fix: check if `gating_policy_latest.pt` exists in configured checkpoint_dir
- Can use `PolicyManager.is_gating_policy_available()` if it exists, or direct path check

**Files to modify:**
- `merlt/merlt/rlcf/policy_gradient.py` — save/load checkpoint methods
- `merlt/merlt/experts/orchestrator.py` — trained flag detection

**Files to create:**
- `merlt/tests/rlcf/test_checkpoint_serialization.py` — new tests

---

## Sprint Allocation

**Sprint B — End-to-End Training Loop (11 points)**

| Order | Story | Points | Rationale |
|-------|-------|--------|-----------|
| 1 | STORY-11-4: Fix TraversalTrainingService | 5 | Unblocks traversal training (currently no-op) |
| 2 | STORY-11-5: Wire WeightStore Persistence | 3 | Completes persistence chain (weights → DB) |
| 3 | STORY-11-6: Checkpoint Serialization + Trained Flag | 3 | Correctness fixes, backward compat |

**Execution order rationale:** Fix traversal training first (biggest impact, unblocks weight extraction), then wire weight persistence (depends on traversal training producing real weights), then fix serialization (independent but benefits from stable test baseline).

---

## Epic Traceability

| Sprint | Stories | Points | Goal |
|--------|---------|--------|------|
| A (done) | 11-1, 11-2, 11-3 | 11 | Make GatingPolicy training incremental |
| B | 11-4, 11-5, 11-6 | 11 | Make full training loop functional end-to-end |
| C (backlog) | 11-7+, 11-8+ | TBD | PolicyManager singleton, TraversalPolicy architecture |

---

## Risks and Mitigation

**Medium:**
- TraversalPolicy architecture may need deeper changes for optimizer integration — mitigation: keep optimizer local to training service, don't modify TraversalPolicy class
- WeightStore requires async DB connection — mitigation: graceful fallback if DB unavailable, training never blocks on persistence

**Low:**
- Backward compat for old checkpoint format in 11-6 — mitigation: detect format at load time, support both
- `get_policy_manager()` singleton still has config issue — deferred to Sprint C, workaround is direct instantiation

---

## Definition of Done

For each story:
- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files)
- [ ] Code reviewed (adversarial review via `/bmad:bmm:workflows:code-review`)
- [ ] No regressions in existing test suite (743+ tests)
- [ ] Acceptance criteria validated

---

## Next Steps

Run `/bmad:create-story 11-4` to create detailed story document for first story, or `/bmad:dev-story 11-4` to begin implementation directly.
