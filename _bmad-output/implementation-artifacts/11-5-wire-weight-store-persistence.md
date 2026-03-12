# STORY-11-5: Wire WeightStore Persistence into Training Loop

**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Sprint:** B (End-to-End Training Loop)
**Created:** 2026-03-12

---

## User Story

As the RLCF training pipeline,
I want learned weight configurations (gating priors, retrieval alpha, traversal weights) to be persisted to the `weight_versions` PostgreSQL table after each training epoch,
So that weight optimizations survive process restarts and can be tracked with version history.

---

## Description

### Background

`WeightStore.save_weights()` was implemented in STORY-11-1 with full async PostgreSQL persistence, version deactivation, and cache invalidation. However, it is never called from `TrainingScheduler.run_training_epoch()`. The training loop saves **checkpoints** (PyTorch `.pt` files via `PolicyManager`) and **buffers** (replay buffer to disk), but the learned **weight configuration** (expert priors, retrieval alpha, traversal relation weights) is only saved when explicit feedback triggers `WeightLearner.update_from_feedback()`. This means that after a training epoch, the optimized weight configuration is lost on restart.

### Scope

**In scope:**
- Call `WeightStore.save_weights()` after successful training in `run_training_epoch()`
- Extract current weight state from trained GatingPolicy (softmax output → expert priors)
- Extract traversal relation weights from `TraversalTrainingService.get_domain_weights_table()`
- Load active weight config from DB on startup (if available) via `_load_from_database()`
- Graceful degradation: if DB unavailable, training completes normally

**Out of scope:**
- Modifying `WeightStore.save_weights()` internals (already implemented in 11-1)
- Changing `WeightConfig` schema
- Frontend weight visualization (Epic 8)

---

## Acceptance Criteria

- [ ] `run_training_epoch()` calls `WeightStore.save_weights()` after successful training (after `_save_checkpoint()` and buffer save)
- [ ] Saved `WeightConfig` includes current GatingPolicy expert weights (softmax output from trained policy)
- [ ] Saved `WeightConfig` includes traversal relation weights from `get_domain_weights_table()`
- [ ] Weight version is tagged with training metadata (epoch count, samples processed, checkpoint version)
- [ ] Previous active version for same `experiment_id` is deactivated (handled by `save_weights()` internals)
- [ ] If database is unavailable (`RLCF_DATABASE_URL` not set), training still completes — log warning, skip DB save
- [ ] `run_training_epoch()` loads active weight config from DB on startup if available (before training starts)
- [ ] Unit tests: save after training, load on startup, DB unavailable fallback, weight extraction from policy
- [ ] No regressions in existing 756 RLCF tests

---

## Technical Notes

### Insertion Point

`training_scheduler.py:run_training_epoch()` — after line 519 (buffer save) and before line 521 (traversal training):

```python
# After checkpoint + buffer save, persist weight config to DB
await self._persist_weight_config(
    policy=policy,
    checkpoint_version=checkpoint_version,
    samples_processed=samples_processed,
)
```

Move this AFTER traversal training (line 536) so traversal weights are also included.

### Weight Extraction from GatingPolicy

GatingPolicy trained output → expert prior weights:

```python
import torch
with torch.no_grad():
    # GatingPolicy stores expert weights in its output layer
    # The softmax of the last layer bias gives expert priors
    priors = policy.get_expert_weights()  # if method exists
    # OR extract from trainer state
    priors = trainer.get_current_weights()
```

Need to check what methods `GatingPolicy` / `PolicyGradientTrainer` expose for weight extraction.

### WeightStore Integration

```python
from merlt.weights.store import WeightStore
from merlt.weights.config import WeightConfig

store = WeightStore(database_url=os.environ.get("RLCF_DATABASE_URL"))
config = WeightConfig(...)  # Build from policy state
await store.save_weights(
    config=config,
    experiment_id="rlcf_training",
    metrics={"loss": total_loss, "reward": total_reward, "samples": samples_processed}
)
```

### Graceful Degradation

`WeightStore.save_weights()` already handles missing DB gracefully (logs warning, returns version_id anyway). The caller just needs to not crash if the call fails.

### Files to Modify

- `merlt/merlt/rlcf/training_scheduler.py` — add `_persist_weight_config()` method, call after training

### Files to Create

- `merlt/tests/rlcf/test_weight_store_wiring.py` — integration tests

### Files to Read (Context)

- `merlt/merlt/weights/store.py` — `WeightStore.save_weights()`, `_load_from_database()`
- `merlt/merlt/weights/config.py` — `WeightConfig`, `WeightCategory`
- `merlt/merlt/rlcf/policy_gradient.py` — how to extract weights from trained GatingPolicy
- `merlt/merlt/rlcf/traversal_training_service.py` — `get_domain_weights_table()`

---

## Dependencies

**Prerequisite Stories:**
- STORY-11-1: WeightStore.save_weights() implemented (done)
- STORY-11-4: TraversalTrainingService fixed (done) — produces real traversal weights

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] Code implemented and committed
- [ ] `_persist_weight_config()` called after training epoch
- [ ] Weight config loaded from DB on startup
- [ ] Graceful degradation tested (no DB → warning only)
- [ ] Unit tests (≥ 80% coverage on new code)
- [ ] Code review passed
- [ ] No regressions (756+ RLCF tests pass)
- [ ] Acceptance criteria validated

---

## Story Points Breakdown

- **Wiring + extraction logic:** 2 points
- **Testing:** 1 point
- **Total:** 3 points

**Rationale:** Low complexity — `save_weights()` already works, just needs to be called from the right place with the right data. Main effort is weight extraction from policy + proper test coverage.

---

## Progress Tracking

**Status History:**
- 2026-03-12: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
