# STORY-11-4: Fix TraversalTrainingService (Silent No-Op)

**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Priority:** Must Have
**Story Points:** 5
**Status:** ready-for-dev
**Assigned To:** gpuzio
**Created:** 2026-03-12
**Sprint:** B (RLCF Persistence — End-to-End Training Loop)

---

## User Story

As the RLCF training pipeline,
I want TraversalPolicy training to actually update the policy weights and save loadable checkpoints,
So that traversal preferences learned from F8 source-quality feedback improve over time instead of being silently discarded.

---

## Description

### Background

Story 7-6 implemented `TraversalTrainingService` with a REINFORCE loop for training `TraversalPolicy` from F8 feedback samples. However, the implementation has **6 bugs** that combine to make the entire traversal training a **silent no-op**:

1. Calls a non-existent `PolicyManager.load_traversal_policy()` method — raises AttributeError
2. The `except Exception` in `training_scheduler.py:527` swallows the error — no crash, no signal
3. The REINFORCE loop checks `hasattr(policy, 'optimizer')` which is always `False` — `loss.backward()` never runs
4. The policy is loaded in `policy.eval()` mode — gradients not computed correctly
5. Checkpoint is saved with a versioned name but never as `latest` — next load finds nothing
6. `input_dim` mismatch: fresh policy uses `1024`, but `_load_traversal_policy()` defaults to `768`

**Net result:** Every `run_training_epoch()` call attempts traversal training, hits AttributeError, logs a warning, and continues. The TraversalPolicy never learns anything.

### Scope

**In scope:**
- Fix method call: `pm.load_traversal_policy()` to `pm.get_traversal_policy()`
- Add local optimizer (Adam) to REINFORCE loop
- Call `policy.train()` before training, `policy.eval()` after
- Save `latest` alias after training (in addition to versioned checkpoint)
- Fix `input_dim` to `1024` (E5-large) consistently
- Include traversal training result in `TrainingResult`

**Out of scope:**
- TraversalPolicy architecture changes (expert_type input — Sprint C)
- Traversal optimizer state persistence across restarts (future story)
- TraversalPolicy checkpoint loading in `_get_or_create_policy()` (GatingPolicy only)

---

## Acceptance Criteria

- [ ] `train_traversal_policy()` calls `pm.get_traversal_policy()` (valid public method) instead of `pm.load_traversal_policy()`
- [ ] REINFORCE loop creates `torch.optim.Adam(policy.parameters(), lr=1e-4)` and calls `zero_grad()`, `backward()`, `step()` on every batch
- [ ] Policy is set to `train()` mode before the loop and `eval()` after
- [ ] After training, checkpoint is saved as both versioned name AND `name="latest"` for next load
- [ ] Fresh TraversalPolicy uses `input_dim=1024` (matching E5-large embeddings)
- [ ] `get_domain_weights_table()` also uses the correct method call
- [ ] Training produces non-zero loss and actual parameter updates (verifiable via weight diff before/after)
- [ ] `TrainingResult` includes traversal training outcome (success/skipped/error + samples_used)
- [ ] Graceful fallback: if no traversal checkpoint exists, fresh TraversalPolicy is created
- [ ] Graceful fallback: if traversal training fails, GatingPolicy training result is still returned
- [ ] No regressions in existing 743 RLCF tests

---

## Technical Notes

### Bug 1 — Non-existent method call

**File:** `traversal_training_service.py:176` and `:255`
```python
# BUG: load_traversal_policy() does not exist on PolicyManager
policy = pm.load_traversal_policy()
```

**Fix:**
```python
policy = pm.get_traversal_policy()
```

`get_traversal_policy()` is the public wrapper (policy_manager.py:611) that calls `_load_traversal_policy()` internally. Returns `Optional[TraversalPolicy]` — `None` if no checkpoint exists.

### Bug 2 — No optimizer in REINFORCE loop

**File:** `traversal_training_service.py:199-204`
```python
# BUG: TraversalPolicy has no .optimizer attribute, always False
if hasattr(policy, 'optimizer'):
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()
```

**Fix:** Create optimizer before the loop:
```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# In the loop:
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Learning rate `1e-4` matches the default in `PolicyGradientTrainer` for consistency.

### Bug 3 — Policy in inference mode during training

**File:** `policy_manager.py:254` sets `policy.eval()` on load.

**Fix:** In `train_traversal_policy()`, after loading the policy:
```python
policy.train()  # Switch to training mode
# ... training loop ...
policy.eval()   # Switch back for checkpoint save
```

### Bug 4 — Checkpoint `latest` alias never saved

**File:** `traversal_training_service.py:212-217`
```python
# BUG: only saves versioned name, never "latest"
pm.save_traversal_policy(policy, name=checkpoint_name)
```

**Fix:** Add a second save for the `latest` alias:
```python
pm.save_traversal_policy(policy, name=checkpoint_name)   # versioned
pm.save_traversal_policy(policy, name="latest")          # alias for next load
```

### Bug 5 — input_dim mismatch

**File:** `traversal_training_service.py:179` — fresh policy uses `input_dim=1024`
**File:** `policy_manager.py:244` — `_load_traversal_policy()` defaults `input_dim` to `768`

Both should use `1024` (E5-large). The `_load_traversal_policy()` reads `input_dim` from the checkpoint dict, so this only matters for the **first-ever** training when no checkpoint exists. Verify that the fresh policy in `train_traversal_policy()` uses `1024`.

### Bug 6 — TrainingResult missing traversal info

**File:** `training_scheduler.py:517-528` — traversal result is logged but not included in return.

**Fix:** Add optional traversal fields to `TrainingResult`:
```python
traversal_trained: bool = False
traversal_samples: int = 0
```

### Files to Modify

| File | Change |
|------|--------|
| `merlt/merlt/rlcf/traversal_training_service.py` | Fix method call (x2), add optimizer, train/mode switch, save latest alias, fix input_dim |
| `merlt/merlt/rlcf/training_scheduler.py` | Add traversal fields to TrainingResult, populate from trav_result |

### Files to Create

| File | Purpose |
|------|---------|
| `merlt/tests/rlcf/test_traversal_training_fix.py` | Tests for all 6 fixes |

### Files to Read (context)

| File | Why |
|------|-----|
| `merlt/merlt/rlcf/policy_manager.py` | `get_traversal_policy()`, `save_traversal_policy()`, `_load_traversal_policy()` API |
| `merlt/merlt/rlcf/policy_gradient.py` | Reference REINFORCE pattern (optimizer creation, train/mode switch) |

### Key Interactions

- `PolicyManager.get_traversal_policy()` returns lazy-loaded policy from `traversal_policy_latest.pt`, or `None` if not found
- `PolicyManager.save_traversal_policy(policy, name)` saves to `{checkpoint_dir}/traversal_policy_{name}.pt`
- `TraversalPolicy(input_dim=1024, hidden_dim=128)` — MLP mapping embeddings to relation type weights
- `TraversalTrainingService.prepare_training_data(session)` queries F8 feedback from DB, returns `List[TraversalTrainingSample]`
- `MIN_SAMPLES = 20` — training skipped if fewer samples available

---

## Dependencies

**Prerequisite Stories:**
- STORY-11-3: Training Scheduler Checkpoint Loading (done) — checkpoint infrastructure exists

**Blocked Stories:**
- STORY-11-5: Wire WeightStore Persistence (depends conceptually — traversal must produce real weights first)

**External Dependencies:**
- None (PyTorch already installed)

---

## Definition of Done

- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files):
  - [ ] `get_traversal_policy()` called instead of `load_traversal_policy()`
  - [ ] Optimizer created and used in REINFORCE loop
  - [ ] `policy.train()` before loop, `policy.eval()` after
  - [ ] Both versioned + latest checkpoint saved
  - [ ] Fresh policy uses input_dim=1024
  - [ ] Parameter weights change after training (non-zero update)
  - [ ] Graceful fallback when no checkpoint exists
  - [ ] TrainingResult includes traversal outcome
- [ ] No regressions in existing test suite (`pytest tests/rlcf/`)
- [ ] Acceptance criteria validated

---

## Story Points Breakdown

- **Fix 6 bugs in traversal_training_service.py:** 2 points
- **Add traversal fields to TrainingResult:** 1 point
- **Tests (8+ test cases):** 2 points
- **Total:** 5 points

**Rationale:** All bugs are in a single file with a clear fix pattern. The REINFORCE reference implementation exists in `PolicyGradientTrainer`. Main effort is test coverage for all the edge cases (no checkpoint, corrupted, fresh policy, parameter update verification).
