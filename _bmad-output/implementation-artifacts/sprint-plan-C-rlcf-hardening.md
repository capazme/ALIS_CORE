# Sprint Plan C: RLCF Hardening — Singleton Safety + Critical Fixes

**Date:** 2026-03-12
**Scrum Master:** gpuzio
**Project Level:** 4
**Epic:** 12 — RLCF Hardening (Production Reliability)
**Total Stories:** 4
**Total Points:** 10
**Sprint Goal:** Eliminate production-reliability bugs — fix input_dim mismatch, add singleton thread safety, reset ERROR state, secure torch.load calls.
**Velocity:** 11 pts/sprint (from Epic 11 sprints A+B)

---

## Executive Summary

Epic 11 retrospective gap analysis found 7 critical issues. Sprint C addresses the 4 that affect production reliability directly. These are surgical fixes: small blast radius, high confidence, strong test coverage.

---

## Story Inventory

### STORY-12-1: Fix input_dim 768 Fallback in PolicyManager

**Priority:** Must Have
**Points:** 2
**Depends on:** None

**User Story:**
As the RLCF training pipeline,
I want PolicyManager checkpoint loading to default to input_dim=1024 (E5-large),
So that checkpoint fallback creates networks with the correct architecture.

**Acceptance Criteria:**
- [ ] `PolicyManager._load_traversal_policy()` line 244: fallback 768 -> 1024
- [ ] `PolicyManager._load_gating_policy()` line 297: fallback 768 -> 1024
- [ ] `ValueNetwork.__init__` default parameter: 768 -> 1024
- [ ] Docstring examples in `__init__.py`, `ppo_trainer.py`, `single_step_trainer.py`, `prompt_policy.py` updated to 1024
- [ ] Unit test: verify fresh policy created from fallback has input_dim=1024

**Technical Notes:**
- `PolicyManager._load_traversal_policy()` line 244: `checkpoint.get("input_dim", 768)` -> `checkpoint.get("input_dim", 1024)`
- `PolicyManager._load_gating_policy()` line 297: same change
- `ppo_trainer.py:138` `ValueNetwork.__init__` default `input_dim: int = 768` -> `input_dim: int = 1024`
- Docstring-only changes in 4 files (L1 cleanup bundled here)

**Files to modify:**
- `merlt/rlcf/policy_manager.py` (lines 244, 297)
- `merlt/rlcf/ppo_trainer.py` (line 138 + docstring)
- `merlt/rlcf/__init__.py` (docstring)
- `merlt/rlcf/single_step_trainer.py` (docstring)
- `merlt/rlcf/prompt_policy.py` (docstring)

---

### STORY-12-2: Add Threading Lock to Singletons

**Priority:** Must Have
**Points:** 3
**Depends on:** None

**User Story:**
As the FastAPI server running with multiple workers,
I want get_policy_manager() and get_orchestrator() to be thread-safe,
So that concurrent requests don't create duplicate instances or corrupt state.

**Acceptance Criteria:**
- [ ] `get_policy_manager()` uses `threading.Lock` around the `if _policy_manager is None` check
- [ ] `get_orchestrator()` uses `asyncio.Lock` around the `if _orchestrator_instance is None` check (it's async)
- [ ] `PolicyManager._load_traversal_policy()` and `_load_gating_policy()` lazy loading uses `threading.Lock`
- [ ] `reset_policy_manager()` acquires lock before resetting
- [ ] Unit tests: verify concurrent calls return same instance (threading test)
- [ ] No deadlocks in normal usage patterns

**Technical Notes:**
- `get_policy_manager()` (policy_manager.py:696) — sync function, use `threading.Lock`
- `get_orchestrator()` (orchestrator.py:486) — async function, use `asyncio.Lock`
- `_load_traversal_policy` / `_load_gating_policy` — internal lazy loading, use instance-level `threading.Lock`
- Pattern: `get_scheduler()` at training_scheduler.py:995 already uses `_scheduler_lock = threading.Lock()` — replicate this pattern
- Beware: `asyncio.Lock` must be created in event loop context. Use lazy init pattern.

**Files to modify:**
- `merlt/rlcf/policy_manager.py` — add module-level `_pm_lock`, instance-level `_load_lock`
- `merlt/rlcf/orchestrator.py` — add module-level `_orchestrator_lock` (asyncio.Lock)

---

### STORY-12-3: Fix TrainingScheduler ERROR State Recovery

**Priority:** Must Have
**Points:** 2
**Depends on:** None

**User Story:**
As the RLCF training scheduler,
I want to recover from ERROR state after a configurable cooldown period,
So that a single training failure doesn't permanently disable the training loop.

**Acceptance Criteria:**
- [ ] After entering ERROR state, scheduler recovers to IDLE after `error_cooldown_seconds` (default: 300s / 5 min)
- [ ] `should_train()` returns False while in ERROR state (no immediate retry)
- [ ] `should_train()` transitions ERROR -> IDLE when cooldown has elapsed
- [ ] `_error_timestamp` is set when entering ERROR state
- [ ] `get_status()` includes `error_timestamp` and `error_cooldown_remaining_seconds` when in ERROR
- [ ] `error_cooldown_seconds` is configurable via `SchedulerConfig`
- [ ] Unit tests: ERROR -> cooldown -> IDLE transition, should_train behavior during cooldown

**Technical Notes:**
- `training_scheduler.py:591` sets `self._status = TrainingStatus.ERROR` — also set `self._error_timestamp = time.time()`
- `should_train()` at line 331: add check `if self._status == TrainingStatus.ERROR: check cooldown, maybe reset to IDLE`
- Add `error_cooldown_seconds: int = 300` to `SchedulerConfig`
- Add `_error_timestamp: Optional[float] = None` to `TrainingScheduler.__init__`

**Files to modify:**
- `merlt/rlcf/training_scheduler.py` — SchedulerConfig, __init__, should_train, error handling block, get_status

---

### STORY-12-4: Add weights_only=True to torch.load Calls

**Priority:** Must Have
**Points:** 3
**Depends on:** None

**User Story:**
As a security-conscious deployment,
I want all torch.load calls to use weights_only=True where possible,
So that loading checkpoints doesn't execute arbitrary code (PyTorch security best practice).

**Acceptance Criteria:**
- [ ] All 13 `torch.load()` calls audited for `weights_only` compatibility
- [ ] Calls loading only state_dict tensors: add `weights_only=True`
- [ ] Calls loading complex objects (optimizer state with non-tensor data): keep `weights_only=False` with explicit comment
- [ ] FutureWarning from PyTorch >= 2.6 suppressed for legitimate `weights_only=False` cases
- [ ] Unit tests: verify checkpoint load still works with `weights_only=True` where applied
- [ ] No regressions in existing checkpoint tests

**Technical Notes:**
13 torch.load calls found:

**Can use weights_only=True (only tensors/primitives in checkpoint):**
- `hybrid_router.py:312` — loads `model_state_dict` (tensors only)
- `neural.py:495` — loads `model_state_dict` (tensors only)
- `prompt_policy.py:307` — loads state_dict directly
- `disagreement/encoder.py:505` — loads state_dict directly

**Need weights_only=False (optimizer state, complex objects):**
- `policy_gradient.py:882` — loads policy_state_dict + optimizer_state_dict + baseline + config dict
- `ppo_trainer.py:853` — same pattern
- `react_ppo_trainer.py:881` — same pattern
- `single_step_trainer.py:555` — same pattern
- `policy_manager.py:240` — loads full checkpoint with config
- `policy_manager.py:293` — same
- `prompt_policy.py:454` — loads full checkpoint with metadata
- `disagreement/detector.py:410` — loads heads state
- `disagreement/trainer.py:676` — loads full checkpoint

For `weights_only=False` cases, use: `torch.load(path, map_location=device, weights_only=False)` — explicit False suppresses FutureWarning in PyTorch >= 2.6.

**Files to modify:**
- All 13 files listed above — add `weights_only=True` or explicit `weights_only=False`

---

## Sprint Allocation

**Sprint C — RLCF Hardening (10 points / 11 capacity)**

| Order | Story | Points | Rationale |
|-------|-------|--------|-----------|
| 1 | 12-1: Fix input_dim 768 fallback | 2 | Smallest, independent, high-impact correctness fix |
| 2 | 12-3: ERROR state recovery | 2 | Independent, fixes scheduler resilience |
| 3 | 12-2: Singleton thread safety | 3 | Moderate complexity, threading patterns |
| 4 | 12-4: torch.load weights_only | 3 | Widest blast radius (13 files), do last for stable test baseline |

**Utilization:** 10/11 = 91%

---

## Gap Traceability

| Retro Issue | Story | Status |
|-------------|-------|--------|
| C1 (input_dim 768) | 12-1 | Planned |
| C2 (singleton locks) | 12-2 | Planned |
| C4 (ERROR never resets) | 12-3 | Planned |
| C7 (torch.load security) | 12-4 | Planned |
| L1 (docstring 768) | 12-1 (bundled) | Planned |
| L3 (ValueNetwork default) | 12-1 (bundled) | Planned |

---

## Risks and Mitigation

**Medium:**
- `asyncio.Lock` in `get_orchestrator()` must be created within event loop — mitigation: use lazy init pattern (`_lock = None`, create on first use)
- `weights_only=True` may reject checkpoints with non-tensor metadata — mitigation: audit each checkpoint format before applying

**Low:**
- ERROR cooldown might mask persistent failures — mitigation: log every recovery event, add consecutive_errors counter to status

---

## Definition of Done

For each story:
- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files)
- [ ] Code reviewed (adversarial review)
- [ ] No regressions in existing test suite (788+ tests)
- [ ] Acceptance criteria validated

---

## Next Steps

Run `/bmad:dev-story 12-1` to begin with the smallest fix (input_dim 768 -> 1024).

---

**This plan was created using BMAD Method v6 — Phase 4 (Implementation Planning)**
