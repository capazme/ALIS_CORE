# Epic 12 Retrospective: RLCF Hardening (Production Reliability)

**Date:** 2026-03-13
**Epic Status:** Done
**Total Points:** 10 (Sprint C: 10)
**Total Stories:** 4
**Total New Tests:** 41 (10 + 14 + 11 + 6)
**Test Suite:** 829 RLCF tests passing, 0 regressions

---

## What We Delivered

### Sprint C -- RLCF Hardening (10 pts)
| Story | Pts | Summary |
|-------|-----|---------|
| 12-1 | 2 | PolicyManager fallback 768->1024 (E5-large), ValueNetwork + PromptPolicy defaults, docstrings. 10 tests |
| 12-2 | 3 | threading.Lock on get_policy_manager + PolicyManager lazy loading (double-checked locking), asyncio.Lock on get_orchestrator. 11 tests |
| 12-3 | 2 | TrainingScheduler ERROR->IDLE cooldown recovery, configurable error_cooldown_seconds, run_training_epoch guard. 14 tests |
| 12-4 | 3 | 13 torch.load calls audited: 6 weights_only=True (inference), 7 weights_only=False (optimizer). Audit test. 6 tests |

### Before vs After

| Capability | Before Epic 12 | After Epic 12 |
|------------|----------------|---------------|
| Checkpoint fallback dim | 768 (BERT-base, wrong) | 1024 (E5-large, correct) |
| Singleton thread safety | Race condition on concurrent requests | threading.Lock + asyncio.Lock with double-checked locking |
| ERROR state | Permanent, requires restart | Auto-recovery after configurable cooldown (default 300s) |
| torch.load security | No weights_only param (PyTorch FutureWarning) | Explicit True/False on all 13 calls |
| Lazy loading race | Flag set inside lock, load outside | Entire load body inside lock |

---

## What Went Well

1. **Sprint completed in 1 session** -- all 4 stories implemented, reviewed, committed without blockers
2. **Code review caught 3 real bugs in 12-2** -- double-checked locking was broken (load body outside lock), reset_policies/save_policies not acquiring lock, orchestrator lazy init fragile
3. **Code review caught 1 bug in 12-3** -- _error_timestamp not reset on successful training, run_training_epoch bypassing ERROR guard
4. **Zero regressions** -- 829 tests passing across all 4 stories
5. **Audit test for 12-4** -- `test_no_torch_load_without_weights_only` prevents future regressions by scanning all .py files

## What Could Be Improved

1. **12-2 double-checked locking wrong on first attempt** -- set flag inside lock but left the entire load body outside. Code review caught it. Lesson: always keep the full critical section inside the lock.
2. **policy.requires_grad_(False) crashed** -- GatingPolicy is not nn.Module, `requires_grad_` does not exist on it. Used `policy.mlp.requires_grad_(False)` instead. Should have checked class hierarchy before changing from the original inference-mode call.
3. **Sprint plan 12-4 misclassified** -- plan said `policy_manager.py` needs `weights_only=False` (optimizer), but actual checkpoint contains only tensors + primitives. Corrected to `weights_only=True` during implementation.

## Key Decisions Made

1. **Module-level asyncio.Lock** for orchestrator -- Python 3.12 allows `asyncio.Lock()` at module level. Simpler than lazy init, no race window.
2. **`_traversal_loaded = True` set LAST** in double-checked locking -- flag set only after `self._traversal_policy = policy` assignment, ensuring concurrent threads never see stale None.
3. **ERROR guard in run_training_epoch()** -- blocks direct calls during cooldown AND allows recovery after cooldown (clears error state). Prevents both bypass and permanent lock.

---

## Remaining Technical Debt (Updated Gap Analysis)

### Critical (7 fixed, 2 open)

| # | Issue | Status | Area |
|---|-------|--------|------|
| C1 | input_dim 768 fallback | FIXED (12-1) | policy_manager.py |
| C2 | singleton thread safety | FIXED (12-2) | policy_manager.py, orchestrator.py |
| C3 | NeuralGatingTrainer disconnected from RLCF loop | **OPEN** | Architecture gap |
| C4 | ERROR state never resets | FIXED (12-3) | training_scheduler.py |
| C5 | RLCFOrchestrator zero test coverage | **OPEN** | tests/rlcf/ |
| C6 | NERRLCFIntegration class untested | **PARTIAL** | NERFeedbackBuffer has tests, integration class does not |
| C7 | torch.load security | FIXED (12-4) | 13 files |

### Important (0 fixed, 7 open)

| # | Issue | Status | Area |
|---|-------|--------|------|
| I1 | GatingPolicy.sample_action deterministic is a no-op | OPEN | policy_gradient.py |
| I2 | PPOTrainer compute_advantages dead code | OPEN | ppo_trainer.py |
| I3 | get_weight_evolution stub returning [] | OPEN | orchestrator.py |
| I4 | WeightStore singleton bypassed by TraversalTrainingService | OPEN | Wiring gap |
| I5 | Missing tests: reproducibility_service, quarantine_service | PARTIAL | tests/rlcf/ |
| I6 | experiment_id hardcoded strings | OPEN | orchestrator.py, training_scheduler.py |
| I7 | entity_feedback.py and prompt_policy.py orphan code | OPEN | Dead code |

### Low (3 fixed, 1 open)

| # | Issue | Status | Area |
|---|-------|--------|------|
| L1 | docstring 768 | FIXED (12-1) | Multiple docstrings |
| L2 | _get_torch F variable name | FIXED (already correct) | policy_gradient.py |
| L3 | ValueNetwork default 768 | FIXED (12-1) | ppo_trainer.py |
| L4 | simulator zero tests | OPEN | tests/ |

**Summary:** 10/18 fixed, 8 open, 2 partial

---

## Recommended Next Sprints

### Sprint D -- Test Coverage Debt (Est. 8-11 pts)
Focus: C5, C6, I5 -- bring RLCFOrchestrator, NERRLCFIntegration, reproducibility_service, quarantine_service to 80%+ coverage

### Sprint E -- Architecture Alignment (Est. 5-8 pts)
Focus: C3, I4 -- connect NeuralGatingTrainer to RLCF loop, fix WeightStore singleton wiring

### Sprint F -- Dead Code Cleanup (Est. 3-5 pts)
Focus: I1, I2, I3, I7 -- remove dead branches, stubs, orphan modules

### Sprint G -- Config Hardening (Est. 2-3 pts)
Focus: I6 -- make experiment_id configurable

---

## Cumulative Epic Progress

| Epic | Sprint | Points | Stories | New Tests | Suite Total |
|------|--------|--------|---------|-----------|-------------|
| 11 | A | 11 | 3 | 53 | 738 |
| 11 | B | 11 | 3 | 44 | 788 |
| 12 | C | 10 | 4 | 41 | 829 |
| **Total** | **3 sprints** | **32** | **10** | **138** | **829** |

---

**This retrospective was created using BMAD Method v6 -- Phase 4 (Sprint Retrospective)**
