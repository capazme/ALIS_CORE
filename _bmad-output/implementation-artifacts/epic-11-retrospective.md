# Epic 11 Retrospective: RLCF Persistence (Infrastructure Debt)

**Date:** 2026-03-12
**Epic Status:** Done
**Total Points:** 22 (Sprint A: 11, Sprint B: 11)
**Total Stories:** 6
**Total New Tests:** 97 (20 + 21 + 18 + 13 + 15 + 17 — minus overlap = 97 net new)
**Test Suite:** 788 RLCF tests passing, 0 regressions

---

## What We Delivered

### Sprint A — Make Training Incremental (11 pts)
| Story | Pts | Summary |
|-------|-----|---------|
| 11-1 | 5 | WeightStore DB persistence — save_weights + load_from_database with WeightVersion ORM |
| 11-2 | 3 | PrioritizedReplayBuffer save/load, auto-persist on training completion |
| 11-3 | 3 | PolicyManager checkpoint loading, latest alias, loaded_from_checkpoint flag |

### Sprint B — End-to-End Training Loop (11 pts)
| Story | Pts | Summary |
|-------|-----|---------|
| 11-4 | 5 | Fixed 6 bugs in TraversalTrainingService (was complete silent no-op) + 5 code review fixes |
| 11-5 | 3 | Wired WeightStore into training loop — extract_weight_config, persist_weight_config, WeightVersion dedup |
| 11-6 | 3 | state_dict serialization, load_state_dict, orchestrator trained flag, corrupted checkpoint resilience |

### Before vs After

| Capability | Before Epic 11 | After Epic 11 |
|------------|----------------|---------------|
| GatingPolicy persistence | Fresh random policy every run | Loads from checkpoint, saves versioned + latest |
| TraversalPolicy training | Silent no-op (6 bugs) | Real REINFORCE with optimizer, grad clipping, checkpoint save |
| Replay buffer | RAM-only, lost on restart | Auto-save/load JSON, survives restarts |
| Weight DB persistence | save_weights was stub | Full async PostgreSQL persistence with dedup |
| Checkpoint format | named_parameters (misses buffers) | state_dict (standard PyTorch) |
| Orchestrator trained flag | Hardcoded False | Detects from HybridExpertRouter.loaded_from_checkpoint |

---

## What Went Well

1. **Velocity consistent** — 11 pts/sprint across both sprints, predictable throughput
2. **Code review caught real bugs** — mlp mode not restored (11-5), no try/except on checkpoint load (11-6), fragile test regex (11-6)
3. **Backward compatibility preserved** — old checkpoint format loads seamlessly with new code
4. **Zero regressions** — 788 tests pass, no existing test broken across 6 stories
5. **Infrastructure debt eliminated** — the training loop now works end-to-end for the first time

## What Could Be Improved

1. **Sprint B story 11-4 was underestimated** — 6 bugs found vs 2 described in the story, required deeper analysis
2. **WeightVersion ORM duplication** (caught in 11-5) — two files defining same table caused SQLAlchemy InvalidRequestError; should have been caught in 11-1
3. **get_policy_manager singleton issues deferred** — Sprint C backlog item still open, creates potential race conditions

## Key Decisions Made

1. **Keep optimizer local to TraversalTrainingService** — dont modify TraversalPolicy class (simpler, less risk)
2. **Use getattr fallback for trained flag** — backward compatible with old HybridExpertRouter instances
3. **try/except around checkpoint loading** — corrupted checkpoints fall back to warm-start priors instead of crashing

---

## Remaining Technical Debt (Gap Analysis)

### Critical (blocks production reliability)

| # | Issue | Area |
|---|-------|------|
| C1 | input_dim fallback hardcoded to 768 in PolicyManager (should be 1024) | policy_manager.py |
| C2 | get_policy_manager and get_orchestrator singletons have no threading lock | policy_manager.py, orchestrator.py |
| C3 | NeuralGatingTrainer (experts/neural_gating) and PolicyGradientTrainer (rlcf) train separate models — not connected | Architecture gap |
| C4 | TrainingScheduler ERROR status never resets — scheduler enters permanent error state | training_scheduler.py |
| C5 | RLCFOrchestrator zero test coverage | tests/rlcf/ |
| C6 | ner_feedback_buffer + ner_rlcf_integration zero tests, used by production endpoints | tests/rlcf/ |
| C7 | torch.load without weights_only=True — PyTorch security warning | Multiple files |

### Important (correctness/maintainability)

| # | Issue | Area |
|---|-------|------|
| I1 | GatingPolicy.sample_action deterministic branch is dead code | policy_gradient.py |
| I2 | PPOTrainer marked legacy but not deprecated formally, compute_advantages dead code | ppo_trainer.py |
| I3 | get_weight_evolution is a stub returning empty list, exposed as API | rlcf/orchestrator.py |
| I4 | TraversalTrainingService updates checkpoint but doesnt propagate to production singleton | Wiring gap |
| I5 | Missing tests: reproducibility_service, quarantine_service, domain_authority, edit_merge, export_service | tests/rlcf/ |
| I6 | experiment_id and model_version hardcoded strings | Multiple files |
| I7 | entity_feedback.py and prompt_policy.py are orphan code | Dead code |

### Low (cleanup)

| # | Issue | Area |
|---|-------|------|
| L1 | Docstring examples reference input_dim=768 (misleading) | Multiple docstrings |
| L2 | _get_torch unpacking assigns _optim to F variable name | policy_gradient.py |
| L3 | ValueNetwork default input_dim=768 (overridden at runtime, but misleading) | ppo_trainer.py |
| L4 | simulator package has zero tests | tests/ |

---

## Recommended Next Sprints

### Sprint C — Singleton Safety + Critical Fixes (Est. 8-11 pts)
Focus: C1, C2, C4, C7 — fix production reliability issues

### Sprint D — Test Coverage Debt (Est. 8-13 pts)
Focus: C5, C6, I5 — bring critical untested modules to 80%+ coverage

### Sprint E — Architecture Alignment (Est. 5-8 pts)
Focus: C3, I4 — unify NeuralGatingTrainer/PolicyGradientTrainer, fix singleton propagation

### Sprint F — Dead Code Cleanup (Est. 3-5 pts)
Focus: I1, I2, I7, L1-L4 — remove orphan code, fix misleading defaults/docstrings

---

**This retrospective was created using BMAD Method v6 — Phase 4 (Sprint Retrospective)**
