# STORY-11-3: Training Scheduler Checkpoint Loading

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
I want the TrainingScheduler to load the GatingPolicy from the latest checkpoint before training and save a `latest` alias after each run,
So that training is incremental — each epoch builds on the previous one instead of starting from random weights.

---

## Description

### Background

The `TrainingScheduler.run_training_epoch()` method (`merlt/rlcf/training_scheduler.py:433-436`) currently creates a **brand new** `GatingPolicy` with random weights on every call:

```python
# Get or create policy
# In produzione, caricheremmo da checkpoint
policy = GatingPolicy(input_dim=768, hidden_dim=256)
trainer = PolicyGradientTrainer(policy)
```

This means:
- All REINFORCE training from previous epochs is **discarded**
- The optimizer state (Adam moments) is lost
- The baseline (moving average for variance reduction) resets to 0.0
- Every training run starts from scratch

Additionally, the checkpoint save path (line 517) writes with a versioned name like `gating_vYYYYMMDD_HHMMSS_1.pt` but **never writes the `latest` alias** that `PolicyManager._load_gating_policy()` looks for (`gating_policy_latest.pt`). So even if training runs, subsequent PolicyManager inference uses static weights.

### Scope

**In scope:**
- Load GatingPolicy from checkpoint at start of `run_training_epoch()` via `PolicyGradientTrainer.load_checkpoint()`
- Save `latest` alias after each training run (in addition to versioned checkpoint)
- Use singleton `get_policy_manager()` instead of creating a new `PolicyManager()` instance
- Add `checkpoint_dir` to `SchedulerConfig` for configurability
- Track whether policy was loaded from checkpoint vs fresh init (`trained` flag in `TrainingResult`)
- Graceful degradation: fresh policy if no checkpoint exists (first-ever training)

**Out of scope:**
- TraversalPolicy checkpoint loading (separate story if needed)
- Weight Store DB integration for checkpoints (11-1 handles WeightConfig, not torch checkpoints)
- Multi-GPU distributed training
- Checkpoint pruning/cleanup

---

## Acceptance Criteria

- [ ] `run_training_epoch()` loads GatingPolicy from latest checkpoint if one exists, instead of creating a fresh policy with random weights
- [ ] `run_training_epoch()` loads trainer state (optimizer, baseline, num_updates) from checkpoint via `PolicyGradientTrainer.load_checkpoint()`
- [ ] After training, both a versioned checkpoint AND a `latest` alias are saved
- [ ] The `latest` alias is saved via `PolicyManager.save_gating_policy(policy, name="latest")` so that `PolicyManager._load_gating_policy()` can find it
- [ ] When no checkpoint exists (first-ever training), a fresh GatingPolicy is created and training proceeds normally
- [ ] `TrainingResult` includes a `loaded_from_checkpoint` boolean indicating whether training resumed from an existing checkpoint
- [ ] `SchedulerConfig.checkpoint_dir` configures where checkpoints are read from and written to (default: `checkpoints/`)
- [ ] The singleton `get_policy_manager()` is used instead of creating `PolicyManager()` — checkpoint_dir is passed from SchedulerConfig
- [ ] Training is truly incremental: running two consecutive `run_training_epoch()` calls produces a different final loss than running one fresh epoch (the policy state carries over)
- [ ] No regressions in existing RLCF test suite (721 tests)

---

## Technical Notes

### Implementation Approach

**1. Add `checkpoint_dir` to `SchedulerConfig`**

```python
@dataclass
class SchedulerConfig:
    ...
    checkpoint_dir: str = "checkpoints"
```

**2. Refactor `run_training_epoch()` — policy loading**

Replace lines 431-436:

```python
# Before (BUG):
policy = GatingPolicy(input_dim=768, hidden_dim=256)
trainer = PolicyGradientTrainer(policy)

# After:
policy, trainer, loaded_from_checkpoint = self._get_or_create_policy()
```

New helper method:

```python
def _get_or_create_policy(self) -> Tuple[GatingPolicy, PolicyGradientTrainer, bool]:
    from .policy_gradient import PolicyGradientTrainer, GatingPolicy, create_gating_policy

    checkpoint_dir = Path(self.config.checkpoint_dir)
    latest_path = checkpoint_dir / "gating_policy_latest.pt"

    policy = GatingPolicy(input_dim=768, hidden_dim=256)
    trainer = PolicyGradientTrainer(policy)

    if latest_path.exists():
        try:
            trainer.load_checkpoint(str(latest_path))
            log.info("Loaded GatingPolicy from checkpoint", path=str(latest_path))
            return policy, trainer, True
        except Exception as e:
            log.warning("Checkpoint load failed, using fresh policy", error=str(e))

    return policy, trainer, False
```

**3. Refactor checkpoint saving — add `latest` alias**

Replace lines 509-520 to save both versioned checkpoint AND `latest` alias:

```python
if self.config.auto_save_checkpoint and samples_processed > 0:
    checkpoint_dir = Path(self.config.checkpoint_dir)
    version_tag = f"{datetime.now(UTC)...}_{self._training_sessions_today}"
    checkpoint_version = f"v{version_tag}"

    try:
        # Save versioned checkpoint
        versioned_path = checkpoint_dir / f"gating_v{version_tag}.pt"
        trainer.save_checkpoint(str(versioned_path))

        # Save latest alias via PolicyManager
        from .policy_manager import get_policy_manager
        pm = get_policy_manager(checkpoint_dir=checkpoint_dir)
        pm.save_gating_policy(policy, name="latest")

        log.info("Checkpoint saved", version=checkpoint_version, latest=True)
    except Exception as e:
        log.warning("Checkpoint save failed", error=str(e))
```

**4. Add `loaded_from_checkpoint` to `TrainingResult`**

```python
@dataclass
class TrainingResult:
    ...
    loaded_from_checkpoint: bool = False
```

### Files to Modify

| File | Change |
|------|--------|
| `merlt/merlt/rlcf/training_scheduler.py` | Add `checkpoint_dir` to config, refactor policy loading, save `latest` alias, add `loaded_from_checkpoint` to result |

### Files to Read (context)

| File | Why |
|------|-----|
| `merlt/merlt/rlcf/policy_gradient.py` | `PolicyGradientTrainer.save_checkpoint()`, `load_checkpoint()`, `GatingPolicy` constructor |
| `merlt/merlt/rlcf/policy_manager.py` | `save_gating_policy()`, `_load_gating_policy()`, `get_policy_manager()` singleton |

### Key Interactions

- `PolicyGradientTrainer.save_checkpoint(path)` saves: `policy_state_dict`, `optimizer_state_dict`, `baseline`, `num_updates`, `config`, `policy_config`
- `PolicyGradientTrainer.load_checkpoint(path)` restores: policy weights, optimizer state, baseline, num_updates
- `PolicyManager.save_gating_policy(policy, name)` saves: `input_dim`, `hidden_dim`, `num_experts`, `mlp_state_dict` to `{checkpoint_dir}/gating_policy_{name}.pt`
- `PolicyManager._load_gating_policy()` looks for `{checkpoint_dir}/gating_policy_latest.pt`

**Note**: `PolicyGradientTrainer.save_checkpoint()` saves `named_parameters()` as `policy_state_dict`, while `PolicyManager.save_gating_policy()` saves `state_dict()` as `mlp_state_dict`. These are different formats. The save path should use `PolicyManager.save_gating_policy()` for the `latest` alias (so PolicyManager can load it), and `PolicyGradientTrainer.save_checkpoint()` for the versioned checkpoint (includes optimizer state for training resumption).

---

## Dependencies

**Prerequisite Stories:**
- STORY-11-1: Weight Store DB Persistence (done) — WeightVersion model exists
- STORY-11-2: Replay Buffer Auto-Persistence (done) — buffer persistence works

**Blocked Stories:**
- None

**External Dependencies:**
- PyTorch (already installed)

---

## Definition of Done

- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files):
  - [ ] Fresh policy created when no checkpoint exists
  - [ ] Policy loaded from checkpoint when `latest` exists
  - [ ] Versioned checkpoint saved after training
  - [ ] `latest` alias saved after training
  - [ ] `loaded_from_checkpoint` reflects actual state
  - [ ] `checkpoint_dir` config propagates correctly
  - [ ] Corrupted checkpoint → graceful fallback to fresh policy
  - [ ] Incremental training: two consecutive runs differ from single fresh run
- [ ] No regressions in existing test suite (`pytest tests/rlcf/`)
- [ ] Acceptance criteria validated

---

## Story Points Breakdown

- **Config + refactor loading:** 2 points
- **Save latest alias:** 1 point
- **Tests:** 2 points
- **Total:** 5 points

**Rationale:** All building blocks exist (`PolicyGradientTrainer.save/load_checkpoint`, `PolicyManager.save_gating_policy`). Main work is wiring them into `run_training_epoch()` correctly and writing thorough tests that prove incrementality.
