# STORY-11-2: Replay Buffer Auto-Persistence

**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Priority:** Must Have
**Story Points:** 3
**Status:** done
**Assigned To:** gpuzio
**Created:** 2026-03-12
**Sprint:** A (RLCF Persistence)

---

## User Story

As the RLCF training scheduler,
I want the replay buffer to auto-save to disk after training and auto-load on startup,
So that accumulated experiences survive process restarts and training is truly incremental.

---

## Description

### Background

The `ExperienceReplayBuffer` and `PrioritizedReplayBuffer` store training experiences in RAM only. When the MERL-T process restarts, all accumulated experiences are lost — meaning feedback collected between training runs disappears. This makes the RLCF training loop effectively stateless.

The `ExperienceReplayBuffer` already has `save(path)` and `load(path)` methods (lines 311-350 of `replay_buffer.py`), but they are never called. The `PrioritizedReplayBuffer` lacks these methods entirely.

### Scope

**In scope:**
- Add `save()` / `load()` to `PrioritizedReplayBuffer`
- Add `buffer_persistence_path` to `SchedulerConfig`
- Auto-load buffer in `TrainingScheduler.__init__()` if file exists
- Auto-save buffer after successful training in `run_training_epoch()`
- Graceful handling of corrupted/missing files

**Out of scope:**
- Database-backed buffer (file persistence is sufficient for MVP scale)
- Buffer compaction or pruning strategies
- Distributed buffer sharing across processes

---

## Acceptance Criteria

- [ ] `PrioritizedReplayBuffer` has `save(path)` and `load(path)` methods that persist all experiences and rebuild the SumTree on load
- [ ] `SchedulerConfig` has `buffer_persistence_path: Optional[str]` field (default: `"data/rlcf/replay_buffer.json"`)
- [ ] `TrainingScheduler.__init__()` calls `buffer.load(path)` if `buffer_persistence_path` is set and file exists
- [ ] `TrainingScheduler.run_training_epoch()` calls `buffer.save(path)` after successful training (after checkpoint save, before return)
- [ ] If buffer file is missing on startup, scheduler starts with empty buffer (no error)
- [ ] If buffer file is corrupted (invalid JSON), scheduler starts with empty buffer and logs a warning
- [ ] `ExperienceReplayBuffer.save()` / `load()` roundtrip preserves all experience fields (experience_id, trace_data, feedback_data, reward, priority, timestamp, metadata)
- [ ] `PrioritizedReplayBuffer.save()` / `load()` roundtrip preserves experiences AND correctly rebuilds SumTree priorities
- [ ] Unit tests cover: save/load roundtrip (both buffer types), corrupted file handling, auto-save after training, auto-load on init

---

## Technical Notes

### Implementation Approach

**1. `PrioritizedReplayBuffer.save(path)` / `load(path)`**

Strategy: serialize the experience list (not the SumTree itself), then rebuild the tree on load. This is simpler and more robust than serializing the tree structure.

```python
def save(self, path: str) -> None:
    experiences = [
        self.tree.data[i] for i in range(self.tree.n_entries)
        if self.tree.data[i] is not None
    ]
    data = {
        "buffer_type": "prioritized",
        "capacity": self.capacity,
        "alpha": self.alpha,
        "epsilon": self.epsilon,
        "total_added": self._total_added,
        "total_sampled": self._total_sampled,
        "experiences": [exp.to_dict() for exp in experiences]
    }
    # write JSON

def load(self, path: str) -> None:
    # read JSON, rebuild SumTree by re-adding each experience
    for exp_data in data["experiences"]:
        exp = Experience.from_dict(exp_data)
        priority = (abs(exp.reward) + self.epsilon) ** self.alpha
        self.tree.add(priority, exp)
```

**2. `SchedulerConfig` changes**

Add field:
```python
buffer_persistence_path: Optional[str] = "data/rlcf/replay_buffer.json"
```

**3. `TrainingScheduler.__init__()` — auto-load**

After buffer creation (lines 217-225), add:
```python
if self.config.buffer_persistence_path:
    self._try_load_buffer()
```

**4. `TrainingScheduler.run_training_epoch()` — auto-save**

After checkpoint save (line ~512), add:
```python
if self.config.buffer_persistence_path:
    self._try_save_buffer()
```

**5. Error handling helpers**

```python
def _try_load_buffer(self):
    path = self.config.buffer_persistence_path
    if not Path(path).exists():
        log.info("No buffer file found, starting fresh", path=path)
        return
    try:
        self.buffer.load(path)
    except Exception as e:
        log.warning("Buffer file corrupted, starting fresh", path=path, error=str(e))

def _try_save_buffer(self):
    path = self.config.buffer_persistence_path
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.buffer.save(path)
    except Exception as e:
        log.warning("Buffer save failed", path=path, error=str(e))
```

### Files to Modify

| File | Change |
|------|--------|
| `merlt/merlt/rlcf/replay_buffer.py` | Add `save()`/`load()` to `PrioritizedReplayBuffer` |
| `merlt/merlt/rlcf/training_scheduler.py` | Add `buffer_persistence_path` to config, wire auto-save/load |

### Files to Read (context, already loaded)

| File | Why |
|------|-----|
| `merlt/merlt/rlcf/replay_buffer.py` | Existing `ExperienceReplayBuffer.save()/load()` pattern |
| `merlt/merlt/rlcf/training_scheduler.py` | Where to wire save/load calls |

---

## Dependencies

**Prerequisite Stories:** None

**Blocked Stories:**
- STORY-11-3 benefits from this (buffer survives, so checkpoint loading + buffer loading together make training fully resumable)

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed
- [ ] Unit tests written and passing (>= 80% coverage on changed files):
  - [ ] `PrioritizedReplayBuffer.save()` / `load()` roundtrip
  - [ ] `ExperienceReplayBuffer.save()` / `load()` roundtrip (regression)
  - [ ] SumTree priorities rebuilt correctly after load
  - [ ] Corrupted file → empty buffer + warning
  - [ ] Missing file → empty buffer + info log
  - [ ] Auto-save triggers after successful training
  - [ ] Auto-load triggers on scheduler init
- [ ] No regressions in existing test suite (`pytest merlt/tests/`)
- [ ] Acceptance criteria validated

---

## Story Points Breakdown

- **PrioritizedReplayBuffer save/load:** 1 point
- **TrainingScheduler wiring:** 1 point
- **Tests:** 1 point
- **Total:** 3 points

**Rationale:** Most code already exists (`ExperienceReplayBuffer.save/load`). The main work is adapting it for `PrioritizedReplayBuffer` (SumTree rebuild) and wiring into the scheduler lifecycle.
