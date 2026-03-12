# STORY-11-6: Fix Checkpoint Serialization + Orchestrator Trained Flag

**Epic:** 11 — RLCF Persistence (Infrastructure Debt)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Sprint:** B (End-to-End Training Loop)
**Created:** 2026-03-12

---

## User Story

As the RLCF system,
I want checkpoint serialization to use standard PyTorch `state_dict()` and the orchestrator to correctly report whether a trained policy is loaded,
So that checkpoints are future-proof and pipeline traces accurately reflect training state.

---

## Acceptance Criteria

- [ ] `PolicyGradientTrainer.save_checkpoint()` uses `self.policy.mlp.state_dict()` instead of `named_parameters()` dict comprehension
- [ ] `PolicyGradientTrainer.load_checkpoint()` uses `self.policy.mlp.load_state_dict()` instead of manual `.data` assignment
- [ ] Existing checkpoints saved with old format can still be loaded (backward compatibility)
- [ ] Orchestrator `pipeline_trace["trained"]` reflects actual checkpoint presence (not hardcoded `False`)
- [ ] Unit tests: save/load roundtrip with new format, backward compat with old format, trained flag detection
- [ ] No regressions in existing 771 RLCF tests

---

## Technical Notes

### Fix 1 — save_checkpoint (policy_gradient.py:836)
```python
# OLD: named_parameters() — misses buffers, relation_embeddings
"policy_state_dict": {name: param.cpu() for name, param in self.policy.mlp.named_parameters()}

# NEW: state_dict() — standard PyTorch, includes everything
"policy_state_dict": {k: v.cpu() for k, v in self.policy.mlp.state_dict().items()}
```

### Fix 2 — load_checkpoint (policy_gradient.py:886)
```python
# OLD: manual .data assignment
for name, param in self.policy.mlp.named_parameters():
    if name in policy_state:
        param.data = policy_state[name].to(self.policy.device)

# NEW: load_state_dict — handles all edge cases
self.policy.mlp.load_state_dict(
    {k: v.to(self.policy.device) for k, v in policy_state.items()}
)
```

### Fix 3 — Backward compat
Old format has tensor values from `named_parameters()`. New format has tensor values from `state_dict()`. For `mlp` (Sequential), these produce identical keys/values since Sequential has no buffers. So backward compat is automatic — no special handling needed.

### Fix 4 — Trained flag (orchestrator.py:509)
```python
# OLD:
"trained": False,  # hardcoded

# NEW: detect via checkpoint existence
"trained": self._is_gating_policy_trained(),
```

Use `PolicyManager.is_gating_policy_available()` or direct path check on `checkpoints/gating_policy_latest.pt`.

### Files to modify
- `merlt/merlt/rlcf/policy_gradient.py` — save/load checkpoint methods
- `merlt/merlt/experts/orchestrator.py` — trained flag detection

### Files to create
- `merlt/tests/rlcf/test_checkpoint_serialization.py`

---

## Dependencies

- STORY-11-3: Checkpoint infrastructure (done)
- STORY-11-4: TraversalTrainingService fixed (done)

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
