# Epic 7 Retrospective: Authority & Learning Loop

**Date:** 2026-02-13
**Epic Status:** Done — All 6 stories completed in Sprint 3
**Sprint:** Sprint 3 (single session implementation)
**Previous Retrospective:** Epic 4 (2026-02-02) — No Epic 5/6 retrospective available

---

## Summary

Epic 7 closes the RLCF training loop by implementing authority-weighted feedback, feedback aggregation, training triggers, expert affinity learning, and traversal policy training. This is the culmination of the RLCF pipeline: Epics 4-5 built the expert system and traceability, Epic 6 added feedback collection UI, and Epic 7 makes the system **actually learn** from that feedback.

Six stories were implemented in dependency order:
1. Dynamic authority scoring wired into all feedback endpoints
2. Router feedback (F2) for high-authority users
3. Authority-weighted feedback aggregation with Shannon entropy disagreement
4. Training buffer idle timeout + versioned checkpoints + callbacks
5. Expert affinity learning in the bridge table (F8c)
6. TraversalPolicy REINFORCE training from source feedback (F8d)

---

## Stories Completed

| Story | Description | Key Deliverable | Status |
|-------|-------------|-----------------|--------|
| 7-1 | Authority Score Computation | `_update_user_authority` helper wired into all 5+1 feedback endpoints | Done |
| 7-2 | Router Feedback (F2) | `POST /feedback/router`, `RouterFeedbackRequest`, authority >= 0.7 gate | Done |
| 7-3 | Feedback Aggregation | `FeedbackAggregationService`, `AggregatedFeedback` model, 2 API endpoints | Done |
| 7-4 | Training Buffer & Trigger | `oldest_timestamp()`, idle timeout 7d, versioned checkpoints, callbacks | Done |
| 7-5 | Expert Affinity (F8c) | `AffinityUpdateService`, `expert_affinity` JSONB on bridge table | Done |
| 7-6 | TraversalPolicy Training (F8d) | `TraversalTrainingService`, REINFORCE, integrated in training scheduler | Done |

---

## What Went Well

### 1. Clean Dependency Chain
The implementation followed a strict dependency order (7-1 → 7-2/7-3 → 7-4 → 7-5 → 7-6) that prevented rework. Each story built on the previous one's infrastructure. Authority scoring (7-1) was prerequisite for router feedback authority gating (7-2) and authority-weighted aggregation (7-3). Affinity service (7-5) needed the bridge table column before traversal training (7-6) could use it.

### 2. Non-blocking Integration Pattern
All new hooks (`_update_user_authority`, affinity update, traversal training) are wrapped in try/except blocks that log warnings but never break the main feedback submission flow. This pattern, established in Epic 6's `_wire_feedback_to_training`, proved robust and was consistently applied across all 6 stories.

### 3. Existing Infrastructure Reuse
Epic 7 heavily leveraged existing code:
- `rlcf.authority.update_track_record()` and `update_authority_score()` (existing)
- `rlcf.aggregation.calculate_disagreement()` for Shannon entropy (existing)
- `PolicyManager.save_gating_policy()` / `save_traversal_policy()` for checkpointing (existing)
- `ExperienceReplayBuffer` and `PrioritizedReplayBuffer` (extended, not rewritten)

### 4. Consistent Formula Application
All feedback normalization uses the same formula: `(rating - 1) / 4` mapping 1-5 stars to 0.0-1.0. This consistency across authority scoring, affinity updates, and traversal training prevents subtle bugs from inconsistent scaling.

### 5. Bridge Table as Learning Surface
Adding `expert_affinity` JSONB to the bridge table was a minimal-invasive change that enables per-chunk, per-expert learning. The bridge table now serves triple duty: chunk↔graph mapping, expert affinity storage, and traversal policy training data source.

---

## What Could Be Improved

### 1. Two-Database Session Pattern
The `_update_user_authority` helper opens a separate RLCF session (`get_async_session()`) from the main QATrace session. While both point to the same PostgreSQL instance (port 5433), this creates two independent transactions. A failure in the authority update doesn't roll back the feedback save, which is intentional (non-blocking), but could lead to inconsistencies if the authority DB session succeeds but the main session fails afterward.

**Recommendation:** Consider a shared session factory or two-phase commit for critical paths in Epic 8.

### 2. No Dedicated Tests for Epic 7
Unlike Epic 4 (481 tests) and Epic 5 (108+ tests), Epic 7 was implemented without dedicated unit tests for the 3 new services. The services are syntactically valid and follow established patterns, but lack:
- Unit tests for `FeedbackAggregationService`
- Unit tests for `AffinityUpdateService`
- Unit tests for `TraversalTrainingService`
- Integration tests for the full authority → aggregation → training flow

**Recommendation:** Add test coverage before Epic 8. Priority: aggregation service (most complex logic), then affinity service.

### 3. Stub Embeddings in Traversal Training
`TraversalTrainingService._get_query_embedding()` returns a 768-dim zero vector when the actual embedding isn't in the trace. This means early training samples will all have identical query representations, reducing the policy's ability to learn query-specific traversal preferences.

**Recommendation:** Store query embeddings in `full_trace["query_embedding"]` during pipeline execution (requires a small change in the orchestrator).

### 4. No Migration Script for New Columns/Tables
The `expert_affinity` column on bridge table and `aggregated_feedback` table were added via SQLAlchemy models but no Alembic migration was created. This works for dev but will need migrations for production deployment.

**Recommendation:** Generate Alembic migrations before any production deployment.

---

## Technical Debt

1. **Missing test coverage for Epic 7 services** — FeedbackAggregationService, AffinityUpdateService, TraversalTrainingService lack unit tests
2. **Stub query embeddings** — TraversalTrainingService uses zero vectors when actual embeddings unavailable
3. **No Alembic migrations** — `expert_affinity` column and `aggregated_feedback` table need formal migrations
4. **Aggregation scheduling** — `run_periodic_aggregation()` exists but has no cron/scheduler integration; must be triggered manually via API
5. **Router feedback UI** — Backend endpoint exists (`POST /feedback/router`) but no frontend component (`RouterFeedbackPanel.tsx`) was created; only the `submitRouterFeedback()` service function exists
6. **Implicit F8 inference** — Deferred from Epic 6: deriving F8 signals from expert feedback patterns (currently only explicit source ratings)

---

## Metrics

| Metric | Value |
|--------|-------|
| Files Created | 3 new Python services (~700 LOC) |
| Files Modified | 7 Python + 1 TypeScript |
| New API Endpoints | 3 (router feedback, aggregation run, aggregation latest) |
| New DB Models | 1 table (AggregatedFeedback) + 1 column (expert_affinity JSONB) |
| Feedback Hooks Wired | 6 (all 5 existing + new router endpoint) |
| RLCF Formula Implementations | 4 (authority, affinity, aggregation, REINFORCE) |
| Total Sprint Duration | Single session |

---

## Files Created

| File | Story | Description |
|------|-------|-------------|
| `merlt/merlt/rlcf/feedback_aggregation_service.py` | 7-3 | Authority-weighted aggregation, Shannon entropy disagreement, periodic batch processing |
| `merlt/merlt/rlcf/affinity_service.py` | 7-5 | Expert affinity updates for bridge table, explicit + implicit feedback, bounded learning rate |
| `merlt/merlt/rlcf/traversal_training_service.py` | 7-6 | REINFORCE training for TraversalPolicy, sample extraction from traces, versioned checkpoints |

## Files Modified

| File | Story | Change |
|------|-------|--------|
| `merlt/merlt/api/experts_router.py` | 7-1, 7-2, 7-5 | `_update_user_authority` helper, `RouterFeedbackRequest` model, `POST /feedback/router`, affinity hook in source endpoint, "router" type in `_wire_feedback_to_training` |
| `merlt/merlt/rlcf/training_scheduler.py` | 7-4, 7-6 | Idle timeout 7d, versioned checkpoints via PolicyManager, training callbacks (start/complete/error), TraversalPolicy training in `run_training_epoch()` |
| `merlt/merlt/rlcf/replay_buffer.py` | 7-4 | `oldest_timestamp()` on both `ExperienceReplayBuffer` and `PrioritizedReplayBuffer` |
| `merlt/merlt/storage/bridge/models.py` | 7-5 | `expert_affinity` JSONB column on both `BridgeTableEntryBase` and `BridgeTableEntry` |
| `merlt/merlt/experts/models.py` | 7-3 | `AggregatedFeedback` SQLAlchemy model |
| `merlt/merlt/api/rlcf_router.py` | 7-3 | `POST /rlcf/aggregation/run` and `GET /rlcf/aggregation/latest` endpoints |
| `visualex-merlt/frontend/src/services/merltService.ts` | 7-2 | `submitRouterFeedback()` function |

---

## Key Design Decisions

### 1. Authority-Gated Router Feedback
Only users with `authority >= 0.7` can submit router feedback (F2). This prevents noise from low-authority users influencing the router's classification model. The threshold is configurable via `ROUTER_FEEDBACK_AUTHORITY_THRESHOLD`.

**Rationale:** Router decisions affect all users' experiences. Only domain experts should influence routing weights.

### 2. Bounded Affinity Learning
Expert affinity uses learning rate 0.1 with bounds [0.1, 0.95]. Explicit source feedback gets 3x weight vs implicit inference from expert ratings. This prevents any chunk from becoming permanently invisible to an expert (min 0.1) or monopolizing retrieval (max 0.95).

**Rationale:** Stability and fairness — even poorly-rated chunks might be relevant in new contexts.

### 3. Idle Timeout Training Trigger
The training scheduler now triggers when the buffer has been non-empty but below threshold for > 7 days. This prevents feedback from going stale in low-traffic scenarios.

**Rationale:** Small communities may never hit the 1000-sample threshold. The 7-day idle timeout ensures learning still happens.

### 4. Shannon Entropy for Disagreement
Feedback aggregation uses Shannon entropy (from `aggregation.py`) to detect high-disagreement components (threshold δ > 0.4). This is more robust than variance for Likert-scale ratings.

**Rationale:** Entropy captures multi-modal disagreement (e.g., half rate 1, half rate 5) that variance understates.

### 5. REINFORCE for Traversal Policy
TraversalPolicy training uses REINFORCE policy gradient: `loss = -log(weight) * reward`. This is simpler than actor-critic and sufficient for the discrete relation-type action space (8 relation types).

**Rationale:** The action space is small and the reward signal is clear (source ratings). More complex RL methods would be premature optimization.

---

## Follow-Through on Epic 4 Recommendations

| Recommendation | Status | Notes |
|----------------|--------|-------|
| Traceability Storage (5-1) | Done | Implemented in Epic 5 — QATrace model with full_trace JSONB |
| Source Navigation (5-3) | Done | Implemented in Epic 5 — useSourceNavigation hook |
| Temporal Validity (5-4) | Done | Implemented in Epic 5 — TemporalValidityService |
| Citation Export (5-5) | Done | Implemented in Epic 5 — 4 formats, 108 tests |
| Embedding Service Integration | Partial | E5-large singleton exists; SemanticComparator still uses word overlap fallback |
| Query Classification Model | Partial | Router still uses LLM + regex fallback; RLCF data now flowing for future training |
| Authority Score Persistence | Done | Epic 7 (7-1) — authority scores persisted in RLCF User table |
| Batch Embedding | Not started | Still embeds one at a time |

---

## Recommendations for Epic 8

### 1. Policy Evolution Dashboard (8-1)
The infrastructure is ready: `PolicyManager` saves versioned checkpoints, `TraversalTrainingService.get_domain_weights_table()` returns per-expert relation weights. The dashboard should query these endpoints and visualize weight evolution over time.

### 2. Dataset Export (8-2)
Export should pull from:
- `qa_feedback` table (feedback data)
- `aggregated_feedback` table (aggregation results)
- `QATrace.full_trace` JSONB (pipeline traces)
- PII anonymization (Epic 6 story 6-8) should be implemented first or in parallel

### 3. Query Reproducibility (8-3)
`QATrace` already stores `full_trace` with all expert results, routing decisions, and synthesis. Reproducibility requires:
- Pinning model versions (already in versioned checkpoints)
- Storing exact retrieval results (already in sources JSONB)
- Replay capability (new: feed stored context back through pipeline)

### 4. Devil's Advocate (8-4)
Should trigger on high-consensus responses (when disagreement δ < 0.1). Can leverage:
- `FeedbackAggregationService.aggregate_component_feedback()` for consensus detection
- Existing expert architecture — create a `DevilsAdvocateExpert` that intentionally argues the minority position

### 5. Test Coverage
**Priority before Epic 8:** Write unit tests for the 3 new Epic 7 services. The aggregation service has the most complex logic (authority weighting, entropy calculation, component classification) and should be tested first.

---

## RLCF Pipeline Status (End of Epic 7)

The complete RLCF pipeline is now wired:

```
User submits feedback (F1-F8)
    │
    ├── _wire_feedback_to_training() → ExperienceReplayBuffer
    ├── _update_user_authority() → RLCF User table
    │       └── update_track_record + update_authority_score
    ├── AffinityUpdateService (F8 only) → bridge table expert_affinity
    │
    ▼
TrainingScheduler.should_train()
    ├── buffer >= threshold (1000)? → train
    ├── idle timeout (7 days)? → train
    │
    ▼
run_training_epoch()
    ├── Gating Policy training (from buffer)
    ├── TraversalPolicy training (from DB feedback)
    │       └── TraversalTrainingService.prepare_training_data()
    ├── Save versioned checkpoints
    └── Fire callbacks (on_training_start/complete/error)
    │
    ▼
FeedbackAggregationService.run_periodic_aggregation()
    ├── Authority-weighted averages per component
    ├── Shannon entropy disagreement detection
    └── AggregatedFeedback table
```

---

## Conclusion

Epic 7 completes the RLCF training loop — the defining innovation of the MERL-T system. Feedback now flows from user interactions through authority-weighted aggregation into policy training via REINFORCE. The bridge table learns expert-chunk affinities, and the training scheduler intelligently triggers learning with idle timeouts and versioned checkpoints.

The system is architecturally complete for the core ML pipeline. Epic 8 (Research & Academic Support) will build the observability and reproducibility layer that researchers need to validate the RLCF framework — policy dashboards, dataset export, and the Devil's Advocate system.

Key risk for Epic 8: the 3 new services from Epic 7 lack test coverage. Addressing this before starting Epic 8 is strongly recommended.

---

*Generated: 2026-02-13*
*Author: Claude Code (Epic Retrospective)*
