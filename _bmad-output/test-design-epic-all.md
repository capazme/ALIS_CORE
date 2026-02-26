# Test Design: All Epics (Post-Implementation) — ALIS_CORE

**Date:** 2026-02-17
**Author:** Gpuzio
**Status:** Draft
**Baseline:** `test-design-system.md` (2026-02-16)

---

## Executive Summary

**Scope:** Post-implementation comprehensive test design across all 10 epics (64 stories, all done)

**Architecture:** 3-layer service (Express :3001 / Quart :5000 / FastAPI :8000), 4 databases (PostgreSQL, FalkorDB, Qdrant, Redis), 4-Expert ML pipeline, RLCF feedback loop, React + Vite frontends with plugin system.

**What changed since system-level review (2026-02-16):**
- Auth middleware wired to all routers (Sprint 2: `wire auth to all routers + 90 P1 coverage tests`)
- CI pipeline created for merlt backend + frontend (Sprint 0)
- Frontend test frameworks installed (Vitest in both visualex-platform and visualex-merlt)
- Health check endpoints added (`/health` in both app.py and visualex_bridge.py)
- Hardcoded credentials, mock/placeholder issues systematically fixed (code audit session)
- CONSENT_IP_SALT added to env files, Dockerfile nginx envsubst fixed
- 142 test files exist across the monorepo

**Risk Summary (Updated):**

- Total risks identified: 16
- High-priority risks (≥6): 4 (down from 6 — 2 mitigated)
- Critical categories: TECH (2), DATA (1), SEC (1)

**Coverage Summary:**

- Existing tests: ~142 files (116 merlt, 12 platform-backend, 9 platform-frontend, 5 merlt-frontend)
- P0 scenarios needed: 12 tests (24 hours)
- P1 scenarios needed: 28 tests (28 hours)
- P2 scenarios needed: 35 tests (17.5 hours)
- P3 scenarios needed: 12 tests (3 hours)
- **Total new effort**: 87 tests, ~72.5 hours (~9 days)

---

## Risk Assessment (Post-Implementation Update)

### Mitigation Status from System-Level Review

| Original Risk | Original Score | Current Status | Residual Score |
|---------------|---------------|----------------|----------------|
| R-001: Unauthenticated API (26+ routes) | **9** | **MITIGATED** — Auth wired to all routers (Sprint 2), 90 P1 coverage tests | 3 |
| R-002: JWT forgery in ws_router | **6** | **OPEN** — Status unverified | 6 |
| R-003: 40+ bare `except:` handlers | **6** | **PARTIAL** — Some fixed during code audit, many likely remain | 4 |
| R-004: Cross-store data inconsistency | **6** | **OPEN** — Structural issue, no transactional wrapper added | 6 |
| R-005: Unbounded `_runs` dict | **6** | **PARTIAL** — Warning log added, but no maxlen eviction | 4 |
| R-006: GDPR PII gap | **6** | **PARTIAL** — PII masking wired to feedback endpoints + export | 4 |
| R-008: No CI for merlt | **4** | **MITIGATED** — CI pipeline exists (pytest, black, ruff, bandit, vitest) | 1 |
| R-009: Zero frontend tests | **4** | **PARTIAL** — 14 test files exist (9 platform + 5 merlt), but E2E still empty | 2 |
| R-010: No health check | **4** | **MITIGATED** — `/health` endpoint in both app.py and visualex_bridge.py | 1 |
| R-011: Hardcoded credentials | **3** | **MITIGATED** — Env vars, .env.example, dev defaults | 1 |

### Current High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Prob | Impact | Score | Mitigation | Owner | Timeline |
|---------|----------|-------------|------|--------|-------|------------|-------|----------|
| R-002 | SEC | JWT token parsing in `ws_router.py` without signature verification — WebSocket auth can be forged | 2 | 3 | **6** | Add PyJWT verified decode with algorithm whitelist; unit tests for invalid/expired tokens | Dev | Week 1 |
| R-004 | DATA | Cross-store inconsistency — bridge table can desync from Qdrant/FalkorDB during partial ingestion failures | 2 | 3 | **6** | Add transactional ingestion wrapper with rollback; consistency-check endpoint | Dev | Week 2 |
| R-015 | TECH | Zero E2E test coverage — Playwright config exists but `e2e/` directory is empty; no end-to-end user flow validation | 3 | 2 | **6** | Write P0 E2E tests: registration → login → search → MERL-T analysis → feedback | Dev | Week 1-2 |
| R-017 | TECH | In-memory-only persistence — `weights/store.py`, `training_scheduler` buffer, `regression_router._runs` all lose data on restart | 3 | 2 | **6** | Add Redis/PostgreSQL persistence for critical state; maxlen eviction for _runs | Dev | Week 2-3 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Prob | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|------|--------|-------|------------|-------|
| R-003 | TECH | Remaining bare `except:` handlers — reduced but not eliminated | 2 | 2 | 4 | Continue systematic replacement; add ruff E722 rule to CI | Dev |
| R-005 | PERF | Unbounded collections (`_runs` dict, audit log query) — gradual memory/performance degradation | 2 | 2 | 4 | OrderedDict with maxlen=100; paginated audit query | Dev |
| R-006 | SEC | GDPR PII gap in non-feedback paths — trace viewer, graph query responses may expose PII | 2 | 2 | 4 | Audit all data egress; wire consent check | Dev |
| R-016 | DATA | RLCF data pipeline not end-to-end tested — statistics endpoints return empty data, no real training loop validated | 2 | 2 | 4 | Integration test: synthetic feedback → aggregation → training → weight update | Dev |
| R-018 | OPS | Docker config changes untested — nginx envsubst template, MERLT_API_KEY passthrough | 2 | 2 | 4 | Docker Compose build + smoke test in CI | Dev |
| R-019 | BUS | Plugin system untested — EventBus, PluginSlot, slot rendering, cross-component events | 2 | 2 | 4 | Component tests for plugin registration, event delivery, slot rendering | Dev |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Prob | Impact | Score | Action |
|---------|----------|-------------|------|--------|-------|--------|
| R-001 | SEC | Residual auth gaps (already wired, need verification) | 1 | 3 | 3 | Verify via auth enforcement test sweep |
| R-013 | BUS | Expert router regex fallback misclassification | 1 | 2 | 2 | Monitor classification accuracy |
| R-014 | OPS | Docker-only deployment (no production orchestration) | 1 | 2 | 2 | Document |
| R-020 | TECH | Archived tests (28 files in _archive/) may contain useful coverage | 1 | 1 | 1 | Triage: restore useful, delete broken |

---

## Existing Test Coverage Analysis

### Current Inventory (142 test files)

| Area | Files | Framework | Coverage Quality | Key Gaps |
|------|-------|-----------|-----------------|----------|
| **merlt/tests/rlcf/** | 30 | pytest | ✅ Comprehensive | Training end-to-end, persistence |
| **merlt/tests/api/** | 14 | pytest | ✅ Good | Contract validation, error shapes |
| **merlt/tests/experts/** | 9 | pytest | ✅ Good | Neural gating edge cases |
| **merlt/tests/storage/** | 10 | pytest | ✅ Solid | Cross-store consistency |
| **merlt/tests/disagreement/** | 11 | pytest | ✅ Specialized | Authority conflict resolution |
| **merlt/tests/integration/** | 6 | pytest | ⚠️ Basic | Full pipeline E2E, RLCF flow |
| **merlt/tests/pipeline/** | 4 | pytest | ⚠️ Basic | WebSocket, batch orchestration |
| **merlt/tests/tools/** | 5 | pytest | ⚠️ Adequate | Search edge cases |
| **merlt/tests/benchmark/** | 3 | pytest | ⚠️ Basic | Performance baselines |
| **merlt/tests/ner/** | 4 | pytest | ⚠️ Basic | Edge cases |
| **merlt/tests/unit/** | **0** | — | ❌ **Empty** | Directory exists but no tests |
| **merlt/tests/e2e/** | **0** | — | ❌ **Empty** | Directory exists but no tests |
| **platform/backend/tests/** | 12 | Jest | ✅ Basic | Transaction, validation, rate limiting |
| **platform/frontend/src/test/** | 9 | Vitest | ⚠️ Minimal | Forms, routing, state, a11y, E2E |
| **merlt-frontend/__tests__/** | 5 | Vitest | ⚠️ Minimal | Plugin system, EventBus, slots |

### What's Well-Tested

1. **RLCF subsystem** (30 tests): Policy gradient, PPO trainer, replay buffer, traversal training, persistence
2. **API auth middleware** (14 tests): Auth enforcement, consent filtering, feedback endpoints
3. **Expert implementations** (9 tests): Literal, systemic, principles, precedent, synthesizer
4. **Storage layer** (10 tests): PostgreSQL, FalkorDB, Qdrant, hybrid retrieval
5. **Platform auth** (12 tests): JWT, password, auth flows, consent, privacy, profile

### What's NOT Tested (Critical Gaps)

1. **E2E user flows** — Zero Playwright tests (config exists, directory empty)
2. **Full pipeline orchestration** — query → route → retrieve → analyze → synthesize (no E2E test)
3. **Plugin system** — EventBus, PluginSlot, slot rendering, cross-component events
4. **Frontend forms/validation** — No form validation tests
5. **Frontend routing/navigation** — No React Router tests
6. **State management** — No Zustand store tests
7. **Error boundaries** — No React ErrorBoundary tests
8. **API contract validation** — No schema/shape enforcement tests
9. **Docker deployment** — No container build/smoke tests
10. **Cross-store consistency** — No bridge table integrity verification tests

---

## Test Coverage Plan

### P0 (Critical) — Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| JWT WebSocket authentication (verified decode) | Unit | R-002 | 2 | Dev | Valid token, invalid/expired/tampered token rejection |
| Full expert query pipeline E2E | E2E | R-015 | 2 | Dev | Happy path: query → trace. Partial failure: 1 expert down → degraded response |
| Bridge table cross-store consistency | Integration | R-004 | 2 | Dev | Insert validates all 3 stores in sync; partial failure → clean rollback |
| Auth enforcement sweep (all sensitive endpoints) | API | R-001 | 2 | Dev | Verify every route returns 401 without valid key |
| PII masking correctness (all 4 patterns) | Unit | R-006 | 2 | Dev | CF, email, phone, dates — all masked correctly |
| Circuit breaker state transitions | Unit | — | 2 | Dev | closed→open→half-open, threshold, recovery |

**Total P0**: 12 tests, 24 hours

### P1 (High) — Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| E2E: Registration → Login → Search → View Article | E2E (Playwright) | R-015 | 3 | Dev | Happy path, validation errors, session persistence |
| E2E: MERL-T Analysis → Trace → Feedback | E2E (Playwright) | R-015 | 2 | Dev | Query, view trace, submit inline feedback |
| API endpoint contracts (all 30+ routes) | API | R-003 | 5 | Dev | Status codes, response shapes, error format |
| RLCF feedback collection (F1-F8) | Integration | R-016 | 4 | Dev | Each feedback type stored + authority updated |
| Training pipeline: feedback → aggregate → train | Integration | R-016 | 2 | Dev | Synthetic feedback → weight update cycle |
| Alembic migrations up/down (all 3) | Integration | — | 2 | Dev | Apply and rollback cleanly |
| Plugin system: registration + events | Component | R-019 | 3 | Dev | PluginSlot renders, EventBus delivers, slot context passed |
| Dataset export anonymization | Integration | R-006 | 2 | Dev | GDPR salt, PII removal, CSV/JSON format validation |
| API key lifecycle | API | — | 2 | Dev | Bootstrap → create → use → revoke |
| Docker build + smoke test | OPS | R-018 | 1 | Dev | `docker compose build && curl /health` |

**Total P1**: 28 tests, 28 hours (1 hour/test avg)

### P2 (Medium) — Run nightly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| Expert-specific analysis quality (4 experts) | Unit | — | 4 | Dev | Query rewriting, source selection per expert |
| Gating network aggregation (4 methods) | Unit | — | 4 | Dev | WeightedAvg, Bayesian, MajorityVote, MaxConf |
| Confidence calibration α-blending | Unit | — | 2 | Dev | Edge: zero disagreement, max intensity |
| Synthesizer profile-aware formatting | Unit | — | 2 | Dev | Progressive disclosure, citation format |
| Authority score computation | Unit | — | 2 | Dev | Track record, weight decay, threshold |
| NER pipeline entity extraction | Unit | — | 3 | Dev | URN patterns, entity types, edge cases |
| Graph search traversal | Integration | — | 2 | Dev | Neighbor lookup, path finding |
| Feedback aggregation (Shannon entropy) | Unit | — | 2 | Dev | Authority-weighted avg, disagreement detection |
| Policy evolution queries | API | — | 2 | Dev | Time-series, expert-evolution |
| Devil's advocate check + feedback | API | — | 2 | Dev | Trigger conditions, effectiveness |
| Quarantine service | API | — | 2 | Dev | Flag/approve/reject, filter from training |
| Frontend: Dashboard tab rendering | Component | — | 5 | Dev | Overview, Experts, RLCF, Architecture, Pipeline |
| Frontend: form validation | Component | — | 3 | Dev | Login, Register, Search forms |

**Total P2**: 35 tests, 17.5 hours (0.5 hours/test avg)

### P3 (Low) — Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
|-------------|-----------|------------|-------|-------|
| Vector search latency benchmark | Perf | 2 | Dev | p50/p95/p99 for query set |
| Full pipeline latency benchmark | Perf | 2 | Dev | Cold start vs warm cache |
| Load test: concurrent API queries | Perf | 2 | Dev | 10/50/100 concurrent users via k6 |
| Frontend visual regression | E2E | 2 | Dev | Dashboard screenshot comparison |
| Accessibility audit (axe-core) | Component | 2 | Dev | WCAG 2.1 AA compliance |
| OWASP security scan | Security | 2 | Dev | Injection, XSS, auth bypass |

**Total P3**: 12 tests, 3 hours (0.25 hours/test avg)

---

## Execution Order

### Smoke Tests (<2 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Backend imports: `python -c "from merlt.app import app"` (10s)
- [ ] Frontend compiles: `tsc --noEmit` (30s)
- [ ] Database health: `GET /health` returns `{"status": "healthy"}` (5s)
- [ ] Auth enforcement: unauthenticated request → 401 (5s)

**Total**: 4 scenarios, ~1 min

### P0 Tests (<10 min)

**Purpose**: Critical path validation

- [ ] JWT: valid token accepted, invalid rejected, expired rejected (Unit)
- [ ] Expert pipeline: query returns valid trace with 4 expert responses (Integration)
- [ ] Expert pipeline: one expert fails, other 3 succeed with degraded confidence (Integration)
- [ ] Bridge table: insert → verify Qdrant + FalkorDB + PostgreSQL consistent (Integration)
- [ ] PII masking: CF, email, phone, dates all masked (Unit)
- [ ] Circuit breaker: threshold → open → cooldown → half-open → close (Unit)
- [ ] Auth sweep: all sensitive endpoints return 401 without key (API)

**Total**: 12 scenarios, ~8 min

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] E2E: User registers → logs in → searches → views article (Playwright)
- [ ] E2E: User queries MERL-T → views trace → submits feedback (Playwright)
- [ ] API: All 30+ endpoints return correct status codes and response shapes (API)
- [ ] RLCF: F1-F8 feedback types persisted and authority updated (Integration)
- [ ] Training: synthetic feedback → aggregation → weight update (Integration)
- [ ] Plugin: PluginSlot renders component, EventBus delivers events (Component)
- [ ] Alembic: all migrations up and down cleanly (Integration)
- [ ] API keys: bootstrap → create → authenticate → revoke (API)
- [ ] Docker: `docker compose build` succeeds, `/health` responds (OPS)

**Total**: 28 scenarios, ~25 min

### P2/P3 Tests (<60 min)

**Purpose**: Full regression + benchmarks

- [ ] Expert algorithms: all 4 experts produce valid analysis (Unit ×4)
- [ ] Gating: all 4 aggregation methods (Unit ×4)
- [ ] NER: entity extraction from legal text (Unit ×3)
- [ ] Dashboard tabs: render without errors (Component ×5)
- [ ] Form validation: Login, Register, Search (Component ×3)
- [ ] Performance: pipeline latency, vector search latency (Perf ×4)
- [ ] Accessibility: axe-core scan (Component ×2)

**Total**: 47 scenarios, ~40 min

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
|----------|-------|-----------|-------------|-------|
| P0 | 12 | 2.0 | 24 | Complex setup, security, cross-store |
| P1 | 28 | 1.0 | 28 | E2E flows, API contracts, plugin system |
| P2 | 35 | 0.5 | 17.5 | Unit algorithms, component rendering |
| P3 | 12 | 0.25 | 3 | Benchmarks, accessibility, exploratory |
| **Total** | **87** | **—** | **72.5** | **~9 days** |

### Prerequisites

**Test Data:**
- `UserFactory` — faker-based user with authority scores, profile type
- `FeedbackFactory` — all 8 feedback types with configurable ratings
- `TraceFactory` — QATrace with expert responses and sources
- `ApiKeyFactory` — keys with various roles and tiers
- `ArticleFactory` — URN, rubrica, text, relations

**Tooling (Already Installed):**
- pytest + pytest-asyncio + pytest-cov (merlt backend ✅)
- Vitest + React Testing Library (platform frontend ✅, merlt frontend ✅)
- Playwright (platform frontend — config exists ✅, needs test files)
- bandit + ruff + black (merlt CI ✅)
- Jest + ts-jest (platform backend ✅)

**Tooling (To Install):**
- k6 — performance load testing
- @axe-core/playwright — accessibility testing

**Environment:**
- Docker Compose with all 4 services (PostgreSQL, FalkorDB, Qdrant, Redis)
- E5-large model pre-cached (2.3GB RAM)
- `MERLT_ENV=test` for test isolation
- OpenRouter API key for integration tests (mockable for unit tests)

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Security modules** (auth, rate_limit, pii_masking): ≥80%
- **Core pipeline** (experts, synthesizer, gating): ≥70%
- **API endpoints**: ≥90% (every endpoint has at least one test)
- **Frontend components**: ≥50% (critical paths)
- **E2E flows**: ≥3 critical user journeys covered

### Non-Negotiable Requirements

- [ ] All P0 tests pass (100%)
- [ ] No high-risk (≥6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] `tsc --noEmit` passes with zero errors
- [ ] `ruff check` passes in CI
- [ ] `bandit` security scan passes
- [ ] At least 1 E2E Playwright test validates full user flow

### Gate Decision

**Current Assessment: CONCERNS (Improved)**

Rationale: Significant progress since system-level review — auth wired, CI pipeline operational, frontend test frameworks installed, hardcoded issues fixed. However, 4 high-priority risks remain (R-002 JWT, R-004 consistency, R-015 zero E2E, R-017 in-memory persistence). The most critical gap is **zero E2E test files** despite Playwright being configured. Writing P0 Playwright tests should be the immediate priority.

---

## Mitigation Plans

### R-002: JWT Token Forgery in ws_router (Score: 6)

**Mitigation Strategy:** Replace bare `jwt.decode()` with verified decode using PyJWT with HS256 algorithm whitelist. Reject tokens without valid signature. Add rate limiting to WebSocket connection endpoint.
**Owner:** Dev
**Timeline:** Week 1
**Status:** Open
**Verification:** Unit tests: `test_ws_rejects_unsigned_token`, `test_ws_rejects_expired_token`, `test_ws_accepts_valid_token`

### R-004: Cross-Store Data Inconsistency (Score: 6)

**Mitigation Strategy:** Add transactional wrapper for ingestion pipeline (Qdrant insert + FalkorDB insert + bridge table insert as atomic unit). On partial failure, rollback all stores. Add `GET /pipeline/consistency-check` endpoint for admin verification.
**Owner:** Dev
**Timeline:** Week 2
**Status:** Open
**Verification:** Integration test: inject failure after Qdrant insert → verify FalkorDB and bridge table not modified

### R-015: Zero E2E Test Coverage (Score: 6)

**Mitigation Strategy:** Write 5 Playwright E2E tests covering critical user journeys:
1. Registration → email verification → login
2. Search by keyword → view article → citation highlighting
3. MERL-T query → trace viewer → source navigation
4. Submit feedback (inline + detailed) → verify persistence
5. Admin: pipeline dashboard → trigger ingest → view status

**Owner:** Dev
**Timeline:** Week 1-2
**Status:** Open — Playwright config exists, needs test files
**Verification:** `npx playwright test` passes in CI with ≥5 test files

### R-017: In-Memory Persistence (Score: 6)

**Mitigation Strategy:**
- `weights/store.py`: Persist to PostgreSQL on save, load from DB on startup
- `training_scheduler` buffer: Persist to Redis with TTL
- `regression_router._runs`: Replace with `collections.OrderedDict` maxlen=100

**Owner:** Dev
**Timeline:** Week 2-3
**Status:** Open (warning logs added, persistence not implemented)
**Verification:** Unit tests: restart simulation → verify data survives restart

---

## Implementation Roadmap

### Week 1: Foundation + P0

| Task | Effort | Priority |
|------|--------|----------|
| Fix JWT verification in ws_router.py | 2h | P0 |
| Write P0 unit tests (JWT, PII, circuit breaker) | 6h | P0 |
| Write first 2 Playwright E2E tests | 8h | P0 |
| Create test data factories (`tests/factories/`) | 4h | P1 |
| **Week 1 total** | **20h** | |

### Week 2: P1 Coverage

| Task | Effort | Priority |
|------|--------|----------|
| Write 3 more Playwright E2E tests | 6h | P1 |
| Write API contract tests (30+ endpoints) | 5h | P1 |
| Write RLCF integration tests (F1-F8) | 4h | P1 |
| Write plugin system component tests | 3h | P1 |
| Add cross-store consistency test | 4h | P0 |
| Add bridge table transactional wrapper | 4h | R-004 fix |
| **Week 2 total** | **26h** | |

### Week 3: P2 + Hardening

| Task | Effort | Priority |
|------|--------|----------|
| Write expert algorithm unit tests | 4h | P2 |
| Write gating/synthesizer unit tests | 3h | P2 |
| Write frontend component tests | 5h | P2 |
| Add in-memory persistence fixes | 4h | R-017 fix |
| Remaining bare `except:` cleanup | 3h | R-003 fix |
| Docker build verification in CI | 2h | P1 |
| **Week 3 total** | **21h** | |

### Week 4: Polish + Gate Check

| Task | Effort | Priority |
|------|--------|----------|
| Performance benchmarks (k6) | 3h | P3 |
| Accessibility audit (axe-core) | 2h | P3 |
| Gate check: run full suite, triage failures | 3h | Gate |
| Documentation update | 2h | Gate |
| **Week 4 total** | **10h** | |

**Grand Total: ~77 hours (~10 working days)**

---

## Assumptions and Dependencies

### Assumptions

1. Docker Compose services available for all integration tests
2. E5-large model pre-cached locally (not downloaded during tests)
3. OpenRouter API key available for integration tests (mockable for unit)
4. Single developer performing test implementation
5. Existing 142 test files are functional and can be extended

### Dependencies

1. **Playwright test files** — Must be created in `visualex-platform/frontend/e2e/`
2. **Test data factories** — Must be created in `merlt/tests/factories/`
3. **Redis** — Required for rate limiting and buffer persistence tests
4. **k6** — Required for performance load testing (Week 4)
5. **@axe-core/playwright** — Required for accessibility testing (Week 4)

### Risks to Plan

- **Risk**: OpenRouter API rate limits slow integration tests
  - **Impact**: LLM-dependent tests become flaky
  - **Contingency**: Mock OpenRouter for P0/P1, real API for P3 only

- **Risk**: E5-large model loading adds 30s+ to test startup
  - **Impact**: Slow test suite discourages frequent runs
  - **Contingency**: Session-scoped fixture, `--skip-embedding` marker

- **Risk**: Cross-store transactional wrapper is architecturally complex
  - **Impact**: R-004 fix takes longer than estimated
  - **Contingency**: Implement consistency-check endpoint first (detect-only), defer transactional wrapper

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests from acceptance criteria
- Run `*automate` to expand test automation coverage
- Run `*testarch-ci` to enhance CI pipeline with test gates
- Run `*testarch-nfr` before release for non-functional requirements validation
- Run `*testarch-trace` to generate requirements-to-tests traceability matrix

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: _________________ Date: _________
- [ ] Tech Lead: _________________ Date: _________
- [ ] QA Lead: _________________ Date: _________

**Comments:**

---

## Appendix

### Knowledge Base References

- `risk-governance.md` — Risk classification framework (6 categories, scoring matrix)
- `probability-impact.md` — Risk scoring methodology (probability × impact)
- `test-levels-framework.md` — Unit/Integration/E2E decision matrix
- `test-priorities-matrix.md` — P0-P3 prioritization criteria

### Related Documents

- PRD: `_bmad-output/planning-artifacts/prd.md`
- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Sprint Status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- System-Level Test Design: `_bmad-output/test-design-system.md` (baseline)
- Traceability Matrix: `_bmad-output/traceability-matrix.md`

---

**Generated by**: BMad TEA Agent — Test Architect Module
**Workflow**: `_bmad/bmm/testarch/test-design` (Epic-Level Mode, All Epics)
**Version**: 4.0 (BMad v6)
