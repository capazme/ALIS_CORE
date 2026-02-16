# System-Level Test Design: ALIS_CORE

**Date:** 2026-02-16
**Author:** Gpuzio
**Status:** Draft

---

## Executive Summary

**Scope:** System-level testability review across all 10 epics of the ALIS_CORE platform

**Architecture:** 3-layer service (Express :3001 / Quart :5000 / FastAPI :8000), 4 databases (PostgreSQL, FalkorDB, Qdrant, Redis), 4-Expert ML pipeline, RLCF feedback loop, React dashboard frontend.

**Risk Summary:**

- Total risks identified: 14
- High-priority risks (≥6): 6
- Critical categories: SEC (3), TECH (1), DATA (1), PERF (1)

**Coverage Summary:**

- P0 scenarios: 18 tests (36 hours)
- P1 scenarios: 35 tests (35 hours)
- P2 scenarios: 45 tests (22.5 hours)
- P3 scenarios: 15 tests (3.75 hours)
- **Total effort**: 113 tests, ~97 hours (~12 days)

**Current State:**
- Backend (merlt): pytest infrastructure exists with 100+ test files, real DB isolation (zero-mock policy), but import issues may prevent collection
- Frontend (visualex-merlt): **Zero test infrastructure** — no Vitest, Jest, or Playwright configured
- CI/CD: GitHub Actions covers visualex-platform only, **not merlt framework**

---

## Testability Assessment

### Controllability: CONCERNS

**Strengths:**
- FastAPI Depends() enables straightforward dependency injection for mocking
- Docker Compose (`merlt/docker-compose.dev.yml`) provisions all 4 databases reproducibly
- Existing test fixtures use real databases with transaction rollback (PostgreSQL) and separate test collections (FalkorDB: `merl_t_test`, Qdrant: `merl_t_test_chunks`)
- `SemanticSearchTool.clear_cache()` provides cache reset per pipeline run
- `BaseTool.clone()` creates isolated tool instances per expert

**Concerns:**
- External LLM service (OpenRouter) is not abstracted behind a mockable interface in all call sites — `OpenRouterService` is used directly
- No factory pattern for generating test entities (users, feedback, traces) — each test builds data manually
- FalkorDB graph seeding requires Cypher queries; no seed fixture exists for complex graph topologies
- API key bootstrap endpoint (`POST /api-keys/bootstrap`) only works on empty table — no reset mechanism for test isolation
- RLCF training pipeline has side effects (writes checkpoints, updates policy) — no dry-run mode

**Recommendations:**
1. Create `tests/factories/` with faker-based factories for QATrace, QAFeedback, ApiKey, IngestionSchedule
2. Add `--dry-run` flag to TrainingScheduler for testing without side effects
3. Abstract LLM calls behind `AIServiceProtocol` interface where not already done

### Observability: CONCERNS

**Strengths:**
- `structlog` configured at WARNING level, stderr-only — stdout clean for JSON trace output
- QATrace model captures full reasoning chain (expert responses, sources, confidence)
- AuditService provides SHA-256 hash chain for tamper-evident audit trail
- Circuit breaker state callbacks provide real-time health signals
- `get_last_usage()` on OpenRouterService tracks token consumption

**Concerns:**
- No Prometheus/metrics endpoint — cannot measure SLOs (latency, throughput, error rate) in production
- No structured health check endpoint (only implicit via FastAPI `/docs`)
- Frontend has no error boundary reporting — React errors silently crash components
- No distributed tracing (OpenTelemetry) across the 3-service architecture
- Audit log query has no pagination — will degrade with scale

**Recommendations:**
1. Add `/health` endpoint with dependency checks (PostgreSQL, FalkorDB, Qdrant, Redis)
2. Add Prometheus metrics middleware for request latency, error rate, active connections
3. Implement React ErrorBoundary with centralized error logging

### Reliability: CONCERNS

**Strengths:**
- Circuit breaker pattern implemented with thread-safe registry and configurable thresholds
- Test database isolation via transaction rollback (PostgreSQL) and separate collections
- `pytest-asyncio` auto mode handles async test execution
- Redis rate limiting has graceful degradation (allow if Redis unavailable)

**Concerns:**
- **40+ bare `except:` handlers** across backend code — silently swallow errors, making failures non-deterministic
- **No CI/CD pipeline** for merlt — regressions are manually detected
- Frontend test suite is completely absent — 50+ components untested
- `regression_router._runs` dict is unbounded — OOM risk in long-running instances
- `training_scheduler.py` uses in-memory buffer with no persistence — data lost on restart
- APScheduler for ingestion runs in-process — no supervisor/restart mechanism

**Recommendations:**
1. Replace all bare `except:` with specific exception types + logging
2. Add CI pipeline: `pytest` on PR, `tsc --noEmit` for frontend
3. Add TTL-based eviction to `_runs` dict (max 100 entries, FIFO)
4. Persist training buffer to Redis or PostgreSQL for crash recovery

---

## Architecturally Significant Requirements (ASRs)

| ASR ID | NFR Category | Requirement | Architecture Impact | Risk Score |
|--------|-------------|-------------|-------------------|------------|
| ASR-1 | Security | API authentication required for all external endpoints (FR45) | Auth middleware, API key model, rate limiting | 9 |
| ASR-2 | Performance | Sub-second vector retrieval for semantic search | Qdrant index tuning, E5-large model caching, bridge table indexing | 4 |
| ASR-3 | Reliability | Circuit breaker prevents cascade failures across experts | Per-expert circuit breaker, state callbacks, graceful degradation | 4 |
| ASR-4 | Data Integrity | Bridge table consistency between Qdrant, FalkorDB, PostgreSQL | Transactional ingestion, consistency checks, reconciliation | 6 |
| ASR-5 | Compliance | GDPR: PII anonymization, consent-aware filtering, data export/erasure | PIIMaskingService, consent model, audit trail | 6 |
| ASR-6 | Performance | LLM pipeline must complete within 60s timeout | Async orchestration, per-expert timeout, cached retrieval | 4 |
| ASR-7 | Security | Audit trail with 7-year retention, tamper-evident | SHA-256 hash chain, immutable logs | 4 |
| ASR-8 | Reliability | Training pipeline must handle partial failures gracefully | Training buffer, checkpoint versioning, error callbacks | 3 |

---

## Test Levels Strategy

**Architecture Profile:** API-heavy backend with ML pipeline, monitoring dashboard frontend.

### Recommended Split: 45 / 40 / 15

| Level | Share | Count | Rationale |
|-------|-------|-------|-----------|
| **Unit** | 45% | ~51 | Expert algorithms, confidence calibration, authority scoring, PII masking, audit hashing, risk scoring, query rewriting — all pure or near-pure logic |
| **Integration/API** | 40% | ~45 | Endpoint contracts (26+ routes), database operations, cross-store consistency, auth middleware chain, rate limiting, training triggers |
| **E2E** | 15% | ~17 | Full query pipeline (query→route→retrieve→analyze→synthesize), feedback loop, admin dashboard flows |

### Justification

- **45% Unit**: The MERL-T pipeline has significant algorithmic complexity: 4 expert analysis methods, gating network with 4 aggregation strategies, confidence calibration with α-blending, authority scoring with track record. These are testable as pure functions.
- **40% Integration**: The system has 26+ API endpoints, 4 databases, and complex data flows (ingestion, retrieval, feedback aggregation). Contract testing and cross-store consistency require integration-level validation.
- **15% E2E**: The frontend is a monitoring dashboard (not user-facing critical). E2E tests cover the full query pipeline and key admin workflows but most logic is backend API-testable.

### Technology Stack

| Level | Backend (merlt) | Frontend (visualex-merlt) |
|-------|----------------|--------------------------|
| Unit | pytest + pytest-asyncio | Vitest (to be installed) |
| Integration | pytest + httpx (TestClient) + real DB | Vitest + MSW (API mocking) |
| E2E | pytest integration markers | Playwright (to be installed) |
| Performance | k6 or locust | Lighthouse CI |
| Security | bandit + safety | npm audit |

---

## NFR Testing Approach

### Security (SEC)

**Requirements:** API authentication (FR45), RBAC (admin/user/guest), PII anonymization, GDPR compliance, audit trail integrity.

**Approach:**
- **Unit tests**: `hash_api_key()` correctness, `PIIMaskingService` pattern matching (CF, email, phone, dates), `AuditService` hash chain verification
- **API tests**: Auth middleware enforcement (401 without key, 403 insufficient role, 200 valid key), rate limiting (429 when exceeded, headers correct), bootstrap endpoint (only works on empty table)
- **Integration tests**: Consent-aware data filtering, PII masking in feedback pipeline, audit trail immutability
- **Tools**: bandit (static analysis), safety (dependency vulnerabilities), custom script for hardcoded credentials scan

**Critical Gaps to Address:**
- JWT parsing in `ws_router.py` lacks signature verification — requires fix + test
- PII masking only wired in feedback endpoints — trace/export endpoints need audit
- Hardcoded dev credentials found in codebase — need parameterization via env vars

### Performance (PERF)

**Requirements:** Sub-second vector search, 60s pipeline timeout, rate limiting per tier.

**Approach:**
- **Unit tests**: Timeout enforcement in orchestrator config, rate limit quota calculations
- **Benchmark tests**: Vector search latency (p50/p95/p99), full pipeline latency, E5-large embedding throughput
- **Load tests (P3)**: k6 scripts for concurrent query pipeline (10/50/100 users), rate limiting under load
- **Monitoring**: Prometheus metrics for response time distribution, active workers, queue depth

**Critical Gaps to Address:**
- No performance baseline established — first run establishes benchmarks
- `regression_router._runs` unbounded dict — memory leak under sustained use

### Reliability (REL)

**Requirements:** Circuit breaker activation/recovery, graceful degradation, training pipeline crash recovery.

**Approach:**
- **Unit tests**: Circuit breaker state transitions (closed→open→half-open), threshold calculations, callback invocation
- **Integration tests**: Expert failure isolation (one expert fails, others continue), rate limiter Redis fallback (graceful degradation when Redis unavailable)
- **Chaos tests (P3)**: Kill individual services during query pipeline, verify degraded but functional response
- **Tools**: pytest fixtures for service failure simulation

**Critical Gaps to Address:**
- 40+ bare except handlers mask real failures — systematic replacement needed
- Training buffer not persisted — restart loses accumulated feedback
- APScheduler ingestion has no supervisor — silent death undetected

### Maintainability (MAINT)

**Requirements:** Code coverage targets, code quality gates, observability.

**Approach:**
- **Coverage**: pytest-cov target ≥70% for core modules (experts, pipeline, rlcf), ≥80% for security modules (auth, rate_limit, pii)
- **Static Analysis**: ruff + mypy (backend), tsc --noEmit + ESLint (frontend)
- **Code Quality Gates**: No `any` types in TypeScript, no bare `except:` in Python, no `console.log` in production frontend
- **Observability**: Health check endpoint validates all 4 database connections

**Critical Gaps to Address:**
- Frontend has no ESLint config — 18 files with console.log, 4 files with `any` type
- Backend has 40+ TODOs marking incomplete implementations
- No `mypy` CI gate — type errors not caught

---

## Test Environment Requirements

### Local Development (Primary)

```
Docker Compose (merlt/docker-compose.dev.yml):
├── PostgreSQL  :5433  (test DB: merlt_test)
├── FalkorDB    :6380  (test graph: merl_t_test)
├── Qdrant      :6333  (test collection: merl_t_test_chunks)
├── Redis       :6379  (test namespace: rate_limit:test_*)
└── FastAPI     :8000  (test mode: MERLT_ENV=test)

Frontend:
├── Vite dev    :5173  (HMR for component testing)
└── Vitest      (unit + component tests)
```

### CI Environment (To Be Created)

```
GitHub Actions:
├── Backend job:
│   ├── Services: PostgreSQL, Redis (GitHub Actions services)
│   ├── FalkorDB + Qdrant: Docker containers in workflow
│   ├── Run: pytest -m "not integration" (unit tests)
│   └── Run: pytest -m integration (with services)
├── Frontend job:
│   ├── Run: tsc --noEmit
│   ├── Run: vitest run --coverage
│   └── Run: eslint src/
└── Security job:
    ├── Run: bandit -r merlt/merlt/
    ├── Run: safety check
    └── Run: npm audit (frontend)
```

### Performance Testing (On-Demand)

```
Dedicated environment with:
├── Same Docker Compose stack
├── E5-large model loaded (2.3GB RAM)
├── k6 for API load testing
└── Isolated from development (no shared DB)
```

---

## Testability Concerns

### Blockers (Must Fix Before Testing)

| ID | Concern | Impact | Recommendation |
|----|---------|--------|----------------|
| TC-1 | **40+ bare `except:` handlers** | Tests pass when they should fail — non-deterministic behavior | Replace with specific exceptions + `structlog.error()` |
| TC-2 | **No frontend test framework** | 50+ React components completely untestable | Install Vitest + React Testing Library |
| TC-3 | **JWT parsing without verification** | Security tests for WebSocket auth are meaningless | Fix `ws_router.py` to verify JWT signatures |

### Concerns (Should Fix for Quality)

| ID | Concern | Impact | Recommendation |
|----|---------|--------|----------------|
| TC-4 | No test data factories | Each test manually constructs data — slow, inconsistent | Create `tests/factories/` with faker-based builders |
| TC-5 | No CI pipeline for merlt | Regressions detected manually, post-merge | Add GitHub Actions workflow for merlt backend + frontend |
| TC-6 | Unbounded `_runs` dict | Memory leak makes long-running performance tests unreliable | Add maxlen + FIFO eviction |
| TC-7 | OpenRouter not mockable in all paths | Integration tests hit real LLM API — slow, costly, flaky | Ensure all LLM calls go through `AIServiceProtocol` |
| TC-8 | Frontend console.log pollution | 18 files with debug logging — noisy test output | ESLint rule: no-console in production code |

### Information (Monitor)

| ID | Concern | Impact | Recommendation |
|----|---------|--------|----------------|
| TC-9 | 40+ TODO markers in codebase | Incomplete implementations may affect test expectations | Triage TODOs: fix, defer, or remove |
| TC-10 | Single-process APScheduler | Ingestion scheduler can't be tested for restart recovery | Document limitation, test happy path only |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Prob | Impact | Score | Mitigation | Owner | Timeline |
|---------|----------|-------------|------|--------|-------|------------|-------|----------|
| R-001 | SEC | Unauthenticated access to non-query API endpoints (26+ routes open) — only `POST /experts/query` has auth middleware wired | 3 | 3 | **9** | Wire `verify_api_key` Depends into all sensitive endpoints (CRUD, admin, export, training) | Dev | Sprint 0 |
| R-002 | SEC | JWT token forgery via unverified parsing in `ws_router.py` — attacker can impersonate any user | 2 | 3 | **6** | Add `PyJWT` signature verification with HS256/RS256, reject invalid tokens | Dev | Sprint 0 |
| R-003 | TECH | Silent error propagation via 40+ bare `except:` handlers — errors swallowed, system returns incorrect results silently | 3 | 2 | **6** | Systematic replacement with typed exceptions + error logging | Dev | Sprint 0-1 |
| R-004 | DATA | Cross-store data inconsistency — bridge table can desync from Qdrant/FalkorDB during partial ingestion failures | 2 | 3 | **6** | Add transactional ingestion with rollback, consistency reconciliation check endpoint | Dev | Sprint 1 |
| R-005 | PERF | Memory leak via unbounded `_runs` dict in `regression_router.py` — OOM crash under sustained use | 2 | 3 | **6** | Add `collections.OrderedDict` with maxlen=100, FIFO eviction | Dev | Sprint 0 |
| R-006 | SEC | GDPR compliance gap — PII masking only wired in feedback endpoints, traces and exports may leak PII | 2 | 3 | **6** | Audit all data egress points, wire PIIMaskingService into trace viewer and dataset export | Dev | Sprint 1 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Prob | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|------|--------|-------|------------|-------|
| R-007 | PERF | LLM calls may hang without per-request timeout — blocks FastAPI worker thread | 2 | 2 | 4 | Enforce `asyncio.wait_for()` with 30s timeout per expert call | Dev |
| R-008 | OPS | No CI/CD pipeline for merlt — regressions undetected until manual testing | 2 | 2 | 4 | Add GitHub Actions workflow: pytest + tsc + eslint on PR | DevOps |
| R-009 | TECH | Frontend has zero test coverage — 50+ components with no regression detection | 2 | 2 | 4 | Install Vitest + RTL, write P1 tests for critical components | Dev |
| R-010 | OPS | No health check endpoint — unhealthy services undetected | 2 | 2 | 4 | Add `GET /health` with DB dependency checks | Dev |
| R-011 | SEC | Hardcoded dev credentials in codebase — risk if deployed without override | 1 | 3 | 3 | Replace with env vars, add `.env.example`, scan in CI | Dev |
| R-012 | DATA | Alembic migration chain untested — rollback may fail | 1 | 3 | 3 | Add migration up/down test in CI | Dev |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Prob | Impact | Score | Action |
|---------|----------|-------------|------|--------|-------|--------|
| R-013 | BUS | Expert router regex fallback may misclassify queries when LLM unavailable | 1 | 2 | 2 | Monitor classification accuracy metrics |
| R-014 | OPS | Docker Compose only deployment — no production orchestration | 1 | 2 | 2 | Document production deployment requirements |

---

## Test Coverage Plan

### P0 (Critical) — Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| API authentication enforcement | API | R-001 | 4 | Dev | 401/403/200/expired key |
| Rate limiting enforcement | API + Unit | R-001 | 3 | Dev | Quota exceeded, headers, Redis fallback |
| Expert query pipeline (query→route→retrieve→analyze→synthesize) | Integration | R-003 | 3 | Dev | Happy path, partial failure, timeout |
| PII masking correctness | Unit | R-006 | 3 | Dev | CF, email, phone, dates patterns |
| Bridge table consistency check | Integration | R-004 | 2 | Dev | Insert + verify cross-store, partial failure rollback |
| Circuit breaker state transitions | Unit | R-003 | 3 | Dev | Closed→open→half-open, threshold, recovery |

**Total P0**: 18 tests, 36 hours

### P1 (High) — Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| All API endpoint contracts (26+ routes) | API | R-008 | 10 | Dev | Status codes, response shapes, error handling |
| RLCF feedback collection (F1-F8) | Integration | — | 5 | Dev | Each feedback type stored + authority updated |
| Authority score computation | Unit | — | 3 | Dev | Track record, weight decay, threshold |
| Training scheduler triggers | Unit | R-007 | 3 | Dev | Buffer threshold, idle timeout, checkpoint |
| Alembic migrations up/down | Integration | R-012 | 2 | Dev | All 3 migrations, rollback verification |
| Frontend TypeScript compilation | Build | R-009 | 1 | Dev | `tsc --noEmit` passes |
| Audit trail hash chain integrity | Unit | — | 2 | Dev | Sequential hashing, tamper detection |
| Dataset export anonymization | Integration | R-006 | 2 | Dev | GDPR salt, PII removal, format validation |
| API key CRUD operations | API | R-001 | 3 | Dev | Create/list/revoke/bootstrap |
| Ingestion schedule CRUD | API | — | 2 | Dev | Create/list/delete |
| Consent-aware data filtering | Integration | R-006 | 2 | Dev | Only consented data returned |

**Total P1**: 35 tests, 35 hours

### P2 (Medium) — Run nightly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|-----------|-----------|------------|-------|-------|
| Expert-specific analysis quality | Unit | — | 8 | Dev | Each expert's query rewriting, source selection |
| Gating network aggregation methods | Unit | — | 4 | Dev | WeightedAvg, BayesianFusion, MajorityVote, MaxConf |
| Confidence calibration α-blending | Unit | — | 3 | Dev | Edge cases: zero disagreement, max intensity |
| Synthesizer profile-aware formatting | Unit | — | 3 | Dev | Progressive disclosure, citation formatting |
| Policy evolution queries | API | — | 3 | Dev | Time-series, expert-evolution, aggregation-history |
| Devil's advocate check + feedback | API | — | 3 | Dev | Trigger conditions, effectiveness tracking |
| Reproducibility scoring | Unit | — | 3 | Dev | Diff, Jaccard, composite score |
| NER pipeline entity extraction | Unit | — | 4 | Dev | URN patterns, entity types, edge cases |
| Graph search traversal | Integration | — | 3 | Dev | Neighbor lookup, path finding, relation weights |
| Feedback aggregation | Unit | — | 3 | Dev | Authority-weighted avg, Shannon entropy |
| Quarantine service | API | — | 3 | Dev | Flag/approve/reject, filter from training |
| Frontend component rendering | Component | R-009 | 5 | Dev | Dashboard tabs, key interactive components |

**Total P2**: 45 tests, 22.5 hours

### P3 (Low) — Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
|-------------|-----------|------------|-------|-------|
| Vector search latency benchmark | Perf | 2 | Dev | p50/p95/p99 for 1K/5K queries |
| Full pipeline latency benchmark | Perf | 2 | Dev | Cold start, warm cache, concurrent |
| Load test: concurrent API queries | Perf | 3 | Dev | 10/50/100 concurrent users |
| OWASP top 10 security scan | Security | 2 | Dev | Injection, XSS, CSRF, auth bypass |
| Frontend visual regression | E2E | 2 | Dev | Dashboard screenshot comparison |
| Training pipeline end-to-end | Integration | 2 | Dev | Full RLCF loop: feedback→aggregate→train→deploy |
| E5-large embedding throughput | Perf | 2 | Dev | Batch embedding speed, memory usage |

**Total P3**: 15 tests, 3.75 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Backend imports successfully (`python -c "from merlt.app import app"`) (10s)
- [ ] Frontend compiles (`tsc --noEmit`) (30s)
- [ ] Database connections healthy (`GET /health`) (5s)
- [ ] Auth middleware rejects unauthenticated request (5s)

**Total**: 4 scenarios, ~1 min

### P0 Tests (<10 min)

**Purpose**: Critical path validation

- [ ] API authentication: 401 without key, 403 wrong role, 200 valid key, reject expired (API)
- [ ] Rate limiting: quota enforced, headers returned, Redis fallback allows (API + Unit)
- [ ] Expert query pipeline: happy path returns valid trace (Integration)
- [ ] Expert query pipeline: one expert fails, others succeed (Integration)
- [ ] PII masking: all 4 patterns correctly masked (Unit)
- [ ] Bridge table: insert maintains cross-store consistency (Integration)
- [ ] Circuit breaker: activates on threshold, recovers after cooldown (Unit)

**Total**: 18 scenarios, ~8 min

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] All 26+ API endpoints return correct status codes and shapes (API)
- [ ] Feedback collection: each F1-F8 type persisted correctly (Integration)
- [ ] Authority scoring: computation matches expected values (Unit)
- [ ] Training scheduler: triggers on buffer threshold (Unit)
- [ ] Alembic: all migrations apply and roll back cleanly (Integration)
- [ ] Audit trail: hash chain is valid, tamper detected (Unit)
- [ ] Dataset export: PII anonymized, formats valid (Integration)
- [ ] API key lifecycle: bootstrap→create→use→revoke (API)

**Total**: 35 scenarios, ~25 min

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage + benchmarks

- [ ] Expert algorithms: all 4 experts produce valid analysis (Unit)
- [ ] Gating network: all 4 aggregation methods compute correctly (Unit)
- [ ] Policy evolution: time-series queries return correct data (API)
- [ ] NER pipeline: entities extracted from legal text (Unit)
- [ ] Frontend components: render without errors (Component)
- [ ] Performance benchmarks: latency within thresholds (Perf)

**Total**: 60 scenarios, ~45 min

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
|----------|-------|-----------|-------------|-------|
| P0 | 18 | 2.0 | 36 | Complex setup, security, cross-store |
| P1 | 35 | 1.0 | 35 | Standard API/unit coverage |
| P2 | 45 | 0.5 | 22.5 | Simple scenarios, edge cases |
| P3 | 15 | 0.25 | 3.75 | Benchmarks, exploratory |
| **Total** | **113** | **—** | **97.25** | **~12 days** |

### Prerequisites

**Test Data:**
- `UserFactory` — faker-based user generation with authority scores
- `FeedbackFactory` — all 8 feedback types with configurable ratings
- `TraceFactory` — QATrace with expert responses and sources
- `ApiKeyFactory` — keys with various roles and tiers
- Database seed fixtures for graph topology and vector embeddings

**Tooling:**
- pytest + pytest-asyncio + pytest-cov (backend — already installed)
- Vitest + React Testing Library (frontend — **to install**)
- k6 (performance load testing — **to install**)
- bandit + safety (security static analysis — **to install**)

**Environment:**
- Docker Compose with all 4 services running
- `MERLT_ENV=test` environment variable for test isolation
- Redis available for rate limiting tests
- E5-large model loaded for embedding tests (2.3GB RAM)

### Infrastructure Setup (Sprint 0)

| Task | Effort | Priority |
|------|--------|----------|
| Fix bare `except:` handlers (40+ locations) | 4 hours | Critical |
| Install Vitest + RTL in frontend | 2 hours | Critical |
| Create GitHub Actions CI workflow for merlt | 4 hours | Critical |
| Create test data factories | 4 hours | High |
| Add `/health` endpoint | 1 hour | High |
| Fix JWT verification in ws_router.py | 2 hours | Critical |
| Fix unbounded `_runs` dict | 1 hour | Critical |
| Add ESLint config for frontend | 1 hour | Medium |
| **Total Sprint 0** | **19 hours** | |

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
- **Frontend components**: ≥50% (critical dashboard tabs)

### Non-Negotiable Requirements

- [ ] All P0 tests pass (100%)
- [ ] No high-risk (≥6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] No bare `except:` in production code
- [ ] No `any` type in frontend TypeScript
- [ ] No hardcoded credentials in codebase
- [ ] `tsc --noEmit` passes with zero errors

### Gate Decision

**Current Assessment: CONCERNS**

Rationale: Architecture is fundamentally testable (DI, Docker isolation, structured traces) but 6 high-priority risks require mitigation before release confidence. The 3 security risks (R-001, R-002, R-006) are the most urgent. Sprint 0 infrastructure work (19 hours) is prerequisite for meaningful test execution.

---

## Mitigation Plans

### R-001: Unauthenticated API Access (Score: 9)

**Mitigation Strategy:** Wire `verify_api_key` and `check_rate_limit` Depends into all sensitive endpoints. Create endpoint audit checklist. Add integration test verifying every route requires authentication.
**Owner:** Dev
**Timeline:** Sprint 0
**Status:** Partially mitigated (query endpoint has auth, 25+ others do not)
**Verification:** `pytest -k test_auth_required` — every endpoint returns 401 without valid API key

### R-002: JWT Token Forgery (Score: 6)

**Mitigation Strategy:** Replace bare `jwt.decode()` in `ws_router.py` with verified decode using `PyJWT` library with algorithm whitelist (HS256). Add unit test for invalid/expired/tampered tokens.
**Owner:** Dev
**Timeline:** Sprint 0
**Status:** Planned
**Verification:** Unit test: `test_jwt_rejects_unsigned_token`, `test_jwt_rejects_expired_token`

### R-003: Silent Error Propagation (Score: 6)

**Mitigation Strategy:** Systematic grep for `except:` and `except Exception`, replace with specific exception types. Add `structlog.error()` with context. Add ruff rule to prevent bare except in CI.
**Owner:** Dev
**Timeline:** Sprint 0-1
**Status:** Planned
**Verification:** `ruff check --select E722` returns 0 violations

### R-004: Cross-Store Data Inconsistency (Score: 6)

**Mitigation Strategy:** Add transactional wrapper for ingestion pipeline (Qdrant insert + FalkorDB insert + bridge table insert). Add `GET /pipeline/consistency-check` endpoint that validates cross-store referential integrity.
**Owner:** Dev
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Integration test: partial ingestion failure results in clean rollback

### R-005: Unbounded Memory (Score: 6)

**Mitigation Strategy:** Replace `dict` with `collections.OrderedDict` with maxlen=100 in `regression_router.py`. Add FIFO eviction when limit reached. Log eviction events.
**Owner:** Dev
**Timeline:** Sprint 0
**Status:** Planned
**Verification:** Unit test: insert 101 entries, verify oldest evicted, memory stable

### R-006: GDPR Compliance Gap (Score: 6)

**Mitigation Strategy:** Audit all data egress points (trace viewer, dataset export, API responses). Wire `PIIMaskingService` into trace serialization and export pipeline. Add consent check before any PII-containing response.
**Owner:** Dev
**Timeline:** Sprint 1
**Status:** Partially mitigated (feedback endpoints have PII masking)
**Verification:** Integration test: request trace without consent → PII fields masked/absent

---

## Assumptions and Dependencies

### Assumptions

1. Docker Compose services are available for all integration tests (PostgreSQL, FalkorDB, Qdrant, Redis)
2. E5-large embedding model is pre-cached locally (not downloaded during tests)
3. OpenRouter API key is available for LLM integration tests (can be mocked for unit tests)
4. Single developer performing test implementation (~12 days effort)
5. Existing 100+ backend test files are partially functional and can be adapted

### Dependencies

1. **Vitest + React Testing Library** — Required for frontend test suite (Sprint 0)
2. **GitHub Actions workflow** — Required for CI gate enforcement (Sprint 0)
3. **Redis** — Required for rate limiting tests (already in Docker Compose)
4. **bandit + safety** — Required for security static analysis (Sprint 0)
5. **k6** — Required for performance load testing (Sprint 1)

### Risks to Plan

- **Risk**: Existing 100+ backend tests may have widespread import issues preventing collection
  - **Impact**: Sprint 0 effort increases by 4-8 hours to fix test infrastructure
  - **Contingency**: Triage test files: fix critical paths, archive or delete broken tests

- **Risk**: OpenRouter API rate limits may slow integration tests
  - **Impact**: LLM-dependent tests become flaky or slow
  - **Contingency**: Mock OpenRouter for P0/P1 tests, use real API only for P3 integration

- **Risk**: E5-large model loading adds 30s+ to test startup
  - **Impact**: Test suite slow to start, discourages frequent runs
  - **Contingency**: Use fixture with session scope for model loading, skip embedding tests with marker

---

## Recommendations for Sprint 0

### Critical (Must complete before test writing)

1. **Fix bare `except:` handlers** — Systematic replacement across 40+ locations
2. **Fix JWT verification** — `ws_router.py` must verify signatures
3. **Fix unbounded `_runs` dict** — Add maxlen to `regression_router._runs`
4. **Install frontend test framework** — Vitest + React Testing Library + ESLint
5. **Create CI pipeline** — GitHub Actions for pytest + tsc + eslint on PR

### High Priority (Complete during Sprint 0)

6. **Create test data factories** — `tests/factories/` with faker-based builders
7. **Add `/health` endpoint** — Database dependency checks
8. **Wire auth into all endpoints** — Not just query, but all sensitive routes
9. **Add ESLint config** — Enforce no-console, no-any rules

### Medium Priority (Can start in Sprint 1)

10. **Add Prometheus metrics** — Request latency, error rate, active connections
11. **Add consistency check endpoint** — Cross-store referential integrity
12. **Audit PII egress points** — Ensure all data paths have consent filtering

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.
- Run `*testarch-framework` to scaffold test directory structure and fixtures.
- Run `*testarch-ci` to generate GitHub Actions workflow configuration.
- Run `*testarch-nfr` before release to validate non-functional requirements with evidence.

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
- `test-levels-framework.md` — Unit/Integration/E2E decision matrix
- `test-quality.md` — Quality standards (deterministic, isolated, explicit, <300 lines, <1.5 min)
- `nfr-criteria.md` — NFR validation patterns (security, performance, reliability, maintainability)

### Related Documents

- PRD: `_bmad-output/planning-artifacts/prd.md`
- Epics: `_bmad-output/planning-artifacts/epics.md`
- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Sprint Status: `_bmad-output/implementation-artifacts/sprint-status.yaml`

---

**Generated by**: BMad TEA Agent — Test Architect Module
**Workflow**: `_bmad/bmm/testarch/test-design` (System-Level Mode)
**Version**: 4.0 (BMad v6)
