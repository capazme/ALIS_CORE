# Test Automation Expansion — Summary Report

**Generated**: 2026-02-17
**Workflow**: testarch-automate (Standalone Mode, Auto-discover)
**Coverage Target**: critical-paths
**Project**: ALIS_CORE (Full Monorepo)

---

## Execution Overview

5 parallel agents generated **130 new tests** across 18 files in ~12 minutes:

| Agent | Area | Files | Tests | Status |
|-------|------|-------|-------|--------|
| factory-builder | Python test factories | 6 | — | ✅ Complete |
| unit-test-builder | Python unit tests | 4 | 60 | ✅ 60/60 passing |
| integration-test-builder | Python integration tests | 1 | 4 | ✅ Complete |
| e2e-builder | Playwright E2E tests | 4 | 30 | ✅ Complete |
| plugin-test-builder | Vitest component/service | 3 | 36 | ✅ 36/36 passing |

---

## 1. Infrastructure Created

### Python Test Factories — `merlt/tests/factories/`

| File | Functions | Purpose |
|------|-----------|---------|
| `__init__.py` | exports all | Central import point |
| `user_factory.py` | `create_user()`, `create_users()` | Users with Italian names, authority scores, profile types |
| `feedback_factory.py` | `create_feedback(type)` | All 8 feedback types (F1-F8) with domain-specific fields |
| `trace_factory.py` | `create_trace()`, `create_expert_response()` | QATrace with 4 expert responses, sources, synthesis |
| `api_key_factory.py` | `create_api_key()` | API keys with roles, tiers, SHA-256 hashes |
| `article_factory.py` | `create_article()` | Italian legal articles with realistic URN:NIR format |

**Design:** Python stdlib only (no faker dep), `**overrides` pattern, matches existing SQLAlchemy/Pydantic models.

---

## 2. Tests Created

### Python Unit Tests — `merlt/tests/unit/` (was EMPTY → 60 tests)

| File | Priority | Tests | Coverage |
|------|----------|-------|----------|
| `test_pii_masking.py` | **P0** | 21 | CF, email, phone, dates, edge cases, consent levels |
| `test_circuit_breaker.py` | **P0** | 17 | State transitions, async context, registry, callbacks |
| `test_confidence_calibration.py` | P2 | 9 | α-blending, weight normalization, edge cases |
| `test_query_rewriting.py` | P2 | 13 | All 4 expert rewriting strategies |

**All 60 tests passing.** Pure unit tests — no DB, no network, no LLM.

### Python Integration Tests — `merlt/tests/integration/`

| File | Priority | Tests | Coverage |
|------|----------|-------|----------|
| `test_bridge_consistency.py` | **P0** | 4 | Health check, Qdrant mapping, FalkorDB mapping, orphan detection |

Uses httpx AsyncClient + real DB fixtures. 5% tolerance on cross-store checks, 10% orphan threshold.

### Playwright E2E Tests — `visualex-platform/frontend/e2e/` (was EMPTY → 30 tests)

| File | Priority | Tests | Coverage |
|------|----------|-------|----------|
| `smoke.spec.ts` | **P0** | 5 | Homepage redirect, login/register pages, health check, JS errors |
| `auth.spec.ts` | **P0** | 10 | Registration, login (valid/invalid), logout, validation, navigation |
| `search.spec.ts` | **P1** | 7 | Search form, NDJSON results, range search, empty state, errors, Cmd+K |
| `profile.spec.ts` | **P1** | 8 | Settings sections, consent, authority, privacy, save, preferences |

Network-first mocking, data-testid selectors, Given-When-Then format.

### Vitest Component/Service Tests — `visualex-merlt/frontend/`

| File | Priority | Tests | Coverage |
|------|----------|-------|----------|
| `services/__tests__/merltService.test.ts` | **P1** | 15 | queryExperts, all feedback types, error handling (401/500/network) |
| `hooks/__tests__/useTraceData.test.ts` | P2 | 9 | Fetch, loading, error, partial success, refetch, traceId changes |
| `components/__tests__/MerltToolbar.test.tsx` | **P1** | 12 | Click handler, pulse animation, badge, active/processed styles |

**All 36 tests passing** (1.14s).

---

## 3. Summary Statistics

### Total New Tests: **130**

| Priority | Tests | % |
|----------|-------|---|
| **P0** (every commit) | 56 | 43% |
| **P1** (PR to main) | 43 | 33% |
| **P2** (nightly) | 31 | 24% |

| Test Level | Tests | % |
|------------|-------|---|
| Unit (Python) | 60 | 46% |
| Component (Vitest) | 36 | 28% |
| E2E (Playwright) | 30 | 23% |
| Integration (Python) | 4 | 3% |

### Combined Inventory (Before → After)

| Area | Before | After | Delta |
|------|--------|-------|-------|
| merlt/tests/ | 116 files | 127 files | **+11** |
| platform/backend/tests/ | 12 files | 12 files | 0 |
| platform/frontend/ tests+e2e | 9+0 files | 9+4 files | **+4** |
| merlt-frontend/ tests | 5 files | 8 files | **+3** |
| **Total** | **142** | **160** | **+18** |

---

## 4. Risk Mitigation Status

| Risk | Score | Before | After |
|------|-------|--------|-------|
| R-015: Zero E2E tests | 6 | ❌ 0 tests | ✅ **30 Playwright tests** |
| R-004: Cross-store consistency | 6 | ❌ untested | ✅ **4 integration tests** |
| R-006: PII masking gaps | 4 | ❌ untested | ✅ **21 unit tests** |
| R-019: Plugin system untested | 4 | ❌ untested | ✅ **36 component tests** |
| R-003: Bare except/reliability | 4 | partial | ✅ **17 circuit breaker tests** |

---

## 5. Run Commands

```bash
# === Python (merlt) ===
cd merlt && python -m pytest tests/unit/ -v                              # Unit tests (60)
cd merlt && python -m pytest tests/unit/test_pii_masking.py tests/unit/test_circuit_breaker.py -v  # P0 only
cd merlt && python -m pytest tests/integration/test_bridge_consistency.py -v -m integration        # Integration

# === Playwright E2E (visualex-platform) ===
cd visualex-platform/frontend && npx playwright test                      # All E2E (30)
cd visualex-platform/frontend && npx playwright test e2e/smoke.spec.ts    # Smoke P0
cd visualex-platform/frontend && npx playwright test --grep "P0"          # P0 only

# === Vitest (visualex-merlt) ===
cd visualex-merlt/frontend && npx vitest run                              # All plugin tests (36)
```

---

## 6. Remaining Gaps

| Gap | Priority | Next Action |
|-----|----------|-------------|
| JWT verification in ws_router (R-002) | P0 | Fix code + add unit test |
| In-memory persistence (R-017) | P1 | Add Redis/PostgreSQL persistence |
| RLCF end-to-end training loop | P1 | Integration test: feedback → aggregate → train |
| API contract tests (all 30+ routes) | P1 | Run `*atdd` workflow |
| Alembic migration up/down | P1 | Add migration test |
| Performance benchmarks (k6) | P3 | Install k6, write load tests |
| Accessibility audit (axe-core) | P3 | Install @axe-core/playwright |

---

## 7. Quality Checklist

- [x] All tests follow Given-When-Then format
- [x] All tests have priority tags ([P0], [P1], [P2])
- [x] Playwright tests use data-testid / role selectors
- [x] Playwright tests use network-first mocking
- [x] No hard waits in any test
- [x] Python unit tests are pure (no DB, no network)
- [x] Python integration tests use existing fixture patterns
- [x] Vitest tests mock API calls with vi.mock()
- [x] All test files under 300 lines
- [x] Factory pattern with **overrides for customization
- [x] No shared state between tests

---

## 8. Known Issues

1. **lucide-react Proxy mock** — Existing tab tests use a Proxy-based mock that hangs on Node v25. New tests use explicit named exports instead.
2. **Playwright E2E** — Need backend (:3001) and frontend (:5173) running for real integration; mocked tests work standalone.
3. **Bridge consistency tests** — Need Docker services (PostgreSQL, FalkorDB, Qdrant, Redis).

---

## 9. Next Steps

1. Run `*testarch-trace` — Generate requirements-to-tests traceability matrix
2. Run `*testarch-ci` — Integrate new tests into CI pipeline quality gates
3. Fix R-002 — JWT verification in ws_router.py
4. Run `*testarch-nfr` — Non-functional requirements validation before release

---

*Generated by BMAD TEA testarch-automate workflow v4.0 (Standalone Mode)*
