# System-Level Test Design: ALIS_CORE

**Date:** 2026-01-24
**Author:** TEA (Test Engineering Architect) Agent
**Status:** Draft
**Mode:** System-Level (Phase 3 - Solutioning)

---

## Executive Summary

**Scope:** System-level testability review for ALIS_CORE architecture before implementation phase.

**Architecture Classification:**
- **Type:** Multi-service distributed system (3 backends + 4 databases)
- **Domain:** LegalTech AI with RLCF learning loop
- **Complexity:** High (ML pipeline, graph DB, vector search, legal compliance)
- **Brownfield:** Yes - migrating from Legacy/MERL-T_alpha

**Testability Assessment:**

| Dimension | Score | Status |
|-----------|-------|--------|
| **Controllability** | 7/10 | Good - explicit APIs, dependency injection |
| **Observability** | 6/10 | Adequate - needs telemetry improvements |
| **Isolatability** | 8/10 | Strong - service boundaries well-defined |
| **Determinism** | 5/10 | Concern - LLM responses, async pipeline |
| **Coverage Potential** | 8/10 | Good - clear interfaces, testable units |

**Key Findings:**
- 5 high-risk ASRs identified (see Risk Assessment)
- Existing tests in Legacy need migration/validation
- LLM non-determinism requires special handling
- RLCF feedback loop needs synthetic data strategy
- 7-year audit trail immutability requires specific testing

---

## Architecturally Significant Requirements (ASRs)

### High-Priority ASRs (Score ≥6)

| ASR ID | Category | Requirement | PRD Reference | Risk Score | Testability |
|--------|----------|-------------|---------------|------------|-------------|
| **ASR-001** | SEC | GDPR consent verification on every learning action | NFR-C1, NFR-S7 | 6 | Automatable |
| **ASR-002** | DATA | 7-year immutable audit trail | NFR-R5, NFR-S6 | 6 | Automatable |
| **ASR-003** | PERF | Expert enrichment <3min (first), <500ms (cached) | NFR-P2, NFR-P3 | 6 | Automatable |
| **ASR-004** | REL | Expert pipeline graceful degradation (circuit breaker) | NFR-R3, ADR-001 | 6 | Automatable |
| **ASR-005** | DATA | 100% traceability - every response to Expert + Sources | Success Criteria | 9 | Automatable |

### Medium-Priority ASRs (Score 3-5)

| ASR ID | Category | Requirement | PRD Reference | Risk Score | Testability |
|--------|----------|-------------|---------------|------------|-------------|
| **ASR-006** | PERF | KG query <200ms | NFR-P4 | 4 | Automatable |
| **ASR-007** | SEC | PII anonymization before RLCF storage | NFR-S5 | 4 | Automatable |
| **ASR-008** | REL | Historical query reproducibility | NFR-R6 | 4 | Manual validation + automation |
| **ASR-009** | INT | LLM provider abstraction (switchable) | NFR-I2 | 3 | Integration test |
| **ASR-010** | MAINT | Docker Compose single-command deployment | NFR-M1 | 3 | Smoke test |

---

## Test Levels Strategy

### Proposed Test Pyramid

```
                    ┌───────────┐
                    │    E2E    │  ~10% (User Journeys)
                    │    (5)    │
               ┌────┴───────────┴────┐
               │    Integration      │  ~25% (API contracts, DB ops)
               │       (25)          │
          ┌────┴─────────────────────┴────┐
          │           Unit               │  ~65% (Business logic, utils)
          │           (65)               │
          └──────────────────────────────┘
```

### Test Level Assignment by Component

| Component | Unit % | Integration % | E2E % | Justification |
|-----------|--------|---------------|-------|---------------|
| **NER (A1)** | 80% | 15% | 5% | Pure function extraction + spaCy model validation |
| **Expert Router (A2)** | 60% | 35% | 5% | Rule logic + routing decisions need integration |
| **4 Experts (A3-A6)** | 50% | 40% | 10% | LLM integration critical, prompts need E2E validation |
| **Gating Network (A7)** | 70% | 25% | 5% | Neural weighting is unit-testable, policy needs integration |
| **Synthesizer (A8)** | 50% | 40% | 10% | Aggregation logic + LLM prompts |
| **RLCF Orchestrator (A9)** | 40% | 50% | 10% | Heavy DB interaction, training loops |
| **Policy Gradient Trainer (A10)** | 30% | 60% | 10% | Training requires full pipeline context |
| **Bridge Table (F8)** | 40% | 55% | 5% | Vector↔Graph mapping needs both DBs |
| **TraversalPolicy** | 50% | 45% | 5% | Weight learning needs integration |
| **PostgreSQL schemas** | 10% | 85% | 5% | Mostly integration (migrations, queries) |
| **FalkorDB KG** | 10% | 80% | 10% | Graph queries need real Cypher execution |
| **Qdrant vectors** | 10% | 80% | 10% | Semantic search needs real embeddings |

### Test Type Breakdown

| Test Type | Count | Purpose | Tools |
|-----------|-------|---------|-------|
| **Unit** | ~65 | Business logic, utils, validators | pytest, vitest |
| **Integration API** | ~20 | REST endpoint contracts | pytest + httpx, Playwright APIRequest |
| **Integration DB** | ~15 | PostgreSQL/FalkorDB/Qdrant operations | pytest + testcontainers |
| **Component** | ~10 | React components in isolation | Playwright Component Testing |
| **E2E User Journey** | ~5 | Critical paths (7 journeys from PRD) | Playwright |
| **E2E API** | ~5 | Full pipeline (NER→Experts→Synth→RLCF) | pytest + httpx |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|-------------|--------|-------|------------|-------|
| **R-001** | DATA | Traceability breaks (response without full trace) | 2 | 3 | 6 | Trace validation on every response, schema enforcement | QA |
| **R-002** | SEC | GDPR consent bypass (learning without consent) | 2 | 3 | 6 | Consent middleware test, integration test on RLCF write | Security |
| **R-003** | DATA | Audit trail corruption (mutable or lost) | 2 | 3 | 6 | Append-only DB test, backup restoration test | DevOps |
| **R-004** | PERF | Expert timeout cascade (3min breach) | 3 | 2 | 6 | Circuit breaker test, k6 load test with slow LLM mock | QA |
| **R-005** | REL | Expert pipeline crash (no graceful degradation) | 2 | 3 | 6 | Fault injection test, circuit breaker verification | QA |
| **R-006** | TECH | LLM non-determinism breaks reproducibility | 3 | 2 | 6 | Snapshot testing with fuzzy matching, seed control | QA |

### Medium-Priority Risks (Score 3-5)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation |
|---------|----------|-------------|-------------|--------|-------|------------|
| **R-007** | PERF | KG query >200ms (graph too large) | 2 | 2 | 4 | Index optimization, query plan validation |
| **R-008** | SEC | PII leak in RLCF data | 1 | 3 | 3 | Anonymization unit test, regex scan in CI |
| **R-009** | INT | Normattiva scraper breaks (site change) | 3 | 1 | 3 | Contract test, retry backoff test |
| **R-010** | TECH | Bridge Table mapping drift | 2 | 2 | 4 | F8 feedback validation, consistency check |

---

## NFR Testing Strategy

### Performance (NFR-P1 to NFR-P7)

**Tool:** k6 for load testing (NOT Playwright)

| NFR | Test Approach | Threshold | Automation |
|-----|---------------|-----------|------------|
| NFR-P1: Norm display <500ms | k6 load test `/api/v1/norms/{urn}` | p95 < 500ms | CI nightly |
| NFR-P2: Expert first visit <3min | k6 stress test `/api/v1/analyze` | p99 < 180s | CI nightly |
| NFR-P3: Expert cached <500ms | k6 with warm cache | p95 < 500ms | CI nightly |
| NFR-P4: KG query <200ms | k6 graph endpoint | p95 < 200ms | CI nightly |
| NFR-P5: 20 concurrent users | k6 staged ramp to 20 VUs | 0% error rate | CI weekly |
| NFR-P6: Feedback <1s | k6 POST `/api/v1/feedback` | p95 < 1000ms | CI nightly |
| NFR-P7: 80% cache hit | Cache hit counter after warm start | >80% | CI nightly |

**k6 Script Structure:**
```javascript
// tests/nfr/performance.k6.js
export const options = {
  scenarios: {
    cached_response: { /* 20 VUs, 5 min, cached queries */ },
    cold_start: { /* 5 VUs, 3 min, new queries */ },
    spike: { /* Ramp to 50 VUs briefly */ },
  },
  thresholds: {
    'http_req_duration{scenario:cached_response}': ['p(95)<500'],
    'http_req_duration{scenario:cold_start}': ['p(99)<180000'],
    errors: ['rate<0.01'],
  },
};
```

### Security (NFR-S1 to NFR-S7)

**Tools:** Playwright E2E + OWASP ZAP + custom scripts

| NFR | Test Approach | Automation |
|-----|---------------|------------|
| NFR-S1: AES-256 at rest | DB encryption verification script | CI deploy |
| NFR-S2: TLS 1.3 | SSL Labs API check, certificate validation | CI deploy |
| NFR-S3: JWT rotation | Token expiry E2E test, refresh flow | Playwright |
| NFR-S4: API key + rate limit | Rate limit exhaustion test (429 handling) | k6 |
| NFR-S5: PII anonymization | Regex scan on RLCF data export | pytest |
| NFR-S6: Audit immutability | Attempt audit modification → expect fail | pytest |
| NFR-S7: Consent verification | RLCF write without consent → expect 403 | Playwright + pytest |

### Reliability (NFR-R1 to NFR-R6)

**Tools:** Playwright E2E + pytest + chaos engineering scripts

| NFR | Test Approach | Automation |
|-----|---------------|------------|
| NFR-R1: 99% uptime | Health check endpoint monitoring | External (UptimeRobot) |
| NFR-R2: Daily backup | Backup restoration test | Weekly manual + script |
| NFR-R3: Graceful degradation | Fault injection (kill Expert) → partial response | pytest + Docker |
| NFR-R4: LLM failover | Mock primary provider failure → backup kicks in | pytest |
| NFR-R5: 7-year retention | Retention policy test (mock time advance) | pytest |
| NFR-R6: Query reproducibility | Same query + model version → same trace structure | pytest snapshot |

### Maintainability (NFR-M1 to NFR-M5)

**Tools:** CI scripts + coverage tools

| NFR | Test Approach | Automation |
|-----|---------------|------------|
| NFR-M1: Docker Compose deploy | `docker-compose up` → health check all services | CI on PR |
| NFR-M2: Config externalization | Start with custom env → validate settings applied | pytest |
| NFR-M3: Structured logging | Log format validation (JSON schema) | CI |
| NFR-M4: API docs | OpenAPI spec generation + validation | CI |
| NFR-M5: 80% coverage | pytest-cov, vitest coverage | CI gate |

---

## Testability Concerns & Recommendations

### Concern 1: LLM Non-Determinism

**Issue:** LLM responses are inherently non-deterministic. Same query can produce different reasoning traces.

**Impact:** R-006 (Score 6), NFR-R6 (historical reproducibility)

**Recommendations:**
1. **Seed control:** Use `temperature=0` and fixed random seeds where possible
2. **Fuzzy matching:** Compare trace structure, not exact text
3. **Golden response snapshots:** Validate structure, not content
4. **Model versioning:** Lock model version, log with every response
5. **Stub LLM in unit tests:** Mock LLM calls with deterministic responses

### Concern 2: Distributed System Complexity

**Issue:** 3 backends + 4 databases create complex failure modes and integration points.

**Impact:** R-004, R-005 (Score 6)

**Recommendations:**
1. **Testcontainers:** Use testcontainers-python for isolated DB instances
2. **Service virtualization:** WireMock for external services (Normattiva, LLM providers)
3. **Health check contracts:** Every service exposes `/health` with dependency status
4. **Chaos engineering:** Fault injection tests (kill service, network partition)

### Concern 3: RLCF Feedback Loop Testing

**Issue:** RLCF requires real user feedback to train. Thesis timeline limits real data.

**Impact:** Success criteria (RLCF operational), NFR-SC4 (1000 entries/month)

**Recommendations:**
1. **Synthetic feedback generator:** Script to create realistic feedback distributions
2. **Authority weighting validation:** Unit test weighting math
3. **Policy gradient trainer:** Mock feedback, validate weight updates
4. **A/B comparison:** RLCF vs RLHF with synthetic data

### Concern 4: 7-Year Audit Trail

**Issue:** Cannot wait 7 years to validate retention. Need time-based testing.

**Impact:** ASR-002, NFR-R5 (Score 6)

**Recommendations:**
1. **Mock clock:** Use `freezegun` (Python) or `jest.useFakeTimers()` to simulate time
2. **Retention policy test:** Create record, advance time 7 years, verify present
3. **Immutability test:** Attempt UPDATE/DELETE on audit table → expect failure
4. **Backup restoration test:** Restore from backup, verify audit trail intact

### Concern 5: Bridge Table (F8) Learning Validation

**Issue:** F8 is a new feedback point. No existing tests or patterns.

**Impact:** R-010 (Score 4), TraversalPolicy learning

**Recommendations:**
1. **Unit test:** Bridge Table CRUD operations
2. **Integration test:** chunk_id ↔ graph_node_urn mapping consistency
3. **Feedback collection test:** Implicit (correlation) + Explicit (🎓 Contributore)
4. **Weight update test:** F8 feedback → TraversalPolicy weights change

### Concern 6: Legacy Test Migration

**Issue:** 80+ test files in `Legacy/MERL-T_alpha/tests/`. Need to assess and migrate.

**Recommendations:**
1. **Audit existing tests:** Categorize by component, identify coverage gaps
2. **Migration plan:** Prioritize core pipeline tests (Experts, RLCF, Bridge)
3. **Deprecation:** Mark legacy tests that won't migrate with reason
4. **Coverage baseline:** Establish baseline before migration

---

## Existing Test Inventory (Legacy)

### Test Categories Found

| Category | File Count | Status | Priority |
|----------|------------|--------|----------|
| `tests/experts/` | 8 files | Active | High - migrate |
| `tests/rlcf/` | 7 files | Active | High - migrate |
| `tests/storage/` | 6 files | Active | High - migrate (includes bridge) |
| `tests/pipeline/` | 5 files | Active | High - migrate |
| `tests/disagreement/` | 10 files | Active | Medium - Devil's Advocate |
| `tests/weights/` | 5 files | Active | High - RLCF weights |
| `tests/_archive/` | 38 files | Archived | Low - review for patterns |
| `tests/benchmark/` | 2 files | Active | Medium - gold standard |

### Key Tests to Migrate

| Test File | Purpose | Migration Priority |
|-----------|---------|-------------------|
| `test_experts.py` | Expert base class | P0 |
| `test_literal.py` | LiteralExpert | P0 |
| `test_orchestration.py` | Expert sequencing | P0 |
| `test_synthesizer.py` | Response synthesis | P0 |
| `test_policy_gradient.py` | RLCF training | P0 |
| `test_policy_manager.py` | Policy persistence | P0 |
| `test_bridge_table.py` | F8 bridge | P0 |
| `test_bridge_builder.py` | F8 ingestion | P0 |
| `test_retriever.py` | Vector search | P1 |
| `test_retriever_with_policy.py` | Retrieval + policy | P1 |
| `test_gold_standard.py` | Regression baseline | P1 |

---

## Quality Gate Criteria

### Pre-Implementation Gate (Phase 3 → Phase 4)

- [ ] All P0 test files from Legacy migrated
- [ ] k6 performance baseline established
- [ ] Testcontainers setup for all databases
- [ ] LLM mock/stub infrastructure ready
- [ ] CI pipeline with unit + integration tests
- [ ] Coverage baseline documented

### Pre-Release Gate (Per Epic)

| Criterion | Threshold | Non-Negotiable |
|-----------|-----------|----------------|
| Unit test pass rate | 100% | Yes |
| Integration test pass rate | 100% | Yes |
| E2E critical path pass rate | 100% | Yes |
| Code coverage | ≥80% (core pipeline) | Yes |
| Performance thresholds | All k6 thresholds met | Yes |
| Security tests | All SEC-category pass | Yes |
| Traceability | 100% responses have trace | Yes |

---

## Test Environment Requirements

### Local Development

```yaml
services:
  postgres-test:
    image: postgres:15
    ports: [5433:5432]  # Offset to avoid conflict
    environment:
      POSTGRES_DB: alis_test

  falkordb-test:
    image: falkordb/falkordb:latest
    ports: [6380:6379]  # Offset

  qdrant-test:
    image: qdrant/qdrant:latest
    ports: [6334:6333]  # Offset

  redis-test:
    image: redis:7
    ports: [6381:6379]  # Offset
```

### CI Environment

- **GitHub Actions** with service containers
- **Testcontainers** for isolated DB instances per test run
- **k6 Cloud** or self-hosted for performance tests
- **Playwright containers** for E2E

### Test Data Strategy

| Data Type | Source | Management |
|-----------|--------|------------|
| **Norms** | Seed from `/seed/norms/` JSON | Version-controlled fixtures |
| **Users** | Faker-generated | Per-test cleanup |
| **Feedback** | Synthetic generator | Configurable distributions |
| **Embeddings** | Pre-computed vectors | Stored in test fixtures |
| **Graph** | Cypher seed scripts | Version-controlled |

---

## Appendix: NFR-to-Test Mapping

| NFR | Test Type | Test File | CI Stage |
|-----|-----------|-----------|----------|
| NFR-P1 | k6 | `tests/nfr/performance.k6.js` | Nightly |
| NFR-P2 | k6 | `tests/nfr/performance.k6.js` | Nightly |
| NFR-P3 | k6 | `tests/nfr/performance.k6.js` | Nightly |
| NFR-P4 | k6 | `tests/nfr/performance.k6.js` | Nightly |
| NFR-P5 | k6 | `tests/nfr/load.k6.js` | Weekly |
| NFR-S1 | pytest | `tests/nfr/security.py` | Deploy |
| NFR-S3 | Playwright | `tests/e2e/auth/jwt.spec.ts` | PR |
| NFR-S5 | pytest | `tests/nfr/anonymization.py` | PR |
| NFR-S7 | Playwright + pytest | `tests/e2e/consent.spec.ts` | PR |
| NFR-R3 | pytest | `tests/integration/circuit_breaker.py` | PR |
| NFR-R5 | pytest | `tests/nfr/audit_retention.py` | Weekly |
| NFR-R6 | pytest | `tests/integration/reproducibility.py` | PR |
| NFR-M1 | bash | `scripts/smoke-test.sh` | PR |
| NFR-M5 | pytest-cov | `.github/workflows/coverage.yml` | PR |

---

## Follow-on Workflows

After implementation begins:
1. **`/bmad:bmm:workflows:testarch-atdd`** - Generate failing acceptance tests per epic
2. **`/bmad:bmm:workflows:testarch-automate`** - Expand automation coverage
3. **`/bmad:bmm:workflows:testarch-ci`** - Scaffold CI/CD pipeline
4. **`/bmad:bmm:workflows:testarch-nfr`** - Pre-release NFR validation
5. **`/bmad:bmm:workflows:testarch-trace`** - Requirements-to-tests traceability matrix

---

**Generated by:** BMad TEA Agent - Test Architect Module
**Workflow:** `_bmad/bmm/workflows/testarch/test-design`
**Version:** 4.0 (BMad v6)
