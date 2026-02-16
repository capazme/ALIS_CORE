# VisuaLex Platform — Test Suite Documentation

## Overview

This document describes the complete test architecture for the VisuaLex Platform.

## Test Stack

| Level | Framework | Config | Run Command |
|---|---|---|---|
| **E2E** | Playwright | `playwright.config.ts` | `npm run test:e2e` |
| **Unit/Component** | Vitest | `vitest.config.ts` | `npm run test` |
| **Backend API** | Jest + Supertest | `../backend/jest.config.js` | `cd ../backend && npx jest` |

## Directory Structure

```
visualex-platform/
├── frontend/
│   ├── e2e/                          # Playwright E2E tests
│   │   ├── fixtures.ts               # Shared fixtures, factories, helpers
│   │   ├── auth.spec.ts              # Authentication flows (Epic 1)
│   │   ├── settings.spec.ts          # Settings page (Epic 1)
│   │   ├── search.spec.ts            # Search & Norm Browsing (Epic 3)
│   │   ├── workspace.spec.ts         # Dossier, History, Environments
│   │   ├── admin.spec.ts             # Admin Dashboard (Epic 9)
│   │   └── navigation.spec.ts        # Route protection & sidebar nav
│   ├── src/test/                     # Vitest unit/component tests
│   │   ├── setup.ts                  # Test environment setup
│   │   ├── components/               # Component tests
│   │   │   ├── Button.test.tsx
│   │   │   ├── LoginForm.test.tsx
│   │   │   ├── RegisterForm.test.tsx
│   │   │   ├── SearchForm.test.tsx
│   │   │   └── VerifyEmailPage.test.tsx
│   │   ├── hooks/                    # Hook tests
│   │   │   └── useAuth.test.ts
│   │   └── services/                 # Service tests
│   │       ├── authService.test.ts
│   │       ├── bookmarkService.test.ts
│   │       ├── consent-privacy.test.ts
│   │       └── invitationService.test.ts
│   └── playwright.config.ts
├── backend/
│   └── tests/
│       ├── setup.ts
│       ├── auth.test.ts
│       ├── profile.test.ts
│       ├── consent.test.ts
│       ├── authority.test.ts
│       ├── privacy.test.ts
│       ├── unit/
│       │   ├── jwt.test.ts
│       │   └── password.test.ts
│       └── integration/
│           ├── auth.test.ts
│           ├── bookmarks.test.ts
│           ├── dossiers.test.ts        ← NEW
│           ├── folders.test.ts         ← NEW
│           └── history-highlights-     ← NEW
│               annotations.test.ts
```

## Priority System

Tests are tagged with priority levels `[P0]` through `[P3]`:

| Priority | Meaning | When to Run | Count |
|---|---|---|---|
| **P0** | Critical / Smoke | Every PR, every deploy | ~10 |
| **P1** | High / Regression | Every PR | ~25 |
| **P2** | Medium | Nightly / Full suite | ~15 |
| **P3** | Low / Edge cases | Weekly / On demand | ~5 |

### Running by Priority

```bash
# P0 only (smoke tests, ~30s)
npm run test:e2e:smoke

# P0 + P1 (regression, ~2min)
npm run test:e2e:regression

# Full suite (all priorities)
npm run test:e2e
```

## E2E Test Conventions

### ✅ DO

- Use `data-testid`, ARIA roles, or text selectors
- Use the `authenticatedPage` / `adminPage` fixtures from `fixtures.ts`
- Use Given-When-Then comments for clarity
- Use explicit waits: `waitForURL`, `waitForSelector`, `toBeVisible`
- Tag every test with `[P0]`, `[P1]`, `[P2]`, or `[P3]`
- Make tests self-cleaning (fixtures handle cleanup)

### ❌ DON'T

- Use `page.waitForTimeout()` (hard waits → flakiness)
- Use `waitForLoadState('networkidle')` (unreliable)
- Create Page Object classes
- Use `try-catch` for test logic
- Use `if (await element.isVisible())` conditional flows
- Test the same behavior at multiple levels

### Network-First Pattern

For tests that rely on API responses, set up route interception **before** navigation:

```typescript
// ✅ CORRECT: Route interception before navigation
await page.route('**/api/search', (route) => {
  route.fulfill({ status: 200, body: JSON.stringify({ results: [] }) });
});
await page.goto('/');

// ❌ WRONG: Navigation before interception (race condition)
await page.goto('/');
await page.route('**/api/search', ...);
```

## Running Tests

### Frontend Unit Tests (Vitest)
```bash
cd visualex-platform/frontend
npm run test            # Watch mode
npm run test -- --run   # Single run
npm run test:coverage   # With coverage report
```

### Frontend E2E Tests (Playwright)
```bash
cd visualex-platform/frontend

# Prerequisites: backend must be running
#   cd ../backend && npm run dev

npm run test:e2e            # Full suite (headless)
npm run test:e2e:headed     # With browser visible
npm run test:e2e:ui         # Interactive UI mode
npm run test:e2e:debug      # Debug mode

# By spec file
npm run test:e2e:auth
npm run test:e2e:search
npm run test:e2e:admin
npm run test:e2e:workspace
npm run test:e2e:nav
```

### Backend Tests (Jest)
```bash
cd visualex-platform/backend

# Prerequisites: database must be running
#   docker-compose up -d db

npx jest                    # Full suite
npx jest --testPathPattern integration  # Integration only
npx jest --testPathPattern unit         # Unit only
```

### All Tests
```bash
cd visualex-platform/frontend
npm run test:all    # Vitest (unit) + Playwright (E2E)
```

## Pre-seeded Test Data

E2E tests require pre-seeded users in the database:

```bash
cd visualex-platform/backend
npx tsx src/utils/seed.ts
```

| User | Email | Password | Role |
|---|---|---|---|
| Standard | `e2e-test@visualex.it` | `TestPassword123!` | user |
| Admin | `admin@visualex.it` | `AdminPassword123!` | admin |

## Extending the Test Suite

1. **New E2E test**: Create `e2e/<feature>.spec.ts`, import from `fixtures.ts`
2. **New unit test**: Create `src/test/<type>/<name>.test.ts`
3. **New backend test**: Create `tests/<unit|integration>/<name>.test.ts`
4. **New fixture**: Add to `e2e/fixtures.ts` in the `test.extend<>` block
