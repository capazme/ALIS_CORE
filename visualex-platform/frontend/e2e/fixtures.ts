/**
 * E2E Test Fixtures — BMAD TEA Automate
 * =======================================
 *
 * Shared fixtures, helpers, and data factories for E2E tests.
 * All fixtures have auto-cleanup in teardown.
 *
 * Priority: [P0] Critical  [P1] High  [P2] Medium  [P3] Low
 */
import { test as base, expect, type Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Constants — Pre-seeded credentials (created by prisma/seed.ts)
// ---------------------------------------------------------------------------
export const SEEDED_USER = {
  email: 'e2e-test@visualex.it',
  password: 'TestPassword123!',
} as const;

export const ADMIN_USER = {
  email: 'admin@visualex.it',
  password: 'AdminPassword123!',
} as const;

// ---------------------------------------------------------------------------
// Data Factories
// ---------------------------------------------------------------------------
let _seq = 0;
const seq = () => ++_seq;

/** Generate a unique test user payload (no faker dependency). */
export function createUserData(overrides: Record<string, string> = {}) {
  const id = `${Date.now()}-${seq()}`;
  return {
    email: `e2e-auto-${id}@example.com`,
    username: `e2eauto${id}`,
    password: 'TestPassword123!',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Helpers — deterministic, no hard waits
// ---------------------------------------------------------------------------

/** Login with given credentials, waits for navigation away from /login. */
export async function login(page: Page, email: string, password: string) {
  await page.goto('/login');
  await page.getByPlaceholder(/email|@.*\.com/i).fill(email);
  await page.locator('input[type="password"]').fill(password);
  await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15_000 });
}

/** Login with the pre-seeded standard user. */
export async function loginAsUser(page: Page) {
  await login(page, SEEDED_USER.email, SEEDED_USER.password);
}

/** Login with the pre-seeded admin user. */
export async function loginAsAdmin(page: Page) {
  await login(page, ADMIN_USER.email, ADMIN_USER.password);
}

/** Dismiss the driver.js onboarding tour overlay if present. */
export async function dismissTourOverlay(page: Page) {
  const overlay = page.locator('.driver-overlay, .driver-popover');
  const count = await overlay.count();
  if (count > 0) {
    const closeBtn = page.locator(
      '.driver-popover-close-btn, button:has-text("Skip"), button:has-text("Chiudi")',
    );
    const closeBtnCount = await closeBtn.count();
    if (closeBtnCount > 0) {
      await closeBtn.first().click({ force: true });
    } else {
      await page.keyboard.press('Escape');
    }
    // Wait for overlay to detach
    await overlay.first().waitFor({ state: 'detached', timeout: 3_000 }).catch(() => { });
  }
}

/** Navigate and wait for network activity to settle. */
export async function navigateTo(page: Page, path: string) {
  await page.goto(path);
  await page.waitForLoadState('domcontentloaded');
}

// ---------------------------------------------------------------------------
// Extended Fixtures with auto-cleanup
// ---------------------------------------------------------------------------
type TestFixtures = {
  /** A Page already authenticated as the seeded standard user. */
  authenticatedPage: Page;
  /** A Page already authenticated as the seeded admin user. */
  adminPage: Page;
};

export const test = base.extend<TestFixtures>({
  authenticatedPage: async ({ page }, use) => {
    // Setup: login
    await loginAsUser(page);
    await dismissTourOverlay(page);

    // Provide to test
    await use(page);

    // Cleanup: clear storage so next test starts fresh
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  },

  adminPage: async ({ page }, use) => {
    // Setup: login as admin
    await loginAsAdmin(page);
    await dismissTourOverlay(page);

    // Provide to test
    await use(page);

    // Cleanup
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  },
});

export { expect };
