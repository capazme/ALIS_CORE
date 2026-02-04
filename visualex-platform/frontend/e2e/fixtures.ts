/**
 * E2E Test Fixtures
 * ==================
 *
 * Shared test data and helper functions for E2E tests.
 */
import { test as base, expect } from '@playwright/test';

// Test user data
export const testUser = {
  email: `e2e-test-${Date.now()}@example.com`,
  username: `e2euser${Date.now()}`,
  password: 'TestPassword123!',
};

// Extend base test with custom fixtures
export const test = base.extend<{
  registeredUser: { email: string; username: string; password: string };
}>({
  // Fixture that registers a new user and provides credentials
  registeredUser: async ({ page }, use) => {
    const user = {
      email: `e2e-${Date.now()}@example.com`,
      username: `e2e${Date.now()}`,
      password: 'TestPassword123!',
    };

    // Register the user
    await page.goto('/register');
    await page.getByLabel('Email').fill(user.email);
    await page.getByLabel('Username').fill(user.username);
    await page.getByLabel('Password', { exact: true }).fill(user.password);
    await page.getByLabel('Conferma Password').fill(user.password);
    await page.getByRole('button', { name: /registrati/i }).click();

    // Wait for redirect to login or dashboard
    await page.waitForURL(/\/(login|dashboard|settings)/);

    await use(user);
  },
});

export { expect };

/**
 * Helper: Login with credentials
 */
export async function login(
  page: import('@playwright/test').Page,
  email: string,
  password: string
) {
  await page.goto('/login');
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: /accedi/i }).click();
  // Wait for navigation away from login
  await page.waitForURL((url) => !url.pathname.includes('/login'));
}

/**
 * Helper: Logout
 */
export async function logout(page: import('@playwright/test').Page) {
  // Click user menu or logout button
  const logoutButton = page.getByRole('button', { name: /logout|esci/i });
  if (await logoutButton.isVisible()) {
    await logoutButton.click();
  }
}

/**
 * Helper: Navigate to settings
 */
export async function goToSettings(page: import('@playwright/test').Page) {
  await page.goto('/settings');
  await page.waitForLoadState('networkidle');
}

/**
 * Helper: Wait for toast/notification
 */
export async function waitForToast(
  page: import('@playwright/test').Page,
  text: string | RegExp
) {
  await expect(page.getByText(text)).toBeVisible({ timeout: 10000 });
}
