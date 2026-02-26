/**
 * E2E Smoke Tests
 * ===============
 *
 * [P0] Quick sanity checks to verify the app is alive and rendering.
 * These tests run first and fast — if they fail, nothing else matters.
 *
 * No authentication required for public routes.
 */
import { test, expect } from '@playwright/test';

test.describe('Smoke Tests', () => {
  test('[P0] should load the login page successfully', async ({ page }) => {
    // Given: A fresh browser with no session
    // When: Navigating to the login route
    await page.goto('/login');

    // Then: The page renders the VisuaLex heading and login form
    await expect(page.getByText('VisuaLex')).toBeVisible();
    await expect(page.getByPlaceholder('name@company.com')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(
      page.getByRole('button', { name: /sign\s*in/i }),
    ).toBeVisible();
  });

  test('[P0] should load the register page', async ({ page }) => {
    // Given: A fresh browser with no session
    // When: Navigating to the register route
    await page.goto('/register');

    // Then: The page renders the VisuaLex heading
    await expect(page.getByText('VisuaLex')).toBeVisible();
  });

  test('[P0] should redirect unauthenticated users from / to /login', async ({
    page,
  }) => {
    // Given: No stored auth tokens
    // When: Navigating to the root (protected route)
    await page.goto('/');

    // Then: The ProtectedRoute redirects to /login
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    await expect(page).toHaveURL(/\/login/);
  });

  test('[P0] should respond to the health check endpoint', async ({
    request,
  }) => {
    // Given: The backend is running on port 3001
    // When: Hitting the health endpoint
    const response = await request.get('http://localhost:3001/api/health');

    // Then: It returns 200 with status "ok"
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe('ok');
    expect(body).toHaveProperty('timestamp');
  });

  test('[P0] should serve the frontend without errors', async ({ page }) => {
    // Given: A fresh browser
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    // When: Loading the login page
    await page.goto('/login');
    await page.waitForLoadState('domcontentloaded');

    // Then: No fatal JS errors in the console
    // (filter out expected network errors like failed API calls without auth)
    const fatalErrors = consoleErrors.filter(
      (e) => !e.includes('401') && !e.includes('403') && !e.includes('net::'),
    );
    expect(fatalErrors).toHaveLength(0);
  });
});
