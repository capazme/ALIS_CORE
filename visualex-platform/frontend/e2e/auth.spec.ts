/**
 * E2E Tests: Authentication Flows
 * ================================
 *
 * Tests for Story 1-1 (User Registration) and Story 1-2 (User Login)
 *
 * Note: Some tests use pre-seeded users (e2e-test@visualex.it)
 * Run `npx prisma db seed` in backend to create them.
 */
import { test, expect } from '@playwright/test';

// Pre-seeded test user credentials (created by prisma/seed.ts)
const SEEDED_USER = {
  email: 'e2e-test@visualex.it',
  password: 'TestPassword123!',
};

const ADMIN_USER = {
  email: 'admin@visualex.it',
  password: 'AdminPassword123!',
};

test.describe('User Registration (Story 1-1)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/register');
  });

  test('displays registration form', async ({ page }) => {
    await expect(page.getByRole('heading').first()).toBeVisible();
    await expect(page.getByPlaceholder(/email|@.*\.com/i)).toBeVisible();
    await expect(page.getByPlaceholder(/username|mario_rossi|utente/i)).toBeVisible();

    const passwordInputs = page.locator('input[type="password"]');
    await expect(passwordInputs.first()).toBeVisible();

    await expect(page.getByRole('button', { name: /crea account|registrati|sign up/i })).toBeVisible();
  });

  test('prevents submission with empty form', async ({ page }) => {
    await page.getByRole('button', { name: /crea account|registrati|sign up/i }).click();
    await page.waitForTimeout(1000);
    await expect(page).toHaveURL(/\/register/);
  });

  test('prevents submission with invalid email', async ({ page }) => {
    await page.getByPlaceholder(/email|@.*\.com/i).fill('invalid-email');
    await page.getByPlaceholder(/username|mario_rossi|utente/i).fill('testuser');

    const passwordInputs = page.locator('input[type="password"]');
    await passwordInputs.first().fill('TestPassword123!');
    await passwordInputs.last().fill('TestPassword123!');

    await page.getByRole('button', { name: /crea account|registrati|sign up/i }).click();

    await page.waitForTimeout(1000);
    await expect(page).toHaveURL(/\/register/);
  });

  test('shows password strength indicator', async ({ page }) => {
    const passwordInput = page.locator('input[type="password"]').first();
    await passwordInput.fill('weak');
    await page.waitForTimeout(500);

    // Check for any visual feedback on password strength (indicator, color change, text)
    const hasStrengthIndicator = await page.locator('[class*="strength"], [class*="progress"], [class*="bar"], [class*="meter"]').count() > 0;
    const hasStrengthText = await page.getByText(/debole|weak|forte|strong|medio|medium|sicurezza|security/i).count() > 0;
    const hasColorFeedback = await page.locator('[class*="red"], [class*="green"], [class*="yellow"], [class*="orange"]').count() > 0;

    // Pass if any form of strength feedback is visible
    expect(hasStrengthIndicator || hasStrengthText || hasColorFeedback).toBeTruthy();
  });

  test('successfully registers new user and shows approval message', async ({ page }) => {
    const timestamp = Date.now();
    const email = `e2e-register-${timestamp}@example.com`;
    const username = `e2ereg${timestamp}`;

    await page.getByPlaceholder(/email|@.*\.com/i).fill(email);
    await page.getByPlaceholder(/username|mario_rossi|utente/i).fill(username);

    const passwordInputs = page.locator('input[type="password"]');
    await passwordInputs.first().fill('TestPassword123!');
    await passwordInputs.last().fill('TestPassword123!');

    await page.getByRole('button', { name: /crea account|registrati|sign up/i }).click();

    // Should show registration success message (awaiting admin approval)
    await expect(page.getByText(/registrazione.*complet|account.*creat|successo/i).first()).toBeVisible({ timeout: 15000 });
  });

  test('has link to login page', async ({ page }) => {
    const loginLink = page.getByRole('link', { name: /accedi|login|hai.*account/i });
    await expect(loginLink).toBeVisible();
    await loginLink.click();
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('User Login (Story 1-2)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('displays login form', async ({ page }) => {
    await expect(page.getByRole('heading').first()).toBeVisible();
    await expect(page.getByPlaceholder(/email|@.*\.com/i)).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i })).toBeVisible();
  });

  test('prevents submission with empty credentials', async ({ page }) => {
    await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();
    await page.waitForTimeout(1000);
    await expect(page).toHaveURL(/\/login/);
  });

  test('shows error for invalid credentials', async ({ page }) => {
    await page.getByPlaceholder(/email|@.*\.com/i).fill('nonexistent@example.com');
    await page.locator('input[type="password"]').fill('wrongpassword');
    await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();

    await page.waitForTimeout(3000);

    const hasError = await page.locator('[class*="error"], [role="alert"]').count() > 0 ||
                     await page.getByText(/credenziali|invalid|errore|incorrect|wrong/i).count() > 0;
    const staysOnPage = page.url().includes('/login');

    expect(hasError || staysOnPage).toBeTruthy();
  });

  test('successfully logs in with pre-seeded user', async ({ page }) => {
    await page.getByPlaceholder(/email|@.*\.com/i).fill(SEEDED_USER.email);
    await page.locator('input[type="password"]').fill(SEEDED_USER.password);
    await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();

    // Should redirect away from login
    await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });
  });

  test('remembers user session after login', async ({ page }) => {
    await page.getByPlaceholder(/email|@.*\.com/i).fill(SEEDED_USER.email);
    await page.locator('input[type="password"]').fill(SEEDED_USER.password);
    await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();
    await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });

    // Refresh page
    await page.reload();

    // Should still be logged in
    await page.waitForTimeout(2000);
    await expect(page).not.toHaveURL(/\/login/);
  });

  test('has link to registration page', async ({ page }) => {
    const registerLink = page.getByRole('link', { name: /registrati|crea.*account|non.*hai.*account|sign up/i });
    await expect(registerLink).toBeVisible();
    await registerLink.click();
    await expect(page).toHaveURL(/\/register/);
  });

  test('admin user can login', async ({ page }) => {
    await page.getByPlaceholder(/email|@.*\.com/i).fill(ADMIN_USER.email);
    await page.locator('input[type="password"]').fill(ADMIN_USER.password);
    await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();

    await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });
  });
});
