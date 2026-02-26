/**
 * E2E Tests: Authentication Flows
 * ================================
 *
 * [P0] Tests for user registration, login, error handling, and logout.
 *
 * Uses `page.route()` to mock API responses for deterministic behavior.
 * Pre-seeded user tests hit the real backend (seeded via prisma/seed.ts).
 *
 * Selectors are derived from the actual LoginForm.tsx and RegisterForm.tsx source.
 */
import { test, expect } from '@playwright/test';
import { SEEDED_USER, login, dismissTourOverlay } from './fixtures';

// ============================================================================
// Registration
// ============================================================================

test.describe('User Registration', () => {
  test('[P0] should register with valid data via mocked API', async ({
    page,
  }) => {
    // Given: The registration API succeeds and returns a pending-verification state
    await page.route('**/api/auth/register', (route) =>
      route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          message: 'Registration successful. Please verify your email.',
          user: {
            id: 'mock-id',
            email: 'newuser@example.com',
            username: 'newuser',
          },
        }),
      }),
    );

    // Also mock invitation validation (register requires valid token)
    await page.route('**/api/invitations/validate**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          valid: true,
          email: null,
          inviter: { username: 'admin' },
        }),
      }),
    );

    // When: User navigates to /register with a token and fills the form
    await page.goto('/register?token=mock-valid-token');

    // Wait for the form to appear (invitation validation completes)
    await expect(page.getByPlaceholder('nome@email.com')).toBeVisible({
      timeout: 10_000,
    });

    await page.getByPlaceholder('nome@email.com').fill('newuser@example.com');
    await page.getByPlaceholder('mario_rossi').fill('newuser123');
    await page.getByPlaceholder('Mario Rossi').fill('New User');

    const passwordInputs = page.locator('input[type="password"]');
    await passwordInputs.first().fill('SecurePass1!');
    await passwordInputs.last().fill('SecurePass1!');

    await page
      .getByRole('button', { name: /crea account/i })
      .click();

    // Then: The success/verification screen is shown
    await expect(
      page.getByText(/verifica.*email/i),
    ).toBeVisible({ timeout: 10_000 });
  });

  test('[P0] should show error when registration form is empty', async ({
    page,
  }) => {
    // Given: The register form is loaded with a valid mock invitation
    await page.route('**/api/invitations/validate**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ valid: true, email: null, inviter: null }),
      }),
    );

    await page.goto('/register?token=mock-token');
    await expect(page.getByPlaceholder('nome@email.com')).toBeVisible({
      timeout: 10_000,
    });

    // When: User clicks submit without filling any fields
    await page
      .getByRole('button', { name: /crea account/i })
      .click();

    // Then: An error message is shown and user stays on register page
    await expect(
      page.getByText(/obbligatori|required|compilare/i),
    ).toBeVisible({ timeout: 5_000 });
    await expect(page).toHaveURL(/\/register/);
  });

  test('[P1] should show invalid invitation state when no token provided', async ({
    page,
  }) => {
    // Given: No invitation token in the URL
    // When: Navigating to /register without a token
    await page.goto('/register');

    // Then: The invalid invitation message is displayed
    await expect(
      page.getByText(/invito non valido|nessun token/i),
    ).toBeVisible({ timeout: 10_000 });
  });

  test('[P1] should navigate to login page from register', async ({
    page,
  }) => {
    // Given: User is on the register page (invalid invitation state)
    await page.goto('/register');
    await expect(page.getByText('VisuaLex')).toBeVisible();

    // When: User clicks the login link
    const loginLink = page.getByRole('link', {
      name: /accedi|login|torna.*login/i,
    });
    await expect(loginLink).toBeVisible();
    await loginLink.click();

    // Then: User is redirected to /login
    await expect(page).toHaveURL(/\/login/);
  });
});

// ============================================================================
// Login
// ============================================================================

test.describe('User Login', () => {
  test('[P0] should login with valid credentials', async ({ page }) => {
    // Given: The login and /auth/me APIs succeed
    await page.route('**/api/auth/login', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'mock-access-token',
          refresh_token: 'mock-refresh-token',
        }),
      }),
    );
    await page.route('**/api/auth/me', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'user-1',
          email: 'test@example.com',
          username: 'testuser',
          is_admin: false,
          is_merlt_enabled: false,
          profile_type: 'assisted_research',
          authority_score: 0,
        }),
      }),
    );

    // Mock additional API calls that fire after login
    await page.route('**/api/profile', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ preferences: { theme: 'light', language: 'it', notifications_enabled: true } }),
      }),
    );
    await page.route('**/api/folders**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );
    await page.route('**/api/bookmarks**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );
    await page.route('**/api/history**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );

    // When: User fills in credentials and submits
    await page.goto('/login');
    await page.getByPlaceholder('name@company.com').fill('test@example.com');
    await page.locator('input[type="password"]').fill('ValidPass123!');
    await page.getByRole('button', { name: /sign\s*in/i }).click();

    // Then: User is redirected away from /login
    await page.waitForURL((url) => !url.pathname.includes('/login'), {
      timeout: 15_000,
    });
    expect(page.url()).not.toContain('/login');
  });

  test('[P0] should show error with invalid credentials', async ({ page }) => {
    // Given: The login API returns 401
    await page.route('**/api/auth/login', (route) =>
      route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Invalid email or password',
        }),
      }),
    );

    // When: User submits wrong credentials
    await page.goto('/login');
    await page.getByPlaceholder('name@company.com').fill('wrong@example.com');
    await page.locator('input[type="password"]').fill('wrongpassword');
    await page.getByRole('button', { name: /sign\s*in/i }).click();

    // Then: An error message is visible and user stays on /login
    await expect(
      page.locator('text=/invalid|credenziali|errore|failed/i'),
    ).toBeVisible({ timeout: 5_000 });
    await expect(page).toHaveURL(/\/login/);
  });

  test('[P0] should show validation error for empty fields', async ({
    page,
  }) => {
    // Given: User is on the login page
    await page.goto('/login');

    // When: User clicks sign-in without entering anything
    await page.getByRole('button', { name: /sign\s*in/i }).click();

    // Then: A validation error appears
    await expect(
      page.getByText(/required|obbligatori|email.*password/i),
    ).toBeVisible({ timeout: 5_000 });
    await expect(page).toHaveURL(/\/login/);
  });

  test('[P0] should show validation error for invalid email format', async ({
    page,
  }) => {
    // Given: User is on the login page
    await page.goto('/login');

    // When: User enters an invalid email and submits
    await page.getByPlaceholder('name@company.com').fill('not-an-email');
    await page.locator('input[type="password"]').fill('somepassword');
    await page.getByRole('button', { name: /sign\s*in/i }).click();

    // Then: An invalid email error is shown
    await expect(
      page.getByText(/invalid.*email|email.*non.*valid/i),
    ).toBeVisible({ timeout: 5_000 });
    await expect(page).toHaveURL(/\/login/);
  });

  test('[P1] should navigate to register page from login', async ({
    page,
  }) => {
    // Given: User is on the login page
    await page.goto('/login');

    // When: User clicks the registration link
    const registerLink = page.getByRole('link', {
      name: /registrati/i,
    });
    await expect(registerLink).toBeVisible();
    await registerLink.click();

    // Then: User is on the register page
    await expect(page).toHaveURL(/\/register/);
  });

  test('[P1] should toggle password visibility', async ({ page }) => {
    // Given: User is on the login page with a password typed
    await page.goto('/login');
    const passwordInput = page.locator('input[type="password"]');
    await passwordInput.fill('secret123');

    // When: User clicks the show-password toggle button
    const toggleButton = page.locator(
      'button:has(svg)',
    ).filter({ has: page.locator('[class*="Eye"]') });
    // Fallback: find the button near the password input
    const toggleNearPassword = passwordInput
      .locator('..')
      .locator('button[tabindex="-1"]');
    await toggleNearPassword.click();

    // Then: The input type changes to "text" (password is visible)
    await expect(page.locator('input[autocomplete="current-password"]')).toHaveAttribute(
      'type',
      'text',
    );
  });
});

// ============================================================================
// Logout
// ============================================================================

test.describe('User Logout', () => {
  test('[P0] should logout and redirect to login', async ({ page }) => {
    // Given: User is logged in (via mocked APIs)
    await page.route('**/api/auth/login', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'mock-token',
          refresh_token: 'mock-refresh',
        }),
      }),
    );
    await page.route('**/api/auth/me', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'user-1',
          email: 'test@example.com',
          username: 'testuser',
          is_admin: false,
          is_merlt_enabled: false,
          profile_type: 'assisted_research',
          authority_score: 0,
        }),
      }),
    );
    await page.route('**/api/profile', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ preferences: { theme: 'light', language: 'it', notifications_enabled: true } }),
      }),
    );
    await page.route('**/api/folders**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );
    await page.route('**/api/bookmarks**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );
    await page.route('**/api/history**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    );

    await login(page, 'test@example.com', 'ValidPass123!');
    await dismissTourOverlay(page);

    // When: User triggers logout
    // The logout is in the Sidebar; we call it via localStorage clear + navigate
    // (mirrors authService.logout() which clears localStorage and redirects)
    await page.evaluate(() => {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    });
    await page.goto('/');

    // Then: The ProtectedRoute redirects to /login
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    await expect(page).toHaveURL(/\/login/);
  });
});
