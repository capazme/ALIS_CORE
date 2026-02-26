/**
 * E2E Tests: Profile & Settings Management
 * ==========================================
 *
 * [P1] Tests for the settings page: viewing profile info, updating
 * account settings, and verifying consent/GDPR sections are present.
 *
 * Uses `page.route()` to mock backend API responses for deterministic tests.
 * Selectors derived from SettingsPage.tsx, ProfileSelector.tsx,
 * ConsentSelector.tsx, AuthorityScoreDisplay.tsx, and PrivacySettings.tsx.
 */
import { test, expect, login, dismissTourOverlay } from './fixtures';

// ---------------------------------------------------------------------------
// Shared mock setup for an authenticated session
// ---------------------------------------------------------------------------

async function setupAuthMocks(page: import('@playwright/test').Page) {
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
        email: 'mario@example.com',
        username: 'mario_rossi',
        name: 'Mario Rossi',
        is_admin: false,
        is_merlt_enabled: true,
        profile_type: 'assisted_research',
        authority_score: 42,
      }),
    }),
  );

  await page.route('**/api/profile', (route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          username: 'mario_rossi',
          email: 'mario@example.com',
          profile_type: 'assisted_research',
          authority_score: 42,
          preferences: {
            theme: 'light',
            language: 'it',
            notifications_enabled: true,
          },
        }),
      });
    }
    // PUT for update
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ message: 'Profile updated' }),
    });
  });

  await page.route('**/api/profile/account', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ message: 'Account updated' }),
    }),
  );

  await page.route('**/api/profile/preferences', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        preferences: {
          theme: 'dark',
          language: 'it',
          notifications_enabled: true,
        },
      }),
    }),
  );

  await page.route('**/api/consent', (route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          consent_level: 'basic',
          updated_at: '2025-01-01T00:00:00Z',
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ consent_level: 'learning' }),
    });
  });

  await page.route('**/api/consent/history', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    }),
  );

  await page.route('**/api/authority**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        score: 42,
        level: 'contributor',
        breakdown: { feedback_count: 10, accuracy: 0.85 },
      }),
    }),
  );

  // Catch-all for other API calls that happen after login
  await page.route('**/api/folders**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
  );
  await page.route('**/api/bookmarks**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
  );
  await page.route('**/api/history**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Profile & Settings Page', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuthMocks(page);
    await login(page, 'mario@example.com', 'TestPass123!');
    await dismissTourOverlay(page);
  });

  test('[P1] should display the settings page with all sections', async ({
    page,
  }) => {
    // Given: User is authenticated
    // When: Navigating to /settings
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // Then: The page heading and key sections are visible
    await expect(
      page.getByText('Impostazioni Account'),
    ).toBeVisible({ timeout: 10_000 });

    // Profile info section
    await expect(
      page.getByText('Informazioni Profilo'),
    ).toBeVisible();

    // Password section
    await expect(page.getByText('Cambia Password')).toBeVisible();

    // Preferences section
    await expect(page.getByText('Preferenze')).toBeVisible();
  });

  test('[P1] should display current profile information', async ({
    page,
  }) => {
    // Given: User navigates to settings
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: The profile section loads

    // Then: The username and email inputs have the current values
    const usernameInput = page.locator('input[placeholder="Il tuo nome utente"]');
    await expect(usernameInput).toHaveValue('mario_rossi');

    const emailInput = page.locator('input[placeholder="email@esempio.com"]');
    await expect(emailInput).toHaveValue('mario@example.com');
  });

  test('[P1] should show profile type selector', async ({ page }) => {
    // Given: User is on the settings page
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: Scrolling to the profile type section

    // Then: The profile type heading and options are visible
    await expect(
      page.getByText('Profilo di Ricerca'),
    ).toBeVisible();

    // Should show profile type options (from ProfileSelector component)
    const hasProfileOptions =
      (await page
        .getByText(
          /ricerca.*assistita|quick|expert|assisted|studente|professionista/i,
        )
        .count()) > 0;
    expect(hasProfileOptions).toBeTruthy();
  });

  test('[P1] should show consent settings section', async ({ page }) => {
    // Given: User is on the settings page
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: The consent section loads

    // Then: GDPR consent heading and options are visible
    await expect(
      page.getByText('Consenso Dati (GDPR)'),
    ).toBeVisible();

    const hasConsentOptions =
      (await page
        .getByText(
          /base|basic|learning|apprendimento|research|ricerca/i,
        )
        .count()) > 0;
    expect(hasConsentOptions).toBeTruthy();
  });

  test('[P1] should show authority score display', async ({ page }) => {
    // Given: User is on the settings page
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: The authority score section loads

    // Then: The authority score heading is visible
    await expect(
      page.getByText(/punteggio.*autorit/i),
    ).toBeVisible();
  });

  test('[P1] should show privacy settings with export and delete options', async ({
    page,
  }) => {
    // Given: User is on the settings page
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: The privacy section loads

    // Then: Privacy heading and action buttons are visible
    await expect(
      page.getByText('Privacy e Dati Personali'),
    ).toBeVisible();

    // Export data functionality
    const hasExportButton =
      (await page
        .getByRole('button', { name: /scarica|download|esporta/i })
        .count()) > 0;
    expect(hasExportButton).toBeTruthy();

    // Delete account functionality
    const hasDeleteButton =
      (await page
        .getByRole('button', { name: /elimina|delete|cancella/i })
        .count()) > 0;
    expect(hasDeleteButton).toBeTruthy();
  });

  test('[P1] should save profile changes', async ({ page }) => {
    // Given: User is on the settings page and mocked update succeeds
    let updateCalled = false;
    await page.route('**/api/profile/account', (route) => {
      updateCalled = true;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Account updated' }),
      });
    });

    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: User modifies the username and clicks save
    const usernameInput = page.locator('input[placeholder="Il tuo nome utente"]');
    await usernameInput.clear();
    await usernameInput.fill('mario_updated');

    // Click the "Salva modifiche" button in the profile section
    const saveButtons = page.getByRole('button', {
      name: /salva modifiche/i,
    });
    await saveButtons.first().click();

    // Then: The save API was called and a success message appears
    await expect(
      page.getByText(/salvate.*successo|modifiche.*salvate/i),
    ).toBeVisible({ timeout: 5_000 });
    expect(updateCalled).toBeTruthy();
  });

  test('[P1] should show preferences with theme and language options', async ({
    page,
  }) => {
    // Given: User is on the settings page
    await page.goto('/settings');
    await page.waitForLoadState('domcontentloaded');

    // When: Looking at the preferences section

    // Then: Theme buttons (Chiaro, Scuro, Sistema) and language options are visible
    await expect(page.getByText('Preferenze')).toBeVisible();
    await expect(
      page.getByRole('button', { name: /chiaro/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('button', { name: /scuro/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('button', { name: /sistema/i }),
    ).toBeVisible();

    // Language options
    await expect(page.getByText(/italiano/i)).toBeVisible();
  });
});
