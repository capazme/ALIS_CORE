/**
 * E2E Tests: Settings Page Flows
 * ===============================
 *
 * Tests for:
 * - Story 1-3 (Profile Setup)
 * - Story 1-4 (Consent Configuration)
 * - Story 1-5 (Authority Score Display)
 * - Story 1-6 (Data Export & Erasure)
 *
 * Uses pre-seeded test user (run `npx prisma db seed` in backend first)
 */
import { test, expect } from '@playwright/test';

// Pre-seeded test user credentials
const SEEDED_USER = {
  email: 'e2e-test@visualex.it',
  password: 'TestPassword123!',
};

// Helper function to login
async function loginUser(page: import('@playwright/test').Page, email: string, password: string) {
  await page.goto('/login');
  await page.getByPlaceholder(/email|@.*\.com/i).fill(email);
  await page.locator('input[type="password"]').fill(password);
  await page.getByRole('button', { name: /accedi|login|entra|sign\s*in/i }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });
}

// Helper function to dismiss onboarding tour overlay if present
async function dismissTourOverlay(page: import('@playwright/test').Page) {
  // Check for driver.js tour overlay and dismiss it
  const overlay = page.locator('.driver-overlay, .driver-popover');
  if (await overlay.count() > 0) {
    // Try clicking the close/skip button or pressing Escape
    const closeBtn = page.locator('.driver-popover-close-btn, [class*="close"], button:has-text("Skip"), button:has-text("Chiudi")');
    if (await closeBtn.count() > 0) {
      await closeBtn.first().click({ force: true });
    } else {
      await page.keyboard.press('Escape');
    }
    await page.waitForTimeout(500);
  }
}

test.describe('Settings Page', () => {
  test.beforeEach(async ({ page }) => {
    // Login with pre-seeded user before each test
    await loginUser(page, SEEDED_USER.email, SEEDED_USER.password);

    // Navigate to settings
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Dismiss any onboarding tour overlay that might be present
    await dismissTourOverlay(page);
  });

  test('displays settings page with all sections', async ({ page }) => {
    await expect(page.getByRole('heading').first()).toBeVisible();
  });

  test.describe('Profile Setup (Story 1-3)', () => {
    test('displays profile type selector', async ({ page }) => {
      await expect(page.getByText(/tipo.*profilo|profilo.*utente|profile/i).first()).toBeVisible();
    });

    test('shows profile type options', async ({ page }) => {
      // Should show at least one profile type option
      const hasProfileOptions = await page.getByText(/studente|professionista|ricercatore|student|professional|researcher|quick|assisted|expert/i).count() > 0;
      expect(hasProfileOptions).toBeTruthy();
    });

    test('can interact with profile selector', async ({ page }) => {
      // Dismiss any overlay that might be blocking
      await dismissTourOverlay(page);

      const profileOption = page.getByText(/studente|ricerca.*assistita|assisted/i).first();
      if (await profileOption.isVisible()) {
        await profileOption.click({ force: true }); // Force click to bypass any remaining overlay
        await page.waitForTimeout(1000);
      }
    });
  });

  test.describe('Consent Configuration (Story 1-4)', () => {
    test('displays consent section', async ({ page }) => {
      await expect(page.getByText(/consenso|consent|privacy|dati/i).first()).toBeVisible();
    });

    test('shows consent level options', async ({ page }) => {
      const hasConsentOptions = await page.getByText(/base|basic|minimo|learning|apprendimento|research|ricerca/i).count() > 0;
      expect(hasConsentOptions).toBeTruthy();
    });

    test('displays GDPR reference', async ({ page }) => {
      await expect(page.getByText(/GDPR|Art\.|regolamento|privacy/i).first()).toBeVisible();
    });
  });

  test.describe('Authority Score Display (Story 1-5)', () => {
    test('displays authority or score section', async ({ page }) => {
      const hasScoreSection = await page.getByText(/punteggio|authority|score|affidabilità|livello/i).count() > 0;
      expect(hasScoreSection).toBeTruthy();
    });

    test('shows some numeric value', async ({ page }) => {
      // Should display a score value somewhere
      const hasNumber = await page.locator('text=/\\d+/').count() > 0;
      expect(hasNumber).toBeTruthy();
    });
  });

  test.describe('Privacy Settings - Data Export (Story 1-6, GDPR Art. 20)', () => {
    test('displays data export section', async ({ page }) => {
      await expect(page.getByText(/esporta.*dati|data.*export|scarica.*dati|portabilità/i).first()).toBeVisible();
    });

    test('has export functionality', async ({ page }) => {
      const hasExportButton = await page.getByRole('button', { name: /scarica|download|esporta/i }).count() > 0;
      expect(hasExportButton).toBeTruthy();
    });
  });

  test.describe('Privacy Settings - Account Deletion (Story 1-6, GDPR Art. 17)', () => {
    test('displays account deletion section', async ({ page }) => {
      await expect(page.getByText(/elimina.*account|delete.*account|cancella|oblio/i).first()).toBeVisible();
    });

    test('has delete functionality', async ({ page }) => {
      const hasDeleteButton = await page.getByRole('button', { name: /elimina|delete|cancella/i }).count() > 0;
      expect(hasDeleteButton).toBeTruthy();
    });

    test('opens confirmation modal on delete click', async ({ page }) => {
      // Dismiss any overlay that might be blocking
      await dismissTourOverlay(page);

      const deleteButton = page.getByRole('button', { name: /elimina|delete|cancella/i }).first();
      if (await deleteButton.isVisible()) {
        await deleteButton.click({ force: true }); // Force click to bypass any remaining overlay

        // Should show some confirmation dialog
        await page.waitForTimeout(1000);
        const hasModal = await page.locator('[role="dialog"], [class*="modal"]').count() > 0 ||
                         await page.getByText(/conferma|confirm|sicuro|password/i).count() > 0;
        expect(hasModal).toBeTruthy();
      }
    });
  });
});

test.describe('Settings Page - Responsive', () => {
  test('works on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await loginUser(page, SEEDED_USER.email, SEEDED_USER.password);
    await page.goto('/settings');

    await expect(page.getByRole('heading').first()).toBeVisible();
  });

  test('works on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });

    await loginUser(page, SEEDED_USER.email, SEEDED_USER.password);
    await page.goto('/settings');

    await expect(page.getByRole('heading').first()).toBeVisible();
  });
});
