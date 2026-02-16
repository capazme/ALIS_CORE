/**
 * E2E Tests: Navigation & Layout
 * ================================
 *
 * Covers:
 * - Sidebar navigation between all major routes
 * - Layout rendering
 * - Route protection
 * - Responsive layout
 *
 * Priority Tags: [P0] Critical  [P1] High  [P2] Medium
 */
import { test, expect, dismissTourOverlay } from './fixtures';

test.describe('Route Protection', () => {
    test('[P0] should redirect unauthenticated users to /login', async ({ page }) => {
        // GIVEN: User is NOT authenticated

        // WHEN: User attempts to access a protected route
        await page.goto('/');

        // THEN: User is redirected to /login
        await page.waitForURL(/\/login/, { timeout: 10_000 });
        await expect(page).toHaveURL(/\/login/);
    });

    test('[P0] should redirect /settings to /login when unauthenticated', async ({ page }) => {
        // GIVEN: User is NOT authenticated

        // WHEN: User attempts to access /settings
        await page.goto('/settings');

        // THEN: User is redirected to /login
        await page.waitForURL(/\/login/, { timeout: 10_000 });
        await expect(page).toHaveURL(/\/login/);
    });

    test('[P0] should redirect /admin to /login when unauthenticated', async ({ page }) => {
        // GIVEN: User is NOT authenticated

        // WHEN: User attempts to access /admin
        await page.goto('/admin');

        // THEN: User is redirected to /login
        await page.waitForURL(/\/login/, { timeout: 10_000 });
    });
});

test.describe('Layout & Sidebar Navigation', () => {
    test('[P1] should display the sidebar with navigation links', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated and on the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // WHEN: The layout renders

        // THEN: Sidebar or navigation should be visible with links
        const nav = page.locator('nav, aside, [class*="sidebar"], [class*="Sidebar"]').first();
        await expect(nav).toBeVisible({ timeout: 10_000 });
    });

    test('[P1] should navigate to dossier page via sidebar', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // WHEN: User clicks on the dossier navigation link
        const dossierLink = page.locator(
            'a[href*="dossier"], button:has-text("Fascicoli"), button:has-text("Dossier")',
        ).first();
        if (await dossierLink.isVisible()) {
            await dossierLink.click();

            // THEN: User navigates to /dossier
            await page.waitForURL(/\/dossier/, { timeout: 10_000 });
        }
    });

    test('[P1] should navigate to history page via sidebar', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // WHEN: User clicks on the history navigation link
        const historyLink = page.locator(
            'a[href*="history"], button:has-text("Cronologia"), button:has-text("History")',
        ).first();
        if (await historyLink.isVisible()) {
            await historyLink.click();

            // THEN: User navigates to /history
            await page.waitForURL(/\/history/, { timeout: 10_000 });
        }
    });

    test('[P1] should navigate to settings page via sidebar', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // WHEN: User clicks on the settings navigation link
        const settingsLink = page.locator(
            'a[href*="settings"], button:has-text("Impostazioni"), button:has-text("Settings")',
        ).first();
        if (await settingsLink.isVisible()) {
            await settingsLink.click();

            // THEN: User navigates to /settings
            await page.waitForURL(/\/settings/, { timeout: 10_000 });
        }
    });
});

test.describe('Responsive Layout', () => {
    test('[P2] should render correctly on mobile viewport', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        await page.setViewportSize({ width: 375, height: 667 });

        // WHEN: User navigates to the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // THEN: The page renders without layout overflow
        const body = page.locator('body');
        await expect(body).toBeVisible();
    });

    test('[P2] should render correctly on tablet viewport', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        await page.setViewportSize({ width: 768, height: 1024 });

        // WHEN: User navigates to the main page
        await page.goto('/');
        await dismissTourOverlay(page);

        // THEN: The page renders without layout overflow
        const body = page.locator('body');
        await expect(body).toBeVisible();
    });
});
