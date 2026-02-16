/**
 * E2E Tests: Workspace Features (Dossiers, History, Environments)
 * =================================================================
 *
 * Covers:
 * - Dossier creation and management
 * - Search history
 * - Shared environments
 *
 * Priority Tags: [P0] Critical  [P1] High  [P2] Medium
 */
import { test, expect, dismissTourOverlay } from './fixtures';

test.describe('Dossier Management', () => {
    test('[P1] should display the dossier page', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        // WHEN: User navigates to the dossier page
        await page.goto('/dossier');
        await dismissTourOverlay(page);

        // THEN: The dossier page loads without errors
        await page.waitForLoadState('domcontentloaded');
        const heading = page.getByRole('heading').first();
        await expect(heading).toBeVisible({ timeout: 10_000 });
    });

    test('[P1] should show create dossier UI or empty state', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the dossier page
        await page.goto('/dossier');
        await dismissTourOverlay(page);

        // WHEN: Page loads

        // THEN: Either a create button or an empty-state message should be visible
        const hasCreateBtn = await page.getByRole('button', { name: /crea|nuovo|new|aggiungi|create/i }).count() > 0;
        const hasEmptyState = await page.getByText(/nessun|vuoto|empty|inizia|crea.*primo/i).count() > 0;
        const hasDossierList = await page.locator('[class*="dossier"], [class*="card"], [class*="list"]').count() > 0;

        expect(hasCreateBtn || hasEmptyState || hasDossierList).toBeTruthy();
    });
});

test.describe('History View', () => {
    test('[P1] should display the history page', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        // WHEN: User navigates to the history page
        await page.goto('/history');
        await dismissTourOverlay(page);

        // THEN: The history page loads without errors
        await page.waitForLoadState('domcontentloaded');
        const heading = page.getByRole('heading').first();
        await expect(heading).toBeVisible({ timeout: 10_000 });
    });

    test('[P2] should show history entries or empty state', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the history page
        await page.goto('/history');
        await dismissTourOverlay(page);

        // WHEN: The page loads

        // THEN: Either history entries or an empty-state message should be present
        const contentArea = page.locator('main, [role="main"], .content, #root');
        await expect(contentArea.first()).toBeVisible();
    });
});

test.describe('Shared Environments', () => {
    test('[P1] should display the environments page', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        // WHEN: User navigates to the environments page
        await page.goto('/environments');
        await dismissTourOverlay(page);

        // THEN: The page loads without errors
        await page.waitForLoadState('domcontentloaded');
        const heading = page.getByRole('heading').first();
        await expect(heading).toBeVisible({ timeout: 10_000 });
    });

    test('[P2] should show environment list or creation interface', async ({ authenticatedPage: page }) => {
        // GIVEN: User is on the environments page
        await page.goto('/environments');
        await dismissTourOverlay(page);

        // WHEN: The page loads

        // THEN: Should show environments or a way to create one
        const hasContent = await page.locator(
            '[class*="environment"], [class*="card"], [class*="list"], button, [class*="empty"]',
        ).count();
        expect(hasContent).toBeGreaterThan(0);
    });
});

test.describe('Bulletin Board', () => {
    test('[P2] should display the bulletin board page', async ({ authenticatedPage: page }) => {
        // GIVEN: User is authenticated
        // WHEN: User navigates to the bulletin page
        await page.goto('/bulletin');
        await dismissTourOverlay(page);

        // THEN: The page loads
        await page.waitForLoadState('domcontentloaded');
        const heading = page.getByRole('heading').first();
        await expect(heading).toBeVisible({ timeout: 10_000 });
    });
});
