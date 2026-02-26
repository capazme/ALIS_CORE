/**
 * E2E Tests: Search Functionality
 * ================================
 *
 * [P1] Tests for the search page: searching by norm type, viewing results,
 * and handling empty/error states.
 *
 * Uses `page.route()` to mock backend streaming responses for determinism.
 * Selectors derived from SearchForm.tsx, SearchPanel.tsx, and CommandPalette.tsx.
 */
import { test, expect, dismissTourOverlay } from './fixtures';

// ---------------------------------------------------------------------------
// Helpers: mock data for streaming search responses
// ---------------------------------------------------------------------------

/** Build a mock NDJSON streaming response for /stream_article_text */
function mockArticleStream(articles: Array<{ numero_articolo: string; testo: string }>) {
  const lines = articles.map((a) =>
    JSON.stringify({
      norma_data: {
        tipo_atto: 'codice civile',
        data: '',
        numero_atto: '',
        numero_articolo: a.numero_articolo,
        tipo_atto_reale: 'Codice Civile',
        url: 'urn:nir:stato:codice.civile:1942-03-16',
      },
      testo: a.testo,
    }),
  );
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Search Functionality', () => {
  test('[P1] should display the search form on the main page', async ({
    authenticatedPage: page,
  }) => {
    // Given: User is authenticated and navigated to the home page
    await page.goto('/');
    await dismissTourOverlay(page);

    // When: The page loads fully

    // Then: The search form with act-type select and article input are visible
    await expect(page.locator('#search-act-type')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('#article')).toBeVisible();
    await expect(
      page.getByRole('button', { name: /estrai contenuto/i }),
    ).toBeVisible();
  });

  test('[P1] should search by norm type and see results', async ({
    authenticatedPage: page,
  }) => {
    // Given: The streaming search endpoint returns a mocked article
    await page.route('**/stream_article_text', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/x-ndjson',
        body: mockArticleStream([
          {
            numero_articolo: '1',
            testo: 'Ogni persona ha la capacita\' giuridica dal momento della nascita.',
          },
        ]),
      }),
    );

    await page.goto('/');
    await dismissTourOverlay(page);

    // When: User selects "Codice Civile" and submits the search
    await page.locator('#search-act-type').selectOption('codice civile');
    await page.locator('#article').clear();
    await page.locator('#article').fill('1');
    await page.getByRole('button', { name: /estrai contenuto/i }).click();

    // Then: A result card appears with article text
    const resultArea = page.locator('[class*="card"], [class*="norma"], article').first();
    await expect(resultArea).toBeVisible({ timeout: 15_000 });

    // The result should contain recognizable article text
    const pageText = await page.textContent('body');
    expect(pageText).toContain('capacita');
  });

  test('[P1] should display article content from search results', async ({
    authenticatedPage: page,
  }) => {
    // Given: The streaming endpoint returns multiple articles
    await page.route('**/stream_article_text', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/x-ndjson',
        body: mockArticleStream([
          { numero_articolo: '1', testo: 'Articolo 1 - Capacita giuridica.' },
          { numero_articolo: '2', testo: 'Articolo 2 - Maggiore eta.' },
        ]),
      }),
    );

    await page.goto('/');
    await dismissTourOverlay(page);

    // When: User searches for a range of articles
    await page.locator('#search-act-type').selectOption('codice civile');
    await page.locator('#article').clear();
    await page.locator('#article').fill('1-2');
    await page.getByRole('button', { name: /estrai contenuto/i }).click();

    // Then: Both articles should appear in the workspace
    await expect(page.getByText(/Capacita giuridica/)).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText(/Maggiore eta/)).toBeVisible({
      timeout: 10_000,
    });
  });

  test('[P1] should show empty state when no search has been performed', async ({
    authenticatedPage: page,
  }) => {
    // Given: User is on the main page without having searched
    await page.goto('/');
    await dismissTourOverlay(page);

    // When: The page loads

    // Then: The empty state with "Ricerca Intelligente" heading is visible
    await expect(
      page.getByText('Ricerca Intelligente'),
    ).toBeVisible({ timeout: 10_000 });
  });

  test('[P1] should handle search errors gracefully', async ({
    authenticatedPage: page,
  }) => {
    // Given: The streaming endpoint returns a server error
    await page.route('**/stream_article_text', (route) =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      }),
    );

    await page.goto('/');
    await dismissTourOverlay(page);

    // When: User submits a search
    await page.locator('#search-act-type').selectOption('codice civile');
    await page.getByRole('button', { name: /estrai contenuto/i }).click();

    // Then: An error message is displayed (not a crash)
    await expect(
      page.getByText(/errore/i),
    ).toBeVisible({ timeout: 10_000 });
  });

  test('[P2] should open command palette with keyboard shortcut', async ({
    authenticatedPage: page,
  }) => {
    // Given: User is on the main page
    await page.goto('/');
    await dismissTourOverlay(page);

    // When: User presses Cmd+K (Meta+K)
    await page.keyboard.press('Meta+k');

    // Then: The command palette overlay appears
    // CommandPalette renders as a modal/dialog
    const palette = page.locator('[class*="CommandPalette"], [role="dialog"]').first();
    await expect(palette).toBeVisible({ timeout: 5_000 });
  });

  test('[P2] should reset the search form', async ({
    authenticatedPage: page,
  }) => {
    // Given: User has selected values in the search form
    await page.goto('/');
    await dismissTourOverlay(page);

    await page.locator('#search-act-type').selectOption('codice civile');
    await page.locator('#article').clear();
    await page.locator('#article').fill('42');

    // When: User clicks the Reset button
    await page
      .getByRole('button', { name: /reset/i })
      .click();

    // Then: The form is cleared
    await expect(page.locator('#search-act-type')).toHaveValue('');
    await expect(page.locator('#article')).toHaveValue('1');
  });
});
