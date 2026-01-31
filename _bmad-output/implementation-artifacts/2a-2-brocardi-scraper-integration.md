# Story 2a-2: Brocardi Scraper Integration

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: in-progress
- **Priority**: High

## Context
Integrate the Brocardi scraper into the ALIS_CORE pipeline. Brocardi provides legal commentary (massime, spiegazioni) which enriches the raw norms. We need to verify the existing scraper and potentially improve the data structure based on the current website layout.

## Requirements
- [ ] **Analysis**: Analyze live Brocardi pages to identify optimal information structure.
- [ ] **Verification**: Verify/Refactor existing `BrocardiScraper` in `visualex-api`.
- [ ] **Data Model**: Ensure extracted data (explanation, maxims, ratios) matches the `NormaVisitata` or related models.
- [ ] **Testing**: Update tests to reflect the current site structure.
