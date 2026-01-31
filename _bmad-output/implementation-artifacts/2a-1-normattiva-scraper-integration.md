# Story 2a-1: Normattiva Scraper Integration

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: in-progress
- **Priority**: High

## Context
Integrate the Normattiva scraper into the ALIS_CORE pipeline. The goal is to ensure reliable fetching and parsing of Italian laws from Normattiva, handling URNs correctly.

## Requirements
- [x] **Verification**: Verify the existing `NormattivaScraper` implementation.
- [x] **Integration**: Ensure the scraper functions within the new architecture.
- [x] **Robustness**: Implement error handling and caching (already present, need verification).
- [x] **Testing**: Add or update tests to validate scraper output.

## Technical Details
- **Source**: `visualex-api/visualex/scrapers/normattiva.py`
- **Dependencies**: `aiocache`, `bs4`, `structlog`
- **Output**: Structured text/metadata from Normattiva HTML.
