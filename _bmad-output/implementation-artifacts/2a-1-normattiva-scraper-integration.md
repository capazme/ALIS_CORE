# Story 2a-1: Normattiva Scraper Integration

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: done
- **Priority**: High

## Context
Integrate the Normattiva scraper into the ALIS_CORE pipeline for fetching and parsing Italian laws. The scraper must handle URNs correctly, provide robust error handling, caching, and retry logic.

## Acceptance Criteria

### AC1: Document Fetching
**Given** the existing Normattiva scraper in visualex-api
**When** I request an article by URN (e.g., Codice Civile Art. 1453)
**Then** it returns the article text, rubrica, and URN
**And** the response is cached for subsequent requests

### AC2: HTML Parsing (Multiple Formats)
**Given** Normattiva returns HTML in various formats
**When** the scraper parses the document
**Then** it correctly extracts text from:
- AKN detailed format (art-comma-div-akn)
- AKN simple format (art-just-text-akn)
- Attachment format (attachment-just-text)
- Fallback for unknown structures

### AC3: Error Handling
**Given** the scraper encounters an error (network, parsing, not found)
**When** the error occurs
**Then** it raises appropriate exceptions:
- `DocumentNotFoundError` for 404/missing articles
- `ParsingError` for malformed HTML
- `NetworkError` for connection issues
**And** errors are logged with structured logging

### AC4: Retry Logic
**Given** the scraper encounters a transient error (network, rate limit)
**When** the error occurs
**Then** it retries with exponential backoff (max 4 retries, factor 2.0)
**And** failed requests after retries raise NetworkError

### AC5: Testing
**Given** the scraper implementation
**When** tests are run
**Then** all parsing scenarios are covered with realistic HTML fixtures
**And** error handling is verified
**And** cache behavior is tested

## Tasks/Subtasks

- [x] **Verification**: Verify the existing `NormattivaScraper` implementation
- [x] **Integration**: Ensure the scraper functions within the new architecture
- [x] **Robustness**: Implement error handling and caching
  - [x] Proper exception hierarchy (DocumentNotFoundError, ParsingError, NetworkError)
  - [x] Structured logging with structlog
  - [x] Retry with exponential backoff via ThrottledHttpClient
- [x] **Testing**: Add comprehensive tests with realistic fixtures
  - [x] AKN detailed format parsing
  - [x] AKN simple format parsing
  - [x] Attachment format parsing
  - [x] Fallback extraction
  - [x] Error handling (ParsingError on malformed HTML)
  - [x] Link extraction

### Review Follow-ups (Code Review 2026-01-31)
- [x] [CRITICAL] Fix `logger` → `log` bug (lines 145, 173, 200, 227)
- [x] [HIGH] Fix error handlers returning strings instead of raising exceptions
- [x] [HIGH] Replace placeholder tests with real assertions

## Technical Details

### Source Files
- **Scraper**: `visualex-api/visualex/scrapers/normattiva.py`
- **Base**: `visualex-api/visualex/utils/sys_op.py`
- **HTTP Client**: `visualex-api/visualex/utils/http_client.py`
- **Selectors**: `visualex-api/visualex/utils/selectors.py`
- **Exceptions**: `visualex-api/visualex/exceptions.py`

### Dependencies
- `aiocache` - Caching decorator
- `bs4` (BeautifulSoup4) - HTML parsing
- `structlog` - Structured logging
- `aiohttp` - Async HTTP client

### Output
- Structured text/metadata from Normattiva HTML
- Format: `(text: str, urn: str)` or `{"testo": str, "link": dict}` with links

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex-api/visualex/scrapers/normattiva.py` | Modified | Fixed logger→log bug, error handling now raises exceptions |
| `visualex-api/tests/integration/test_scrapers.py` | Modified | Replaced placeholder tests with 10 real tests using HTML fixtures |

### Change Log

**2026-01-31 - Code Review Fixes**
- Fixed `logger` → `log` in 4 error handlers (lines 145, 173, 200, 227)
- Changed error handlers from returning error strings to raising `ParsingError`
- Rewrote test suite with realistic Normattiva HTML fixtures
- Added tests for: AKN detailed, AKN simple, attachment, fallback, links, error handling
- Test count: 2 placeholder → 10 real tests (all passing)

**2026-01-25 - Initial Implementation**
- Verified existing NormattivaScraper works with current architecture
- Confirmed retry logic present in ThrottledHttpClient (max 4 retries, factor 2.0)
- Confirmed caching at two levels: @cached decorator + persistent cache
