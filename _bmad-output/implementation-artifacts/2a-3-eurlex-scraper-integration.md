# Story 2a-3: EUR-Lex Scraper Integration

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: done
- **Priority**: Medium

## Context
Integrate the EUR-Lex scraper for EU directives related to contract law. This enables SystemicExpert to reference EU law when interpreting Italian implementation. Lower priority than Normattiva/Brocardi for thesis MVP, but important for comprehensive legal analysis.

## Acceptance Criteria

### AC1: Directive Retrieval
**Given** the existing EUR-Lex scraper
**When** I configure it for directives related to Libro IV topics
**Then** it retrieves key EU directives (e.g., Consumer Rights Directive 2011/83/EU)
**And** each directive includes: CELEX number, title, text

### AC2: Article Extraction
**Given** an EU directive document
**When** I request a specific article
**Then** the scraper extracts the article text correctly
**And** handles multiple article formats (Articolo, Article, Art.)

### AC3: WAF Bypass with Playwright
**Given** EUR-Lex uses CloudFront WAF protection
**When** the scraper requests a document
**Then** it uses Playwright with proper user-agent
**And** waits for network idle before extracting content

### AC4: Error Handling & Retry
**Given** EUR-Lex API rate limits are hit or network errors occur
**When** the error occurs
**Then** the scraper raises appropriate exceptions (DocumentNotFoundError, NetworkError)
**And** errors are logged with structured logging

### AC5: Caching
**Given** a document has been fetched
**When** the same document is requested again
**Then** it returns from cache without making HTTP request

### AC6: Testing
**Given** the scraper implementation
**When** tests are run
**Then** all key scenarios are covered with realistic fixtures
**And** tests pass without making real HTTP requests

## Tasks/Subtasks

- [x] **Analysis**: Analyze existing EurlexScraper for issues
- [x] **M1-MEDIUM**: Remove unused `import os`
- [x] **M2-MEDIUM**: Use EURLexSelectors consistently (ARTICLE_CLASS, SUBDIVISION)
- [x] **Testing**: Add comprehensive tests (11 tests with HTML fixtures)
- [x] **Documentation**: Update story with Dev Agent Record

### Code Review (Adversarial)
- [x] **CR-M1**: Extract hardcoded browser args to constants (`BROWSER_ARGS`)
- [x] **CR-M2**: Extract hardcoded user-agent to constant (`USER_AGENT`)
- [x] **CR-L1**: Extract timeout magic number to constant (`PAGE_TIMEOUT_MS`)
- [x] **CR-L2**: Extract min HTML length to constant (`MIN_VALID_HTML_LENGTH`)
- [x] **CR-L3**: Extract viewport dimensions to constants (`VIEWPORT_WIDTH/HEIGHT`)
- [x] **CR-L4**: Consolidate article detection pattern (`ARTICLE_PATTERN`)
- [x] **CR-L5**: Use structlog key-value format for logging
- [x] **Live Tests**: Added 10 live tests for all scrapers (Normattiva, Brocardi, EUR-Lex)

## Technical Details

### Source Files
- **Scraper**: `visualex-api/visualex/scrapers/eurlex.py`
- **Selectors**: `visualex-api/visualex/utils/selectors.py`
- **Map**: `visualex-api/visualex/utils/map.py` (EURLEX dict)
- **Exceptions**: `visualex-api/visualex/exceptions.py`

### Dependencies
- `playwright` - Browser automation for WAF bypass
- `bs4` (BeautifulSoup4) - HTML parsing
- `structlog` - Structured logging

### Key Directives (Scope)
- Consumer Rights Directive 2011/83/EU
- Unfair Contract Terms Directive 93/13/EEC
- E-Commerce Directive 2000/31/EC
- Digital Content Directive 2019/770/EU

### Features
- **Playwright WAF Bypass**: Headless Chromium with proper user-agent
- **Multi-Strategy Article Search**: 4 strategies (ti-art class, art/title class, regex, eli-subdivision)
- **Table Extraction**: Handles tabular data within articles
- **Treaty Support**: Direct URLs for TUE, TFUE, CDFUE

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex-api/visualex/scrapers/eurlex.py` | Modified | Removed unused os import, use selectors, add constants |
| `visualex-api/tests/integration/test_scrapers.py` | Modified | Added 11 tests for EUR-Lex with HTML fixtures |
| `visualex-api/tests/integration/test_scrapers_live.py` | Created | Live tests for all scrapers (10 tests) |
| `visualex-api/tests/conftest.py` | Modified | Added `--run-live` option and `live` marker |

### Change Log

**2026-01-31 - Implementation & Code Review**
- Removed unused `import os`
- Updated `extract_article_text` to use `self.selectors.ARTICLE_CLASS` instead of hardcoded `'ti-art'`
- Updated `extract_article_text` to use `self.selectors.SUBDIVISION` instead of hardcoded `'eli-subdivision'`
- Added comprehensive test suite with realistic EUR-Lex HTML fixtures:
  - `test_get_uri_regulation` - Regulation URI generation
  - `test_get_uri_directive` - Directive URI generation
  - `test_get_uri_treaty_tfue` - Treaty direct URL
  - `test_get_uri_extracts_year_from_date` - Year extraction from full date
  - `test_extract_article_text_finds_article` - Article text extraction
  - `test_extract_article_text_stops_at_next_article` - Boundary detection
  - `test_extract_article_text_with_table` - Table content extraction
  - `test_extract_article_text_not_found_raises` - Error handling
  - `test_extract_table_text` - Table parsing
  - `test_parse_document` - Document parsing
  - `test_extract_article_english_format` - English article format support
- Test count: 2 → 11 tests (all passing)
- Total test suite: 35 tests passing

**2026-02-01 - Adversarial Code Review Fixes**
- Added Playwright configuration constants at module level:
  - `BROWSER_ARGS` - Chrome sandbox flags
  - `USER_AGENT` - Browser user-agent string
  - `PAGE_TIMEOUT_MS` - Page load timeout (30s)
  - `MIN_VALID_HTML_LENGTH` - Minimum valid response size (1000 bytes)
  - `VIEWPORT_WIDTH/HEIGHT` - Browser viewport dimensions (1920x1080)
  - `ARTICLE_PATTERN` - Compiled regex for article detection
- Updated `_fetch_with_playwright` and `request_document` to use constants
- Consolidated article pattern regex (previously duplicated in 2 places)
- Improved structlog calls to use key-value format instead of f-strings
- Added live integration tests (`test_scrapers_live.py`) for all 3 scrapers:
  - Normattiva: 3 tests (Codice Civile, Costituzione)
  - Brocardi: 2 tests (Codice Civile, Codice Penale)
  - EUR-Lex: 5 tests (GDPR, Consumer Rights, TFUE, URI generation)
- Tests are skipped by default, run with `pytest --run-live`
- Total test count: 35 mocked + 10 live = 45 tests
