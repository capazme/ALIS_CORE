# Story 2a-2: Brocardi Scraper Integration

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: done
- **Priority**: High

## Context
Integrate the Brocardi scraper into the ALIS_CORE pipeline. Brocardi provides legal commentary (massime, spiegazioni) which enriches the raw norms. We need to verify the existing scraper and potentially improve the data structure based on the current website layout.

## Acceptance Criteria

### AC1: Section Extraction
**Given** the existing Brocardi scraper in visualex-api
**When** I request an article (e.g., Codice Civile Art. 1453)
**Then** it extracts:
- Brocardi (Latin maxim)
- Ratio (legal rationale)
- Spiegazione (explanation)
- Massime (case law references)
**And** all sections are cleaned of extra whitespace

### AC2: Massime Parsing (Structured)
**Given** Brocardi returns massime from various courts
**When** the scraper parses the massime section
**Then** it correctly extracts from each massima:
- Autorità (Cassazione, Corte Cost., TAR, etc.)
- Numero sentenza
- Anno
- Testo della massima
**And** supports all Italian court authority patterns

### AC3: Cross-Reference Extraction
**Given** article pages with links to other articles
**When** the scraper processes the page
**Then** it extracts:
- Related articles (precedente/successivo)
- Cross-references to other codes (Civile, Penale, Costituzione)
- Footnotes with references

### AC4: Async-First Architecture
**Given** CLAUDE.md specifies async-only HTTP
**When** making HTTP requests
**Then** uses `urllib.parse.urljoin` instead of `requests.compat`
**And** does not import the synchronous `requests` library
**And** uses the async `http_client` for all network operations

### AC5: Testing
**Given** the scraper implementation
**When** tests are run
**Then** all parsing scenarios are covered with realistic HTML fixtures
**And** at least 14 test cases for Brocardi logic
**And** all tests pass without `pytest.skip()` fallbacks

## Tasks/Subtasks

- [x] **Analysis**: Analyze Brocardi scraper for issues
- [x] **C1-CRITICAL**: Replace `import requests` with `urllib.parse.urljoin`
- [x] **H1-HIGH**: Clean up unused imports (os, ParsingError, NetworkError)
- [x] **H2-HIGH**: Update tests from placeholders to real tests (14 tests)
- [x] **M1-MEDIUM**: Use BrocardiSelectors consistently
  - [x] BREADCRUMB_ID for position extraction
  - [x] MAIN_CONTENT_CLASS for main content
  - [x] BROCARDI_CONTENT_CLASS for brocardi section
  - [x] RATIO_CONTAINER_CLASS for ratio section
- [x] **Documentation**: Update story file with AC and Dev Agent Record

## Technical Details

### Source Files
- **Scraper**: `visualex-api/visualex/scrapers/brocardi.py`
- **Selectors**: `visualex-api/visualex/utils/selectors.py`
- **HTTP Client**: `visualex-api/visualex/utils/http_client.py`
- **Exceptions**: `visualex-api/visualex/exceptions.py`

### Dependencies
- `aiocache` - Caching decorator
- `bs4` (BeautifulSoup4) - HTML parsing
- `structlog` - Structured logging
- `aiohttp` - Async HTTP client (via http_client)

### Data Structures
- Massima dict: `{autorita, numero, anno, massima}`
- Cross-reference dict: `{articolo, tipo_atto, url, sezione, testo}`
- Related articles dict: `{previous: {...}, next: {...}}`

### Features
- **Court Authority Patterns**: Regex patterns for all Italian courts (Cassazione, Corte Cost., Cons. Stato, TAR, Appello, CGUE, CEDU)
- **Footnote Extraction**: Multiple patterns (nota-ref, corpoDelTesto.nota, legacy sup/div)
- **Cross-Reference Extraction**: Links to Codice Civile, Codice Penale, Costituzione, etc.
- **Relazioni Storiche**: Extracts Guardasigilli relations (1941/1942)

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex-api/visualex/scrapers/brocardi.py` | Modified | Replaced requests with urllib.parse, cleaned imports, use selectors |
| `visualex-api/visualex/utils/selectors.py` | Modified | Added CORPO_DEL_TESTO_CLASS selector |
| `visualex-api/tests/integration/test_scrapers.py` | Modified | Added 14 real tests for Brocardi with HTML fixtures |

### Change Log

**2026-01-31 - Formal Code Review (Adversarial)**
- [M1] Removed unused `import os`
- [M2] Removed unused `ParsingError, NetworkError` imports (kept only `DocumentNotFoundError`)
- [M3] Added `CORPO_DEL_TESTO_CLASS` to BrocardiSelectors and used it in `_extract_sections`
- [L1] Removed unused `RelazioneContent` dataclass (dict used instead)
- [L2] Added explanatory comment for `[17:]` magic number in `_extract_position`
- All 26 tests passing

**2026-01-31 - Initial Code Review & Fixes**
- Replaced `import requests` with `from urllib.parse import urljoin`
- Changed `requests.compat.urljoin` → `urljoin` (2 occurrences)
- Updated `_extract_position` to use `self.selectors.BREADCRUMB_ID`
- Updated `_extract_sections` to use:
  - `self.selectors.MAIN_CONTENT_CLASS`
  - `self.selectors.BROCARDI_CONTENT_CLASS`
  - `self.selectors.RATIO_CONTAINER_CLASS`
- Rewrote test suite with realistic Brocardi HTML fixtures
- Added tests for:
  - `_clean_text` (whitespace normalization, empty handling)
  - `_extract_sections` (Brocardi, Ratio, Spiegazione, Massime)
  - `_parse_massima` (Cassazione parsing)
  - `_extract_footnotes`
  - `_extract_cross_references`
  - `_extract_related_articles`
  - `_extract_position`
  - `_extract_article_links`
  - `do_know`, `_build_norma_string`
- Test count: 2 placeholder → 14 real tests (all passing)
