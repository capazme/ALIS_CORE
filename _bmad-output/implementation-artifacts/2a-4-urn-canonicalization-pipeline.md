# Story 2a-4: URN Canonicalization Pipeline

## Status
- **Epic**: Epic 2a: Scraping & URN Pipeline
- **Status**: done
- **Priority**: High

## Context
Generate and validate canonical URNs for all scraped content following the official NIR (Norme In Rete) standard. URN is the primary key linking all systems (Graph, Vector, Bridge).

## NIR URN Specification (Official Normattiva)

### Base Format
```
urn:nir:stato:[tipo_atto]:[AAAA-MM-GG];[numero]
```

### Full Format with all components
```
urn:nir:stato:[tipo_atto]:[data];[numero]:[allegato]~art[N][estensione][@originale|!vig=[data]]
```

### Component Order (CRITICAL)
1. `urn:nir:stato:` - prefix (constant)
2. `[tipo_atto]` - act type (e.g., `legge`, `decreto.legge`, `regio.decreto`)
3. `:[data]` - date in YYYY-MM-DD format
4. `;[numero]` - act number
5. `:[allegato]` - annex number (optional, e.g., `:1`, `:2`)
6. `~art[N][estensione]` - article (optional, e.g., `~art1453`, `~art16bis`)
7. `@originale` OR `!vig=[data]` - version (optional)

### Examples
| URN | Description |
|-----|-------------|
| `urn:nir:stato:legge:2020-12-30;178` | Legge 178/2020 |
| `urn:nir:stato:decreto.legge:2008-11-10;180~art2` | D.L. 180/2008, art. 2 |
| `urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453` | Codice Civile art. 1453 |
| `urn:nir:stato:costituzione:1947-12-27~art7` | Costituzione art. VII (as 7) |
| `urn:nir:stato:decreto.legge:2008-11-10;180@originale` | Original version |
| `urn:nir:stato:decreto.legge:2008-11-10;180!vig=2009-11-10` | Vigente at date |

### Article Extensions (Latin ordinals)
1=base, 2=bis, 3=ter, 4=quater, 5=quinquies, 6=sexies, 7=septies, 8=octies,
9=novies, 10=decies, 11=undecies, 12=duodecies, ... up to 49=undequinquagies

### Special Cases
- Roman numerals in source → Arabic in URN (art. VII → ~art7)
- Extended numbering: art. 314/15 → ~art314.15
- Zero-prefixed: art. 01 → ~art01
- Extension with dot: art. 79 octies.1 → ~art79octies.1

## Acceptance Criteria

### AC1: URN Validation
**Given** a URN string
**When** I validate it
**Then** it returns True if valid NIR format
**And** returns detailed error for invalid URNs

### AC2: URN Parsing
**Given** a valid URN
**When** I parse it
**Then** I extract all components: tipo_atto, data, numero, allegato, articolo, estensione, versione, data_versione

### AC3: URN Canonicalization
**Given** different URN variants referring to same norm
**When** I canonicalize them
**Then** they produce identical canonical URN
**And** component order matches NIR standard

### AC4: Duplicate Detection
**Given** URNs from different sources
**When** compared for equality
**Then** duplicates are detected even with different formatting

### AC5: URL Generation
**Given** a canonical URN
**When** I generate Normattiva URL
**Then** it produces valid `https://www.normattiva.it/uri-res/N2Ls?[urn]`

## Tasks/Subtasks

- [x] **T1**: Create `urn_pipeline.py` with URNComponents dataclass
- [x] **T2**: Implement `parse_urn()` - extract all components from URN
- [x] **T3**: Implement `validate_urn()` - validate against NIR standard
- [x] **T4**: Implement `canonicalize_urn()` - normalize to canonical form
- [x] **T5**: Implement `urn_to_dict()` and `dict_to_urn()` conversion
- [x] **T6**: Add comprehensive tests for all edge cases (60 tests)
- [x] **T7**: Code review completed

### Code Review (Adversarial)
- [x] **CR-M1**: Removed unused `URN_QUICK_PATTERN` (dead code)
- [x] **CR-M2**: Documented comma/lettera limitation (Normattiva supports only article level)
- [x] **CR-L1**: Added `__all__` for explicit public API exports
- [x] **CR-L2**: Optimized `number_to_extension()` with O(1) reverse lookup dict
- [x] **CR-L3**: Removed confusing `pass` statement, clarified comment
- [x] **CR-L4**: Extended `VALID_ACT_TYPES` with `legge.regionale`

## Technical Details

### URNComponents Dataclass
```python
@dataclass
class URNComponents:
    tipo_atto: str          # e.g., "legge", "decreto.legge"
    data: str               # YYYY-MM-DD
    numero: str             # act number
    allegato: str | None    # annex number
    articolo: str | None    # article number
    estensione: str | None  # bis, ter, quater...
    versione: str | None    # "originale" or "vigente"
    data_versione: str | None  # for !vig=YYYY-MM-DD
```

### Regex Patterns
```python
# Full URN pattern
URN_PATTERN = re.compile(
    r'^urn:nir:stato:'
    r'(?P<tipo_atto>[a-z.]+):'
    r'(?P<data>\d{4}-\d{2}-\d{2})'
    r'(?:;(?P<numero>\d+))?'
    r'(?::(?P<allegato>\d+))?'
    r'(?:~art(?P<articolo>[\d.]+)(?P<estensione>[a-z]*))?'
    r'(?:@(?P<versione_orig>originale)|!vig=(?P<data_versione>\d{4}-\d{2}-\d{2})?)?$'
)
```

### Validation Rules
1. Must start with `urn:nir:stato:`
2. tipo_atto must be valid (from NORMATTIVA map)
3. Date must be valid YYYY-MM-DD
4. Number required except for costituzione
5. Allegato must be positive integer if present
6. Article extensions must be valid Latin ordinals

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/utils/urn_pipeline.py` | Created | URN parsing, validation, canonicalization pipeline |
| `tests/unit/test_urn_pipeline.py` | Created | 60 unit tests covering all edge cases |

### Change Log

**2026-02-01 - Implementation**
- Created `URNComponents` dataclass with all NIR fields
- Implemented `parse_urn()` with regex supporting:
  - Full URL prefix (`https://www.normattiva.it/uri-res/N2Ls?`)
  - All act types (legge, decreto.legge, regio.decreto, etc.)
  - Allegato numbering (`:1`, `:2`)
  - Article with extensions (`~art16bis`, `~art79octies.1`)
  - Version markers (`@originale`, `!vig=`, `!vig=YYYY-MM-DD`)
- Implemented `validate_urn()` with comprehensive validation
- Implemented `canonicalize_urn()` for normalization
- Implemented `urns_are_equivalent()` for duplicate detection
- Added utility functions: `dict_to_urn()`, `extract_article_info()`, `extension_to_number()`
- Test coverage: 60 tests including real-world Normattiva examples

**2026-02-01 - Code Review Fixes**
- Extended `VALID_ACT_TYPES` with additional types and documentation
- Added support for subpart notation (`art79octies.1`)
- Validation handles Latin extensions with subparts
