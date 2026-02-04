# Story 3-7: Modification Detection Alert (Backend)

## Status
- **Epic**: Epic 3: Norm Browsing & Search
- **Status**: done
- **Priority**: High

## Context
Backend API for detecting and alerting on norm modifications. Supports recent modification detection, newer version availability, and batch checking for Dossier.

## Existing Code
- `visualex/graph/temporal.py` - Version timeline and abrogation info

## Acceptance Criteria

### AC1: Recent Modification Alert
**Given** an article was modified in the last 30 days
**When** checking for alerts
**Then** return alert with modification date and modifying legislation

### AC2: Batch Checking (Dossier Support)
**Given** multiple URNs saved in a Dossier
**When** checking batch for modifications
**Then** return alerts for all recently modified norms

### AC3: Newer Version Available
**Given** querying with as_of_date
**When** a newer version exists
**Then** return alert indicating newer version available

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/alerts.py` module
- [x] **T2**: Create `AlertType` constants
- [x] **T3**: Create `ModificationAlert` dataclass
- [x] **T4**: Implement `ModificationAlertService` class
- [x] **T5**: Implement `check_recent_modification()` (AC1)
- [x] **T6**: Implement `check_newer_version()` (AC3)
- [x] **T7**: Implement `check_abrogation()`
- [x] **T8**: Implement `check_batch()` (AC2)
- [x] **T9**: Implement `get_all_alerts()` for combined checks
- [x] **T10**: Update `visualex/graph/__init__.py` exports
- [x] **T11**: Write tests (21 tests passing)

## Technical Details

### Alert Types
- `recent_modification`: Norm was modified within N days
- `newer_version`: A newer version exists than queried date
- `abrogated`: Norm has been abrogated

### API Methods
```python
service = ModificationAlertService(falkor_client)

# Check single norm
alert = await service.check_recent_modification(urn, days=30)

# Check for newer version
alert = await service.check_newer_version(urn, as_of_date)

# Batch check for Dossier
alerts = await service.check_batch([urn1, urn2, ...], days=30)

# Get all applicable alerts
alerts = await service.get_all_alerts(urn, as_of_date, days=30)
```

### Italian Messages
- Recent: "Questo articolo è stato modificato il {date} da {norm}"
- Newer: "Versione più recente disponibile"
- Abrogated: "Questo articolo è stato abrogato il {date}"

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/alerts.py` | Create | ModificationAlertService |
| `visualex/graph/__init__.py` | Modify | Add exports |
| `tests/unit/test_graph_alerts.py` | Create | 21 tests |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implemented ModificationAlertService with all AC criteria
- **2026-02-02**: All 21 tests passing - Story completed
