# Story 1.6: Data Export & Erasure

Status: done

## Story

As a **user**,
I want to **export my personal data and request account deletion**,
So that **I can exercise my GDPR rights (Art. 20 portability, Art. 17 erasure)**.

## Acceptance Criteria

1. **Data Export Request:**
   - **Given** I am logged in and access privacy settings
   - **When** I request data export
   - **Then** I see a confirmation that export is being prepared
   - **And** I can download a JSON file containing my data

2. **Export Contents:**
   - **Given** my data export is ready
   - **When** I download it
   - **Then** it contains:
     - My profile information (email, username, profile type)
     - My consent history
     - My authority score breakdown
     - My preferences

3. **Account Deletion Request:**
   - **Given** I request account deletion
   - **When** I confirm with my password
   - **Then** I see a warning about irreversibility
   - **And** my account enters "deletion_pending" status
   - **And** I am logged out

4. **Deletion Cancellation:**
   - **Given** my account is in "deletion_pending" status
   - **When** I log in within 30 days
   - **Then** I can cancel the deletion request
   - **And** my account is restored to active status

## Tasks / Subtasks

- [x] **Backend: Database Schema**
  - [x] Add `deletionRequestedAt` field to User model
  - [x] Add `deletionReason` field (optional)
  - [x] Apply schema changes

- [x] **Backend: Data Export Endpoint**
  - [x] `POST /api/privacy/export` - Request data export
  - [x] Collect all user data (profile, consent, authority, preferences)
  - [x] Return JSON file for download

- [x] **Backend: Account Deletion Endpoints**
  - [x] `POST /api/privacy/delete-account` - Request deletion (requires password)
  - [x] `POST /api/privacy/cancel-deletion` - Cancel pending deletion
  - [x] Mark account as deletion_pending
  - [x] Implement actual deletion logic (for future scheduled job)

- [x] **Frontend: Privacy Settings Component**
  - [x] Create `PrivacySettings.tsx` with export and delete sections
  - [x] Add password confirmation modal for deletion
  - [x] Show deletion warning
  - [x] Show pending deletion status with cancel option

- [x] **Frontend: Settings Integration**
  - [x] Add privacy section to Settings page
  - [x] Create privacy service for API calls

- [x] **Tests**
  - [x] Backend: Privacy API endpoint tests (21 tests)
  - [x] Frontend: PrivacySettings component tests (15 tests)

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5)
- **GDPR Compliance:** Art. 17 (erasure), Art. 20 (portability)
- **API Path:** `/api/privacy/*`

### Data Export Contents
```typescript
interface DataExport {
  exported_at: string;
  user: {
    id: string;
    email: string;
    username: string;
    profile_type: string;
    created_at: string;
  };
  preferences: {
    theme: string;
    language: string;
    notifications_enabled: boolean;
  };
  consent: {
    current_level: string;
    granted_at: string;
    history: ConsentHistoryEntry[];
  };
  authority: {
    score: number;
    baseline: number;
    track_record: number;
    recent_performance: number;
    feedback_count: number;
  };
}
```

### Deletion Flow
1. User requests deletion with password confirmation
2. Account marked as `deletion_pending` with timestamp
3. User logged out (all refresh tokens revoked)
4. 30-day grace period (can cancel by logging in)
5. After 30 days: scheduled job permanently deletes data
6. Audit logs anonymized (userId replaced with hash)

### Dependencies
- Story 1.2 (User Login) - for authentication
- Story 1.4 (Consent Configuration) - for consent history export
- Story 1.5 (Authority Score) - for authority data export

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5

### File List

**Backend (6 files):**
1. `visualex-platform/backend/prisma/schema.prisma` - Added deletionRequestedAt, deletionReason fields to User model
2. `visualex-platform/backend/src/controllers/privacyController.ts` - NEW: Privacy controller with exportData, requestDeletion, cancelDeletion, getPrivacyStatus
3. `visualex-platform/backend/src/routes/privacy.ts` - NEW: Privacy routes registration
4. `visualex-platform/backend/src/index.ts` - Added privacy routes import and registration
5. `visualex-platform/backend/tests/privacy.test.ts` - NEW: 21 integration tests for privacy API

**Frontend (6 files):**
1. `visualex-platform/frontend/src/types/api.ts` - Added PrivacyStatusResponse, DeleteAccountRequest, DeleteAccountResponse, CancelDeletionResponse, DataExport types
2. `visualex-platform/frontend/src/services/privacyService.ts` - NEW: Privacy service with getPrivacyStatus, exportData, requestDeletion, cancelDeletion
3. `visualex-platform/frontend/src/components/features/privacy/PrivacySettings.tsx` - NEW: Privacy settings component with export and deletion UI
4. `visualex-platform/frontend/src/components/features/privacy/PrivacySettings.test.tsx` - NEW: 15 component tests
5. `visualex-platform/frontend/src/components/features/privacy/index.ts` - NEW: Barrel export
6. `visualex-platform/frontend/src/pages/SettingsPage.tsx` - Integrated PrivacySettings component

### Change Log

1. Added `deletionRequestedAt` DateTime field to User model (GDPR Art. 17)
2. Added `deletionReason` String field to User model (optional)
3. Created privacyController.ts with 4 endpoints
4. Implemented exportData() - collects user, preferences, consent history, authority data
5. Implemented requestDeletion() - password verification, 30-day grace period
6. Implemented cancelDeletion() - restores account to active
7. Implemented getPrivacyStatus() - returns deletion status and days remaining
8. Created privacy routes with authenticate middleware
9. Registered privacy routes in index.ts
10. Added 5 TypeScript interfaces for privacy API
11. Created privacyService.ts with typed API calls
12. Created PrivacySettings.tsx with data export and account deletion sections
13. Implemented deletion confirmation modal with password input
14. Added pending deletion warning banner with cancel option
15. Integrated PrivacySettings into SettingsPage
16. Created 21 backend tests covering all privacy endpoints
17. Created 15 frontend tests covering component behavior

### Completion Notes List

- Story created from Epic 1 definition in epics.md
