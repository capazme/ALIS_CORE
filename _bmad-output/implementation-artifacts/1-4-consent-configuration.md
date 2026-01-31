# Story 1.4: Consent Configuration

Status: done

## Story

As a **user**,
I want to **choose my data consent level for RLCF participation**,
So that **I control how my interactions contribute to system learning (GDPR Art. 6)**.

## Acceptance Criteria

1. **Consent Level Display:**
   - **Given** I am in settings or first-time setup
   - **When** I view consent options
   - **Then** I see 3 levels clearly explained:
     - **Basic**: "No data collected beyond session. System use only."
     - **Learning**: "Anonymized queries + feedback used for RLCF training."
     - **Research**: "Aggregated data available for academic analysis."

2. **Consent Selection:**
   - **Given** I select a consent level
   - **When** I confirm my choice
   - **Then** my consent is recorded with timestamp
   - **And** a consent audit log entry is created (immutable)
   - **And** the system respects my choice for all future interactions

3. **Consent Downgrade:**
   - **Given** I change my consent level to a lower level
   - **When** I confirm the change
   - **Then** my new preference is applied immediately
   - **And** I am informed that previously collected data (if any) remains until I request erasure

## Tasks / Subtasks

- [x] **Backend: Database Schema**
  - [x] Add `ConsentLevel` enum to Prisma schema (basic, learning, research)
  - [x] Create `UserConsent` model (id, userId, consentLevel, grantedAt, ipHash)
  - [x] Create `ConsentAuditLog` model (append-only, 7-year retention)
  - [x] Apply schema changes with Prisma migration

- [x] **Backend: Consent API Endpoints**
  - [x] `GET /api/consent` - Get current user consent level
  - [x] `PUT /api/consent` - Update consent level (with audit logging)
  - [x] `GET /api/consent/history` - Get consent change history
  - [x] Hash IP address for privacy (SHA-256 + salt)
  - [x] Create consent audit log entry on every change

- [x] **Frontend: Consent Selector Component**
  - [x] Create `ConsentSelector.tsx` with 3 consent level cards
  - [x] Add clear descriptions and icons for each level
  - [x] Show warning when downgrading consent level
  - [x] Style with Tailwind (consistent with ProfileSelector)

- [x] **Frontend: Settings Integration**
  - [x] Add consent section to Settings page
  - [x] Create consent service for API calls
  - [x] Show current consent level with timestamp

- [x] **Tests**
  - [x] Backend: Consent API endpoint tests (17 tests)
  - [x] Frontend: ConsentSelector component tests (15 tests)

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5)
- **Database:** PostgreSQL (user_consents + consent_audit_log tables)
- **API Path:** `/api/consent`
- **GDPR Compliance:** Art. 6 (lawful basis), Art. 7 (consent conditions)

### Consent Levels (Enum)
```typescript
enum ConsentLevel {
  BASIC = 'basic',           // No data collected beyond session
  LEARNING = 'learning',     // Anonymized queries + feedback for RLCF
  RESEARCH = 'research'      // Aggregated data for academic analysis
}
```

### Database Schema
```typescript
// User consent record
model UserConsent {
  id           String       @id @default(uuid())
  userId       String       @unique @map("user_id")
  consentLevel ConsentLevel @default(basic) @map("consent_level")
  grantedAt    DateTime     @default(now()) @map("granted_at")
  ipHash       String?      @map("ip_hash") // SHA-256 hashed IP

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
}

// Append-only audit log (7-year retention per NFR-R5)
model ConsentAuditLog {
  id              String       @id @default(uuid())
  userId          String       @map("user_id")
  previousLevel   ConsentLevel? @map("previous_level")
  newLevel        ConsentLevel @map("new_level")
  ipHash          String?      @map("ip_hash")
  userAgent       String?      @map("user_agent")
  changedAt       DateTime     @default(now()) @map("changed_at")

  // No foreign key - audit log survives user deletion
  // No UPDATE/DELETE permissions on this table
}
```

### IP Hashing
```typescript
// Privacy-preserving IP hash
const hashIP = (ip: string, salt: string) => {
  return crypto.createHash('sha256').update(ip + salt).digest('hex');
};
```

### Dependencies
- Story 1.2 (User Login) - for authentication
- Story 1.3 (Profile Setup) - for settings page integration

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5

### File List

**Backend (6 files):**
- `visualex-platform/backend/prisma/schema.prisma` - Added ConsentLevel enum, UserConsent, ConsentAuditLog models
- `visualex-platform/backend/src/config.ts` - Added consentIpSalt configuration
- `visualex-platform/backend/src/controllers/consentController.ts` - NEW: Consent API controller
- `visualex-platform/backend/src/routes/consent.ts` - NEW: Consent routes
- `visualex-platform/backend/src/index.ts` - Registered consent routes
- `visualex-platform/backend/tests/consent.test.ts` - NEW: Backend consent tests (17 tests)

**Frontend (6 files):**
- `visualex-platform/frontend/src/types/api.ts` - Added consent types (ConsentLevel, ConsentResponse, etc.)
- `visualex-platform/frontend/src/services/consentService.ts` - NEW: Consent API service
- `visualex-platform/frontend/src/components/features/consent/ConsentSelector.tsx` - NEW: Consent selector component
- `visualex-platform/frontend/src/components/features/consent/ConsentSelector.test.tsx` - NEW: Component tests (15 tests)
- `visualex-platform/frontend/src/components/features/consent/index.ts` - NEW: Barrel export
- `visualex-platform/frontend/src/pages/SettingsPage.tsx` - Added consent section

### Change Log

1. Added ConsentLevel enum to Prisma schema (basic, learning, research)
2. Created UserConsent model with ipHash for privacy
3. Created ConsentAuditLog model (append-only, no FK to User)
4. Applied schema with `prisma db push`
5. Created consentController.ts with getConsent, updateConsent, getConsentHistory
6. Implemented IP hashing with SHA-256 + separate salt (config.consentIpSalt)
7. Created consent routes and registered in index.ts
8. Added consent types to frontend api.ts
9. Created consentService.ts for API calls
10. Created ConsentSelector.tsx with 3 consent level cards
11. Implemented downgrade confirmation dialog with warning
12. Added collapsible history section
13. Integrated ConsentSelector into SettingsPage.tsx
14. Created backend tests (17 tests covering all endpoints)
15. Created frontend tests (15 tests covering all interactions)
16. Code review: Fixed IP hash salt to use separate env var (CONSENT_IP_SALT)

### Completion Notes List

- Story created from Epic 1 definition in epics.md
- Database schema applied with `prisma db push` (no migration file - dev environment)
- IP hashing uses separate CONSENT_IP_SALT env var for security isolation from JWT
- Audit log has no FK to User - survives user deletion for GDPR compliance
- History limited to 50 entries per user to prevent unbounded growth
- All 32 tests passing (17 backend + 15 frontend)
