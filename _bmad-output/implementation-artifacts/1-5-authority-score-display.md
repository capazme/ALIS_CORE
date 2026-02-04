# Story 1.5: Authority Score Display

Status: done

## Story

As a **user**,
I want to **view my authority score and understand how it's calculated**,
So that **I know my influence on RLCF training and can work to increase it**.

## Acceptance Criteria

1. **Authority Score Display:**
   - **Given** I am logged in and view my profile/settings
   - **When** I access the authority score section
   - **Then** I see my current authority score (0.0 - 1.0 scale)
   - **And** I see a visual progress indicator

2. **Score Component Breakdown:**
   - **Given** I view my authority score
   - **When** I see the breakdown section
   - **Then** I see three components:
     - Baseline credentials (30%): qualifications, role, experience
     - Track record (50%): historical feedback accuracy
     - Recent performance (20%): last N feedback quality

3. **New User Experience:**
   - **Given** I am a new user with no feedback history
   - **When** I view my authority score
   - **Then** I see my baseline score based on credentials
   - **And** I see a message "Contribuisci feedback per aumentare la tua autorità"

4. **Component Tooltips:**
   - **Given** I hover over or click on a score component
   - **When** the tooltip appears
   - **Then** I see a plain-language explanation of that component
   - **And** I understand how to improve it

## Tasks / Subtasks

- [x] **Backend: Database Schema**
  - [x] Create `UserAuthority` model (baseline, trackRecord, recentPerformance, computedScore)
  - [x] Add relation to User model
  - [x] Apply schema changes

- [x] **Backend: Authority API Endpoints**
  - [x] `GET /api/authority` - Get current user authority breakdown
  - [x] Calculate computed score using formula: A_u = 0.3·B + 0.5·T + 0.2·P
  - [x] Create authority record on first access (lazy initialization)
  - [x] Recalculate baseline when profile type changes

- [x] **Frontend: Authority Display Component**
  - [x] Create `AuthorityScoreDisplay.tsx` with score visualization
  - [x] Create progress bar/gauge component
  - [x] Add breakdown cards for each component (baseline, track record, recent)
  - [x] Add tooltips with explanations

- [x] **Frontend: Settings Integration**
  - [x] Add authority score section to Settings page
  - [x] Create authority service for API calls

- [x] **Tests**
  - [x] Backend: Authority API endpoint tests (11 tests)
  - [x] Frontend: AuthorityScoreDisplay component tests (13 tests)

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5)
- **Database:** PostgreSQL (user_authority table)
- **API Path:** `/api/authority`
- **Formula:** A_u(t) = 0.3·B_u + 0.5·T_u(t) + 0.2·P_u(t)

### Authority Score Components
```typescript
// Component weights (must sum to 1.0)
const WEIGHTS = {
  baseline: 0.3,      // α: Baseline credentials
  trackRecord: 0.5,   // β: Historical feedback accuracy
  recentPerformance: 0.2, // γ: Last N feedback quality
};

// Each component is 0.0 - 1.0 scale
// Computed score = Σ(weight_i × component_i)
```

### Database Schema
```typescript
model UserAuthority {
  id                 String   @id @default(uuid())
  userId             String   @unique @map("user_id")
  baselineScore      Float    @default(0.1) @map("baseline_score")     // From credentials
  trackRecordScore   Float    @default(0.0) @map("track_record_score") // From feedback history
  recentPerformance  Float    @default(0.0) @map("recent_performance") // Last N feedbacks
  computedScore      Float    @default(0.03) @map("computed_score")    // Weighted sum
  feedbackCount      Int      @default(0) @map("feedback_count")
  updatedAt          DateTime @updatedAt @map("updated_at")

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@map("user_authority")
}
```

### Baseline Score Mapping
```typescript
// Based on profile type selected during setup
const BASELINE_BY_PROFILE: Record<ProfileType, number> = {
  quick_consultation: 0.1,   // Casual user
  assisted_research: 0.2,    // Student/researcher
  expert_analysis: 0.4,      // Professional
  active_contributor: 0.5,   // Expert (requires authority ≥ 0.5 to select)
};
```

### Dependencies
- Story 1.2 (User Login) - for authentication
- Story 1.3 (Profile Setup) - for profile type (affects baseline)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5

### File List

**Backend (5 files):**
- `visualex-platform/backend/prisma/schema.prisma` - Added UserAuthority model with relation to User
- `visualex-platform/backend/src/controllers/authorityController.ts` - NEW: Authority API controller with lazy init
- `visualex-platform/backend/src/controllers/profileController.ts` - Added recalculateBaseline call on profile change
- `visualex-platform/backend/src/routes/authority.ts` - NEW: Authority routes
- `visualex-platform/backend/src/index.ts` - Registered authority routes
- `visualex-platform/backend/tests/authority.test.ts` - NEW: Backend authority tests (11 tests)

**Frontend (7 files):**
- `visualex-platform/frontend/src/types/api.ts` - Added authority types (AuthorityResponse, AuthorityComponents)
- `visualex-platform/frontend/src/services/authorityService.ts` - NEW: Authority API service
- `visualex-platform/frontend/src/components/features/authority/AuthorityScoreDisplay.tsx` - NEW: Main component
- `visualex-platform/frontend/src/components/features/authority/AuthorityScoreDisplay.test.tsx` - NEW: Tests (13)
- `visualex-platform/frontend/src/components/features/authority/index.ts` - NEW: Barrel export
- `visualex-platform/frontend/src/pages/SettingsPage.tsx` - Added authority section

### Change Log

1. Added UserAuthority model to Prisma schema (baseline, trackRecord, recentPerformance, computedScore)
2. Added User → UserAuthority relation
3. Applied schema with `prisma db push`
4. Created authorityController.ts with getAuthority endpoint
5. Implemented lazy initialization (creates authority record on first access)
6. Implemented weighted score calculation: A = 0.3·B + 0.5·T + 0.2·P
7. Created BASELINE_BY_PROFILE mapping (profile type → baseline score)
8. Created authority routes and registered in index.ts
9. Updated profileController to recalculate baseline on profile change
10. Added authority types to frontend api.ts
11. Created authorityService.ts for API calls
12. Created AuthorityScoreDisplay.tsx with score gauge and component cards
13. Implemented color-coded progress bars based on score level
14. Added tooltips with component descriptions
15. Added "new user" message when feedback_count=0
16. Integrated into SettingsPage.tsx
17. Created backend tests (11 tests)
18. Created frontend tests (13 tests)

### Completion Notes List

- Story created from Epic 1 definition in epics.md
- Database schema applied with `prisma db push` (no migration file - dev environment)
- Baseline score is recalculated when profile type changes (maintains consistency)
- Authority record is lazily created on first GET /api/authority call
- User.authorityScore field synced with UserAuthority.computedScore for quick access
- All 24 tests passing (11 backend + 13 frontend)
