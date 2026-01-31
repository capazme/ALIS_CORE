# Story 1.3: Profile Setup

Status: done

## Story

As a **logged-in user**,
I want to **select my usage profile and configure basic preferences**,
So that **ALIS adapts its interface and feedback options to my expertise level**.

## Acceptance Criteria

1. **Profile Selection:**
   - **Given** I am logged in for the first time (or access profile settings)
   - **When** I view the profile selector
   - **Then** I see 4 profile options with descriptions:
     - ⚡ Consultazione Rapida: "Risposte veloci, minima interazione"
     - 📖 Ricerca Assistita: "Esplorazione guidata con suggerimenti"
     - 🔍 Analisi Esperta: "Accesso completo a Expert trace e feedback"
     - 🎓 Contributore Attivo: "Feedback granulare, impatto sul training"

2. **Profile Saving:**
   - **Given** I select a profile
   - **When** I confirm my selection
   - **Then** my profile is saved to my user record
   - **And** the UI adapts to show/hide features based on profile
   - **And** I can change profile anytime from settings

3. **Contributore Restriction:**
   - **Given** I am a new user without expertise credentials
   - **When** I try to select 🎓 Contributore Attivo
   - **Then** I see a message explaining this profile requires authority score ≥ 0.5
   - **And** I am offered 🔍 Analisi Esperta as alternative

## Tasks / Subtasks

- [x] **Backend: Database Schema**
  - [x] Add `ProfileType` enum to Prisma schema
  - [x] Add `profileType` column to User model
  - [x] Add `UserPreferences` model (theme, language, notifications)
  - [x] Apply schema changes with Prisma db push

- [x] **Backend: Profile API Endpoints**
  - [x] `GET /api/profile` - Get current user profile and preferences
  - [x] `PUT /api/profile` - Update profile type
  - [x] `PUT /api/profile/preferences` - Update user preferences
  - [x] Add profile type validation (Contributore requires authority ≥ 0.5)

- [x] **Frontend: Profile Selector Component**
  - [x] Create `ProfileSelector.tsx` with 4 profile cards
  - [x] Add icons and descriptions for each profile
  - [x] Handle Contributore restriction with warning message
  - [x] Style with Tailwind (glassmorphism, consistent with login)

- [x] **Frontend: Settings Integration**
  - [x] Add profile section to Settings page
  - [x] Create profile service for API calls
  - [x] Update useAuth hook to include profile info
  - [x] Add preferences section (theme, language, notifications)

- [x] **Tests**
  - [x] Backend: Profile API endpoint tests (`profile.test.ts`)
  - [x] Frontend: ProfileSelector component tests (`ProfileSelector.test.tsx`)

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5)
- **Database:** PostgreSQL (users table + user_preferences table)
- **API Path:** `/api/profile/*`

### Profile Types (Enum)
```typescript
enum ProfileType {
  QUICK_CONSULTATION = 'quick_consultation',    // ⚡
  ASSISTED_RESEARCH = 'assisted_research',      // 📖
  EXPERT_ANALYSIS = 'expert_analysis',          // 🔍
  ACTIVE_CONTRIBUTOR = 'active_contributor'     // 🎓
}
```

### User Preferences Schema
```typescript
interface UserPreferences {
  userId: string;
  theme: 'light' | 'dark' | 'system';
  language: 'it' | 'en';
  notificationsEnabled: boolean;
}
```

### Dependencies
- Story 1.2 (User Login) - for authentication
- Authority score system (future Epic 7) - for Contributore restriction

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5

### File List

**Backend (Modified):**
- `visualex-platform/backend/prisma/schema.prisma` - Added ProfileType enum, UserPreferences model, profileType/authorityScore to User
- `visualex-platform/backend/src/index.ts` - Registered profile routes
- `visualex-platform/backend/src/controllers/authController.ts` - Added profile_type/authority_score to login response

**Backend (Created):**
- `visualex-platform/backend/src/controllers/profileController.ts` - Profile API controller (getProfile, updateProfile, updatePreferences)
- `visualex-platform/backend/src/routes/profile.ts` - Profile route definitions
- `visualex-platform/backend/tests/profile.test.ts` - Profile API endpoint tests (19 tests)

**Frontend (Modified):**
- `visualex-platform/frontend/src/types/api.ts` - Added ProfileType, UserPreferences, ProfileResponse types
- `visualex-platform/frontend/src/hooks/useAuth.ts` - Added profileType, authorityScore exports
- `visualex-platform/frontend/src/pages/SettingsPage.tsx` - Integrated ProfileSelector and Preferences sections

**Frontend (Created):**
- `visualex-platform/frontend/src/services/profileService.ts` - Profile API service
- `visualex-platform/frontend/src/components/features/profile/ProfileSelector.tsx` - Profile selector component
- `visualex-platform/frontend/src/components/features/profile/ProfileSelector.test.tsx` - Component tests (12 tests)
- `visualex-platform/frontend/src/components/features/profile/index.ts` - Barrel exports

### Change Log

| Date | Change | Files |
|------|--------|-------|
| 2026-01-31 | Created story file | 1-3-profile-setup.md |
| 2026-01-31 | Added ProfileType enum and UserPreferences model to schema | schema.prisma |
| 2026-01-31 | Applied schema with `prisma db push` | - |
| 2026-01-31 | Created profile controller with 3 endpoints | profileController.ts |
| 2026-01-31 | Created profile routes, registered in index.ts | profile.ts, index.ts |
| 2026-01-31 | Updated auth response to include profile info | authController.ts |
| 2026-01-31 | Added frontend types for profile system | api.ts |
| 2026-01-31 | Created profile service for API calls | profileService.ts |
| 2026-01-31 | Created ProfileSelector component with 4 profile cards | ProfileSelector.tsx |
| 2026-01-31 | Integrated ProfileSelector and Preferences in Settings | SettingsPage.tsx |
| 2026-01-31 | Updated useAuth hook with profile exports | useAuth.ts |
| 2026-01-31 | Created frontend component tests (12 tests) | ProfileSelector.test.tsx |
| 2026-01-31 | Created backend API tests (19 tests) | profile.test.ts |
| 2026-01-31 | Code review fixes: removed console.log, fixed `any` types | SettingsPage.tsx, api.ts |

### Completion Notes List

- Story created from Epic 1 definition in epics.md
- All 4 profile types implemented with correct descriptions
- Contributore restriction working with authority score >= 0.5 validation
- Preferences section includes theme (light/dark/system), language (it/en), notifications toggle
- Used `prisma db push` instead of formal migration for development speed (migration recommended for production)
- All tests passing: 12 frontend tests, 19 backend tests
