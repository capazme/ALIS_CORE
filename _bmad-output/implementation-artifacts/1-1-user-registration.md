# Story 1.1: User Registration

Status: completed

## Story

As a **new member**,
I want to **register with my email and password after receiving an invitation**,
So that **I can access the ALIS platform as an association member**.

## Acceptance Criteria

1. **Given** I have received an invitation link from an existing member
   **When** I click the invitation link and enter my email, password, and basic info (name, role)
   **Then** my account is created with "pending" status
   **And** I receive a verification email
   **And** my password is hashed with bcrypt before storage

2. **Given** I try to register without a valid invitation
   **When** I submit the registration form
   **Then** I see an error "Registration requires an invitation from an existing member"

3. **Given** I try to register with an email already in use
   **When** I submit the registration form
   **Then** I see an error "This email is already registered"

4. **Given** I submit with an expired invitation (>7 days)
   **When** I complete the form
   **Then** I see an error "This invitation has expired"

5. **Given** I use an invitation that was already used
   **When** I try to register
   **Then** I see an error "This invitation has already been used"

## Tasks / Subtasks

- [x] Task 1: Create `invitations` table and Prisma migration (AC: #1, #2, #4, #5)
  - [x] 1.1 Add Invitation model to `schema.prisma`
  - [x] 1.2 Run `npx prisma migrate dev --name add_invitations`
  - [x] 1.3 Add `role` field to User model
  - [x] 1.4 Generate Prisma client

- [x] Task 2: Create invitation generation endpoint (AC: #1)
  - [x] 2.1 Create `invitationController.ts`
  - [x] 2.2 Add `POST /api/invitations` route (authenticated users only)
  - [x] 2.3 Generate secure token (crypto.randomUUID)
  - [x] 2.4 Set 7-day expiration
  - [x] 2.5 Store inviter_id for tracking

- [x] Task 3: Modify registration endpoint for invitation-based flow (AC: #1, #2, #3, #4, #5)
  - [x] 3.1 Update `registerSchema` to require `invitation_token`
  - [x] 3.2 Validate invitation exists and is valid
  - [x] 3.3 Check invitation not expired
  - [x] 3.4 Check invitation not already used
  - [x] 3.5 Mark invitation as used on successful registration
  - [x] 3.6 Add `role` to user creation

- [x] Task 4: Create email verification flow (AC: #1)
  - [x] 4.1 Create `email_verifications` table
  - [x] 4.2 Generate verification token on registration
  - [x] 4.3 Create email sending service (stub for MVP)
  - [x] 4.4 Add `GET /api/auth/verify-email/:token` endpoint
  - [x] 4.5 Set `isVerified: true` on successful verification

- [x] Task 5: Frontend registration form with invitation (AC: #1, #2)
  - [x] 5.1 Modify `RegisterForm.tsx` component for invitation flow
  - [x] 5.2 Parse invitation token from URL (`/register?token=xxx`)
  - [x] 5.3 Form fields: email, password, confirm password, name, role
  - [x] 5.4 Client-side validation (password strength, email format)
  - [x] 5.5 Error handling and display
  - [x] 5.6 Create `VerifyEmailPage.tsx` for email verification callback

- [x] Task 6: Tests (AC: all)
  - [x] 6.1 Unit tests for invitation service (8 tests - all pass)
  - [x] 6.2 Component tests for RegisterForm (15 tests - all pass)
  - [x] 6.3 Component tests for VerifyEmailPage (7 tests - all pass)

## Dev Notes

### 🔴 CRITICAL: Brownfield Context - Existing Code!

**This story MODIFIES existing code, not greenfield.** The following components already exist:

| File | Status | Action Required |
|------|--------|-----------------|
| `backend/src/controllers/authController.ts` | ✅ Exists | **MODIFY** `register()` function |
| `backend/src/routes/auth.ts` | ✅ Exists | **ADD** invitation routes |
| `backend/prisma/schema.prisma` | ✅ Exists | **ADD** Invitation model, role field |
| `backend/src/utils/password.ts` | ✅ Exists | **REUSE** - already uses bcrypt |
| `backend/src/utils/jwt.ts` | ✅ Exists | **REUSE** - no changes needed |

### Current Registration Flow (to be replaced)

```typescript
// CURRENT: Open registration with admin approval
router.post('/auth/register', authLimiter, authController.register);
// Creates user with isActive: false, awaits admin approval

// NEW: Invitation-based registration
router.post('/auth/register', authLimiter, authController.register);
// Requires valid invitation_token, still creates with isActive: false
```

### Database Schema Changes

```prisma
// ADD to schema.prisma

model Invitation {
  id        String    @id @default(uuid())
  email     String    // Target email (optional, can be open)
  token     String    @unique
  expiresAt DateTime  @map("expires_at")
  usedAt    DateTime? @map("used_at")
  inviterId String    @map("inviter_id")
  createdAt DateTime  @default(now()) @map("created_at")

  inviter   User      @relation("InvitationsSent", fields: [inviterId], references: [id])

  @@index([token])
  @@index([inviterId])
  @@map("invitations")
}

// MODIFY User model - add:
model User {
  // ... existing fields ...
  role        String    @default("member") // member, researcher, admin

  // ... existing relations ...
  invitationsSent Invitation[] @relation("InvitationsSent")
}

model EmailVerification {
  id        String    @id @default(uuid())
  userId    String    @unique @map("user_id")
  token     String    @unique
  expiresAt DateTime  @map("expires_at")
  createdAt DateTime  @default(now()) @map("created_at")

  user      User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@map("email_verifications")
}
```

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/invitations` | Required | Create invitation (returns token) |
| GET | `/api/invitations/:token/validate` | None | Check if invitation is valid |
| POST | `/api/auth/register` | None | Register with invitation token |
| GET | `/api/auth/verify-email/:token` | None | Verify email address |

### Request/Response Examples

**Create Invitation:**
```json
POST /api/invitations
Authorization: Bearer <token>
{ "email": "newuser@example.com" }  // email optional

Response:
{
  "invitation_url": "https://alis.app/register?token=abc123",
  "token": "abc123",
  "expires_at": "2026-02-01T00:00:00Z"
}
```

**Register:**
```json
POST /api/auth/register
{
  "invitation_token": "abc123",
  "email": "newuser@example.com",
  "password": "SecurePass123!",
  "username": "newuser",
  "name": "New User",
  "role": "member"
}

Response (201):
{
  "message": "Registration successful. Please verify your email.",
  "verification_pending": true
}
```

### Project Structure Notes

Follow existing patterns in `visualex-platform/backend/`:

```
backend/src/
├── controllers/
│   ├── authController.ts      # MODIFY
│   └── invitationController.ts # CREATE
├── routes/
│   ├── auth.ts                # MODIFY
│   └── invitations.ts         # CREATE
├── services/
│   └── emailService.ts        # CREATE (stub)
├── middleware/
│   └── auth.ts                # REUSE
└── utils/
    ├── jwt.ts                 # REUSE
    └── password.ts            # REUSE
```

### Security Requirements

- Password: min 8 chars, 1 uppercase, 1 lowercase, 1 number (already enforced)
- Invitation token: crypto.randomUUID() - 36 chars
- Email verification token: crypto.randomBytes(32).toString('hex') - 64 chars
- Rate limiting: 5 attempts/15min (already configured)
- Token expiration: Invitation 7 days, Email verification 24 hours

### NFR Compliance

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| NFR-S3 | JWT with rotation | ✅ Already implemented in jwt.ts |
| NFR-C1 | GDPR explicit consent | 🔲 Consent captured at registration (Story 1.4) |
| NFR-S5 | PII protection | Password hashed with bcrypt (cost 12) |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-1.1]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-002-GDPR-Consent]
- [Source: visualex-platform/CLAUDE.md]
- [Existing Code: visualex-platform/backend/src/controllers/authController.ts]
- [Existing Code: visualex-platform/backend/prisma/schema.prisma]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Prisma v7 incompatibility issue: downgraded to v5.22.0
- node_modules corruption in frontend: clean reinstall resolved

### Completion Notes List

1. **Backend Implementation:**
   - Added Invitation and EmailVerification models to Prisma schema
   - Created invitationController.ts with CRUD operations
   - Modified authController.ts for invitation-based registration
   - Created emailService.ts stub (console logging in dev mode)
   - Added rate limiting for invitation endpoints (10/hour)

2. **Frontend Implementation:**
   - Rewrote RegisterForm.tsx with invitation validation flow
   - Created VerifyEmailPage.tsx for email verification callback
   - Created invitationService.ts for invitation API calls
   - Updated types in api.ts for new request/response shapes
   - Added new route in App.tsx for /verify-email

3. **Tests:**
   - 30 new tests added (all pass):
     - invitationService.test.ts: 8 tests
     - RegisterForm.test.tsx: 15 tests
     - VerifyEmailPage.test.tsx: 7 tests

### File List

**Files created:**
- `backend/src/controllers/invitationController.ts`
- `backend/src/routes/invitations.ts`
- `backend/src/services/emailService.ts`
- `frontend/src/pages/VerifyEmailPage.tsx`
- `frontend/src/services/invitationService.ts`
- `frontend/src/test/services/invitationService.test.ts`
- `frontend/src/test/components/RegisterForm.test.tsx`
- `frontend/src/test/components/VerifyEmailPage.test.tsx`

**Files modified:**
- `backend/prisma/schema.prisma` (added Invitation, EmailVerification models; added name, role to User)
- `backend/src/controllers/authController.ts` (invitation-based registration, email verification endpoints)
- `backend/src/routes/auth.ts` (added verify-email and resend-verification routes)
- `backend/src/index.ts` (added invitation routes import)
- `frontend/src/App.tsx` (added /verify-email route)
- `frontend/src/components/auth/RegisterForm.tsx` (complete rewrite for invitation flow)
- `frontend/src/services/authService.ts` (added verifyEmail, resendVerificationEmail)
- `frontend/src/types/api.ts` (added invitation and verification types)
