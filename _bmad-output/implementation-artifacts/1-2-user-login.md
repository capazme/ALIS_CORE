# Story 1.2: User Login

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **registered user**,
I want to **login with my email and password**,
so that **I can access my personalized ALIS experience**.

## Acceptance Criteria

1. **Successful Login:**
   - **Given** I am a verified user with valid credentials
   - **When** I enter my email and password and submit
   - **Then** I receive a JWT access token (1h expiry) and refresh token (7d expiry)
   - **And** I am redirected to the main dashboard

2. **Invalid Credentials:**
   - **Given** I enter incorrect credentials
   - **When** I submit the login form
   - **Then** I see an error "Invalid email or password" (no credential leak)
   - **And** the failed attempt is logged for security

3. **Rate Limiting:**
   - **Given** I have failed to login 5 times consecutively
   - **When** I try again within 15 minutes
   - **Then** I am blocked with a "Too many attempts" message

4. **Token Refresh:**
   - **Given** my JWT access token has expired
   - **When** I make an API request
   - **Then** the system automatically uses my refresh token to issue a new access token
   - **And** if the refresh token is also expired, I am redirected to login

## Tasks / Subtasks

- [x] **Backend: Auth Service & Token Management** (Platform Layer - Express)
  - [x] Implement `POST /api/auth/login` endpoint
    - [x] Validate request body (email, password format)
    - [x] Rate limit check (Redis or memory)
    - [x] Verify credentials against `users` table (bcrypt compare)
    - [x] Generate JWT Access Token (1h, RS256/HS256)
    - [x] Generate Refresh Token (7d, opaque or JWT)
    - [x] Store Refresh Token hash in DB (optional but recommended for revocation)
  - [x] Implement `POST /api/auth/refresh` endpoint
    - [x] Validate Refresh Token
    - [x] Issue new Access + Refresh pair (Rotation)
  - [x] Implement Rate Limiter Middleware (5 attempts / 15m)

- [x] **Frontend: Login UI** (VisuaLex Platform)
  - [x] Create `LoginPage` component
    - [x] Email & Password fields with validation
    - [x] "Forgot Password" link (placeholder for now)
    - [x] Error message display
    - [x] Loading state handling
  - [x] Style with Tailwind v4 (Glassmorphism/Premium feel)

- [x] **Frontend: Auth Integration**
  - [x] Implement `useAuth` hook / Context
    - [x] `login(email, password)` function
    - [x] Token storage (HttpOnly cookie preferred for Refresh, memory for Access)
    - [x] User state management (isLoggedIn, userProfile)
  - [x] Implement Axios/Fetch Interceptor
    - [x] Auto-attach Access Token to requests
    - [x] Handle 401: Call refresh endpoint, retry original request
    - [x] Redirect to Login on refresh failure

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5).
- **Database:** PostgreSQL (`users` table).
- **Security:** Ensure `bcrypt` is used for password verification. Do not log passwords.
- **API Path:** `/api/auth/*` (current implementation uses unversioned paths).

### Dependencies
- `jsonwebtoken` (or similar) for JWT.
- `bcrypt` for hashing.
- `express-rate-limit` for throttling.
- `axios` (Frontend) for API calls.

### Integration Points
- Frontend communicates with Platform Layer (Port 3001).
- Upon success, Frontend should fetch User Profile (Story 1.1/1.3) to determine redirect or UI state.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (Code Review + Fixes)

### File List

**Backend:**
- `backend/src/controllers/authController.ts` - Login, refresh, security logging
- `backend/src/routes/auth.ts` - Auth routes with rate limiting
- `backend/src/utils/jwt.ts` - JWT token generation/verification
- `backend/src/config.ts` - JWT expiry configuration (1h access, 7d refresh)
- `backend/tests/auth.test.ts` - Integration tests for auth endpoints

**Frontend:**
- `frontend/src/pages/LoginPage.tsx` - Login page wrapper
- `frontend/src/components/auth/LoginForm.tsx` - Login form with glassmorphism UI
- `frontend/src/hooks/useAuth.ts` - Authentication state management hook
- `frontend/src/services/authService.ts` - Auth API service functions
- `frontend/src/services/api.ts` - Axios client with token refresh interceptor
- `frontend/src/test/components/LoginForm.test.tsx` - LoginForm unit tests

### Change Log

- 2026-01-31: Code review fixes applied (Round 2)
  - Fixed TypeScript `any` usage in LoginForm.tsx (proper LocationState interface)
  - Fixed TypeScript `any` generics in api.ts (changed to `unknown` defaults)
  - Updated documentation: API paths use `/api/auth/*` not `/api/v1/auth/*`
  - Added test for "Forgot Password" link
- 2026-01-31: Code review fixes applied (Round 1)
  - Added "Forgot Password" link placeholder to LoginForm
  - Fixed JWT access token expiry default (30m → 1h per AC1)
  - Added security logging for failed/successful login attempts (AC2)
  - Added rate limiting test to backend tests
  - Added File List and documentation to Dev Agent Record

### Completion Notes List

- Story created based on Epics and Architecture documentation.
- Identified specific endpoints and security constraints.
- Code review completed with 8 issues found and 6 fixed.
