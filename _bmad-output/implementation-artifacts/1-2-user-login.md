# Story 1.2: User Login

Status: ready-for-dev

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

- [ ] **Backend: Auth Service & Token Management** (Platform Layer - Express)
  - [ ] Implement `POST /api/v1/auth/login` endpoint
    - [ ] Validate request body (email, password format)
    - [ ] Rate limit check (Redis or memory)
    - [ ] Verify credentials against `users` table (bcrypt compare)
    - [ ] Generate JWT Access Token (1h, RS256/HS256)
    - [ ] Generate Refresh Token (7d, opaque or JWT)
    - [ ] Store Refresh Token hash in DB (optional but recommended for revocation)
  - [ ] Implement `POST /api/v1/auth/refresh` endpoint
    - [ ] Validate Refresh Token
    - [ ] Issue new Access + Refresh pair (Rotation)
  - [ ] Implement Rate Limiter Middleware (5 attempts / 15m)

- [ ] **Frontend: Login UI** (VisuaLex Platform)
  - [ ] Create `LoginPage` component
    - [ ] Email & Password fields with validation
    - [ ] "Forgot Password" link (placeholder for now)
    - [ ] Error message display
    - [ ] Loading state handling
  - [ ] Style with Tailwind v4 (Glassmorphism/Premium feel)

- [ ] **Frontend: Auth Integration**
  - [ ] Implement `useAuth` hook / Context
    - [ ] `login(email, password)` function
    - [ ] Token storage (HttpOnly cookie preferred for Refresh, memory for Access)
    - [ ] User state management (isLoggedIn, userProfile)
  - [ ] Implement Axios/Fetch Interceptor
    - [ ] Auto-attach Access Token to requests
    - [ ] Handle 401: Call refresh endpoint, retry original request
    - [ ] Redirect to Login on refresh failure

## Dev Notes

### Architecture Patterns
- **Layer:** Platform Backend (Express 5).
- **Database:** PostgreSQL (`users` table).
- **Security:** Ensure `bcrypt` is used for password verification. Do not log passwords.
- **API Versioning:** Use `/api/v1/` prefix (ADR-003).

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

Antigravity (simulating BMad workflow)

### Completion Notes List

- Story created based on Epics and Architecture documentation.
- Identified specific endpoints and security constraints.
