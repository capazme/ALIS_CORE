/**
 * Unit Tests: Auth Service
 * =========================
 *
 * Tests for:
 * - isAuthenticated() — token check
 * - getAccessToken() — retrieve from localStorage
 * - getRefreshToken() — retrieve from localStorage
 * - logout() — clear tokens
 *
 * Priority Tags: [P0] Critical  [P1] High  [P2] Medium
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock api module
vi.mock('../../services/api', () => ({
    post: vi.fn(),
    get: vi.fn(),
    put: vi.fn(),
}));

describe('Auth Service', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    describe('[P0] isAuthenticated', () => {
        it('should return false when no token is stored', async () => {
            // GIVEN: No access_token in localStorage
            // WHEN: Checking auth status
            const { isAuthenticated } = await import('../../services/authService');

            // THEN: Returns false
            expect(isAuthenticated()).toBe(false);
        });

        it('should return true when token exists', async () => {
            // GIVEN: An access_token in localStorage
            localStorage.setItem('access_token', 'test-jwt-token');

            // WHEN: Checking auth status
            const { isAuthenticated } = await import('../../services/authService');

            // THEN: Returns true
            expect(isAuthenticated()).toBe(true);
        });
    });

    describe('[P1] getAccessToken', () => {
        it('should return null when no token is stored', async () => {
            // GIVEN: No token in localStorage
            // WHEN: Getting access token
            const { getAccessToken } = await import('../../services/authService');

            // THEN: Returns null
            expect(getAccessToken()).toBeNull();
        });

        it('should return the stored token', async () => {
            // GIVEN: A token is stored
            localStorage.setItem('access_token', 'my-jwt-token');

            // WHEN: Getting access token
            const { getAccessToken } = await import('../../services/authService');

            // THEN: Returns the token
            expect(getAccessToken()).toBe('my-jwt-token');
        });
    });

    describe('[P1] getRefreshToken', () => {
        it('should return null when no refresh token is stored', async () => {
            // GIVEN: No refresh token
            // WHEN: Getting refresh token
            const { getRefreshToken } = await import('../../services/authService');

            // THEN: Returns null
            expect(getRefreshToken()).toBeNull();
        });

        it('should return the stored refresh token', async () => {
            // GIVEN: A refresh token is stored
            localStorage.setItem('refresh_token', 'my-refresh-token');

            // WHEN: Getting refresh token
            const { getRefreshToken } = await import('../../services/authService');

            // THEN: Returns the token
            expect(getRefreshToken()).toBe('my-refresh-token');
        });
    });

    describe('[P0] logout', () => {
        it('should clear tokens from localStorage', async () => {
            // GIVEN: Tokens are stored
            localStorage.setItem('access_token', 'test-token');
            localStorage.setItem('refresh_token', 'test-refresh');

            // Mock window.location.href (jsdom doesn't support navigation)
            const originalHref = window.location.href;
            Object.defineProperty(window, 'location', {
                value: { ...window.location, href: originalHref },
                writable: true,
            });

            // WHEN: Logging out
            const { logout } = await import('../../services/authService');
            logout();

            // THEN: Tokens are removed
            expect(localStorage.getItem('access_token')).toBeNull();
            expect(localStorage.getItem('refresh_token')).toBeNull();
        });
    });
});
