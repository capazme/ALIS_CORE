/**
 * Tests for JWT utilities
 *
 * Tests the actual implementation:
 * - generateAccessToken(userId, email)
 * - generateRefreshToken(userId, email)
 * - verifyToken(token) -> TokenPayload | null
 * - verifyTokenType(payload, expectedType)
 */
import { describe, it, expect, beforeEach } from '@jest/globals';

// Mock environment before importing
process.env.JWT_SECRET = 'test-secret-key-for-testing';
process.env.JWT_ACCESS_EXPIRY = '15m';
process.env.JWT_REFRESH_EXPIRY = '7d';

describe('JWT Utilities', () => {
  let jwtUtils: typeof import('../../src/utils/jwt');

  beforeEach(async () => {
    // Clear module cache to reset state
    jest.resetModules();
    jwtUtils = await import('../../src/utils/jwt');
  });

  describe('generateAccessToken', () => {
    it('generates a valid JWT token', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3); // JWT has 3 parts
    });

    it('includes userId and email in token payload', () => {
      const userId = 'user-123';
      const email = 'test@example.com';
      const token = jwtUtils.generateAccessToken(userId, email);
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      expect(decoded?.userId).toBe(userId);
      expect(decoded?.email).toBe(email);
    });

    it('sets token type to access', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded?.type).toBe('access');
    });

    it('includes jti (JWT ID) for uniqueness', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded?.jti).toBeDefined();
      expect(typeof decoded?.jti).toBe('string');
    });

    it('generates unique tokens for same user', () => {
      const token1 = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const token2 = jwtUtils.generateAccessToken('user-123', 'test@example.com');

      expect(token1).not.toBe(token2);
    });
  });

  describe('generateRefreshToken', () => {
    it('generates a valid JWT token', () => {
      const token = jwtUtils.generateRefreshToken('user-123', 'test@example.com');

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3);
    });

    it('sets token type to refresh', () => {
      const token = jwtUtils.generateRefreshToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded?.type).toBe('refresh');
    });

    it('includes userId and email in payload', () => {
      const userId = 'user-456';
      const email = 'refresh@example.com';
      const token = jwtUtils.generateRefreshToken(userId, email);
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded?.userId).toBe(userId);
      expect(decoded?.email).toBe(email);
    });
  });

  describe('verifyToken', () => {
    it('verifies a valid access token', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      expect(decoded?.userId).toBe('user-123');
    });

    it('verifies a valid refresh token', () => {
      const token = jwtUtils.generateRefreshToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      expect(decoded?.userId).toBe('user-123');
    });

    it('returns null for invalid token', () => {
      const result = jwtUtils.verifyToken('invalid-token');

      expect(result).toBeNull();
    });

    it('returns null for tampered token', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const tamperedToken = token.slice(0, -5) + 'xxxxx';

      const result = jwtUtils.verifyToken(tamperedToken);

      expect(result).toBeNull();
    });

    it('returns null for empty token', () => {
      const result = jwtUtils.verifyToken('');

      expect(result).toBeNull();
    });

    it('returns null for malformed JWT', () => {
      const result = jwtUtils.verifyToken('not.a.valid.jwt.token');

      expect(result).toBeNull();
    });
  });

  describe('verifyTokenType', () => {
    it('returns true when token type matches expected', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      const result = jwtUtils.verifyTokenType(decoded!, 'access');

      expect(result).toBe(true);
    });

    it('returns false when token type does not match', () => {
      const token = jwtUtils.generateAccessToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      const result = jwtUtils.verifyTokenType(decoded!, 'refresh');

      expect(result).toBe(false);
    });

    it('correctly validates refresh token type', () => {
      const token = jwtUtils.generateRefreshToken('user-123', 'test@example.com');
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).not.toBeNull();
      expect(jwtUtils.verifyTokenType(decoded!, 'refresh')).toBe(true);
      expect(jwtUtils.verifyTokenType(decoded!, 'access')).toBe(false);
    });
  });
});
