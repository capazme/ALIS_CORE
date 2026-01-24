/**
 * Tests for JWT utilities
 */
import { describe, it, expect, beforeEach } from '@jest/globals';

// Mock environment before importing
process.env.JWT_SECRET = 'test-secret-key-for-testing';
process.env.JWT_EXPIRY = '1h';

describe('JWT Utilities', () => {
  let jwtUtils: typeof import('../../src/utils/jwt');

  beforeEach(async () => {
    // Clear module cache to reset state
    jest.resetModules();
    jwtUtils = await import('../../src/utils/jwt');
  });

  describe('generateToken', () => {
    it('generates a valid JWT token', () => {
      const payload = { userId: '123', email: 'test@example.com' };
      const token = jwtUtils.generateToken(payload);

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3); // JWT has 3 parts
    });

    it('includes payload in token', () => {
      const payload = { userId: '123', email: 'test@example.com' };
      const token = jwtUtils.generateToken(payload);
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded.userId).toBe('123');
      expect(decoded.email).toBe('test@example.com');
    });
  });

  describe('verifyToken', () => {
    it('verifies a valid token', () => {
      const payload = { userId: '123' };
      const token = jwtUtils.generateToken(payload);
      const decoded = jwtUtils.verifyToken(token);

      expect(decoded).toBeDefined();
      expect(decoded.userId).toBe('123');
    });

    it('throws error for invalid token', () => {
      expect(() => {
        jwtUtils.verifyToken('invalid-token');
      }).toThrow();
    });

    it('throws error for expired token', async () => {
      // Create a token that expires immediately
      const shortLivedToken = jwtUtils.generateToken(
        { userId: '123' },
        '1ms'
      );

      // Wait for token to expire
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(() => {
        jwtUtils.verifyToken(shortLivedToken);
      }).toThrow();
    });

    it('throws error for tampered token', () => {
      const token = jwtUtils.generateToken({ userId: '123' });
      const tamperedToken = token.slice(0, -5) + 'xxxxx';

      expect(() => {
        jwtUtils.verifyToken(tamperedToken);
      }).toThrow();
    });
  });

  describe('decodeToken', () => {
    it('decodes token without verification', () => {
      const payload = { userId: '123', email: 'test@example.com' };
      const token = jwtUtils.generateToken(payload);
      const decoded = jwtUtils.decodeToken(token);

      expect(decoded).toBeDefined();
      expect(decoded.userId).toBe('123');
    });
  });
});
