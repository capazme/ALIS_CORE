/**
 * Tests for password utilities
 *
 * Tests the actual implementation:
 * - hashPassword(password) -> Promise<string>
 * - verifyPassword(password, hashedPassword) -> Promise<boolean>
 */
import { describe, it, expect, beforeAll } from '@jest/globals';

describe('Password Utilities', () => {
  let passwordUtils: typeof import('../../src/utils/password');

  beforeAll(async () => {
    passwordUtils = await import('../../src/utils/password');
  });

  describe('hashPassword', () => {
    it('hashes a password', async () => {
      const password = 'mySecurePassword123';
      const hash = await passwordUtils.hashPassword(password);

      expect(hash).toBeDefined();
      expect(hash).not.toBe(password);
      expect(hash.length).toBeGreaterThan(20);
    });

    it('generates different hashes for same password (random salt)', async () => {
      const password = 'mySecurePassword123';
      const hash1 = await passwordUtils.hashPassword(password);
      const hash2 = await passwordUtils.hashPassword(password);

      // bcrypt uses random salt, so hashes should differ
      expect(hash1).not.toBe(hash2);
    });

    it('generates bcrypt-formatted hash', async () => {
      const password = 'testPassword';
      const hash = await passwordUtils.hashPassword(password);

      // bcrypt hashes start with $2a$ or $2b$
      expect(hash).toMatch(/^\$2[ab]\$/);
    });

    it('handles empty password', async () => {
      const hash = await passwordUtils.hashPassword('');

      expect(hash).toBeDefined();
      expect(hash.length).toBeGreaterThan(0);
    });

    it('handles unicode passwords', async () => {
      const password = 'пароль密码🔐';
      const hash = await passwordUtils.hashPassword(password);

      expect(hash).toBeDefined();
      // Verify we can still verify it
      const isValid = await passwordUtils.verifyPassword(password, hash);
      expect(isValid).toBe(true);
    });

    it('handles very long passwords', async () => {
      const longPassword = 'a'.repeat(100);
      const hash = await passwordUtils.hashPassword(longPassword);

      expect(hash).toBeDefined();
      const isValid = await passwordUtils.verifyPassword(longPassword, hash);
      expect(isValid).toBe(true);
    });
  });

  describe('verifyPassword', () => {
    it('returns true for matching password', async () => {
      const password = 'mySecurePassword123';
      const hash = await passwordUtils.hashPassword(password);

      const result = await passwordUtils.verifyPassword(password, hash);

      expect(result).toBe(true);
    });

    it('returns false for non-matching password', async () => {
      const password = 'mySecurePassword123';
      const wrongPassword = 'wrongPassword';
      const hash = await passwordUtils.hashPassword(password);

      const result = await passwordUtils.verifyPassword(wrongPassword, hash);

      expect(result).toBe(false);
    });

    it('returns false for empty password against valid hash', async () => {
      const hash = await passwordUtils.hashPassword('somePassword');

      const result = await passwordUtils.verifyPassword('', hash);

      expect(result).toBe(false);
    });

    it('returns false for similar but different password', async () => {
      const password = 'MyPassword123';
      const hash = await passwordUtils.hashPassword(password);

      // Test case sensitivity
      const result = await passwordUtils.verifyPassword('mypassword123', hash);

      expect(result).toBe(false);
    });

    it('returns false for password with extra whitespace', async () => {
      const password = 'MyPassword123';
      const hash = await passwordUtils.hashPassword(password);

      const result = await passwordUtils.verifyPassword('MyPassword123 ', hash);

      expect(result).toBe(false);
    });

    it('handles invalid hash format gracefully', async () => {
      const result = await passwordUtils.verifyPassword('password', 'invalid-hash');

      expect(result).toBe(false);
    });

    it('handles empty hash gracefully', async () => {
      const result = await passwordUtils.verifyPassword('password', '');

      expect(result).toBe(false);
    });
  });

  describe('Password hashing consistency', () => {
    it('can verify password immediately after hashing', async () => {
      const password = 'TestPassword!@#123';
      const hash = await passwordUtils.hashPassword(password);
      const isValid = await passwordUtils.verifyPassword(password, hash);

      expect(isValid).toBe(true);
    });

    it('maintains consistency across multiple verifications', async () => {
      const password = 'ConsistentPassword123';
      const hash = await passwordUtils.hashPassword(password);

      // Verify multiple times
      const results = await Promise.all([
        passwordUtils.verifyPassword(password, hash),
        passwordUtils.verifyPassword(password, hash),
        passwordUtils.verifyPassword(password, hash),
      ]);

      expect(results.every((r) => r === true)).toBe(true);
    });
  });
});
