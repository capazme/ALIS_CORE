/**
 * Tests for password utilities
 */
import { describe, it, expect } from '@jest/globals';

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

    it('generates different hashes for same password', async () => {
      const password = 'mySecurePassword123';
      const hash1 = await passwordUtils.hashPassword(password);
      const hash2 = await passwordUtils.hashPassword(password);

      // bcrypt uses random salt, so hashes should differ
      expect(hash1).not.toBe(hash2);
    });
  });

  describe('comparePassword', () => {
    it('returns true for matching password', async () => {
      const password = 'mySecurePassword123';
      const hash = await passwordUtils.hashPassword(password);

      const result = await passwordUtils.comparePassword(password, hash);

      expect(result).toBe(true);
    });

    it('returns false for non-matching password', async () => {
      const password = 'mySecurePassword123';
      const wrongPassword = 'wrongPassword';
      const hash = await passwordUtils.hashPassword(password);

      const result = await passwordUtils.comparePassword(wrongPassword, hash);

      expect(result).toBe(false);
    });

    it('returns false for empty password', async () => {
      const hash = await passwordUtils.hashPassword('somePassword');

      const result = await passwordUtils.comparePassword('', hash);

      expect(result).toBe(false);
    });
  });

  describe('validatePassword', () => {
    it('accepts strong password', () => {
      const strongPassword = 'MyStr0ngP@ssword!';

      const result = passwordUtils.validatePassword(strongPassword);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('rejects short password', () => {
      const shortPassword = 'Ab1!';

      const result = passwordUtils.validatePassword(shortPassword);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain(expect.stringMatching(/length|caratteri/i));
    });

    it('rejects password without uppercase', () => {
      const noUppercase = 'mypassword123!';

      const result = passwordUtils.validatePassword(noUppercase);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain(expect.stringMatching(/uppercase|maiuscola/i));
    });

    it('rejects password without number', () => {
      const noNumber = 'MyPassword!';

      const result = passwordUtils.validatePassword(noNumber);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain(expect.stringMatching(/number|numero/i));
    });
  });
});
