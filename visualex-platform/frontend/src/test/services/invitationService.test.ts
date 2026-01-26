/**
 * Tests for invitation service
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { validateInvitation, createInvitation, listMyInvitations, revokeInvitation } from '@/services/invitationService';

// Mock the api module
const mockGet = vi.fn();
const mockPost = vi.fn();
const mockDel = vi.fn();

vi.mock('@/services/api', () => ({
  get: (...args: unknown[]) => mockGet(...args),
  post: (...args: unknown[]) => mockPost(...args),
  del: (...args: unknown[]) => mockDel(...args),
}));

describe('invitationService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('validateInvitation', () => {
    it('validates a valid invitation token', async () => {
      const mockResponse = {
        valid: true,
        email: 'invited@example.com',
        expires_at: '2026-02-01T00:00:00Z',
        inviter: { username: 'john_doe' },
      };
      mockGet.mockResolvedValue(mockResponse);

      const result = await validateInvitation('valid-token-uuid');

      expect(mockGet).toHaveBeenCalledWith('/invitations/valid-token-uuid/validate');
      expect(result.valid).toBe(true);
      expect(result.email).toBe('invited@example.com');
      expect(result.inviter?.username).toBe('john_doe');
    });

    it('handles invalid invitation token', async () => {
      mockGet.mockRejectedValue({ message: 'Invalid or expired invitation' });

      await expect(validateInvitation('invalid-token')).rejects.toEqual({
        message: 'Invalid or expired invitation',
      });
    });

    it('handles expired invitation token', async () => {
      mockGet.mockRejectedValue({ message: 'This invitation has expired' });

      await expect(validateInvitation('expired-token')).rejects.toEqual({
        message: 'This invitation has expired',
      });
    });
  });

  describe('createInvitation', () => {
    it('creates an invitation without email', async () => {
      const mockResponse = {
        id: 'inv-123',
        token: 'new-token-uuid',
        expires_at: '2026-02-01T00:00:00Z',
        created_at: '2026-01-25T00:00:00Z',
      };
      mockPost.mockResolvedValue(mockResponse);

      const result = await createInvitation();

      expect(mockPost).toHaveBeenCalledWith('/invitations', {});
      expect(result.token).toBe('new-token-uuid');
    });

    it('creates an invitation with specific email', async () => {
      const mockResponse = {
        id: 'inv-123',
        token: 'new-token-uuid',
        email: 'specific@example.com',
        expires_at: '2026-02-01T00:00:00Z',
        created_at: '2026-01-25T00:00:00Z',
      };
      mockPost.mockResolvedValue(mockResponse);

      const result = await createInvitation({ email: 'specific@example.com' });

      expect(mockPost).toHaveBeenCalledWith('/invitations', { email: 'specific@example.com' });
      expect(result.email).toBe('specific@example.com');
    });
  });

  describe('listMyInvitations', () => {
    it('lists user invitations', async () => {
      const mockResponse = [
        { id: 'inv-1', token: 'token-1', expires_at: '2026-02-01T00:00:00Z', created_at: '2026-01-25T00:00:00Z' },
        { id: 'inv-2', token: 'token-2', email: 'user@example.com', expires_at: '2026-02-01T00:00:00Z', created_at: '2026-01-24T00:00:00Z', used_at: '2026-01-25T00:00:00Z' },
      ];
      mockGet.mockResolvedValue(mockResponse);

      const result = await listMyInvitations();

      expect(mockGet).toHaveBeenCalledWith('/invitations');
      expect(result).toHaveLength(2);
      expect(result[1].used_at).toBeDefined();
    });
  });

  describe('revokeInvitation', () => {
    it('revokes an invitation', async () => {
      mockDel.mockResolvedValue(undefined);

      await revokeInvitation('inv-123');

      expect(mockDel).toHaveBeenCalledWith('/invitations/inv-123');
    });

    it('handles error when revoking non-existent invitation', async () => {
      mockDel.mockRejectedValue({ message: 'Invitation not found' });

      await expect(revokeInvitation('non-existent')).rejects.toEqual({
        message: 'Invitation not found',
      });
    });
  });
});
