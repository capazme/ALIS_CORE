/**
 * Tests for merltService API client
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('merltService', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('submitValidation', () => {
    it('sends validation to API', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, validationId: 'v-1' }),
      });

      const { submitValidation } = await import(
        '../../frontend/src/services/merltService'
      );

      const result = await submitValidation({
        itemId: 'item-1',
        decision: 'approve',
        userId: 'user-1',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/validation'),
        expect.objectContaining({
          method: 'POST',
        })
      );
      expect(result.success).toBe(true);
    });

    it('handles validation error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: 'Invalid validation' }),
      });

      const { submitValidation } = await import(
        '../../frontend/src/services/merltService'
      );

      await expect(
        submitValidation({
          itemId: 'item-1',
          decision: 'approve',
          userId: 'user-1',
        })
      ).rejects.toThrow();
    });
  });

  describe('fetchPendingValidations', () => {
    it('fetches pending items for user', async () => {
      const mockItems = [
        { id: '1', type: 'entity', content: {} },
        { id: '2', type: 'relation', content: {} },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ items: mockItems, total: 2 }),
      });

      const { fetchPendingValidations } = await import(
        '../../frontend/src/services/merltService'
      );

      const result = await fetchPendingValidations('user-1');

      expect(result.items).toHaveLength(2);
      expect(result.total).toBe(2);
    });
  });

  describe('proposeEntity', () => {
    it('submits new entity proposal', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            proposalId: 'prop-1',
          }),
      });

      const { proposeEntity } = await import(
        '../../frontend/src/services/merltService'
      );

      const result = await proposeEntity({
        name: 'New Concept',
        entityType: 'CONCEPT',
        articleUrn: 'urn:nir:stato:codice.civile~art1453',
        description: 'A new legal concept',
        userId: 'user-1',
      });

      expect(result.success).toBe(true);
      expect(result.proposalId).toBe('prop-1');
    });
  });

  describe('proposeRelation', () => {
    it('submits new relation proposal', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            proposalId: 'prop-2',
          }),
      });

      const { proposeRelation } = await import(
        '../../frontend/src/services/merltService'
      );

      const result = await proposeRelation({
        sourceId: 'entity-1',
        targetId: 'entity-2',
        relationType: 'DEFINES',
        userId: 'user-1',
      });

      expect(result.success).toBe(true);
    });
  });
});
