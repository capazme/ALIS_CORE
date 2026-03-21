import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGet = vi.fn();
const mockPost = vi.fn();

vi.mock('../api', () => ({
  get: (...args: unknown[]) => mockGet(...args),
  post: (...args: unknown[]) => mockPost(...args),
}));

import {
  queryExperts,
  submitInlineFeedback,
  submitDetailedFeedback,
  submitSourceFeedback,
  submitRouterFeedback,
  checkArticleInGraph,
  getUserAuthority,
} from '../merltService';

describe('merltService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ===========================================================================
  // queryExperts
  // ===========================================================================

  describe('queryExperts', () => {
    it('[P1] should call correct endpoint and return response', async () => {
      const mockResponse = {
        trace_id: 'trace-123',
        synthesis: 'Test synthesis',
        experts: [],
        confidence: 0.85,
      };
      mockPost.mockResolvedValue(mockResponse);

      const request = {
        query: "Cos'e' la legittima difesa?",
        user_id: 'user-1',
        include_trace: true,
        consent_level: 'basic' as const,
      };

      const result = await queryExperts(request);

      expect(mockPost).toHaveBeenCalledWith('/merlt/experts/query', request);
      expect(result).toEqual(mockResponse);
    });

    it('[P1] should propagate network errors', async () => {
      mockPost.mockRejectedValue(new Error('Network error'));

      await expect(
        queryExperts({
          query: 'test',
          user_id: 'user-1',
          include_trace: false,
          consent_level: 'anonymous',
        }),
      ).rejects.toThrow('Network error');
    });

    it('[P1] should propagate HTTP error objects', async () => {
      const httpError = { status: 500, message: 'Internal Server Error', data: {} };
      mockPost.mockRejectedValue(httpError);

      await expect(
        queryExperts({
          query: 'test',
          user_id: 'user-1',
          include_trace: false,
          consent_level: 'anonymous',
        }),
      ).rejects.toEqual(httpError);
    });
  });

  // ===========================================================================
  // submitInlineFeedback
  // ===========================================================================

  describe('submitInlineFeedback', () => {
    it('[P1] should send inline feedback with correct payload', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      const result = await submitInlineFeedback('trace-1', 'user-1', 5);

      expect(mockPost).toHaveBeenCalledWith('/merlt/experts/feedback/inline', {
        trace_id: 'trace-1',
        user_id: 'user-1',
        rating: 5,
      });
      expect(result.success).toBe(true);
    });

    it('[P1] should accept all valid rating values', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      for (const rating of [1, 2, 3, 4, 5] as const) {
        await submitInlineFeedback('trace-1', 'user-1', rating);
        expect(mockPost).toHaveBeenLastCalledWith(
          '/merlt/experts/feedback/inline',
          expect.objectContaining({ rating }),
        );
      }
    });
  });

  // ===========================================================================
  // submitDetailedFeedback
  // ===========================================================================

  describe('submitDetailedFeedback', () => {
    it('[P1] should transform 1-5 scores to 0-1 range for backend', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      await submitDetailedFeedback({
        trace_id: 'trace-1',
        user_id: 'user-1',
        accuracy: 5,
        completeness: 3,
        relevance: 1,
        comment: 'Good',
      });

      expect(mockPost).toHaveBeenCalledWith('/merlt/experts/feedback/detailed', {
        trace_id: 'trace-1',
        user_id: 'user-1',
        retrieval_score: 1,       // (5-1)/4 = 1
        reasoning_score: 0.5,     // (3-1)/4 = 0.5
        synthesis_score: 0,       // (1-1)/4 = 0
        comment: 'Good',
      });
    });

    it('[P1] should send undefined comment when not provided', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      await submitDetailedFeedback({
        trace_id: 'trace-1',
        user_id: 'user-1',
        accuracy: 3,
        completeness: 3,
        relevance: 3,
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/merlt/experts/feedback/detailed',
        expect.objectContaining({ comment: undefined }),
      );
    });
  });

  // ===========================================================================
  // submitSourceFeedback
  // ===========================================================================

  describe('submitSourceFeedback', () => {
    it('[P1] should map frontend fields to backend fields', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      await submitSourceFeedback({
        trace_id: 'trace-1',
        user_id: 'user-1',
        source_id: 'urn:nir:stato:codice.civile:1942;262~art1218',
        rating: 5,
      });

      expect(mockPost).toHaveBeenCalledWith('/merlt/experts/feedback/source', {
        trace_id: 'trace-1',
        user_id: 'user-1',
        source_id: 'urn:nir:stato:codice.civile:1942;262~art1218',
        relevance: 0.9,
      });
    });
  });

  // ===========================================================================
  // submitRouterFeedback
  // ===========================================================================

  describe('submitRouterFeedback', () => {
    it('[P1] should pass router feedback data directly', async () => {
      mockPost.mockResolvedValue({ success: true, message: 'ok' });

      const data = {
        trace_id: 'trace-1',
        user_id: 'user-1',
        routing_correct: true,
        suggested_weights: { literal: 0.5, systemic: 0.5 },
        comment: 'Routing was correct',
      };

      await submitRouterFeedback(data);

      expect(mockPost).toHaveBeenCalledWith('/merlt/experts/feedback/router', data);
    });
  });

  // ===========================================================================
  // checkArticleInGraph
  // ===========================================================================

  describe('checkArticleInGraph', () => {
    it('[P1] should transform backend response to ArticleGraphStatus', async () => {
      mockGet.mockResolvedValue({
        in_graph: true,
        node_count: 15,
        has_entities: true,
        last_updated: '2026-01-01T00:00:00Z',
        article_urn: 'urn:nir:stato:codice.civile:1942;262~art1218',
      });

      const result = await checkArticleInGraph('codice civile', '1218');

      expect(result).toEqual({
        exists: true,
        node_id: 'urn:nir:stato:codice.civile:1942;262~art1218',
        pending_validation: false,
        entity_count: 15,
      });
    });

    it('[P1] should handle article not in graph', async () => {
      mockGet.mockResolvedValue({
        in_graph: false,
        node_count: 0,
        has_entities: false,
        last_updated: null,
        article_urn: null,
      });

      const result = await checkArticleInGraph('codice civile', '9999');

      expect(result).toEqual({
        exists: false,
        node_id: undefined,
        pending_validation: false,
        entity_count: 0,
      });
    });

    it('[P1] should include optional parameters in query string', async () => {
      mockGet.mockResolvedValue({
        in_graph: false,
        node_count: 0,
        has_entities: false,
        last_updated: null,
        article_urn: null,
      });

      await checkArticleInGraph('legge', '241', '241', '1990-08-07');

      const calledUrl = mockGet.mock.calls[0][0] as string;
      expect(calledUrl).toContain('tipo_atto=legge');
      expect(calledUrl).toContain('articolo=241');
      expect(calledUrl).toContain('numero_atto=241');
      expect(calledUrl).toContain('data=1990-08-07');
    });
  });

  // ===========================================================================
  // getUserAuthority
  // ===========================================================================

  describe('getUserAuthority', () => {
    it('[P1] should call correct endpoint with user_id', async () => {
      const mockAuthority = { score: 0.72, tier: 'expert', domain_scores: {} };
      mockGet.mockResolvedValue(mockAuthority);

      const result = await getUserAuthority('user-1');

      expect(mockGet).toHaveBeenCalledWith('/merlt/authority/user-1');
      expect(result).toEqual(mockAuthority);
    });
  });

  // ===========================================================================
  // Error scenarios
  // ===========================================================================

  describe('error handling', () => {
    it('[P1] should propagate 401 errors', async () => {
      const error = { status: 401, message: 'Unauthorized', data: {} };
      mockPost.mockRejectedValue(error);

      await expect(submitInlineFeedback('t', 'u', 5)).rejects.toEqual(error);
    });

    it('[P1] should propagate 500 errors', async () => {
      const error = { status: 500, message: 'Internal Server Error', data: {} };
      mockGet.mockRejectedValue(error);

      await expect(getUserAuthority('user-1')).rejects.toEqual(error);
    });
  });
});
