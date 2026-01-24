/**
 * useProposals Hook
 *
 * Handles submitting new entity/relation proposals.
 */

import { useState, useCallback } from 'react';
import { getMerltConfig } from '../services/merltInit';

export interface ProposalData {
  type: 'entity' | 'relation';
  articleUrn: string;
  name: string;
  entityType?: string;
  description?: string;
  // For relations
  sourceId?: string;
  targetId?: string;
  relationType?: string;
}

interface UseProposalsResult {
  submitProposal: (data: ProposalData) => Promise<{ id: string } | null>;
  isSubmitting: boolean;
  error: Error | null;
}

export function useProposals(): UseProposalsResult {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const submitProposal = useCallback(async (data: ProposalData) => {
    const config = getMerltConfig();
    if (!config) {
      setError(new Error('MERLT not initialized'));
      return null;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const token = await config.getAuthToken();
      const endpoint =
        data.type === 'entity'
          ? `${config.apiBaseUrl}/merlt/proposals/entity`
          : `${config.apiBaseUrl}/merlt/proposals/relation`;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message ?? `Failed to submit proposal: ${response.statusText}`);
      }

      const result = await response.json();
      return { id: result.proposalId };
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      return null;
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  return {
    submitProposal,
    isSubmitting,
    error,
  };
}
