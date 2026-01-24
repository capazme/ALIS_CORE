/**
 * usePendingValidations Hook
 *
 * Fetches pending validation items for the current user/article.
 */

import { useState, useEffect, useCallback } from 'react';
import { getMerltConfig } from '../services/merltInit';

export interface ValidationItem {
  id: string;
  type: 'entity' | 'relation';
  content: {
    name: string;
    description?: string;
    confidence: number;
  };
  articleUrn: string;
  createdAt: string;
}

interface UsePendingValidationsResult {
  validations: ValidationItem[] | undefined;
  isLoading: boolean;
  error: Error | null;
  submitDecision: (validationId: string, decision: 'approve' | 'reject') => Promise<void>;
  refetch: () => Promise<void>;
}

export function usePendingValidations(articleUrn: string): UsePendingValidationsResult {
  const [validations, setValidations] = useState<ValidationItem[] | undefined>();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchValidations = useCallback(async () => {
    const config = getMerltConfig();
    if (!config) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const token = await config.getAuthToken();
      const response = await fetch(
        `${config.apiBaseUrl}/merlt/validations/pending?articleUrn=${encodeURIComponent(articleUrn)}`,
        {
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch validations: ${response.statusText}`);
      }

      const result = await response.json();
      setValidations(result.items ?? []);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, [articleUrn]);

  const submitDecision = useCallback(
    async (validationId: string, decision: 'approve' | 'reject') => {
      const config = getMerltConfig();
      if (!config) {
        throw new Error('MERLT not initialized');
      }

      const token = await config.getAuthToken();
      const response = await fetch(`${config.apiBaseUrl}/merlt/validations/${validationId}/decision`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ decision }),
      });

      if (!response.ok) {
        throw new Error(`Failed to submit decision: ${response.statusText}`);
      }

      // Remove from local state
      setValidations((prev) => prev?.filter((v) => v.id !== validationId));
    },
    []
  );

  useEffect(() => {
    fetchValidations();
  }, [fetchValidations]);

  return {
    validations,
    isLoading,
    error,
    submitDecision,
    refetch: fetchValidations,
  };
}
