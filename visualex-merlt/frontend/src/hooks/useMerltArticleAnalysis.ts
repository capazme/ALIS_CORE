/**
 * useMerltArticleAnalysis Hook
 *
 * Fetches MERLT analysis data (entities, relations) for an article.
 */

import { useState, useEffect } from 'react';
import type { Entity, Relation } from '../components/EntityList';
import { getMerltConfig } from '../services/merltInit';

interface ArticleAnalysis {
  entities: Entity[];
  relations: Relation[];
}

interface UseMerltArticleAnalysisResult {
  entities: Entity[] | undefined;
  relations: Relation[] | undefined;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useMerltArticleAnalysis(urn: string): UseMerltArticleAnalysisResult {
  const [data, setData] = useState<ArticleAnalysis | undefined>();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchAnalysis = async () => {
    const config = getMerltConfig();
    if (!config) {
      setError(new Error('MERLT not initialized'));
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const token = await config.getAuthToken();
      const response = await fetch(`${config.apiBaseUrl}/merlt/articles/${encodeURIComponent(urn)}/analysis`, {
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch analysis: ${response.statusText}`);
      }

      const result = await response.json();
      setData({
        entities: result.entities ?? [],
        relations: result.relations ?? [],
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis();
  }, [urn]);

  return {
    entities: data?.entities,
    relations: data?.relations,
    isLoading,
    error,
    refetch: fetchAnalysis,
  };
}
