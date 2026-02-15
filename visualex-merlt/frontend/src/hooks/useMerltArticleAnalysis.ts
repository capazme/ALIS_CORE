/**
 * useMerltArticleAnalysis Hook
 *
 * Fetches MERLT analysis data (entities, relations) for an article
 * by calling the graph API endpoints via merltService.
 */

import { useState, useEffect, useCallback } from 'react';
import type { Entity, Relation } from '../components/EntityList';
import { getArticleEntities, getArticleRelations } from '../services/merltService';

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
  const [data, setData] = useState(undefined as unknown as ArticleAnalysis | undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null as Error | null);

  const fetchAnalysis = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [entitiesResult, relationsResult] = await Promise.all([
        getArticleEntities(urn),
        getArticleRelations(urn),
      ]);

      setData({
        entities: entitiesResult.entities ?? [],
        relations: relationsResult.relations ?? [],
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, [urn]);

  useEffect(() => {
    fetchAnalysis();
  }, [fetchAnalysis]);

  return {
    entities: data?.entities,
    relations: data?.relations,
    isLoading,
    error,
    refetch: fetchAnalysis,
  };
}
