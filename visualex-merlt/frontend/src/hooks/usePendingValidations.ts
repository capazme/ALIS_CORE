/**
 * usePendingValidations Hook
 *
 * Fetches pending validation items for the current user/article
 * using the centralized merltService.
 */

import { useState, useEffect, useCallback } from 'react';
import { merltService } from '../services/merltService';
import type { PendingEntity, PendingRelation } from '../types/merlt';

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
  submitDecision: (validationId: string, decision: 'approve' | 'reject', type: 'entity' | 'relation') => Promise<void>;
  refetch: () => Promise<void>;
}

function mapEntityToValidation(entity: PendingEntity): ValidationItem {
  return {
    id: entity.id,
    type: 'entity',
    content: {
      name: entity.nome,
      description: entity.descrizione,
      confidence: entity.llm_confidence,
    },
    articleUrn: entity.articoli_correlati[0] ?? '',
    createdAt: entity.created_at,
  };
}

function mapRelationToValidation(relation: PendingRelation): ValidationItem {
  return {
    id: relation.id,
    type: 'relation',
    content: {
      name: `${relation.source_urn} → ${relation.target_urn}`,
      description: relation.evidence,
      confidence: relation.llm_confidence,
    },
    articleUrn: '',
    createdAt: relation.created_at,
  };
}

export function usePendingValidations(articleUrn: string, userId?: string): UsePendingValidationsResult {
  const [validations, setValidations] = useState(undefined as unknown as ValidationItem[] | undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null as Error | null);

  const effectiveUserId = userId || 'anonymous';

  const fetchValidations = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const queue = await merltService.getPendingQueue(effectiveUserId, {
        article_urn: articleUrn,
      });

      const items: ValidationItem[] = [
        ...queue.pending_entities.map(mapEntityToValidation),
        ...queue.pending_relations.map(mapRelationToValidation),
      ];

      setValidations(items);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, [articleUrn, effectiveUserId]);

  const submitDecision = useCallback(
    async (validationId: string, decision: 'approve' | 'reject', type: 'entity' | 'relation') => {
      const vote = decision === 'approve' ? 'approve' : 'reject';

      if (type === 'entity') {
        await merltService.validateEntity({
          entity_id: validationId,
          user_id: effectiveUserId,
          vote: vote as 'approve' | 'reject',
        });
      } else {
        await merltService.validateRelation({
          relation_id: validationId,
          user_id: effectiveUserId,
          vote: vote as 'approve' | 'reject',
        });
      }

      // Remove from local state
      setValidations((prev: ValidationItem[] | undefined) => prev?.filter((v: ValidationItem) => v.id !== validationId));
    },
    [effectiveUserId]
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
