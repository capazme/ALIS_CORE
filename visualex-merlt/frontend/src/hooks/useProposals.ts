/**
 * useProposals Hook
 *
 * Handles submitting new entity/relation proposals
 * using the centralized merltService.
 */

import { useState, useCallback } from 'react';
import { merltService } from '../services/merltService';
import type { EntityType, RelationType } from '../types/merlt';

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

/** Map UI entity type values to backend EntityType. */
function toEntityType(uiType: string | undefined): EntityType {
  const map: Record<string, EntityType> = {
    'CONCEPT': 'concetto',
    'SUBJECT': 'soggetto_giuridico',
    'CONDITION': 'fatto_giuridico',
    'EFFECT': 'fatto_giuridico',
    'PROCEDURE': 'procedura',
  };
  return map[uiType ?? ''] ?? 'concetto';
}

export function useProposals(userId?: string): UseProposalsResult {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null as Error | null);

  const effectiveUserId = userId || 'anonymous';

  const submitProposal = useCallback(async (data: ProposalData) => {
    setIsSubmitting(true);
    setError(null);

    try {
      if (data.type === 'entity') {
        const result = await merltService.proposeEntity({
          tipo: toEntityType(data.entityType),
          nome: data.name,
          descrizione: data.description ?? '',
          article_urn: data.articleUrn,
          ambito: 'generale',
          evidence: data.description ?? data.name,
          user_id: effectiveUserId,
        });
        return { id: result.pending_entity?.id ?? '' };
      } else {
        const result = await merltService.proposeRelation({
          tipo_relazione: (data.relationType ?? 'CORRELATO') as RelationType,
          source_urn: data.sourceId ?? '',
          target_entity_id: data.targetId ?? '',
          article_urn: data.articleUrn,
          descrizione: data.description ?? '',
          certezza: 0.5,
          user_id: effectiveUserId,
        });
        return { id: result.relation_id ?? '' };
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      return null;
    } finally {
      setIsSubmitting(false);
    }
  }, [effectiveUserId]);

  return {
    submitProposal,
    isSubmitting,
    error,
  };
}
