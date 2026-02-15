/**
 * Hook per lo stato MERL-T di un articolo
 * ========================================
 *
 * Gestisce il caricamento dello stato dell'articolo nel Knowledge Graph
 * e la coda di entità/relazioni pending per validazione.
 *
 * L'enrichment usa un global store (Zustand) per continuare anche
 * se l'utente naviga altrove - la richiesta non si interrompe.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { merltService } from '../services/merltService';
import {
  useEnrichmentStore,
  getArticleKey,
  selectJobForArticle,
  selectResultForArticle,
} from '../store/useEnrichmentStore';
import type {
  ArticleGraphStatus,
  PendingEntity,
  PendingRelation,
  EntityValidationResponse,
  RelationValidationResponse,
  VoteType,
} from '../types/merlt';

interface MerltArticleState {
  // Loading states
  isLoading: boolean;
  isValidating: boolean;
  isEnriching: boolean;
  error: string | null;

  // Streaming progress message
  progressMessage?: string;

  // Article status
  graphStatus: ArticleGraphStatus | null;

  // Pending items for this article
  pendingEntities: PendingEntity[];
  pendingRelations: PendingRelation[];

  // Counts
  pendingCount: number;
  validatedCount: number;

  // Tracks if article has been processed (to avoid duplicate AI extractions)
  hasBeenProcessed: boolean;

  // Track entities/relations voted by user in this session (for UX sorting)
  votedEntityIds: Set<string>;
  votedRelationIds: Set<string>;

  // Track entities/relations "committed" (hidden from view until new proposals)
  committedEntityIds: Set<string>;
  committedRelationIds: Set<string>;
}

interface UseMerltArticleStatusOptions {
  tipo_atto: string;
  articolo: string;
  numero_atto?: string;
  data?: string;
  user_id?: string;
  enabled?: boolean;
}

interface UseMerltArticleStatusReturn extends MerltArticleState {
  // Actions
  refresh: () => Promise<{ graphStatus: ArticleGraphStatus; pendingEntities: PendingEntity[]; hasBeenProcessed: boolean; } | null | undefined>;
  validateEntity: (entityId: string, vote: VoteType, comment?: string) => Promise<EntityValidationResponse | null>;
  validateRelation: (relationId: string, vote: VoteType, comment?: string) => Promise<RelationValidationResponse | null>;
  requestEnrichment: () => void;  // Changed to sync - starts streaming in background

  // Commit feedback - hides voted items from view
  commitFeedback: () => void;

  // Check if user has voted on an item
  hasVotedEntity: (entityId: string) => boolean;
  hasVotedRelation: (relationId: string) => boolean;

  // Count of items voted but not committed
  uncommittedVoteCount: number;
}

export function useMerltArticleStatus(
  options: UseMerltArticleStatusOptions
): UseMerltArticleStatusReturn {
  const { tipo_atto, articolo, numero_atto, data, user_id, enabled = true } = options;

  const articleKey = useMemo(
    () => getArticleKey(tipo_atto, articolo, numero_atto, data),
    [tipo_atto, articolo, numero_atto, data]
  );

  // Selettori reattivi dal global store
  const enrichmentJob = useEnrichmentStore(selectJobForArticle(articleKey));
  const enrichmentResult = useEnrichmentStore(selectResultForArticle(articleKey));
  const startEnrichmentStreaming = useEnrichmentStore(state => state.startEnrichmentStreaming);

  // Derivati dal job globale
  const isEnrichingGlobal = enrichmentJob?.status === 'in_progress';
  const progressMessage = enrichmentJob?.progressMessage;

  const [state, setState] = useState({
    isLoading: false,
    isValidating: false,
    isEnriching: false,
    error: null,
    graphStatus: null,
    pendingEntities: [],
    pendingRelations: [],
    pendingCount: 0,
    validatedCount: 0,
    hasBeenProcessed: false,
    votedEntityIds: new Set(),
    votedRelationIds: new Set(),
    committedEntityIds: new Set(),
    committedRelationIds: new Set(),
  } as MerltArticleState);

  // Track if local changes were made (validation) to prevent global store overwrite
  const [hasLocalChanges, setHasLocalChanges] = useState(false);

  // Fetch article status and pending items
  const fetchStatus = useCallback(async () => {
    if (!enabled || !tipo_atto || !articolo) return;

    // Reset local changes flag when explicitly refreshing
    setHasLocalChanges(false);
    setState((prev: MerltArticleState) =>({ ...prev, isLoading: true, error: null }));

    try {
      // Check if article is in graph
      const graphStatus = await merltService.checkArticleInGraph(
        tipo_atto,
        articolo,
        numero_atto,
        data
      );

      // Fetch pending items if we have a user_id (regardless of graph status)
      // This allows showing LLM-proposed entities even before they're validated
      let pendingEntities: PendingEntity[] = [];
      let pendingRelations: PendingRelation[] = [];

      if (user_id) {
        try {
          // Build article URN pattern for server-side filtering
          const articoloNorm = articolo.toLowerCase().replace(/\s+/g, '').replace(/^art\.?\s*/i, '');
          const articleUrnPattern = `~art${articoloNorm}`;

          const queue = await merltService.getPendingQueue(user_id, {
            tipo_atto: tipo_atto,
            article_urn: articleUrnPattern,
            limit: 50,
          });

          pendingEntities = queue.pending_entities;
          pendingRelations = queue.pending_relations;
        } catch (err) {
          console.warn('Failed to fetch pending queue:', err);
        }
      }

      // Article is considered "processed" if it has validated entities OR pending entities
      const hasBeenProcessed = (graphStatus.entity_count || 0) > 0 || pendingEntities.length > 0;

      setState((prev: MerltArticleState) =>({
        ...prev,
        isLoading: false,
        graphStatus,
        pendingEntities,
        pendingRelations,
        pendingCount: pendingEntities.length + pendingRelations.length,
        validatedCount: graphStatus.entity_count || 0,
        hasBeenProcessed,
      }));

      return { graphStatus, pendingEntities, hasBeenProcessed };
    } catch (err) {
      setState((prev: MerltArticleState) =>({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Errore nel caricamento',
      }));
      return null;
    }
  }, [enabled, tipo_atto, articolo, numero_atto, data, user_id]);

  // Initial load
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Update state when enrichment job changes
  useEffect(() => {
    if (enrichmentJob?.status === 'in_progress') {
      setState((prev: MerltArticleState) =>({ ...prev, isEnriching: true }));
    } else if (enrichmentJob?.status === 'completed') {
      setState((prev: MerltArticleState) =>({ ...prev, isEnriching: false }));

      // Refresh status after enrichment completes
      fetchStatus();
    } else if (enrichmentJob?.status === 'failed') {
      setState((prev: MerltArticleState) =>({
        ...prev,
        isEnriching: false,
        error: enrichmentJob.error || 'Errore enrichment',
      }));
    }
  }, [enrichmentJob, fetchStatus]);

  // If enrichment result arrives and no local changes, update pending queue
  useEffect(() => {
    if (!enrichmentResult || hasLocalChanges) return;

    setState((prev: MerltArticleState) =>({
      ...prev,
      pendingEntities: enrichmentResult.pending_entities || prev.pendingEntities,
      pendingRelations: enrichmentResult.pending_relations || prev.pendingRelations,
      pendingCount: (enrichmentResult.pending_entities?.length || 0) + (enrichmentResult.pending_relations?.length || 0),
    }));
  }, [enrichmentResult, hasLocalChanges]);

  const validateEntity = useCallback(async (entityId: string, vote: VoteType, comment?: string) => {
    if (!user_id) return null;
    setState((prev: MerltArticleState) =>({ ...prev, isValidating: true }));
    setHasLocalChanges(true);

    try {
      const response = await merltService.validateEntity({
        entity_id: entityId,
        user_id,
        vote,
        comment,
      });

      setState((prev: MerltArticleState) =>({
        ...prev,
        isValidating: false,
        votedEntityIds: new Set(prev.votedEntityIds).add(entityId),
      }));

      return response;
    } catch (err) {
      setState((prev: MerltArticleState) =>({
        ...prev,
        isValidating: false,
        error: err instanceof Error ? err.message : 'Errore nella validazione',
      }));
      return null;
    }
  }, [user_id]);

  const validateRelation = useCallback(async (relationId: string, vote: VoteType, comment?: string) => {
    if (!user_id) return null;
    setState((prev: MerltArticleState) =>({ ...prev, isValidating: true }));
    setHasLocalChanges(true);

    try {
      const response = await merltService.validateRelation({
        relation_id: relationId,
        user_id,
        vote,
        comment,
      });

      setState((prev: MerltArticleState) =>({
        ...prev,
        isValidating: false,
        votedRelationIds: new Set(prev.votedRelationIds).add(relationId),
      }));

      return response;
    } catch (err) {
      setState((prev: MerltArticleState) =>({
        ...prev,
        isValidating: false,
        error: err instanceof Error ? err.message : 'Errore nella validazione',
      }));
      return null;
    }
  }, [user_id]);

  const requestEnrichment = useCallback(() => {
    if (!user_id) return;
    setState((prev: MerltArticleState) =>({ ...prev, isEnriching: true }));

    startEnrichmentStreaming({
      tipo_atto,
      articolo,
      numero_atto,
      data,
      user_id,
    });
  }, [startEnrichmentStreaming, tipo_atto, articolo, numero_atto, data, user_id]);

  const commitFeedback = useCallback(() => {
    setState((prev: MerltArticleState) =>({
      ...prev,
      committedEntityIds: new Set(prev.votedEntityIds),
      committedRelationIds: new Set(prev.votedRelationIds),
    }));
  }, []);

  const hasVotedEntity = useCallback((entityId: string) => {
    return state.votedEntityIds.has(entityId);
  }, [state.votedEntityIds]);

  const hasVotedRelation = useCallback((relationId: string) => {
    return state.votedRelationIds.has(relationId);
  }, [state.votedRelationIds]);

  const uncommittedVoteCount = useMemo(() => {
    let count = 0;
    state.votedEntityIds.forEach((id: string) => {
      if (!state.committedEntityIds.has(id)) count += 1;
    });
    state.votedRelationIds.forEach((id: string) => {
      if (!state.committedRelationIds.has(id)) count += 1;
    });
    return count;
  }, [state.votedEntityIds, state.votedRelationIds, state.committedEntityIds, state.committedRelationIds]);

  return {
    ...state,
    isEnriching: state.isEnriching || isEnrichingGlobal,
    progressMessage,
    refresh: fetchStatus,
    validateEntity,
    validateRelation,
    requestEnrichment,
    commitFeedback,
    hasVotedEntity,
    hasVotedRelation,
    uncommittedVoteCount,
  };
}

export default useMerltArticleStatus;
