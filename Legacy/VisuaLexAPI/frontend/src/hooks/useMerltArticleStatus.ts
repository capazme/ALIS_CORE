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

  const [state, setState] = useState<MerltArticleState>({
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
  });

  // Track if local changes were made (validation) to prevent global store overwrite
  const [hasLocalChanges, setHasLocalChanges] = useState(false);

  // Fetch article status and pending items
  const fetchStatus = useCallback(async () => {
    if (!enabled || !tipo_atto || !articolo) return;

    // Reset local changes flag when explicitly refreshing
    setHasLocalChanges(false);
    setState(prev => ({ ...prev, isLoading: true, error: null }));

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
          // Backend uses both tipo_atto (resolved via NORMATTIVA_URN_CODICI) and article pattern
          // This ensures Art. 1 Costituzione is distinguished from Art. 1 Codice Civile
          //
          // URN format: https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1930-10-19;1398:1~art11
          // The article number appears after "~art" at the end
          // Pattern: "~art{numero}" matches the specific article with regex ~art{num}([^0-9]|$)
          const articoloNorm = articolo.toLowerCase().replace(/\s+/g, '').replace(/^art\.?\s*/i, '');
          // Match the URN format which ends with ~art{number}
          const articleUrnPattern = `~art${articoloNorm}`;

          const queue = await merltService.getPendingQueue(user_id, {
            tipo_atto: tipo_atto,  // Pass tipo_atto for precise filtering
            article_urn: articleUrnPattern,
            limit: 50,
          });

          // Server-side filtering handles URN matching, no client-side filter needed
          pendingEntities = queue.pending_entities;
          pendingRelations = queue.pending_relations;
        } catch (err) {
          // Non-critical error, just log
          console.warn('Failed to fetch pending queue:', err);
        }
      }

      // Article is considered "processed" if it has validated entities OR pending entities
      const hasBeenProcessed = (graphStatus.entity_count || 0) > 0 || pendingEntities.length > 0;

      setState(prev => ({
        ...prev,
        isLoading: false,
        graphStatus,
        pendingEntities,
        pendingRelations,
        pendingCount: pendingEntities.length + pendingRelations.length,
        validatedCount: graphStatus.entity_count || 0,
        hasBeenProcessed,
      }));

      // Return status for auto-enrichment check
      return { graphStatus, pendingEntities, hasBeenProcessed };
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Errore nel caricamento',
      }));
      return null;
    }
  }, [tipo_atto, articolo, numero_atto, data, user_id, enabled]);

  // Initial fetch - NO auto-enrichment per controllo costi API
  // L'enrichment è solo su richiesta esplicita dell'utente (requestEnrichment)
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Sync global enrichment state con local state
  // Questo effect si triggera quando enrichmentJob o enrichmentResult cambiano
  // Per streaming: aggiorna progressivamente le entità man mano che arrivano
  // NON sovrascrive se ci sono modifiche locali (validazioni in corso)
  useEffect(() => {
    if (enrichmentResult && !hasLocalChanges) {
      setState(prev => ({
        ...prev,
        pendingEntities: enrichmentResult.pending_entities,
        pendingRelations: enrichmentResult.pending_relations,
        pendingCount: enrichmentResult.pending_entities.length + enrichmentResult.pending_relations.length,
        hasBeenProcessed: enrichmentResult.pending_entities.length > 0 || !isEnrichingGlobal,
        isEnriching: isEnrichingGlobal,
        progressMessage: progressMessage,
      }));
    } else {
      // Sincronizza solo lo stato isEnriching e progressMessage (non sovrascrive entità)
      setState(prev => ({
        ...prev,
        isEnriching: isEnrichingGlobal,
        progressMessage: progressMessage,
      }));
    }
  }, [enrichmentJob, enrichmentResult, isEnrichingGlobal, progressMessage, hasLocalChanges]);

  // Validate entity
  const validateEntity = useCallback(async (
    entityId: string,
    vote: VoteType,
    comment?: string
  ): Promise<EntityValidationResponse | null> => {
    if (!user_id) {
      setState(prev => ({ ...prev, error: 'Devi essere autenticato per votare' }));
      return null;
    }

    setState(prev => ({ ...prev, isValidating: true }));

    try {
      const response = await merltService.validateEntity({
        entity_id: entityId,
        user_id,
        vote,
        comment,
      });

      // Mark that we have local changes to prevent global store overwrite
      setHasLocalChanges(true);

      // Update local state - remove from pending if threshold reached
      if (response.threshold_reached) {
        setState(prev => ({
          ...prev,
          isValidating: false,
          pendingEntities: prev.pendingEntities.filter(e => e.id !== entityId),
          pendingCount: prev.pendingCount - 1,
          validatedCount: response.new_status === 'approved'
            ? prev.validatedCount + 1
            : prev.validatedCount,
          // Track that user voted on this entity
          votedEntityIds: new Set([...prev.votedEntityIds, entityId]),
        }));
      } else {
        // Update the entity's scores locally
        setState(prev => ({
          ...prev,
          isValidating: false,
          pendingEntities: prev.pendingEntities.map(e =>
            e.id === entityId
              ? {
                  ...e,
                  approval_score: response.approval_score,
                  rejection_score: response.rejection_score,
                  votes_count: response.votes_count,
                }
              : e
          ),
          // Track that user voted on this entity
          votedEntityIds: new Set([...prev.votedEntityIds, entityId]),
        }));
      }

      return response;
    } catch (err) {
      setState(prev => ({
        ...prev,
        isValidating: false,
        error: err instanceof Error ? err.message : 'Errore nella validazione',
      }));
      return null;
    }
  }, [user_id]);

  // Validate relation
  const validateRelation = useCallback(async (
    relationId: string,
    vote: VoteType,
    comment?: string
  ): Promise<RelationValidationResponse | null> => {
    if (!user_id) {
      setState(prev => ({ ...prev, error: 'Devi essere autenticato per votare' }));
      return null;
    }

    setState(prev => ({ ...prev, isValidating: true }));

    try {
      const response = await merltService.validateRelation({
        relation_id: relationId,
        user_id,
        vote,
        comment,
      });

      // Mark that we have local changes to prevent global store overwrite
      setHasLocalChanges(true);

      // Update local state
      if (response.threshold_reached) {
        setState(prev => ({
          ...prev,
          isValidating: false,
          pendingRelations: prev.pendingRelations.filter(r => r.id !== relationId),
          pendingCount: prev.pendingCount - 1,
          // Track that user voted on this relation
          votedRelationIds: new Set([...prev.votedRelationIds, relationId]),
        }));
      } else {
        setState(prev => ({
          ...prev,
          isValidating: false,
          pendingRelations: prev.pendingRelations.map(r =>
            r.id === relationId
              ? {
                  ...r,
                  approval_score: response.approval_score,
                  rejection_score: response.rejection_score,
                  votes_count: response.votes_count,
                }
              : r
          ),
          // Track that user voted on this relation
          votedRelationIds: new Set([...prev.votedRelationIds, relationId]),
        }));
      }

      return response;
    } catch (err) {
      setState(prev => ({
        ...prev,
        isValidating: false,
        error: err instanceof Error ? err.message : 'Errore nella validazione',
      }));
      return null;
    }
  }, [user_id]);

  // Commit feedback - moves voted items to "committed" set, hiding them from view
  const commitFeedback = useCallback(() => {
    setState(prev => ({
      ...prev,
      // Move voted IDs to committed set
      committedEntityIds: new Set([...prev.committedEntityIds, ...prev.votedEntityIds]),
      committedRelationIds: new Set([...prev.committedRelationIds, ...prev.votedRelationIds]),
      // Clear voted sets (they're now committed)
      votedEntityIds: new Set(),
      votedRelationIds: new Set(),
    }));
  }, []);

  // Helper functions to check if user has voted
  const hasVotedEntity = useCallback((entityId: string) => {
    return state.votedEntityIds.has(entityId);
  }, [state.votedEntityIds]);

  const hasVotedRelation = useCallback((relationId: string) => {
    return state.votedRelationIds.has(relationId);
  }, [state.votedRelationIds]);

  // Count of uncommitted votes
  const uncommittedVoteCount = useMemo(() => {
    return state.votedEntityIds.size + state.votedRelationIds.size;
  }, [state.votedEntityIds, state.votedRelationIds]);

  // Request enrichment for article not in graph (manual trigger)
  // Usa SSE streaming così le entità appaiono una alla volta
  const requestEnrichment = useCallback(() => {
    if (!user_id) {
      setState(prev => ({ ...prev, error: 'Devi essere autenticato per arricchire' }));
      return;
    }

    // Reset local changes flag - we want to receive entities from global store
    setHasLocalChanges(false);

    // Avvia enrichment streaming nel global store
    // Le entità appariranno progressivamente nella UI
    startEnrichmentStreaming(articleKey, tipo_atto, articolo, user_id);
  }, [tipo_atto, articolo, user_id, articleKey, startEnrichmentStreaming]);

  // During streaming, use entities directly from global store for real-time updates
  // After streaming or when validating, use local state
  const effectivePendingEntities = useMemo(() => {
    if (isEnrichingGlobal && enrichmentResult) {
      // During streaming: use global store directly for real-time updates
      return enrichmentResult.pending_entities;
    }
    // After streaming or during validation: use local state
    return state.pendingEntities;
  }, [isEnrichingGlobal, enrichmentResult, state.pendingEntities]);

  const effectivePendingRelations = useMemo(() => {
    if (isEnrichingGlobal && enrichmentResult) {
      return enrichmentResult.pending_relations;
    }
    return state.pendingRelations;
  }, [isEnrichingGlobal, enrichmentResult, state.pendingRelations]);

  // Filter out committed entities (hidden from view)
  const visiblePendingEntities = useMemo(() => {
    return effectivePendingEntities.filter(e => !state.committedEntityIds.has(e.id));
  }, [effectivePendingEntities, state.committedEntityIds]);

  const visiblePendingRelations = useMemo(() => {
    return effectivePendingRelations.filter(r => !state.committedRelationIds.has(r.id));
  }, [effectivePendingRelations, state.committedRelationIds]);

  const visiblePendingCount = visiblePendingEntities.length + visiblePendingRelations.length;

  return {
    ...state,
    // Override pending items with visible items (excluding committed)
    pendingEntities: visiblePendingEntities,
    pendingRelations: visiblePendingRelations,
    pendingCount: visiblePendingCount,
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
