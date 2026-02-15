/**
 * Enrichment Store (Zustand)
 * ==========================
 *
 * Store globale per gestire il live enrichment MERL-T:
 * - Mantiene stato della richiesta anche se l'utente naviga
 * - Permette streaming continuo senza perdere contesto
 * - Cache per evitare richieste duplicate
 */

import { create } from 'zustand';
import { requestLiveEnrichment, requestLiveEnrichmentStreaming, type StreamingEvent } from '../services/merltService';
import type { PendingEntity, PendingRelation } from '../types/merlt';

/** Job status for enrichment process */
export type EnrichmentJobStatus = 'pending' | 'in_progress' | 'completed' | 'failed';

/** Result of an enrichment job */
export interface EnrichmentJobResult {
  pending_entities: PendingEntity[];
  pending_relations: PendingRelation[];
  article_urn: string;
}

export interface EnrichmentJob {
  id: string;
  articleKey: string;
  status: EnrichmentJobStatus;
  progressMessage?: string;
  startedAt: number;
  completedAt?: number;
  error?: string;
}

interface EnrichmentState {
  jobs: Record<string, EnrichmentJob>;
  results: Record<string, EnrichmentJobResult>;

  startEnrichmentStreaming: (params: {
    tipo_atto: string;
    articolo: string;
    numero_atto?: string;
    data?: string;
    user_id: string;
  }) => void;

  updateJob: (articleKey: string, updates: Partial<EnrichmentJob>) => void;
  setResult: (articleKey: string, result: EnrichmentJobResult) => void;
  clearJob: (articleKey: string) => void;
}

export const getArticleKey = (
  tipo_atto: string,
  articolo: string,
  numero_atto?: string,
  data?: string
) => {
  return [tipo_atto, articolo, numero_atto, data].filter(Boolean).join('|');
};

export const useEnrichmentStore = create<EnrichmentState>((set, get) => ({
  jobs: {},
  results: {},

  startEnrichmentStreaming: async ({ tipo_atto, articolo, numero_atto, data, user_id }) => {
    const articleKey = getArticleKey(tipo_atto, articolo, numero_atto, data);

    // Prevent duplicate requests
    if (get().jobs[articleKey]?.status === 'in_progress') {
      return;
    }

    // Initialize job
    set(state => ({
      jobs: {
        ...state.jobs,
        [articleKey]: {
          id: `${articleKey}-${Date.now()}`,
          articleKey,
          status: 'in_progress',
          startedAt: Date.now(),
        },
      },
    }));

    try {
      // Accumulate entities/relations as they arrive
      const entities: PendingEntity[] = [];
      const relations: PendingRelation[] = [];
      let articleUrn = '';

      // Start streaming updates via SSE
      const close = requestLiveEnrichmentStreaming(
        tipo_atto,
        articolo,
        user_id,
        (event: StreamingEvent) => {
          switch (event.type) {
            case 'progress':
            case 'waiting':
              get().updateJob(articleKey, { progressMessage: event.message });
              break;
            case 'start':
              if (event.article) articleUrn = event.article.urn;
              break;
            case 'entity':
              if (event.entity) entities.push(event.entity);
              get().updateJob(articleKey, { progressMessage: `Estratte ${entities.length} entità...` });
              break;
            case 'relation':
              if (event.relation) relations.push(event.relation);
              break;
            case 'complete':
              get().setResult(articleKey, { pending_entities: entities, pending_relations: relations, article_urn: articleUrn });
              get().updateJob(articleKey, { status: 'completed', completedAt: Date.now() });
              break;
            case 'error':
              get().updateJob(articleKey, { status: 'failed', error: event.message, completedAt: Date.now() });
              break;
          }
        }
      );

      // Store close function for potential cancellation (not used yet)
      void close;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Errore durante enrichment';
      get().updateJob(articleKey, {
        status: 'failed',
        error: errorMessage,
        completedAt: Date.now(),
      });
    }
  },

  updateJob: (articleKey, updates) => {
    set(state => ({
      jobs: {
        ...state.jobs,
        [articleKey]: {
          ...(state.jobs[articleKey] || { id: articleKey, articleKey, status: 'pending', startedAt: Date.now() }),
          ...updates,
        },
      },
    }));
  },

  setResult: (articleKey, result) => {
    set(state => ({
      results: {
        ...state.results,
        [articleKey]: result,
      },
    }));
  },

  clearJob: (articleKey) => {
    set(state => {
      const { [articleKey]: _, ...restJobs } = state.jobs;
      const { [articleKey]: __, ...restResults } = state.results;
      return { jobs: restJobs, results: restResults };
    });
  },
}));

export const selectJobForArticle = (articleKey: string) => (state: EnrichmentState) => state.jobs[articleKey];
export const selectResultForArticle = (articleKey: string) => (state: EnrichmentState) => state.results[articleKey];
