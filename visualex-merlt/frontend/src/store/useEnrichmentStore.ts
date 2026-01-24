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
import { merltService, requestLiveEnrichmentStreaming } from '../services/merltService';
import type { EnrichmentJobStatus, EnrichmentJobResult } from '../types/merlt';

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
      // Kick off enrichment on backend
      await merltService.requestLiveEnrichment(tipo_atto, articolo, user_id, numero_atto, data);

      // Start streaming updates
      await requestLiveEnrichmentStreaming({
        tipo_atto,
        articolo,
        numero_atto,
        data,
        user_id,
        onProgress: (message) => {
          get().updateJob(articleKey, { progressMessage: message });
        },
        onComplete: (result) => {
          get().setResult(articleKey, result);
          get().updateJob(articleKey, { status: 'completed', completedAt: Date.now() });
        },
        onError: (error) => {
          get().updateJob(articleKey, { status: 'failed', error: error.message, completedAt: Date.now() });
        },
      });
    } catch (error: any) {
      get().updateJob(articleKey, {
        status: 'failed',
        error: error.message || 'Errore durante enrichment',
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
