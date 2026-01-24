/**
 * Global Enrichment Store
 * =======================
 *
 * Gestisce le richieste di enrichment in modo globale,
 * così continuano anche se l'utente naviga altrove.
 *
 * Supporta:
 * - Batch enrichment (aspetta tutte le entità)
 * - Streaming enrichment (entità una alla volta via SSE)
 *
 * L'enrichment una volta avviato NON si interrompe,
 * indipendentemente dalle azioni dell'utente.
 */

import { create } from 'zustand';
import { merltService, requestLiveEnrichmentStreaming } from '../services/merltService';
import type { PendingEntity, PendingRelation } from '../types/merlt';

interface EnrichmentJob {
  articleKey: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'waiting';
  startedAt: number;
  completedAt?: number;
  entitiesCount?: number;
  error?: string;
  progressMessage?: string;  // Per streaming progress
  waitingForOther?: boolean; // True se in attesa che un altro utente completi
  waitProgress?: number;     // Percentuale stimata di attesa (0-100)
  waitElapsed?: number;      // Secondi trascorsi in attesa
}

interface EnrichmentResult {
  pending_entities: PendingEntity[];
  pending_relations: PendingRelation[];
  article_urn: string;
}

interface EnrichmentState {
  // Jobs in corso o completati (cache per sessione)
  jobs: Record<string, EnrichmentJob>;

  // Risultati cached per articolo
  results: Record<string, EnrichmentResult>;

  // Close functions per streaming attivi (per cleanup)
  _closeFunctions: Record<string, () => void>;

  // Actions
  startEnrichment: (
    articleKey: string,
    tipo_atto: string,
    articolo: string,
    user_id: string
  ) => void;

  // Streaming enrichment - entità arrivano una alla volta
  startEnrichmentStreaming: (
    articleKey: string,
    tipo_atto: string,
    articolo: string,
    user_id: string
  ) => void;

  // Stop streaming enrichment
  stopEnrichment: (articleKey: string) => void;

  getJobStatus: (articleKey: string) => EnrichmentJob | undefined;
  getResult: (articleKey: string) => EnrichmentResult | undefined;
  isEnriching: (articleKey: string) => boolean;
  hasBeenProcessed: (articleKey: string) => boolean;
}

export const useEnrichmentStore = create<EnrichmentState>((set, get) => ({
  jobs: {},
  results: {},
  _closeFunctions: {},

  startEnrichment: (articleKey, tipo_atto, articolo, user_id) => {
    const { jobs } = get();

    // Se già in corso o completato, non rifare
    const existingJob = jobs[articleKey];
    if (existingJob && (existingJob.status === 'in_progress' || existingJob.status === 'completed')) {
      console.log(`⏭️ Enrichment already ${existingJob.status} for ${articleKey}`);
      return;
    }

    // Registra job come in_progress
    const newJob: EnrichmentJob = {
      articleKey,
      status: 'in_progress',
      startedAt: Date.now(),
    };

    set(state => ({
      jobs: { ...state.jobs, [articleKey]: newJob },
    }));

    console.log(`🚀 Starting global enrichment for ${articleKey}`);

    // Avvia enrichment - NON await, così continua in background
    merltService.requestLiveEnrichment(tipo_atto, articolo, user_id)
      .then(result => {
        console.log(`✅ Enrichment completed for ${articleKey}: ${result.pending_entities.length} entities`);

        set(state => ({
          jobs: {
            ...state.jobs,
            [articleKey]: {
              ...newJob,
              status: 'completed',
              completedAt: Date.now(),
              entitiesCount: result.pending_entities.length,
            },
          },
          results: {
            ...state.results,
            [articleKey]: result,
          },
        }));
      })
      .catch(error => {
        console.error(`❌ Enrichment failed for ${articleKey}:`, error);

        set(state => ({
          jobs: {
            ...state.jobs,
            [articleKey]: {
              ...newJob,
              status: 'failed',
              completedAt: Date.now(),
              error: error.message || 'Unknown error',
            },
          },
        }));
      });
  },

  // Streaming enrichment - entità arrivano una alla volta via SSE
  startEnrichmentStreaming: (articleKey, tipo_atto, articolo, user_id) => {
    const { jobs } = get();

    // Se già in corso o completato, non rifare
    const existingJob = jobs[articleKey];
    if (existingJob && (existingJob.status === 'in_progress' || existingJob.status === 'completed')) {
      console.log(`⏭️ Enrichment already ${existingJob.status} for ${articleKey}`);
      return;
    }

    // Registra job come in_progress
    const newJob: EnrichmentJob = {
      articleKey,
      status: 'in_progress',
      startedAt: Date.now(),
      progressMessage: 'Avvio estrazione...',
    };

    // Inizializza risultato vuoto per progressive update
    set(state => ({
      jobs: { ...state.jobs, [articleKey]: newJob },
      results: {
        ...state.results,
        [articleKey]: {
          pending_entities: [],
          pending_relations: [],
          article_urn: '',
        },
      },
    }));

    console.log(`🚀 Starting STREAMING enrichment for ${articleKey}`);

    // Avvia SSE streaming
    const closeFunction = requestLiveEnrichmentStreaming(
      tipo_atto,
      articolo,
      user_id,
      (event) => {
        // Note: We use set() with callback pattern which provides fresh state

        switch (event.type) {
          case 'start':
            console.log(`📄 Article loaded: ${event.article?.urn}`);
            set(state => ({
              jobs: {
                ...state.jobs,
                [articleKey]: {
                  ...state.jobs[articleKey],
                  progressMessage: 'Articolo caricato, estrazione in corso...',
                },
              },
              results: {
                ...state.results,
                [articleKey]: {
                  ...state.results[articleKey],
                  article_urn: event.article?.urn || '',
                },
              },
            }));
            break;

          case 'progress':
            set(state => ({
              jobs: {
                ...state.jobs,
                [articleKey]: {
                  ...state.jobs[articleKey],
                  progressMessage: event.message || 'Estrazione...',
                },
              },
            }));
            break;

          case 'entity':
            console.log(`✨ New entity: ${event.entity?.nome}`);
            if (event.entity) {
              set(state => ({
                results: {
                  ...state.results,
                  [articleKey]: {
                    ...state.results[articleKey],
                    pending_entities: [
                      ...state.results[articleKey].pending_entities,
                      event.entity as PendingEntity,
                    ],
                  },
                },
                jobs: {
                  ...state.jobs,
                  [articleKey]: {
                    ...state.jobs[articleKey],
                    entitiesCount: (state.jobs[articleKey].entitiesCount || 0) + 1,
                    progressMessage: `Estratte ${(state.jobs[articleKey].entitiesCount || 0) + 1} entità...`,
                  },
                },
              }));
            }
            break;

          case 'relation':
            console.log(`🔗 New relation`);
            if (event.relation) {
              set(state => ({
                results: {
                  ...state.results,
                  [articleKey]: {
                    ...state.results[articleKey],
                    pending_relations: [
                      ...state.results[articleKey].pending_relations,
                      event.relation as PendingRelation,
                    ],
                  },
                },
              }));
            }
            break;

          case 'complete':
            console.log(`✅ Streaming complete: ${event.summary?.entities_count} entities`);
            set(state => {
              // Cleanup close function immutably
              const newCloseFunctions = { ...state._closeFunctions };
              delete newCloseFunctions[articleKey];

              return {
                jobs: {
                  ...state.jobs,
                  [articleKey]: {
                    ...state.jobs[articleKey],
                    status: 'completed',
                    completedAt: Date.now(),
                    entitiesCount: event.summary?.entities_count || state.jobs[articleKey]?.entitiesCount || 0,
                    progressMessage: undefined,
                  },
                },
                _closeFunctions: newCloseFunctions,
              };
            });
            break;

          case 'error':
            console.error(`❌ Streaming error: ${event.message}`);
            set(state => {
              // Cleanup close function immutably
              const newCloseFunctions = { ...state._closeFunctions };
              delete newCloseFunctions[articleKey];

              return {
                jobs: {
                  ...state.jobs,
                  [articleKey]: {
                    ...state.jobs[articleKey],
                    status: 'failed',
                    completedAt: Date.now(),
                    error: event.message || 'Unknown error',
                    progressMessage: undefined,
                  },
                },
                _closeFunctions: newCloseFunctions,
              };
            });
            break;

          case 'warning':
            console.warn(`⚠️ ${event.message}`);
            break;

          case 'waiting':
            // Un altro utente sta già estraendo questo articolo
            console.log(`⏳ Waiting for another user: ${event.message}`);
            set(state => ({
              jobs: {
                ...state.jobs,
                [articleKey]: {
                  ...state.jobs[articleKey],
                  status: 'waiting',
                  waitingForOther: true,
                  waitProgress: event.progress || 0,
                  waitElapsed: event.elapsed || 0,
                  progressMessage: event.message || 'In attesa di un altro utente...',
                },
              },
            }));
            break;
        }
      }
    );

    // Store close function for potential cleanup
    set(state => ({
      _closeFunctions: { ...state._closeFunctions, [articleKey]: closeFunction },
    }));
  },

  // Stop enrichment (close SSE connection)
  stopEnrichment: (articleKey) => {
    const { _closeFunctions } = get();
    const closeFunction = _closeFunctions[articleKey];

    if (closeFunction) {
      console.log(`🛑 Stopping enrichment for ${articleKey}`);
      closeFunction();

      set(state => {
        const newCloseFunctions = { ...state._closeFunctions };
        delete newCloseFunctions[articleKey];

        return {
          _closeFunctions: newCloseFunctions,
          jobs: {
            ...state.jobs,
            [articleKey]: {
              ...state.jobs[articleKey],
              status: 'failed',
              completedAt: Date.now(),
              error: 'Stopped by user',
              progressMessage: undefined,
            },
          },
        };
      });
    }
  },

  getJobStatus: (articleKey) => {
    return get().jobs[articleKey];
  },

  getResult: (articleKey) => {
    return get().results[articleKey];
  },

  isEnriching: (articleKey) => {
    const job = get().jobs[articleKey];
    return job?.status === 'in_progress' || job?.status === 'waiting';
  },

  hasBeenProcessed: (articleKey) => {
    const job = get().jobs[articleKey];
    return job?.status === 'completed' || job?.status === 'in_progress';
  },
}));

// Helper per generare articleKey consistente
export function getArticleKey(
  tipo_atto: string,
  articolo: string,
  numero_atto?: string,
  data?: string
): string {
  return `${tipo_atto}:${articolo}:${numero_atto || ''}:${data || ''}`;
}

// Selettori per uso reattivo in componenti
export const selectJobForArticle = (articleKey: string) => (state: EnrichmentState) =>
  state.jobs[articleKey];

export const selectResultForArticle = (articleKey: string) => (state: EnrichmentState) =>
  state.results[articleKey];
