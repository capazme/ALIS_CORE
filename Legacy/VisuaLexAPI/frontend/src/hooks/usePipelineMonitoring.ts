/**
 * usePipelineMonitoring Hook
 * ==========================
 *
 * Hook per monitorare pipeline runs in tempo reale.
 * Gestisce sia polling REST che connessioni WebSocket.
 *
 * Features:
 * - Polling periodico lista runs
 * - WebSocket per run specifico
 * - Aggregazione KPI
 * - Stato loading/error
 *
 * @example
 * ```tsx
 * function PipelineDashboard() {
 *   const {
 *     runs,
 *     activeRuns,
 *     completedRuns,
 *     isLoading,
 *     error,
 *     refresh,
 *     subscribeToRun,
 *   } = usePipelineMonitoring();
 *
 *   return (
 *     <div>
 *       <h2>Active: {activeRuns.length}</h2>
 *       {runs.map(run => <RunCard key={run.run_id} run={run} />)}
 *     </div>
 *   );
 * }
 * ```
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  listPipelineRuns,
  getPipelineRun,
  getPipelineErrors,
  connectPipelineWebSocket,
  type PipelineRun,
  type PipelineRunDetails,
  type PipelineError,
  type ProgressUpdate,
  type PipelineStatus,
  type PipelineType,
} from '../services/pipelineService';

// =============================================================================
// TYPES
// =============================================================================

export interface PipelineStats {
  totalRuns: number;
  activeRuns: number;
  completedRuns: number;
  failedRuns: number;
  totalItemsProcessed: number;
  totalErrors: number;
}

export interface UsePipelineMonitoringOptions {
  /** Intervallo polling in ms (default: 5000) */
  pollingInterval?: number;
  /** Abilita polling automatico (default: true) */
  autoPolling?: boolean;
  /** Filtra per status */
  statusFilter?: PipelineStatus;
  /** Filtra per tipo */
  typeFilter?: PipelineType;
  /** Limite risultati */
  limit?: number;
}

export interface UsePipelineMonitoringReturn {
  /** Lista di tutte le pipeline runs */
  runs: PipelineRun[];
  /** Solo runs attive (running) */
  activeRuns: PipelineRun[];
  /** Solo runs completate */
  completedRuns: PipelineRun[];
  /** Solo runs fallite */
  failedRuns: PipelineRun[];
  /** Statistiche aggregate */
  stats: PipelineStats;
  /** Sta caricando */
  isLoading: boolean;
  /** Errore di caricamento */
  error: string | null;
  /** Refresh manuale della lista */
  refresh: () => Promise<void>;
  /** Recupera dettagli singola run */
  getRunDetails: (runId: string) => Promise<PipelineRunDetails | null>;
  /** Recupera errori singola run */
  getRunErrors: (runId: string) => Promise<PipelineError[]>;
  /** Sottoscrivi a updates real-time per una run */
  subscribeToRun: (
    runId: string,
    onProgress: (data: ProgressUpdate) => void,
    onError?: (error: PipelineError) => void
  ) => () => void;
}

// =============================================================================
// HOOK
// =============================================================================

export function usePipelineMonitoring(
  options: UsePipelineMonitoringOptions = {}
): UsePipelineMonitoringReturn {
  const {
    pollingInterval = 5000,
    autoPolling = true,
    statusFilter,
    typeFilter,
    limit = 50,
  } = options;

  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Track active WebSocket subscriptions
  const subscriptionsRef = useRef<Map<string, () => void>>(new Map());

  // Fetch runs
  const fetchRuns = useCallback(async () => {
    try {
      const data = await listPipelineRuns({
        status: statusFilter,
        pipeline_type: typeFilter,
        limit,
      });

      setRuns(data);
      setError(null);
    } catch (err) {
      console.error('[usePipelineMonitoring] Failed to fetch runs:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch pipeline runs');
    } finally {
      setIsLoading(false);
    }
  }, [statusFilter, typeFilter, limit]);

  // Initial fetch and polling
  useEffect(() => {
    fetchRuns();

    if (autoPolling && pollingInterval > 0) {
      const intervalId = setInterval(fetchRuns, pollingInterval);
      return () => clearInterval(intervalId);
    }
  }, [fetchRuns, autoPolling, pollingInterval]);

  // Cleanup subscriptions on unmount
  useEffect(() => {
    return () => {
      subscriptionsRef.current.forEach((unsubscribe) => unsubscribe());
      subscriptionsRef.current.clear();
    };
  }, []);

  // Computed values
  const activeRuns = runs.filter((r) => r.status === 'running');
  const completedRuns = runs.filter((r) => r.status === 'completed');
  const failedRuns = runs.filter((r) => r.status === 'failed');

  const stats: PipelineStats = {
    totalRuns: runs.length,
    activeRuns: activeRuns.length,
    completedRuns: completedRuns.length,
    failedRuns: failedRuns.length,
    totalItemsProcessed: runs.reduce(
      (sum, r) => sum + (r.summary.successful || r.summary.processed || 0),
      0
    ),
    totalErrors: runs.reduce(
      (sum, r) => sum + (r.summary.failed || r.summary.errors || 0),
      0
    ),
  };

  // Get run details
  const getRunDetails = useCallback(async (runId: string): Promise<PipelineRunDetails | null> => {
    try {
      return await getPipelineRun(runId);
    } catch (err) {
      console.error('[usePipelineMonitoring] Failed to get run details:', err);
      return null;
    }
  }, []);

  // Get run errors
  const getRunErrors = useCallback(async (runId: string): Promise<PipelineError[]> => {
    try {
      return await getPipelineErrors(runId);
    } catch (err) {
      console.error('[usePipelineMonitoring] Failed to get run errors:', err);
      return [];
    }
  }, []);

  // Subscribe to real-time updates
  const subscribeToRun = useCallback(
    (
      runId: string,
      onProgress: (data: ProgressUpdate) => void,
      onError?: (error: PipelineError) => void
    ): (() => void) => {
      // Unsubscribe from previous subscription for this run
      const existingUnsubscribe = subscriptionsRef.current.get(runId);
      if (existingUnsubscribe) {
        existingUnsubscribe();
      }

      // Create new subscription
      const unsubscribe = connectPipelineWebSocket(runId, {
        onInitialState: onProgress,
        onProgress,
        onError,
        onConnectionClose: () => {
          subscriptionsRef.current.delete(runId);
        },
      });

      subscriptionsRef.current.set(runId, unsubscribe);

      // Return unsubscribe function
      return () => {
        unsubscribe();
        subscriptionsRef.current.delete(runId);
      };
    },
    []
  );

  return {
    runs,
    activeRuns,
    completedRuns,
    failedRuns,
    stats,
    isLoading,
    error,
    refresh: fetchRuns,
    getRunDetails,
    getRunErrors,
    subscribeToRun,
  };
}

export default usePipelineMonitoring;
