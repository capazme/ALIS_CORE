/**
 * useExpertPipelineStatus - Polling-based hook for expert pipeline status.
 *
 * Derives pipeline status from the trace endpoint (GET /traces/{id}).
 * Falls back gracefully if the trace is not yet available.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { ExpertPipelineStatus, ExpertStatus } from '../types/pipeline';
import { EXPERT_IDS } from '../types/pipeline';
import { get } from '../services/api';

const PREFIX = '/merlt';
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 45; // 90s total

interface TraceExpertResult {
  expert_name: string;
  status?: string;
  confidence?: number;
  answer?: string;
}

interface TraceResponse {
  trace_id: string;
  status?: string;
  experts?: TraceExpertResult[];
  synthesis?: {
    mode?: string;
    confidence?: number;
  };
  execution_time_ms?: number;
}

function deriveStatusFromTrace(queryId: string, trace: TraceResponse): ExpertPipelineStatus {
  const experts: ExpertStatus[] = EXPERT_IDS.map(id => {
    const expertResult = trace.experts?.find(
      (e: TraceExpertResult) => e.expert_name === id || e.expert_name?.toLowerCase().includes(id)
    );
    if (!expertResult) {
      return { id, status: 'pending' as const };
    }
    return {
      id,
      status: 'completed' as const,
      confidence: expertResult.confidence,
    };
  });

  const completedCount = experts.filter((e: ExpertStatus) => e.status === 'completed').length;
  const hasSynthesis = !!trace.synthesis;
  const overallProgress = Math.round((completedCount / experts.length) * 80) + (hasSynthesis ? 20 : 0);

  let phase: ExpertPipelineStatus['phase'] = 'routing';
  if (completedCount > 0 && completedCount < experts.length) phase = 'expert_analysis';
  if (completedCount === experts.length && !hasSynthesis) phase = 'synthesis';
  if (hasSynthesis) phase = 'completed';
  if (trace.status === 'failed') phase = 'failed';

  return {
    queryId,
    overallProgress: Math.min(overallProgress, 100),
    phase,
    experts,
  };
}

function createInitialStatus(queryId: string): ExpertPipelineStatus {
  return {
    queryId,
    overallProgress: 0,
    phase: 'routing',
    experts: EXPERT_IDS.map(id => ({
      id,
      status: 'pending' as const,
    })),
  };
}

export interface UseExpertPipelineStatusReturn {
  status: ExpertPipelineStatus | null;
  isActive: boolean;
  error: string | null;
}

export function useExpertPipelineStatus(queryId: string | null): UseExpertPipelineStatusReturn {
  const [status, setStatus] = useState(null as ExpertPipelineStatus | null);
  const [error, setError] = useState(null as string | null);
  const pollIntervalRef = useRef(null as number | null);
  const pollAttemptsRef = useRef(0);

  const isActive = status !== null &&
    status.phase !== 'completed' &&
    status.phase !== 'failed';

  const cleanup = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!queryId) {
      setStatus(null);
      setError(null);
      cleanup();
      return;
    }

    setStatus(createInitialStatus(queryId));
    setError(null);
    pollAttemptsRef.current = 0;

    // Guard against double invocation
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    const poll = async () => {
      pollAttemptsRef.current += 1;

      if (pollAttemptsRef.current > MAX_POLL_ATTEMPTS) {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        setStatus((prev: ExpertPipelineStatus | null) =>
          prev ? { ...prev, phase: 'failed' as const, overallProgress: prev.overallProgress } : null
        );
        setError('Timeout: il server non ha risposto entro 90 secondi');
        return;
      }

      try {
        const trace = await get<TraceResponse>(
          `${PREFIX}/traces/${encodeURIComponent(queryId)}`
        );
        const derived = deriveStatusFromTrace(queryId, trace);
        setStatus(derived);

        if (derived.phase === 'completed' || derived.phase === 'failed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
        }
      } catch {
        // Ignore poll errors - trace may not be stored yet
      }
    };

    // B5: initial delay before first poll to allow backend to commit the trace
    const initialDelay = window.setTimeout(() => {
      poll();
      pollIntervalRef.current = window.setInterval(poll, POLL_INTERVAL_MS);
    }, 500);

    return () => {
      window.clearTimeout(initialDelay);
      cleanup();
    };
  }, [queryId, cleanup]);

  return { status, isActive, error };
}
