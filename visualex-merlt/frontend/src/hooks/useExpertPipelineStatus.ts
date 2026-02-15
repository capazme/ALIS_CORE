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

    // Guard against double invocation
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    const poll = async () => {
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

    poll();
    pollIntervalRef.current = window.setInterval(poll, 2000);

    return cleanup;
  }, [queryId, cleanup]);

  return { status, isActive, error };
}
