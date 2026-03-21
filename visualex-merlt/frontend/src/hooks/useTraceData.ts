/**
 * useTraceData - Fetches trace, sources, and validity data in parallel.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getTraceWithSources, getTraceValidity } from '../services/traceService';
import type { TraceDetail, SourceResolution, ValiditySummary } from '../types/trace';

export interface UseTraceDataReturn {
  trace: TraceDetail | null;
  sources: SourceResolution[];
  validity: ValiditySummary | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useTraceData(traceId: string | null): UseTraceDataReturn {
  const [trace, setTrace] = useState(null as TraceDetail | null);
  const [sources, setSources] = useState([] as SourceResolution[]);
  const [validity, setValidity] = useState(null as ValiditySummary | null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [fetchCounter, setFetchCounter] = useState(0);
  const controllerRef = useRef(undefined as AbortController | undefined);

  const refetch = useCallback(() => {
    setFetchCounter((c: number) => c + 1);
  }, []);

  useEffect(() => {
    controllerRef.current?.abort();
    controllerRef.current = new AbortController();
    const signal = controllerRef.current.signal;

    if (!traceId) {
      setTrace(null);
      setSources([]);
      setValidity(null);
      return;
    }

    const doFetch = async () => {
      setIsLoading(true);
      setError(null);

      const [traceAndSourcesResult, validityResult] = await Promise.allSettled([
        getTraceWithSources(traceId),
        getTraceValidity(traceId),
      ]);

      if (signal.aborted) return;

      if (traceAndSourcesResult.status === 'fulfilled') {
        setTrace(traceAndSourcesResult.value.trace);
        setSources(traceAndSourcesResult.value.sourcesResponse.sources);
      } else {
        if (!signal.aborted) {
          setError('Errore nel caricamento del trace');
        }
      }

      if (validityResult.status === 'fulfilled') {
        setValidity(validityResult.value.validity);
      }

      if (!signal.aborted) {
        setIsLoading(false);
      }
    };

    doFetch();

    return () => {
      controllerRef.current?.abort();
    };
  }, [traceId, fetchCounter]);

  return { trace, sources, validity, isLoading, error, refetch };
}
