/**
 * usePolicyEvolution
 * ===================
 *
 * Hook per caricare i dati di evoluzione policy (time-series, expert usage, aggregation).
 * Fetch parallelo dei 3 endpoint con stato loading/error.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getTimeSeries,
  getExpertEvolution,
  getAggregationHistory,
} from '../services/policyEvolutionService';
import type {
  TimeSeriesPoint,
  ExpertEvolutionPoint,
  AggregationHistoryPoint,
} from '../services/policyEvolutionService';

interface PolicyEvolutionState {
  timeSeries: TimeSeriesPoint[];
  expertEvolution: ExpertEvolutionPoint[];
  aggregationHistory: AggregationHistoryPoint[];
  loading: boolean;
  error: string | null;
}

export function usePolicyEvolution(days: number = 30) {
  const [state, setState] = useState({
    timeSeries: [],
    expertEvolution: [],
    aggregationHistory: [],
    loading: true,
    error: null,
  } as PolicyEvolutionState);

  const fetchAll = useCallback(async () => {
    setState((prev: PolicyEvolutionState) => ({ ...prev, loading: true, error: null }));
    try {
      const [ts, ee, ah] = await Promise.all([
        getTimeSeries('confidence', days * 2),
        getExpertEvolution(days),
        getAggregationHistory(days),
      ]);
      setState({
        timeSeries: ts,
        expertEvolution: ee,
        aggregationHistory: ah,
        loading: false,
        error: null,
      });
    } catch (err) {
      console.error('Failed to load policy evolution data:', err);
      setState((prev: PolicyEvolutionState) => ({
        ...prev,
        loading: false,
        error: 'Errore caricamento dati evoluzione policy',
      }));
    }
  }, [days]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  return { ...state, refresh: fetchAll };
}
