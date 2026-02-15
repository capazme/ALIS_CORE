/**
 * Hook for fetching Policy Evolution data from MERL-T API.
 *
 * Uses @tanstack/react-query for caching and refetching.
 */

import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const ORCHESTRATION_URL = import.meta.env.VITE_ORCHESTRATION_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: `${ORCHESTRATION_URL}/api/v1/policy-evolution`,
  timeout: 15000,
});

export interface TimeSeriesPoint {
  timestamp: string;
  confidence: number | null;
  reward: number | null;
  query_count: number;
}

export interface ExpertEvolutionPoint {
  timestamp: string;
  literal: number;
  systemic: number;
  principles: number;
  precedent: number;
}

export interface AggregationHistoryPoint {
  timestamp: string;
  component: string;
  avg_rating: number | null;
  disagreement_score: number | null;
  total_feedback: number;
}

export function useTimeSeries(metric = 'confidence', window = 50) {
  return useQuery<TimeSeriesPoint[]>({
    queryKey: ['policy-evolution', 'time-series', metric, window],
    queryFn: () =>
      api.get('/time-series', { params: { metric, window } }).then((r) => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });
}

export function useExpertEvolution(days = 30) {
  return useQuery<ExpertEvolutionPoint[]>({
    queryKey: ['policy-evolution', 'expert-evolution', days],
    queryFn: () =>
      api.get('/expert-evolution', { params: { days } }).then((r) => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });
}

export function useAggregationHistory(days = 30) {
  return useQuery<AggregationHistoryPoint[]>({
    queryKey: ['policy-evolution', 'aggregation-history', days],
    queryFn: () =>
      api.get('/aggregation-history', { params: { days } }).then((r) => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });
}
