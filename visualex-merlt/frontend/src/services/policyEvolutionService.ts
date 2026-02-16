/**
 * Policy Evolution Service
 * =========================
 *
 * Service per time-series di evoluzione policy, expert usage e aggregation trends.
 *
 * Endpoints backend:
 * - GET /policy-evolution/time-series
 * - GET /policy-evolution/expert-evolution
 * - GET /policy-evolution/aggregation-history
 */

import { get } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

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

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function getTimeSeries(
  metric: string = 'confidence',
  window: number = 50,
): Promise<TimeSeriesPoint[]> {
  return get(`${PREFIX}/policy-evolution/time-series`, { metric, window });
}

export async function getExpertEvolution(
  days: number = 30,
): Promise<ExpertEvolutionPoint[]> {
  return get(`${PREFIX}/policy-evolution/expert-evolution`, { days });
}

export async function getAggregationHistory(
  days: number = 30,
): Promise<AggregationHistoryPoint[]> {
  return get(`${PREFIX}/policy-evolution/aggregation-history`, { days });
}
