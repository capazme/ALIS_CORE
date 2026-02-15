/**
 * Expert Metrics Service
 * ======================
 *
 * Service per interagire con l'API Expert Metrics di MERL-T.
 * Fornisce funzioni per metriche performance, query stats, reasoning trace.
 *
 * API Endpoints:
 * - GET /expert-metrics/performance - Performance per expert
 * - GET /expert-metrics/queries/stats - Statistiche query
 * - GET /expert-metrics/queries/recent - Query recenti
 * - GET /expert-metrics/trace/{trace_id} - Reasoning trace
 * - GET /expert-metrics/aggregation - Statistiche aggregazione
 */

import { get } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface ExpertPerformance {
  name: string;
  display_name: string;
  accuracy: number;
  accuracy_ci: [number, number];
  latency_ms: number;
  latency_p95: number;
  usage_percentage: number;
  feedback_score: number;
  feedback_count: number;
  queries_handled: number;
}

export interface ExpertPerformanceResponse {
  experts: ExpertPerformance[];
  period_days: number;
  total_queries: number;
  last_updated?: string | null;
}

export interface QueryTypeStats {
  type: string;
  count: number;
  percentage: number;
  avg_latency_ms: number;
  avg_confidence: number;
}

export interface QueryStatsResponse {
  total_queries: number;
  by_type: QueryTypeStats[];
  avg_latency_ms: number;
  avg_confidence: number;
  period_days: number;
}

export interface RecentQuery {
  trace_id: string;
  query: string;
  timestamp: string;
  experts_used: string[];
  confidence: number;
  latency_ms: number;
  mode: string;
  feedback_received: boolean;
}

export interface RecentQueriesResponse {
  queries: RecentQuery[];
  total_count: number;
  has_more: boolean;
}

export interface ExpertContribution {
  expert_name: string;
  confidence: number;
  weight: number;
  sources_cited: number;
  key_points: string[];
  excerpt?: string;
}

export interface ReasoningTrace {
  trace_id: string;
  query: string;
  timestamp: string;
  contributions: ExpertContribution[];
  aggregation_method: string;
  final_confidence: number;
  confidence_ci: [number, number];
  synthesis: string;
  mode: string;
  has_alternatives: boolean;
  total_latency_ms: number;
  sources_count: number;
}

export interface AggregationStats {
  method: string;
  total_responses: number;
  agreement_rate: number;
  divergence_count: number;
  divergence_rate: number;
  avg_confidence: number;
  confidence_ci: [number, number];
  avg_experts_per_query: number;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Recupera metriche performance per ogni expert.
 */
export async function getExpertPerformance(
  periodDays: number = 7
): Promise<ExpertPerformanceResponse> {
  return get(`${PREFIX}/expert-metrics/performance?period_days=${periodDays}`);
}

/**
 * Recupera statistiche classificazione query.
 */
export async function getQueryStats(periodDays: number = 7): Promise<QueryStatsResponse> {
  return get(`${PREFIX}/expert-metrics/queries/stats?period_days=${periodDays}`);
}

/**
 * Recupera query recenti con summary.
 */
export async function getRecentQueries(
  limit: number = 10,
  offset: number = 0
): Promise<RecentQueriesResponse> {
  return get(`${PREFIX}/expert-metrics/queries/recent?limit=${limit}&offset=${offset}`);
}

/**
 * Recupera reasoning trace per una query specifica.
 */
export async function getReasoningTrace(traceId: string): Promise<ReasoningTrace> {
  return get(`${PREFIX}/expert-metrics/trace/${encodeURIComponent(traceId)}`);
}

/**
 * Recupera statistiche di aggregazione.
 */
export async function getAggregationStats(): Promise<AggregationStats> {
  return get(`${PREFIX}/expert-metrics/aggregation`);
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Formatta accuracy con confidence interval.
 */
export function formatAccuracy(accuracy: number, ci: [number, number]): string {
  return `${accuracy.toFixed(1)}% [${ci[0].toFixed(1)}, ${ci[1].toFixed(1)}]`;
}

/**
 * Ritorna colore per feedback score.
 */
export function getFeedbackColor(score: number): string {
  if (score >= 0.7) return 'text-green-600 dark:text-green-400';
  if (score >= 0.4) return 'text-yellow-600 dark:text-yellow-400';
  if (score >= 0) return 'text-slate-600 dark:text-slate-400';
  return 'text-red-600 dark:text-red-400';
}

/**
 * Ritorna badge per query mode.
 */
export function getModeBadge(mode: string): { text: string; color: string } {
  if (mode === 'convergent') {
    return { text: 'Convergent', color: 'bg-green-100 text-green-800' };
  }
  if (mode === 'divergent') {
    return { text: 'Divergent', color: 'bg-orange-100 text-orange-800' };
  }
  return { text: mode, color: 'bg-slate-100 text-slate-800' };
}

/**
 * Calcola latenza media ponderata.
 */
export function calculateWeightedLatency(stats: QueryTypeStats[]): number {
  const totalCount = stats.reduce((sum, s) => sum + s.count, 0);
  if (totalCount === 0) return 0;

  return stats.reduce((sum, s) => sum + s.avg_latency_ms * s.count, 0) / totalCount;
}
