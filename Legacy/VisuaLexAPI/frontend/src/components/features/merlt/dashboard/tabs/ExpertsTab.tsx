/**
 * ExpertsTab
 * ==========
 *
 * Tab per metriche e performance del sistema Multi-Expert con:
 * - Tabella performance per expert (accuracy, latency, feedback)
 * - Statistiche classificazione query
 * - Reasoning trace visualization
 * - Aggregation stats
 *
 * @example
 * ```tsx
 * <ExpertsTab />
 * ```
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Users,
  RefreshCw,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Brain,
  GitBranch,
  FileText,
  Scale,
} from 'lucide-react';
import { cn } from '../../../../../lib/utils';
import { get } from '../../../../../services/api';

// =============================================================================
// TYPES (from expert_metrics_router.py)
// =============================================================================

interface ExpertPerformance {
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

interface ExpertPerformanceResponse {
  experts: ExpertPerformance[];
  period_days: number;
  total_queries: number;
  last_updated: string;
}

interface QueryTypeStats {
  type: string;
  count: number;
  percentage: number;
  avg_latency_ms: number;
  avg_confidence: number;
}

interface QueryStatsResponse {
  total_queries: number;
  by_type: QueryTypeStats[];
  avg_latency_ms: number;
  avg_confidence: number;
  period_days: number;
}

interface RecentQuery {
  trace_id: string;
  query: string;
  timestamp: string;
  experts_used: string[];
  confidence: number;
  latency_ms: number;
  mode: string;
  feedback_received: boolean;
}

interface RecentQueriesResponse {
  queries: RecentQuery[];
  total_count: number;
  has_more: boolean;
}

interface AggregationStats {
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
// SERVICE FUNCTIONS
// =============================================================================

const PREFIX = 'merlt';

async function getExpertPerformance(periodDays = 7): Promise<ExpertPerformanceResponse> {
  return get(`${PREFIX}/expert-metrics/performance?period_days=${periodDays}`);
}

async function getQueryStats(periodDays = 7): Promise<QueryStatsResponse> {
  return get(`${PREFIX}/expert-metrics/queries/stats?period_days=${periodDays}`);
}

async function getRecentQueries(limit = 10): Promise<RecentQueriesResponse> {
  return get(`${PREFIX}/expert-metrics/queries/recent?limit=${limit}`);
}

async function getAggregationStats(): Promise<AggregationStats> {
  return get(`${PREFIX}/expert-metrics/aggregation`);
}

// =============================================================================
// EXPERT PERFORMANCE TABLE
// =============================================================================

interface ExpertTableProps {
  experts: ExpertPerformance[];
  totalQueries: number;
}

function ExpertTable({ experts, totalQueries }: ExpertTableProps) {
  const expertIcons: Record<string, React.ReactNode> = {
    literal: <FileText size={16} className="text-blue-500" />,
    systemic: <GitBranch size={16} className="text-green-500" />,
    principles: <Scale size={16} className="text-yellow-500" />,
    precedent: <Brain size={16} className="text-orange-500" />,
  };

  const getFeedbackColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600 dark:text-green-400';
    if (score >= 0.4) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Expert Performance (Last 7 days)
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {totalQueries.toLocaleString()} query totali
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-700/50 text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              <th className="px-4 py-3 text-left">Expert</th>
              <th className="px-4 py-3 text-right">Accuracy</th>
              <th className="px-4 py-3 text-right">Latency</th>
              <th className="px-4 py-3 text-right">Usage</th>
              <th className="px-4 py-3 text-right">Feedback</th>
              <th className="px-4 py-3 text-right">Queries</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {experts.map((expert) => (
              <tr key={expert.name} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    {expertIcons[expert.name] || <Users size={16} />}
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {expert.display_name}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {expert.accuracy.toFixed(1)}%
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                      [{expert.accuracy_ci[0].toFixed(1)}, {expert.accuracy_ci[1].toFixed(1)}]
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {(expert.latency_ms / 1000).toFixed(1)}s
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                      (p95: {(expert.latency_p95 / 1000).toFixed(1)}s)
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${expert.usage_percentage}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {expert.usage_percentage.toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <div>
                    <span className={cn('font-medium', getFeedbackColor(expert.feedback_score))}>
                      {expert.feedback_score >= 0 ? '+' : ''}
                      {expert.feedback_score.toFixed(2)}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                      ({expert.feedback_count})
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-gray-100">
                  {expert.queries_handled.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =============================================================================
// QUERY CLASSIFICATION CARD
// =============================================================================

interface QueryClassificationCardProps {
  stats: QueryStatsResponse;
}

function QueryClassificationCard({ stats }: QueryClassificationCardProps) {
  const typeColors: Record<string, string> = {
    definitional: 'bg-blue-500',
    interpretive: 'bg-green-500',
    applicative: 'bg-yellow-500',
    comparative: 'bg-purple-500',
  };

  const typeLabels: Record<string, string> = {
    definitional: 'Definizionale',
    interpretive: 'Interpretativa',
    applicative: 'Applicativa',
    comparative: 'Comparativa',
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Classificazione Query
      </h3>

      <div className="mb-4">
        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          {stats.total_queries.toLocaleString()}
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">Query totali</p>
      </div>

      {/* Stacked bar */}
      <div className="h-4 rounded-full overflow-hidden flex mb-4">
        {stats.by_type.map((type) => (
          <div
            key={type.type}
            className={cn(typeColors[type.type] || 'bg-gray-500')}
            style={{ width: `${type.percentage}%` }}
            title={`${typeLabels[type.type] || type.type}: ${type.count} (${type.percentage.toFixed(1)}%)`}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="space-y-2">
        {stats.by_type.map((type) => (
          <div key={type.type} className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <span
                className={cn('w-3 h-3 rounded-full', typeColors[type.type] || 'bg-gray-500')}
              />
              <span className="text-gray-700 dark:text-gray-300">
                {typeLabels[type.type] || type.type}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-gray-500 dark:text-gray-400">
                {type.count} ({type.percentage.toFixed(1)}%)
              </span>
              <span className="text-gray-400 dark:text-gray-500 text-xs">
                {type.avg_latency_ms}ms
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// AGGREGATION STATS CARD
// =============================================================================

interface AggregationStatsCardProps {
  stats: AggregationStats;
}

function AggregationStatsCard({ stats }: AggregationStatsCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Response Aggregation
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-500 dark:text-gray-400">Method</p>
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
            {stats.method.replace(/_/g, ' ')}
          </p>
        </div>
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-500 dark:text-gray-400">Avg Experts/Query</p>
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {stats.avg_experts_per_query.toFixed(1)}
          </p>
        </div>
      </div>

      {/* Agreement vs Divergence */}
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600 dark:text-gray-400">Agreement Rate</span>
            <span className="text-green-600 dark:text-green-400 font-medium">
              {stats.agreement_rate.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500"
              style={{ width: `${stats.agreement_rate}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600 dark:text-gray-400">Divergence Rate</span>
            <span className="text-orange-600 dark:text-orange-400 font-medium">
              {stats.divergence_rate.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-orange-500"
              style={{ width: `${stats.divergence_rate}%` }}
            />
          </div>
        </div>
      </div>

      {/* Confidence */}
      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <div className="flex items-center justify-between">
          <span className="text-sm text-blue-700 dark:text-blue-300">Avg Confidence</span>
          <span className="text-lg font-semibold text-blue-600 dark:text-blue-400">
            {(stats.avg_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
          95% CI [{(stats.confidence_ci[0] * 100).toFixed(1)}%, {(stats.confidence_ci[1] * 100).toFixed(1)}%]
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// RECENT QUERIES LIST
// =============================================================================

interface RecentQueriesListProps {
  queries: RecentQuery[];
}

function RecentQueriesList({ queries }: RecentQueriesListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('it-IT', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short',
    });
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Query Recenti
        </h3>
      </div>

      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {queries.map((query) => (
          <div key={query.trace_id} className="p-4">
            <button
              onClick={() => setExpandedId(expandedId === query.trace_id ? null : query.trace_id)}
              className="w-full text-left"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {query.query}
                  </p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatTime(query.timestamp)}
                    </span>
                    <span
                      className={cn(
                        'text-xs px-2 py-0.5 rounded',
                        query.mode === 'convergent'
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400'
                      )}
                    >
                      {query.mode}
                    </span>
                    {query.feedback_received && (
                      <span className="text-xs text-blue-600 dark:text-blue-400">
                        Feedback
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3 ml-4">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {(query.confidence * 100).toFixed(0)}%
                  </span>
                  {expandedId === query.trace_id ? (
                    <ChevronUp size={16} className="text-gray-400" />
                  ) : (
                    <ChevronDown size={16} className="text-gray-400" />
                  )}
                </div>
              </div>
            </button>

            {expandedId === query.trace_id && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700"
              >
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Experts usati</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {query.experts_used.map((expert) => (
                        <span
                          key={expert}
                          className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs capitalize"
                        >
                          {expert}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Latenza</p>
                    <p className="text-gray-900 dark:text-gray-100">{query.latency_ms}ms</p>
                  </div>
                </div>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                  Trace ID: {query.trace_id}
                </p>
              </motion.div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExpertsTab() {
  const [performance, setPerformance] = useState<ExpertPerformanceResponse | null>(null);
  const [queryStats, setQueryStats] = useState<QueryStatsResponse | null>(null);
  const [recentQueries, setRecentQueries] = useState<RecentQueriesResponse | null>(null);
  const [aggregationStats, setAggregationStats] = useState<AggregationStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [perf, stats, recent, agg] = await Promise.all([
        getExpertPerformance(),
        getQueryStats(),
        getRecentQueries(),
        getAggregationStats(),
      ]);
      setPerformance(perf);
      setQueryStats(stats);
      setRecentQueries(recent);
      setAggregationStats(agg);
    } catch (err) {
      setError('Errore nel caricamento delle metriche expert');
      console.error('Failed to load expert metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw size={24} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">{error}</p>
        <button
          onClick={fetchAllData}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Performance Table */}
      {performance && (
        <ExpertTable
          experts={performance.experts}
          totalQueries={performance.total_queries}
        />
      )}

      {/* Query Stats + Aggregation */}
      <div className="grid md:grid-cols-2 gap-6">
        {queryStats && <QueryClassificationCard stats={queryStats} />}
        {aggregationStats && <AggregationStatsCard stats={aggregationStats} />}
      </div>

      {/* Recent Queries */}
      {recentQueries && <RecentQueriesList queries={recentQueries.queries} />}
    </div>
  );
}

export default ExpertsTab;
