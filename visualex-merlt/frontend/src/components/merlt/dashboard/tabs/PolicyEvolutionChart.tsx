/**
 * PolicyEvolutionChart
 * =====================
 *
 * Visualizzazione evoluzione policy nel tempo:
 * - Confidence/reward time-series (line chart)
 * - Expert usage distribution (stacked bars)
 * - Aggregation trends (disagreement/rating)
 *
 * Consuma dati da usePolicyEvolution hook.
 *
 * @example
 * ```tsx
 * <PolicyEvolutionChart />
 * ```
 */

import { useState } from 'react';
import {
  RefreshCw,
  TrendingUp,
  BarChart3,
  AlertCircle,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import { usePolicyEvolution } from '../../../../hooks/usePolicyEvolution';
import type { TimeSeriesPoint, ExpertEvolutionPoint } from '../../../../services/policyEvolutionService';

// =============================================================================
// CONFIDENCE TIME-SERIES
// =============================================================================

function ConfidenceTimeSeries({ data }: { data: TimeSeriesPoint[] }) {
  if (data.length === 0) {
    return (
      <p className="text-sm text-slate-400 dark:text-slate-500 text-center py-4">
        Nessun dato disponibile.
      </p>
    );
  }

  const maxQueries = Math.max(...data.map((d: TimeSeriesPoint) => d.query_count), 1);

  return (
    <div className="space-y-1">
      {data.slice(-20).map((point: TimeSeriesPoint) => {
        const conf = point.confidence ?? 0;
        const reward = point.reward ?? 0;
        return (
          <div key={point.timestamp} className="flex items-center gap-2 text-xs">
            <span className="w-20 text-slate-500 dark:text-slate-400 font-mono flex-shrink-0">
              {point.timestamp.slice(5)}
            </span>
            <div className="flex-1 flex gap-1 items-center">
              {/* Confidence bar */}
              <div className="flex-1 h-4 bg-slate-100 dark:bg-slate-700 rounded overflow-hidden relative">
                <div
                  className="h-full bg-blue-500 dark:bg-blue-400 rounded transition-all"
                  style={{ width: `${conf * 100}%` }}
                />
                <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-slate-700 dark:text-slate-200">
                  {(conf * 100).toFixed(0)}%
                </span>
              </div>
              {/* Reward indicator */}
              {point.reward !== null && (
                <div
                  className={cn(
                    'w-8 text-center rounded px-1 py-0.5 text-[10px] font-medium',
                    reward >= 0.6
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                      : reward >= 0.3
                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                  )}
                >
                  {(reward * 100).toFixed(0)}
                </div>
              )}
            </div>
            {/* Query count */}
            <span className="w-8 text-right text-slate-400 dark:text-slate-500">
              {point.query_count}
            </span>
          </div>
        );
      })}
      {/* Legend */}
      <div className="flex items-center gap-4 pt-2 text-[10px] text-slate-400 dark:text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-blue-500 rounded" /> Confidence
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-green-500 rounded" /> Reward
        </span>
        <span className="ml-auto">n = query count</span>
      </div>
    </div>
  );
}

// =============================================================================
// EXPERT USAGE DISTRIBUTION
// =============================================================================

const EXPERT_COLORS: Record<string, string> = {
  literal: 'bg-blue-500',
  systemic: 'bg-green-500',
  principles: 'bg-yellow-500',
  precedent: 'bg-orange-500',
};

const EXPERT_COLORS_LIGHT: Record<string, string> = {
  literal: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
  systemic: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  principles: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
  precedent: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
};

function ExpertUsageChart({ data }: { data: ExpertEvolutionPoint[] }) {
  if (data.length === 0) {
    return (
      <p className="text-sm text-slate-400 dark:text-slate-500 text-center py-4">
        Nessun dato disponibile.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {data.slice(-15).map((point: ExpertEvolutionPoint) => (
        <div key={point.timestamp} className="flex items-center gap-2 text-xs">
          <span className="w-20 text-slate-500 dark:text-slate-400 font-mono flex-shrink-0">
            {point.timestamp.slice(5)}
          </span>
          <div className="flex-1 h-5 flex rounded overflow-hidden">
            {point.literal > 0 && (
              <div
                className="bg-blue-500 transition-all"
                style={{ width: `${point.literal * 100}%` }}
                title={`Literal: ${(point.literal * 100).toFixed(1)}%`}
              />
            )}
            {point.systemic > 0 && (
              <div
                className="bg-green-500 transition-all"
                style={{ width: `${point.systemic * 100}%` }}
                title={`Systemic: ${(point.systemic * 100).toFixed(1)}%`}
              />
            )}
            {point.principles > 0 && (
              <div
                className="bg-yellow-500 transition-all"
                style={{ width: `${point.principles * 100}%` }}
                title={`Principles: ${(point.principles * 100).toFixed(1)}%`}
              />
            )}
            {point.precedent > 0 && (
              <div
                className="bg-orange-500 transition-all"
                style={{ width: `${point.precedent * 100}%` }}
                title={`Precedent: ${(point.precedent * 100).toFixed(1)}%`}
              />
            )}
          </div>
        </div>
      ))}
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 pt-2 text-[10px]">
        {(['literal', 'systemic', 'principles', 'precedent'] as const).map((expert) => (
          <span key={expert} className={cn('px-1.5 py-0.5 rounded', EXPERT_COLORS_LIGHT[expert])}>
            {expert}
          </span>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function PolicyEvolutionChart() {
  const [days, setDays] = useState(30);
  const { timeSeries, expertEvolution, aggregationHistory, loading, error, refresh } =
    usePolicyEvolution(days);

  if (loading) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
        <div className="flex items-center justify-center py-8" role="status">
          <RefreshCw size={20} className="animate-spin text-blue-500" aria-hidden="true" />
          <span className="sr-only">Caricamento evoluzione policy...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
        <div className="text-center py-6" role="alert">
          <AlertCircle size={24} className="mx-auto text-red-400 mb-2" aria-hidden="true" />
          <p className="text-sm text-slate-500">{error}</p>
          <button
            onClick={refresh}
            className="mt-3 px-4 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
          >
            Riprova
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp size={20} className="text-slate-500" aria-hidden="true" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Policy Evolution
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {/* Period selector */}
          <select
            value={days}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setDays(Number(e.target.value))}
            className="px-2 py-1 text-xs rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-label="Periodo"
          >
            <option value={7}>7 giorni</option>
            <option value={30}>30 giorni</option>
            <option value={90}>90 giorni</option>
          </select>
          <button
            onClick={refresh}
            className="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            aria-label="Aggiorna"
          >
            <RefreshCw size={16} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Confidence / Reward Time-Series */}
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-1.5">
            <TrendingUp size={14} aria-hidden="true" />
            Confidence & Reward
          </h4>
          <ConfidenceTimeSeries data={timeSeries} />
        </div>

        {/* Expert Usage Distribution */}
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-1.5">
            <BarChart3 size={14} aria-hidden="true" />
            Expert Usage Distribution
          </h4>
          <ExpertUsageChart data={expertEvolution} />
        </div>

        {/* Aggregation Trends Summary */}
        {aggregationHistory.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
              Aggregation Trends
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    <th className="px-2 py-1.5 text-left">Component</th>
                    <th className="px-2 py-1.5 text-right">Avg Rating</th>
                    <th className="px-2 py-1.5 text-right">Disagreement</th>
                    <th className="px-2 py-1.5 text-right">Feedback</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                  {aggregationHistory.slice(-10).map((row: { component: string; avg_rating: number | null; disagreement_score: number | null; total_feedback: number }, idx: number) => (
                    <tr key={`${row.component}-${idx}`}>
                      <td className="px-2 py-1.5 text-slate-700 dark:text-slate-300">{row.component}</td>
                      <td className="px-2 py-1.5 text-right text-slate-600 dark:text-slate-400">
                        {row.avg_rating !== null ? row.avg_rating.toFixed(2) : '-'}
                      </td>
                      <td className="px-2 py-1.5 text-right">
                        {row.disagreement_score !== null ? (
                          <span
                            className={cn(
                              'px-1.5 py-0.5 rounded',
                              row.disagreement_score > 0.5
                                ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                : row.disagreement_score > 0.2
                                  ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                  : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                            )}
                          >
                            {row.disagreement_score.toFixed(2)}
                          </span>
                        ) : '-'}
                      </td>
                      <td className="px-2 py-1.5 text-right text-slate-600 dark:text-slate-400">
                        {row.total_feedback}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PolicyEvolutionChart;
