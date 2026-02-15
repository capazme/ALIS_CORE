/**
 * StatisticsTab
 * =============
 *
 * Tab Statistics con rigore accademico per tesi di laurea:
 * - Hypothesis testing H1-H4 con p-values, effect sizes, CI
 * - Distribuzioni con test di normalità
 * - Matrice di correlazione
 * - Export CSV/JSON/LaTeX
 *
 * @example
 * ```tsx
 * <StatisticsTab />
 * ```
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircle2,
  XCircle,
  RefreshCw,
  AlertCircle,
  Download,
  FileText,
  FileJson,
  FileSpreadsheet,
  TrendingUp,
  Grid3X3,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  getStatisticsOverview,
  exportStatistics,
  formatPValue,
  formatEffectSize,
  formatCI,
  getSignificanceColor,
  getEffectSizeColor,
  type HypothesisTestResult,
  type HypothesisTestSummary,
  type CorrelationMatrix,
  type ExportFormat,
} from '../../../../services/statisticsService';

// =============================================================================
// HYPOTHESIS TEST CARD
// =============================================================================

interface HypothesisTestCardProps {
  test: HypothesisTestResult;
}

function HypothesisTestCard({ test }: HypothesisTestCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white dark:bg-slate-800 rounded-xl border-2 p-6',
        test.supported
          ? 'border-green-200 dark:border-green-800'
          : 'border-red-200 dark:border-red-800'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          {test.supported ? (
            <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
              <CheckCircle2 size={20} className="text-green-600 dark:text-green-400" aria-hidden="true" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <XCircle size={20} className="text-red-600 dark:text-red-400" aria-hidden="true" />
            </div>
          )}
          <div>
            <h3 className="font-bold text-slate-900 dark:text-slate-100">
              {test.hypothesis_id}
            </h3>
            <span
              className={cn(
                'text-xs font-medium',
                test.supported ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}
            >
              {test.supported ? 'SUPPORTED' : 'NOT SUPPORTED'}
            </span>
          </div>
        </div>
        <span className={cn('text-lg font-bold', getSignificanceColor(test.significance))}>
          {test.significance}
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        {test.description}
      </p>

      {/* Pre/Post Stats */}
      {test.pre_stats && test.post_stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Pre-training</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {(test.pre_stats.mean * 100).toFixed(1)}% ± {(test.pre_stats.std * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-slate-400">n = {test.pre_stats.n}</p>
          </div>
          <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Post-training</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {(test.post_stats.mean * 100).toFixed(1)}% ± {(test.post_stats.std * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-slate-400">n = {test.post_stats.n}</p>
          </div>
        </div>
      )}

      {/* Delta */}
      {test.delta !== undefined && (
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={16} className={test.delta >= 0 ? 'text-green-500' : 'text-red-500'} aria-hidden="true" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Δ = {test.delta >= 0 ? '+' : ''}{(test.delta * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Statistical test */}
      <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 mb-4 font-mono text-sm">
        <p className="text-slate-700 dark:text-slate-300">
          {test.test_type === 't-test' && `t(${test.df}) = ${test.statistic.toFixed(2)}`}
          {test.test_type === 'paired-t-test' && `t(${test.df}) = ${test.statistic.toFixed(2)}`}
          {test.test_type === 'anova' && `F(${test.df}, ${test.df2}) = ${test.statistic.toFixed(2)}`}
          {test.test_type === 'mann-whitney' && `U = ${test.statistic.toFixed(0)}`}
          {test.test_type === 'wilcoxon' && `W = ${test.statistic.toFixed(0)}`}
          {test.test_type === 'correlation' && `r = ${test.statistic.toFixed(2)}`}
          {test.test_type === 'chi-square' && `χ²(${test.df}) = ${test.statistic.toFixed(2)}`}
          , <span className={getSignificanceColor(test.significance)}>
            {formatPValue(test.p_value, test.significance)}
          </span>
        </p>
      </div>

      {/* Effect size */}
      <div className="flex items-center justify-between text-sm">
        <span className={test.effect_size ? getEffectSizeColor(test.effect_size.interpretation) : 'text-slate-500'}>
          {formatEffectSize(test.effect_size)}
        </span>
        {test.ci_lower != null && test.ci_upper != null && (
          <span className="text-slate-500 dark:text-slate-400">
            {formatCI(test.ci_lower, test.ci_upper, test.ci_level)}
          </span>
        )}
      </div>

      {/* Notes */}
      {test.notes && (
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-3 italic">
          {test.notes}
        </p>
      )}
    </motion.div>
  );
}

// =============================================================================
// CORRELATION MATRIX
// =============================================================================

interface CorrelationMatrixCardProps {
  matrix: CorrelationMatrix;
}

function CorrelationMatrixCard({ matrix }: CorrelationMatrixCardProps) {
  const getCorrelationColor = (r: number) => {
    const intensity = Math.abs(r);
    if (r > 0) {
      if (intensity > 0.7) return 'bg-green-600 text-white';
      if (intensity > 0.4) return 'bg-green-400 text-white';
      return 'bg-green-200 text-green-800';
    } else {
      if (intensity > 0.7) return 'bg-red-600 text-white';
      if (intensity > 0.4) return 'bg-red-400 text-white';
      return 'bg-red-200 text-red-800';
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Grid3X3 size={20} className="text-blue-500" aria-hidden="true" />
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Matrice di Correlazione
        </h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="p-2"></th>
              {matrix.variables.map((v) => (
                <th key={v} className="p-2 text-center text-slate-600 dark:text-slate-400 font-medium">
                  {v}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.variables.map((rowVar, i) => (
              <tr key={rowVar}>
                <td className="p-2 text-slate-600 dark:text-slate-400 font-medium">
                  {rowVar}
                </td>
                {matrix.matrix[i].map((r, j) => (
                  <td key={j} className="p-1">
                    <div
                      className={cn(
                        'w-12 h-12 flex items-center justify-center rounded text-xs font-medium',
                        i === j
                          ? 'bg-slate-200 dark:bg-slate-700 text-slate-500'
                          : getCorrelationColor(r)
                      )}
                      title={`p = ${matrix.p_values[i][j].toFixed(4)}`}
                    >
                      {r.toFixed(2)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-4 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-red-500 rounded" /> Negativa
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-green-500 rounded" /> Positiva
        </span>
      </div>

      {/* Significant pairs */}
      {matrix.significant_pairs.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
          <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Correlazioni significative:
          </p>
          <div className="flex flex-wrap gap-2">
            {matrix.significant_pairs.map((pair, idx) => (
              <span
                key={idx}
                className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400 rounded"
              >
                {pair.var1} ↔ {pair.var2}: r={pair.r.toFixed(2)} {pair.significance}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// EXPORT PANEL
// =============================================================================

interface ExportPanelProps {
  onExport: (format: ExportFormat) => void;
  loading: boolean;
}

function ExportPanel({ onExport, loading }: ExportPanelProps) {
  const formats: { id: ExportFormat; label: string; icon: React.ReactNode; description: string }[] = [
    {
      id: 'csv',
      label: 'CSV',
      icon: <FileSpreadsheet size={24} className="text-green-500" />,
      description: 'Raw data per analisi esterna',
    },
    {
      id: 'json',
      label: 'JSON',
      icon: <FileJson size={24} className="text-blue-500" />,
      description: 'Full export strutturato',
    },
    {
      id: 'latex',
      label: 'LaTeX',
      icon: <FileText size={24} className="text-purple-500" />,
      description: 'Tabelle per tesi accademica',
    },
  ];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Download size={20} className="text-blue-500" aria-hidden="true" />
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Export Statistiche
        </h3>
      </div>

      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        Esporta i risultati statistici per analisi esterne o inclusione nella tesi.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {formats.map((format) => (
          <button
            key={format.id}
            onClick={() => onExport(format.id)}
            disabled={loading}
            className={cn(
              'flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors',
              'border-slate-200 dark:border-slate-700',
              'hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
            )}
          >
            <span aria-hidden="true">{format.icon}</span>
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              {format.label}
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 text-center">
              {format.description}
            </span>
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex items-center justify-center gap-2 mt-4 text-slate-500" role="status">
          <RefreshCw size={16} className="animate-spin" aria-hidden="true" />
          <span className="text-sm">Generazione in corso...</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function StatisticsTab() {
  const [hypothesisTests, setHypothesisTests] = useState(null as HypothesisTestSummary | null);
  const [correlations, setCorrelations] = useState(null as CorrelationMatrix | null);
  const [loading, setLoading] = useState(true);
  const [exportLoading, setExportLoading] = useState(false);
  const [error, setError] = useState(null as string | null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const overview = await getStatisticsOverview();
      setHypothesisTests(overview.hypothesis_tests);
      setCorrelations(overview.correlations);
    } catch (err) {
      setError('Errore nel caricamento delle statistiche');
      console.error('Failed to load statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: ExportFormat) => {
    setExportLoading(true);
    try {
      const response = await exportStatistics({
        format,
        include_hypothesis_tests: true,
        include_descriptive_stats: true,
        include_confidence_intervals: true,
        include_effect_sizes: true,
      });

      if (response.success && response.download_url) {
        // Trigger download
        const a = document.createElement('a');
        a.href = `/api/merlt${response.download_url}`;
        a.download = response.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Failed to export statistics:', err);
    } finally {
      setExportLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12" role="status">
        <RefreshCw size={24} className="animate-spin text-blue-500" aria-hidden="true" />
        <span className="sr-only">Caricamento statistiche in corso...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12" role="alert">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" aria-hidden="true" />
        <p className="text-slate-500 dark:text-slate-400">{error}</p>
        <button
          onClick={fetchData}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary */}
      {hypothesisTests && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-blue-900 dark:text-blue-100">
                Hypothesis Testing Summary
              </h2>
              <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                {hypothesisTests.supported_count}/{hypothesisTests.total_count} ipotesi supportate
                (α = {hypothesisTests.alpha})
              </p>
            </div>
            <div className="text-4xl font-bold text-blue-600 dark:text-blue-400">
              {((hypothesisTests.supported_count / hypothesisTests.total_count) * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      )}

      {/* Hypothesis Tests Grid */}
      {hypothesisTests && (
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
            Test di Ipotesi
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {hypothesisTests.tests.map((test: HypothesisTestResult) => (
              <HypothesisTestCard key={test.hypothesis_id} test={test} />
            ))}
          </div>
        </div>
      )}

      {/* Correlation Matrix */}
      {correlations && <CorrelationMatrixCard matrix={correlations} />}

      {/* Export Panel */}
      <ExportPanel onExport={handleExport} loading={exportLoading} />

      {/* Academic note */}
      <div className="text-center text-sm text-slate-500 dark:text-slate-400">
        <p>
          Tutti i test statistici seguono gli standard APA 7th Edition.
          <br />
          Gli effect size sono interpretati secondo le convenzioni di Cohen (1988).
        </p>
      </div>
    </div>
  );
}

export default StatisticsTab;
