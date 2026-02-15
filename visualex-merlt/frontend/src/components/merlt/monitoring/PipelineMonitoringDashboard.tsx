/**
 * PipelineMonitoringDashboard
 * ===========================
 *
 * Dashboard principale per monitorare pipeline di ingestion/enrichment.
 * Mostra statistiche aggregate, lista runs, e dettagli in tempo reale.
 *
 * Features:
 * - Statistiche KPI aggregate
 * - Lista runs con filtri per status/tipo
 * - Progress real-time via WebSocket
 * - Dettagli errori e retry
 *
 * @example
 * ```tsx
 * <PipelineMonitoringDashboard />
 * ```
 */

import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Filter,
  Activity,
  CheckCircle2,
  XCircle,
  Database,
  Play,
  Download,
  BarChart3,
  X,
  FileJson,
  FileSpreadsheet,
  FileCode,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { usePipelineMonitoring } from '../../../hooks/usePipelineMonitoring';
import {
  type PipelineRun,
  type PipelineError,
  type ProgressUpdate,
  type PipelineStatus,
  type PipelineType,
  type DatasetStats,
  type StartPipelineRequest,
  getPipelineErrors,
  startPipeline,
  getDatasetStats,
  exportDataset,
} from '../../../services/pipelineService';
import { PipelineStatsPanel } from './PipelineStatsPanel';
import { PipelineRunCard } from './PipelineRunCard';

// =============================================================================
// TYPES
// =============================================================================

interface PipelineMonitoringDashboardProps {
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function PipelineMonitoringDashboard({
  className,
}: PipelineMonitoringDashboardProps) {
  // State
  const [statusFilter, setStatusFilter] = useState('all' as PipelineStatus | 'all');
  const [typeFilter, setTypeFilter] = useState('all' as PipelineType | 'all');
  const [expandedRunId, setExpandedRunId] = useState(null as string | null);
  const [runErrors, setRunErrors] = useState({} as Record<string, PipelineError[]>);
  const [liveProgress, setLiveProgress] = useState({} as Record<string, ProgressUpdate>);

  // New state for start pipeline dialog
  const [showStartDialog, setShowStartDialog] = useState(false);
  const [startLoading, setStartLoading] = useState(false);
  const [startForm, setStartForm] = useState({
    tipo_atto: 'codice civile',
    libro: 'IV',
    batch_size: 10,
    skip_existing: true,
    with_enrichment: true,
  });

  // Dataset state
  const [datasetStats, setDatasetStats] = useState(null as DatasetStats | null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);

  // Hook
  const {
    runs,
    activeRuns,
    stats,
    isLoading,
    error,
    refresh,
    subscribeToRun,
  } = usePipelineMonitoring({
    statusFilter: statusFilter === 'all' ? undefined : statusFilter,
    typeFilter: typeFilter === 'all' ? undefined : typeFilter,
    pollingInterval: 5000,
  });

  // Subscribe to active runs for real-time updates
  useEffect(() => {
    const unsubscribers: (() => void)[] = [];

    activeRuns.forEach((run) => {
      const unsubscribe = subscribeToRun(
        run.run_id,
        (progress) => {
          setLiveProgress((prev: Record<string, ProgressUpdate>) => ({
            ...prev,
            [run.run_id]: progress,
          }));
        },
        (err) => {
          setRunErrors((prev: Record<string, PipelineError[]>) => ({
            ...prev,
            [run.run_id]: [...(prev[run.run_id] || []), err],
          }));
        }
      );
      unsubscribers.push(unsubscribe);
    });

    return () => {
      unsubscribers.forEach((unsub) => unsub());
    };
  }, [activeRuns.map((r) => r.run_id).join(','), subscribeToRun]);

  // Load errors when expanding a run
  const handleExpand = useCallback(async (runId: string) => {
    if (expandedRunId === runId) {
      setExpandedRunId(null);
      return;
    }

    setExpandedRunId(runId);

    // Load errors if not already loaded
    if (!runErrors[runId]) {
      try {
        const errors = await getPipelineErrors(runId);
        setRunErrors((prev: Record<string, PipelineError[]>) => ({
          ...prev,
          [runId]: errors,
        }));
      } catch (err) {
        // Error loading run errors - handled silently
      }
    }
  }, [expandedRunId, runErrors]);

  // Load dataset stats
  const loadDatasetStats = useCallback(async () => {
    setDatasetLoading(true);
    try {
      const stats = await getDatasetStats();
      setDatasetStats(stats);
    } catch (err) {
      // Error loading dataset stats - handled silently
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  // Load stats on mount
  useEffect(() => {
    loadDatasetStats();
  }, [loadDatasetStats]);

  // Handle start pipeline
  const handleStartPipeline = async () => {
    setStartLoading(true);
    try {
      const response = await startPipeline(startForm);
      if (response.success) {
        setShowStartDialog(false);
        refresh(); // Refresh the runs list
      } else {
        alert(response.message);
      }
    } catch (err) {
      // Error starting pipeline - alert shown to user
      alert('Errore nell\'avvio della pipeline');
    } finally {
      setStartLoading(false);
    }
  };

  // Handle export
  const handleExport = async (format: 'json' | 'csv' | 'cypher') => {
    setExportLoading(true);
    try {
      const response = await exportDataset({ format, limit: 10000 });
      if (response.success && response.download_url) {
        // Download using fetch with auth headers, then trigger download
        const token = localStorage.getItem('access_token');
        const downloadUrl = `/api/merlt/pipeline/dataset/download/${response.download_url.split('/').pop()}`;

        const downloadResponse = await fetch(downloadUrl, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {},
        });

        if (downloadResponse.ok) {
          const blob = await downloadResponse.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = response.download_url.split('/').pop() || `export.${format}`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          setShowExportDialog(false);
        } else {
          alert('Errore nel download del file');
        }
      } else {
        alert(response.message);
      }
    } catch (err) {
      // Error exporting dataset - alert shown to user
      alert('Errore nell\'export del dataset');
    } finally {
      setExportLoading(false);
    }
  };

  // Filter runs
  const filteredRuns = runs;

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Pipeline Monitoring
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Monitora lo stato delle pipeline di ingestion e enrichment
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setShowStartDialog(true)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
              'bg-green-600 text-white hover:bg-green-700',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
            )}
          >
            <Play size={16} aria-hidden="true" />
            Avvia Pipeline
          </button>

          <button
            onClick={() => setShowExportDialog(true)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
              'bg-purple-600 text-white hover:bg-purple-700',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
            )}
          >
            <Download size={16} aria-hidden="true" />
            Esporta Dataset
          </button>

          <button
            onClick={() => refresh()}
            disabled={isLoading}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
              'bg-blue-600 text-white hover:bg-blue-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            <RefreshCw size={16} className={cn(isLoading && 'animate-spin')} aria-hidden="true" />
            Aggiorna
          </button>
        </div>
      </div>

      {/* Dataset Stats Panel */}
      {datasetStats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <Database size={14} />
              Nodi Totali
            </div>
            <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
              {datasetStats.total_nodes.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <Activity size={14} />
              Relazioni
            </div>
            <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
              {datasetStats.total_edges.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <FileJson size={14} />
              Articoli
            </div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {datasetStats.articles_count.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <BarChart3 size={14} />
              Entità
            </div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {datasetStats.entities_count.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <Database size={14} />
              Embeddings
            </div>
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {datasetStats.embeddings_count.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
              <Activity size={14} />
              Bridge Mappings
            </div>
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {datasetStats.bridge_mappings.toLocaleString()}
            </div>
          </div>
        </div>
      )}

      {/* Stats Panel */}
      <PipelineStatsPanel stats={stats} />

      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Filter size={16} className="text-slate-400" />
          <span className="text-sm text-slate-600 dark:text-slate-400">Filtri:</span>
        </div>

        {/* Status Filter */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-1 overflow-x-auto">
          {(['all', 'running', 'completed', 'failed'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                statusFilter === status
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              )}
            >
              {status === 'all' && <Database size={14} className="inline mr-1" />}
              {status === 'running' && <Activity size={14} className="inline mr-1" />}
              {status === 'completed' && <CheckCircle2 size={14} className="inline mr-1" />}
              {status === 'failed' && <XCircle size={14} className="inline mr-1" />}
              {status === 'all' ? 'Tutti' : status === 'running' ? 'Attivi' : status === 'completed' ? 'Completati' : 'Falliti'}
            </button>
          ))}
        </div>

        {/* Type Filter */}
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as PipelineType | 'all')}
          className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-sm"
        >
          <option value="all">Tutti i tipi</option>
          <option value="ingestion">Ingestion</option>
          <option value="enrichment">Enrichment</option>
          <option value="batch_ingestion">Batch Ingestion</option>
        </select>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4" role="alert">
          <p className="text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Runs List */}
      <div className="space-y-4">
        {isLoading && filteredRuns.length === 0 ? (
          <div className="flex items-center justify-center py-12" role="status" aria-label="Caricamento pipeline">
            <RefreshCw size={24} className="animate-spin text-slate-400" aria-hidden="true" />
            <span className="sr-only">Caricamento...</span>
          </div>
        ) : filteredRuns.length === 0 ? (
          <div className="text-center py-12 bg-slate-50 dark:bg-slate-800/50 rounded-xl" role="status">
            <Database size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-4" />
            <p className="text-slate-500 dark:text-slate-400">
              Nessuna pipeline run trovata
            </p>
            <p className="text-sm text-slate-400 dark:text-slate-500 mt-1">
              Avvia una pipeline di ingestion o enrichment per vedere i dati qui
            </p>
          </div>
        ) : (
          filteredRuns.map((run) => (
            <PipelineRunCard
              key={run.run_id}
              run={run}
              liveProgress={liveProgress[run.run_id]}
              errors={runErrors[run.run_id]}
              isExpanded={expandedRunId === run.run_id}
              onExpand={() => handleExpand(run.run_id)}
            />
          ))
        )}
      </div>

      {/* Start Pipeline Dialog */}
      {showStartDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" role="dialog" aria-label="Avvia nuova pipeline" aria-modal="true">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                Avvia Nuova Pipeline
              </h2>
              <button
                onClick={() => setShowStartDialog(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                aria-label="Chiudi dialogo"
              >
                <X size={20} aria-hidden="true" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Tipo Atto
                </label>
                <select
                  value={startForm.tipo_atto}
                  onChange={(e) => setStartForm({ ...startForm, tipo_atto: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
                >
                  <option value="codice civile">Codice Civile</option>
                  <option value="codice penale">Codice Penale</option>
                  <option value="codice di procedura civile">Codice di Procedura Civile</option>
                  <option value="codice di procedura penale">Codice di Procedura Penale</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Libro (opzionale)
                </label>
                <select
                  value={startForm.libro || ''}
                  onChange={(e) => setStartForm({ ...startForm, libro: e.target.value || undefined })}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
                >
                  <option value="">Tutti i libri</option>
                  <option value="I">Libro I</option>
                  <option value="II">Libro II</option>
                  <option value="III">Libro III</option>
                  <option value="IV">Libro IV</option>
                  <option value="V">Libro V</option>
                  <option value="VI">Libro VI</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={startForm.batch_size}
                  onChange={(e) => setStartForm({ ...startForm, batch_size: parseInt(e.target.value) || 10 })}
                  min={1}
                  max={50}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
                />
              </div>

              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={startForm.skip_existing}
                    onChange={(e) => setStartForm({ ...startForm, skip_existing: e.target.checked })}
                    className="rounded"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Salta articoli esistenti
                  </span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={startForm.with_enrichment}
                    onChange={(e) => setStartForm({ ...startForm, with_enrichment: e.target.checked })}
                    className="rounded"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Con enrichment
                  </span>
                </label>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => setShowStartDialog(false)}
                className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700"
              >
                Annulla
              </button>
              <button
                onClick={handleStartPipeline}
                disabled={startLoading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
              >
                {startLoading ? (
                  <RefreshCw size={16} className="animate-spin" />
                ) : (
                  <Play size={16} />
                )}
                Avvia
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Dialog */}
      {showExportDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" role="dialog" aria-label="Esporta dataset" aria-modal="true">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                Esporta Dataset
              </h2>
              <button
                onClick={() => setShowExportDialog(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                aria-label="Chiudi dialogo"
              >
                <X size={20} aria-hidden="true" />
              </button>
            </div>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Scegli il formato di esportazione per il dataset del knowledge graph.
            </p>

            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => handleExport('json')}
                disabled={exportLoading}
                className={cn(
                  'flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors',
                  'border-slate-200 dark:border-slate-700 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20',
                  'disabled:opacity-50'
                )}
              >
                <FileJson size={32} className="text-blue-500" />
                <span className="text-sm font-medium">JSON</span>
              </button>

              <button
                onClick={() => handleExport('csv')}
                disabled={exportLoading}
                className={cn(
                  'flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors',
                  'border-slate-200 dark:border-slate-700 hover:border-green-500 hover:bg-green-50 dark:hover:bg-green-900/20',
                  'disabled:opacity-50'
                )}
              >
                <FileSpreadsheet size={32} className="text-green-500" />
                <span className="text-sm font-medium">CSV</span>
              </button>

              <button
                onClick={() => handleExport('cypher')}
                disabled={exportLoading}
                className={cn(
                  'flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors',
                  'border-slate-200 dark:border-slate-700 hover:border-purple-500 hover:bg-purple-50 dark:hover:bg-purple-900/20',
                  'disabled:opacity-50'
                )}
              >
                <FileCode size={32} className="text-purple-500" />
                <span className="text-sm font-medium">Cypher</span>
              </button>
            </div>

            {exportLoading && (
              <div className="flex items-center justify-center gap-2 mt-4 text-slate-500" role="status">
                <RefreshCw size={16} className="animate-spin" aria-hidden="true" />
                <span>Esportazione in corso...</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default PipelineMonitoringDashboard;
