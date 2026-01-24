/**
 * PipelineRunCard
 * ===============
 *
 * Card per visualizzare una singola pipeline run.
 * Mostra status, progress bar, statistiche e azioni.
 */

import { useState, useEffect } from 'react';
import {
  Activity,
  CheckCircle2,
  XCircle,
  Pause,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Clock,
  Zap,
  FileText,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  type PipelineRun,
  type ProgressUpdate,
  type PipelineError,
  formatETA,
  formatSpeed,
  formatDuration,
} from '../../../../services/pipelineService';

// =============================================================================
// TYPES
// =============================================================================

interface PipelineRunCardProps {
  run: PipelineRun;
  /** Progress data da WebSocket (opzionale) */
  liveProgress?: ProgressUpdate | null;
  /** Callback per espandere dettagli */
  onExpand?: () => void;
  /** Callback per retry */
  onRetry?: () => void;
  /** Lista errori (se espanso) */
  errors?: PipelineError[];
  /** Card espansa */
  isExpanded?: boolean;
  className?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function getStatusIcon(status: PipelineRun['status']) {
  switch (status) {
    case 'running':
      return <Activity size={18} className="text-blue-500 animate-pulse" />;
    case 'completed':
      return <CheckCircle2 size={18} className="text-green-500" />;
    case 'failed':
      return <XCircle size={18} className="text-red-500" />;
    case 'paused':
      return <Pause size={18} className="text-amber-500" />;
    default:
      return <Activity size={18} className="text-gray-400" />;
  }
}

function getStatusBadge(status: PipelineRun['status']) {
  const classes = {
    running: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    completed: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    failed: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    paused: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
  };

  const labels = {
    running: 'In Esecuzione',
    completed: 'Completato',
    failed: 'Fallito',
    paused: 'In Pausa',
  };

  return (
    <span
      className={cn(
        'px-2.5 py-1 rounded-full text-xs font-medium',
        classes[status] || 'bg-gray-100 text-gray-700'
      )}
    >
      {labels[status] || status}
    </span>
  );
}

function getTypeBadge(type: PipelineRun['type']) {
  const labels = {
    ingestion: 'Ingestion',
    enrichment: 'Enrichment',
    batch_ingestion: 'Batch Ingestion',
  };

  return (
    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 rounded text-xs font-medium">
      {labels[type] || type}
    </span>
  );
}

function formatDateTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleString('it-IT', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoString;
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

export function PipelineRunCard({
  run,
  liveProgress,
  onExpand,
  onRetry,
  errors,
  isExpanded = false,
  className,
}: PipelineRunCardProps) {
  // Use live progress if available, otherwise use run data
  const progress = liveProgress?.progress ?? run.progress;
  const currentItem = liveProgress?.current_item;
  const speed = liveProgress?.speed_per_sec ?? 0;
  const eta = liveProgress?.eta_seconds ?? 0;
  const elapsed = liveProgress?.elapsed_seconds;

  const summary = run.summary || {};
  const successful = summary.successful ?? summary.success ?? 0;
  const failed = summary.failed ?? summary.errors ?? 0;

  return (
    <div
      className={cn(
        'bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700',
        'transition-all duration-200',
        isExpanded && 'ring-2 ring-blue-500/50',
        className
      )}
    >
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Left: Icon + Title */}
          <div className="flex items-start gap-3 min-w-0 flex-1">
            <div className="mt-0.5">{getStatusIcon(run.status)}</div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100 truncate">
                  {run.run_id}
                </h3>
                {getTypeBadge(run.type)}
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Avviato: {formatDateTime(run.started_at)}
                {run.completed_at && ` | Completato: ${formatDateTime(run.completed_at)}`}
              </p>
            </div>
          </div>

          {/* Right: Status + Actions */}
          <div className="flex items-center gap-2 shrink-0">
            {getStatusBadge(run.status)}
            <button
              onClick={onExpand}
              className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={isExpanded ? 'Comprimi' : 'Espandi'}
            >
              {isExpanded ? (
                <ChevronUp size={18} className="text-gray-500" />
              ) : (
                <ChevronDown size={18} className="text-gray-500" />
              )}
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-gray-600 dark:text-gray-400">
              Progresso: {progress.toFixed(1)}%
            </span>
            {currentItem && run.status === 'running' && (
              <span className="text-gray-500 dark:text-gray-500 text-xs truncate max-w-[200px]">
                {currentItem}
              </span>
            )}
          </div>
          <div className="h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all duration-300',
                run.status === 'running' && 'bg-blue-500',
                run.status === 'completed' && 'bg-green-500',
                run.status === 'failed' && 'bg-red-500',
                run.status === 'paused' && 'bg-amber-500'
              )}
              style={{ width: `${Math.min(100, progress)}%` }}
            />
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-3 flex items-center gap-4 text-sm">
          {run.status === 'running' && (
            <>
              <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                <Zap size={14} />
                <span>{formatSpeed(speed)}</span>
              </div>
              <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                <Clock size={14} />
                <span>ETA: {formatETA(eta)}</span>
              </div>
            </>
          )}
          {elapsed !== undefined && (
            <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
              <Clock size={14} />
              <span>{formatDuration(elapsed)}</span>
            </div>
          )}
          {successful > 0 && (
            <div className="flex items-center gap-1 text-green-600 dark:text-green-400">
              <CheckCircle2 size={14} />
              <span>{successful}</span>
            </div>
          )}
          {failed > 0 && (
            <div className="flex items-center gap-1 text-red-600 dark:text-red-400">
              <XCircle size={14} />
              <span>{failed}</span>
            </div>
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50/50 dark:bg-gray-900/30">
          {/* Config */}
          {run.config && Object.keys(run.config).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Configurazione
              </h4>
              <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-3">
                <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
                  {JSON.stringify(run.config, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Errors */}
          {errors && errors.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Errori ({errors.length})
              </h4>
              <div className="space-y-2 max-h-[200px] overflow-y-auto">
                {errors.map((error, i) => (
                  <div
                    key={i}
                    className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3"
                  >
                    <div className="flex items-center gap-2 text-sm">
                      <FileText size={14} className="text-red-500 shrink-0" />
                      <span className="font-medium text-red-700 dark:text-red-400">
                        {error.item_id}
                      </span>
                      <span className="text-red-600 dark:text-red-500">
                        [{error.phase}]
                      </span>
                    </div>
                    <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                      {error.error_message}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          {(run.status === 'failed' || (errors && errors.length > 0)) && onRetry && (
            <div className="flex justify-end">
              <button
                onClick={onRetry}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
              >
                <RefreshCw size={16} />
                Riprova Item Falliti
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PipelineRunCard;
