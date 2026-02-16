/**
 * RLCFTab
 * =======
 *
 * Tab per monitoraggio RLCF Training con:
 * - Training status e progress
 * - Loss curve chart
 * - Policy weights visualization
 * - Feedback buffer status
 * - Start/Stop controls
 *
 * Supporta WebSocket per aggiornamenti real-time.
 *
 * @example
 * ```tsx
 * <RLCFTab />
 * ```
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Play,
  Square,
  RefreshCw,
  AlertCircle,
  TrendingDown,
  Clock,
  Activity,
  CheckCircle2,
  Shield,
  Flag,
  Check,
  Ban,
  Search,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  getTrainingStatus,
  startTraining,
  stopTraining,
  getBufferStatus,
  getPolicyWeights,
  connectTrainingStream,
  formatETA,
  formatLoss,
  getTrainingStatusBadge,
  type TrainingStatus,
  type BufferStatus,
  type PolicyWeightsStatus,
  type TrainingConfig,
} from '../../../../services/rlcfService';
import {
  getFlaggedFeedback,
  getQuarantinedFeedback,
  approveFeedback,
  quarantineFeedback,
  autoDetectOutliers,
} from '../../../../services/quarantineService';
import type { FeedbackItem } from '../../../../services/quarantineService';

// =============================================================================
// TRAINING STATUS CARD
// =============================================================================

interface TrainingStatusCardProps {
  status: TrainingStatus;
  onStart: () => void;
  onStop: () => void;
  loading: boolean;
}

function TrainingStatusCard({
  status,
  onStart,
  onStop,
  loading,
}: TrainingStatusCardProps) {
  const badge = getTrainingStatusBadge(status);
  const progressPercent = status.total_epochs > 0
    ? (status.current_epoch / status.total_epochs) * 100
    : 0;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Training Status
        </h3>
        <div className="flex items-center gap-2">
          <span className={cn('px-3 py-1 rounded-full text-sm font-medium', badge.color)}>
            {badge.text}
          </span>
          <div className="flex gap-2">
            {!status.is_running ? (
              <>
                <button
                  onClick={onStart}
                  disabled={loading}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium',
                    'bg-green-600 text-white hover:bg-green-700 disabled:opacity-50',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2'
                  )}
                >
                  <Play size={14} aria-hidden="true" />
                  Start
                </button>
              </>
            ) : (
              <button
                onClick={onStop}
                disabled={loading}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium',
                  'bg-red-600 text-white hover:bg-red-700 disabled:opacity-50',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2'
                )}
              >
                <Square size={14} aria-hidden="true" />
                Stop
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-600 dark:text-slate-400">
            Epoch {status.current_epoch}/{status.total_epochs}
          </span>
          <span className="text-slate-600 dark:text-slate-400">
            {progressPercent.toFixed(1)}%
          </span>
        </div>
        <div
          className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"
          role="progressbar"
          aria-valuenow={Math.round(progressPercent)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Training progress: ${progressPercent.toFixed(1)}%`}
        >
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            initial={{ width: 0 }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs mb-1">
            <TrendingDown size={12} aria-hidden="true" />
            Current Loss
          </div>
          <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            {formatLoss(status.current_loss)}
          </p>
        </div>

        <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs mb-1">
            <CheckCircle2 size={12} aria-hidden="true" />
            Best Loss
          </div>
          <p className="text-lg font-semibold text-green-600 dark:text-green-400">
            {formatLoss(status.best_loss)}
          </p>
        </div>

        <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs mb-1">
            <Activity size={12} aria-hidden="true" />
            Learning Rate
          </div>
          <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            {status.learning_rate != null ? status.learning_rate.toExponential(1) : '-'}
          </p>
        </div>

        <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs mb-1">
            <Clock size={12} aria-hidden="true" />
            ETA
          </div>
          <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            {formatETA(status.eta_seconds ?? undefined)}
          </p>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// POLICY WEIGHTS CARD
// =============================================================================

interface PolicyWeightsCardProps {
  weights: PolicyWeightsStatus;
}

function PolicyWeightsCard({ weights }: PolicyWeightsCardProps) {
  const gatingColors: Record<string, string> = {
    literal: 'bg-blue-500',
    systemic: 'bg-green-500',
    principles: 'bg-yellow-500',
    precedent: 'bg-orange-500',
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
        Policy Weights
      </h3>

      {/* Gating Policy */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">
          GatingPolicy (Expert Selection)
        </h4>
        <div className="space-y-3">
          {Object.entries(weights.gating).map(([expert, weight]) => (
            <div key={expert}>
              <div className="flex justify-between text-sm mb-1">
                <span className="capitalize text-slate-700 dark:text-slate-300">
                  {expert}
                </span>
                <span className="text-slate-500 dark:text-slate-400">
                  {(weight * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className={cn('h-full', gatingColors[expert] || 'bg-slate-500')}
                  initial={{ width: 0 }}
                  animate={{ width: `${weight * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Traversal Policy */}
      <div>
        <h4 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">
          TraversalPolicy (Graph Navigation)
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.entries(weights.traversal).map(([param, value]) => (
            <div
              key={param}
              className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3"
            >
              <p className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                {param.replace(/_/g, ' ')}
              </p>
              <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {typeof value === 'number' ? value.toFixed(2) : value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// BUFFER STATUS CARD
// =============================================================================

interface BufferStatusCardProps {
  buffer: BufferStatus;
}

function BufferStatusCard({ buffer }: BufferStatusCardProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Feedback Buffer
        </h3>
        <span
          className={cn(
            'px-3 py-1 rounded-full text-sm font-medium',
            buffer.training_ready
              ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
          )}
        >
          {buffer.training_ready ? 'Ready' : 'Accumulating'}
        </span>
      </div>

      {/* Buffer fill */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-600 dark:text-slate-400">
            {(buffer.size ?? 0).toLocaleString()} / {(buffer.capacity ?? 0).toLocaleString()} experiences
          </span>
          <span className="text-slate-600 dark:text-slate-400">
            {(buffer.fill_percentage ?? 0).toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            className={cn(
              'h-full',
              (buffer.fill_percentage ?? 0) >= 80
                ? 'bg-green-500'
                : (buffer.fill_percentage ?? 0) >= 50
                  ? 'bg-yellow-500'
                  : 'bg-blue-500'
            )}
            initial={{ width: 0 }}
            animate={{ width: `${buffer.fill_percentage ?? 0}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Distribution */}
      <div>
        <h4 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">
          Distribuzione Feedback
        </h4>
        <div className="flex h-4 rounded-full overflow-hidden">
          <div
            className="bg-green-500"
            style={{ width: `${buffer.size ? ((buffer.positive_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Positive: ${buffer.positive_count ?? 0}`}
          />
          <div
            className="bg-slate-400"
            style={{ width: `${buffer.size ? ((buffer.neutral_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Neutral: ${buffer.neutral_count ?? 0}`}
          />
          <div
            className="bg-red-500"
            style={{ width: `${buffer.size ? ((buffer.negative_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Negative: ${buffer.negative_count ?? 0}`}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full" />
            Positive ({buffer.positive_count ?? 0})
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-slate-400 rounded-full" />
            Neutral ({buffer.neutral_count ?? 0})
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-red-500 rounded-full" />
            Negative ({buffer.negative_count ?? 0})
          </span>
        </div>
      </div>

      {buffer.last_feedback_at && (
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-4">
          Ultimo feedback: {new Date(buffer.last_feedback_at).toLocaleString('it-IT')}
        </p>
      )}
    </div>
  );
}

// =============================================================================
// FEEDBACK REVIEW SECTION
// =============================================================================

function FeedbackReviewSection() {
  const [activeView, setActiveView] = useState('flagged' as 'flagged' | 'quarantined');
  const [items, setItems] = useState([] as FeedbackItem[]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [detectLoading, setDetectLoading] = useState(false);
  const [error, setError] = useState(null as string | null);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = activeView === 'flagged'
        ? await getFlaggedFeedback(50, 0)
        : await getQuarantinedFeedback(50, 0);
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError('Errore caricamento feedback');
      console.error('Failed to load feedback:', err);
    } finally {
      setLoading(false);
    }
  }, [activeView]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  const handleApprove = async (id: number) => {
    try {
      await approveFeedback(id);
      await fetchItems();
    } catch (err) {
      console.error('Failed to approve:', err);
    }
  };

  const handleQuarantine = async (id: number) => {
    try {
      await quarantineFeedback(id, 'Quarantined by admin');
      await fetchItems();
    } catch (err) {
      console.error('Failed to quarantine:', err);
    }
  };

  const handleAutoDetect = async () => {
    setDetectLoading(true);
    try {
      const result = await autoDetectOutliers();
      if (result.flagged_count > 0) {
        await fetchItems();
      }
    } catch (err) {
      console.error('Failed to auto-detect:', err);
    } finally {
      setDetectLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield size={20} className="text-slate-500" aria-hidden="true" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Feedback Review
          </h3>
          <span className="text-sm text-slate-500 dark:text-slate-400">({total})</span>
        </div>
        <button
          onClick={handleAutoDetect}
          disabled={detectLoading}
          className={cn(
            'flex items-center gap-1 px-3 py-1.5 text-sm rounded-md transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
            detectLoading
              ? 'bg-slate-100 dark:bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-orange-600 text-white hover:bg-orange-700'
          )}
        >
          <Search size={14} aria-hidden="true" />
          {detectLoading ? 'Detecting...' : 'Auto-Detect'}
        </button>
      </div>

      {/* Tab switcher */}
      <div className="flex border-b border-slate-200 dark:border-slate-700">
        <button
          onClick={() => setActiveView('flagged')}
          className={cn(
            'flex-1 px-4 py-2 text-sm font-medium border-b-2 transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
            activeView === 'flagged'
              ? 'border-orange-500 text-orange-700 dark:text-orange-400'
              : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
          )}
        >
          <Flag size={14} className="inline mr-1" aria-hidden="true" />
          Flagged
        </button>
        <button
          onClick={() => setActiveView('quarantined')}
          className={cn(
            'flex-1 px-4 py-2 text-sm font-medium border-b-2 transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
            activeView === 'quarantined'
              ? 'border-red-500 text-red-700 dark:text-red-400'
              : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
          )}
        >
          <Ban size={14} className="inline mr-1" aria-hidden="true" />
          Quarantined
        </button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8" role="status">
          <RefreshCw size={20} className="animate-spin text-blue-500" aria-hidden="true" />
          <span className="sr-only">Caricamento...</span>
        </div>
      )}

      {error && (
        <div className="p-4 text-center text-red-500 text-sm">{error}</div>
      )}

      {!loading && !error && items.length === 0 && (
        <div className="p-8 text-center text-slate-400 dark:text-slate-500">
          <p className="text-sm">
            {activeView === 'flagged' ? 'Nessun feedback flaggato.' : 'Nessun feedback quarantinato.'}
          </p>
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="divide-y divide-slate-200 dark:divide-slate-700">
          {items.map((item: FeedbackItem) => (
            <div key={item.id} className="p-4 flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
                    #{item.id}
                  </span>
                  <span className="text-xs text-slate-500">trace: {item.trace_id}</span>
                  {item.inline_rating != null && (
                    <span className={cn(
                      'px-2 py-0.5 rounded text-xs',
                      item.inline_rating >= 4
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                        : item.inline_rating <= 2
                          ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                          : 'bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-300'
                    )}>
                      Rating: {item.inline_rating}
                    </span>
                  )}
                  {item.user_authority != null && (
                    <span className="text-xs text-slate-400">
                      Authority: {item.user_authority.toFixed(2)}
                    </span>
                  )}
                </div>
                {item.quarantine_reason && (
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">
                    {item.quarantine_reason}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-1 ml-4">
                {activeView === 'flagged' && (
                  <>
                    <button
                      onClick={() => handleApprove(item.id)}
                      className="p-1.5 text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500"
                      title="Approva"
                      aria-label="Approva"
                    >
                      <Check size={16} aria-hidden="true" />
                    </button>
                    <button
                      onClick={() => handleQuarantine(item.id)}
                      className="p-1.5 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                      title="Quarantina"
                      aria-label="Quarantina"
                    >
                      <Ban size={16} aria-hidden="true" />
                    </button>
                  </>
                )}
                {activeView === 'quarantined' && (
                  <button
                    onClick={() => handleApprove(item.id)}
                    className="p-1.5 text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500"
                    title="Riabilita"
                    aria-label="Riabilita"
                  >
                    <Check size={16} aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function RLCFTab() {
  const [trainingStatus, setTrainingStatus] = useState(null as TrainingStatus | null);
  const [bufferStatus, setBufferStatus] = useState(null as BufferStatus | null);
  const [policyWeights, setPolicyWeights] = useState(null as PolicyWeightsStatus | null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState(null as string | null);
  const disconnectRef = useRef(null as (() => void) | null);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [training, buffer, weights] = await Promise.all([
        getTrainingStatus(),
        getBufferStatus(),
        getPolicyWeights(),
      ]);
      setTrainingStatus(training);
      setBufferStatus(buffer);
      setPolicyWeights(weights);
    } catch (err) {
      setError('Errore nel caricamento dei dati RLCF');
      console.error('Failed to load RLCF data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    disconnectRef.current = connectTrainingStream({
      onInitialState: (data) => {
        setTrainingStatus(data.training);
        setBufferStatus(data.buffer);
        setPolicyWeights(data.weights);
      },
      onEpochComplete: (data) => {
        setTrainingStatus((prev: TrainingStatus | null) =>
          prev
            ? {
                ...prev,
                current_epoch: data.epoch,
                current_loss: data.loss,
                best_loss: data.best_loss,
                eta_seconds: data.eta_seconds,
              }
            : prev
        );
      },
      onTrainingStart: () => {
        setTrainingStatus((prev: TrainingStatus | null) => (prev ? { ...prev, is_running: true } : prev));
      },
      onTrainingStop: () => {
        setTrainingStatus((prev: TrainingStatus | null) => (prev ? { ...prev, is_running: false } : prev));
      },
    });

    fetchAllData();

    return () => {
      if (disconnectRef.current) {
        disconnectRef.current();
      }
    };
  }, []);

  const handleStartTraining = async () => {
    setActionLoading(true);
    try {
      const config: TrainingConfig = {
        epochs: 50,
        learning_rate: 0.001,
        batch_size: 32,
        buffer_threshold: 500,
      };
      await startTraining(config);
      await fetchAllData();
    } catch (err) {
      console.error('Failed to start training:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setActionLoading(true);
    try {
      await stopTraining();
      await fetchAllData();
    } catch (err) {
      console.error('Failed to stop training:', err);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12" role="status">
        <RefreshCw size={24} className="animate-spin text-blue-500" aria-hidden="true" />
        <span className="sr-only">Caricamento dati RLCF in corso...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12" role="alert">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" aria-hidden="true" />
        <p className="text-slate-500 dark:text-slate-400">{error}</p>
        <button
          onClick={fetchAllData}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Training Status */}
      {trainingStatus && (
        <TrainingStatusCard
          status={trainingStatus}
          onStart={handleStartTraining}
          onStop={handleStopTraining}
          loading={actionLoading}
        />
      )}

      {/* Policy Weights + Buffer Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {policyWeights && <PolicyWeightsCard weights={policyWeights} />}
        {bufferStatus && <BufferStatusCard buffer={bufferStatus} />}
      </div>

      {/* Feedback Review */}
      <FeedbackReviewSection />

      {/* Info text */}
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center">
        Il training aggiorna automaticamente i pesi delle policy in base al feedback collettivo
      </p>
    </div>
  );
}

export default RLCFTab;
