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

import { useState, useEffect, useRef } from 'react';
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
} from 'lucide-react';
import { cn } from '../../../../../lib/utils';
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
} from '../../../../../services/rlcfService';

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
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
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
                    'bg-green-600 text-white hover:bg-green-700 disabled:opacity-50'
                  )}
                >
                  <Play size={14} />
                  Start
                </button>
              </>
            ) : (
              <button
                onClick={onStop}
                disabled={loading}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium',
                  'bg-red-600 text-white hover:bg-red-700 disabled:opacity-50'
                )}
              >
                <Square size={14} />
                Stop
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600 dark:text-gray-400">
            Epoch {status.current_epoch}/{status.total_epochs}
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            {progressPercent.toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            initial={{ width: 0 }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-xs mb-1">
            <TrendingDown size={12} />
            Current Loss
          </div>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {formatLoss(status.current_loss)}
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-xs mb-1">
            <CheckCircle2 size={12} />
            Best Loss
          </div>
          <p className="text-lg font-semibold text-green-600 dark:text-green-400">
            {formatLoss(status.best_loss)}
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-xs mb-1">
            <Activity size={12} />
            Learning Rate
          </div>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {status.learning_rate != null ? status.learning_rate.toExponential(1) : '-'}
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-xs mb-1">
            <Clock size={12} />
            ETA
          </div>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
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
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Policy Weights
      </h3>

      {/* Gating Policy */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">
          GatingPolicy (Expert Selection)
        </h4>
        <div className="space-y-3">
          {Object.entries(weights.gating).map(([expert, weight]) => (
            <div key={expert}>
              <div className="flex justify-between text-sm mb-1">
                <span className="capitalize text-gray-700 dark:text-gray-300">
                  {expert}
                </span>
                <span className="text-gray-500 dark:text-gray-400">
                  {(weight * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className={cn('h-full', gatingColors[expert] || 'bg-gray-500')}
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
        <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">
          TraversalPolicy (Graph Navigation)
        </h4>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(weights.traversal).map(([param, value]) => (
            <div
              key={param}
              className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3"
            >
              <p className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                {param.replace(/_/g, ' ')}
              </p>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
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
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
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
          <span className="text-gray-600 dark:text-gray-400">
            {(buffer.size ?? 0).toLocaleString()} / {(buffer.capacity ?? 0).toLocaleString()} experiences
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            {(buffer.fill_percentage ?? 0).toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
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
        <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">
          Distribuzione Feedback
        </h4>
        <div className="flex h-4 rounded-full overflow-hidden">
          <div
            className="bg-green-500"
            style={{ width: `${buffer.size ? ((buffer.positive_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Positive: ${buffer.positive_count ?? 0}`}
          />
          <div
            className="bg-gray-400"
            style={{ width: `${buffer.size ? ((buffer.neutral_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Neutral: ${buffer.neutral_count ?? 0}`}
          />
          <div
            className="bg-red-500"
            style={{ width: `${buffer.size ? ((buffer.negative_count ?? 0) / buffer.size) * 100 : 0}%` }}
            title={`Negative: ${buffer.negative_count ?? 0}`}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full" />
            Positive ({buffer.positive_count ?? 0})
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-gray-400 rounded-full" />
            Neutral ({buffer.neutral_count ?? 0})
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-red-500 rounded-full" />
            Negative ({buffer.negative_count ?? 0})
          </span>
        </div>
      </div>

      {buffer.last_feedback_at && (
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-4">
          Ultimo feedback: {new Date(buffer.last_feedback_at).toLocaleString('it-IT')}
        </p>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function RLCFTab() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [bufferStatus, setBufferStatus] = useState<BufferStatus | null>(null);
  const [policyWeights, setPolicyWeights] = useState<PolicyWeightsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const disconnectRef = useRef<(() => void) | null>(null);

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
        setTrainingStatus((prev) =>
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
        setTrainingStatus((prev) => (prev ? { ...prev, is_running: true } : prev));
      },
      onTrainingStop: () => {
        setTrainingStatus((prev) => (prev ? { ...prev, is_running: false } : prev));
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
      <div className="grid md:grid-cols-2 gap-6">
        {policyWeights && <PolicyWeightsCard weights={policyWeights} />}
        {bufferStatus && <BufferStatusCard buffer={bufferStatus} />}
      </div>

      {/* Info text */}
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
        Il training aggiorna automaticamente i pesi delle policy in base al feedback collettivo
      </p>
    </div>
  );
}

export default RLCFTab;
