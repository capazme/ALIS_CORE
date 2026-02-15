/**
 * RLCF Service
 * ============
 *
 * Service per interagire con l'API RLCF Training di MERL-T.
 * Fornisce funzioni per training status, start/stop, buffer e policy weights.
 *
 * API Endpoints:
 * - GET /rlcf/training/status - Stato training
 * - POST /rlcf/training/start - Avvia training
 * - POST /rlcf/training/stop - Ferma training
 * - GET /rlcf/buffer/status - Stato buffer feedback
 * - GET /rlcf/policies/weights - Pesi policy
 * - GET /rlcf/policies/history - Storia pesi
 * - WS /rlcf/training/stream - WebSocket metriche
 */

import { get, post } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface TrainingConfig {
  epochs: number;
  learning_rate: number;
  batch_size: number;
  buffer_threshold: number;
}

export interface TrainingStatus {
  is_running: boolean;
  is_paused: boolean;
  current_epoch: number;
  total_epochs: number;
  current_loss: number | null;
  best_loss: number | null;
  learning_rate: number;
  started_at?: string | null;
  eta_seconds?: number | null;
  last_updated?: string | null;
  training_sessions_today?: number;
}

export interface BufferStatus {
  size: number;
  capacity: number;
  fill_percentage: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  training_ready: boolean;
  last_feedback_at?: string | null;
  avg_reward?: number;
}

export interface PolicyWeightsStatus {
  gating: Record<string, number>;
  traversal: Record<string, number>;
  timestamp?: string | null;
}

export interface PolicyWeightsHistory {
  history: PolicyWeightsStatus[];
  epochs: number[];
}

export interface TrainingStartResponse {
  success: boolean;
  training_id: string;
  message: string;
  config?: TrainingConfig;
}

export interface TrainingStopResponse {
  success: boolean;
  epochs_completed: number;
  final_loss: number;
  message: string;
}

export interface TrainingStreamMessage {
  event: 'initial_state' | 'epoch_complete' | 'training_start' | 'training_stop' | 'keepalive';
  data: Record<string, unknown>;
  timestamp?: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Recupera stato corrente del training.
 */
export async function getTrainingStatus(): Promise<TrainingStatus> {
  return get(`${PREFIX}/rlcf/training/status`);
}

/**
 * Avvia training con configurazione.
 */
export async function startTraining(config: TrainingConfig): Promise<TrainingStartResponse> {
  return post(`${PREFIX}/rlcf/training/start`, config);
}

/**
 * Ferma training in corso.
 */
export async function stopTraining(): Promise<TrainingStopResponse> {
  return post(`${PREFIX}/rlcf/training/stop`, {});
}

/**
 * Recupera stato del feedback buffer.
 */
export async function getBufferStatus(): Promise<BufferStatus> {
  return get(`${PREFIX}/rlcf/buffer/status`);
}

/**
 * Recupera pesi correnti delle policy.
 */
export async function getPolicyWeights(): Promise<PolicyWeightsStatus> {
  return get(`${PREFIX}/rlcf/policies/weights`);
}

/**
 * Recupera storia dei pesi.
 */
export async function getPolicyHistory(limit?: number): Promise<PolicyWeightsHistory> {
  const query = limit ? `?limit=${limit}` : '';
  return get(`${PREFIX}/rlcf/policies/history${query}`);
}

// =============================================================================
// WEBSOCKET
// =============================================================================

export type TrainingStreamHandler = {
  onInitialState?: (data: {
    training: TrainingStatus;
    buffer: BufferStatus;
    weights: PolicyWeightsStatus;
  }) => void;
  onEpochComplete?: (data: {
    epoch: number;
    loss: number;
    best_loss: number;
    eta_seconds?: number;
    weights: Record<string, number>;
  }) => void;
  onTrainingStart?: () => void;
  onTrainingStop?: () => void;
  onConnectionClose?: () => void;
  onConnectionError?: (error: Event) => void;
};

/**
 * Costruisce l'URL WebSocket corretto per l'ambiente.
 */
function getWebSocketUrl(path: string): string {
  // In development con Vite proxy, usa il path relativo
  // Il proxy Vite gestisce il forwarding al backend
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}${path}`;
}

/**
 * Connetti al WebSocket per training updates real-time.
 *
 * @param handlers - Callbacks per gestire gli eventi
 * @returns Funzione per chiudere la connessione
 *
 * @example
 * ```ts
 * const close = connectTrainingStream({
 *   onEpochComplete: (data) => console.log(`Epoch ${data.epoch}: loss=${data.loss}`),
 * });
 *
 * // Per chiudere:
 * close();
 * ```
 */
export function connectTrainingStream(handlers: TrainingStreamHandler): () => void {
  const wsUrl = getWebSocketUrl('/api/merlt/rlcf/training/stream');

  let ws: WebSocket | null = null;
  let pingInterval: number | null = null;
  let isClosedIntentionally = false;

  const connect = () => {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[RLCFTrainingWS] Connected');

      // Keep-alive ping every 25 seconds
      pingInterval = window.setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      if (event.data === 'pong') {
        return;
      }

      try {
        const message: TrainingStreamMessage = JSON.parse(event.data);

        switch (message.event) {
          case 'initial_state':
            handlers.onInitialState?.(
              message.data as {
                training: TrainingStatus;
                buffer: BufferStatus;
                weights: PolicyWeightsStatus;
              }
            );
            break;

          case 'epoch_complete':
            handlers.onEpochComplete?.(
              message.data as {
                epoch: number;
                loss: number;
                best_loss: number;
                eta_seconds?: number;
                weights: Record<string, number>;
              }
            );
            break;

          case 'training_start':
            handlers.onTrainingStart?.();
            break;

          case 'training_stop':
            handlers.onTrainingStop?.();
            break;

          case 'keepalive':
            // Ignore keepalive
            break;

          default:
            console.warn('[RLCFTrainingWS] Unknown event:', message.event);
        }
      } catch (e) {
        console.error('[RLCFTrainingWS] Failed to parse message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[RLCFTrainingWS] Error:', error);
      handlers.onConnectionError?.(error);
    };

    ws.onclose = () => {
      console.log('[RLCFTrainingWS] Disconnected');

      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }

      if (!isClosedIntentionally) {
        console.log('[RLCFTrainingWS] Reconnecting in 3 seconds...');
        setTimeout(connect, 3000);
      } else {
        handlers.onConnectionClose?.();
      }
    };
  };

  connect();

  return () => {
    isClosedIntentionally = true;

    if (pingInterval) {
      clearInterval(pingInterval);
      pingInterval = null;
    }

    if (ws) {
      ws.close();
      ws = null;
    }
  };
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Formatta ETA in formato leggibile.
 */
export function formatETA(seconds: number | undefined): string {
  if (seconds === undefined || seconds <= 0) return '-';

  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);

  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds}s`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  return `${hours}h ${remainingMinutes}m`;
}

/**
 * Formatta loss per visualizzazione.
 */
export function formatLoss(loss: number | null | undefined): string {
  if (loss === null || loss === undefined) {
    return '-';
  }
  if (loss >= 1) {
    return loss.toFixed(2);
  }
  if (loss >= 0.01) {
    return loss.toFixed(4);
  }
  return loss.toExponential(2);
}

/**
 * Calcola percentuale buffer fill.
 */
export function getBufferFillPercentage(status: BufferStatus): number {
  if (!status.capacity || status.capacity === 0) {
    return 0;
  }
  return (status.size / status.capacity) * 100;
}

/**
 * Ritorna colore per training status.
 */
export function getTrainingStatusColor(status: TrainingStatus): string {
  if (status.is_running) {
    return 'text-green-500';
  }
  if (status.is_paused) {
    return 'text-yellow-500';
  }
  return 'text-slate-500';
}

/**
 * Ritorna badge per training status.
 */
export function getTrainingStatusBadge(status: TrainingStatus): {
  text: string;
  color: string;
} {
  if (status.is_running) {
    return { text: 'In corso', color: 'bg-green-100 text-green-800' };
  }
  if (status.is_paused) {
    return { text: 'In pausa', color: 'bg-yellow-100 text-yellow-800' };
  }
  if (status.current_epoch > 0) {
    return { text: 'Completato', color: 'bg-blue-100 text-blue-800' };
  }
  return { text: 'Non avviato', color: 'bg-slate-100 text-slate-800' };
}
