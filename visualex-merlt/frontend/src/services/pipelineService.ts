/**
 * Pipeline Monitoring Service
 * ===========================
 *
 * Service per interagire con l'API di Pipeline Monitoring di MERL-T.
 * Fornisce funzioni per REST endpoints e WebSocket real-time.
 *
 * API Endpoints:
 * - GET /pipeline/runs - Lista pipeline run
 * - GET /pipeline/run/{run_id} - Dettagli run
 * - GET /pipeline/run/{run_id}/errors - Errori run
 * - POST /pipeline/run/{run_id}/retry - Retry item falliti
 * - WS /pipeline/ws/{run_id} - Progress real-time
 */

import { get, post } from './api';

const PREFIX = 'merlt';

// =============================================================================
// TYPES
// =============================================================================

export type PipelineType = 'ingestion' | 'enrichment' | 'batch_ingestion';
export type PipelineStatus = 'running' | 'completed' | 'failed' | 'paused';

export interface PipelineRun {
  run_id: string;
  type: PipelineType;
  status: PipelineStatus;
  started_at: string;
  completed_at?: string;
  progress: number;
  summary: Record<string, number>;
  config: Record<string, unknown>;
}

export interface PipelineRunDetails extends PipelineRun {
  total_items: number;
  processed: number;
  errors: number;
}

export interface PipelineError {
  item_id: string;
  phase: string;
  error_message: string;
  stack_trace?: string;
  timestamp: string;
  retryable?: boolean;
}

export interface RetryRequest {
  item_ids?: string[];
}

export interface RetryResponse {
  retried: number;
  message: string;
}

export interface ProgressUpdate {
  run_id: string;
  processed: number;
  total: number;
  progress: number;
  current_item: string;
  speed_per_sec: number;
  eta_seconds: number;
  elapsed_seconds: number;
  step_progress: Record<string, unknown>;
}

export interface WebSocketMessage {
  event: 'initial_state' | 'progress_update' | 'error' | 'warning';
  data: ProgressUpdate | PipelineError | { message: string };
}

// =============================================================================
// START PIPELINE TYPES
// =============================================================================

export interface StartPipelineRequest {
  tipo_atto: string;
  libro?: string;
  articoli?: string[];
  batch_size?: number;
  skip_existing?: boolean;
  with_enrichment?: boolean;
}

export interface StartPipelineResponse {
  success: boolean;
  run_id?: string;
  message: string;
  estimated_items?: number;
}

// =============================================================================
// DATASET TYPES
// =============================================================================

export interface DatasetStats {
  total_nodes: number;
  total_edges: number;
  articles_count: number;
  entities_count: number;
  relations_by_type: Record<string, number>;
  embeddings_count: number;
  bridge_mappings: number;
  last_updated?: string;
  storage_size_mb?: number;
}

export interface DatasetExportRequest {
  format: 'json' | 'csv' | 'cypher';
  include_embeddings?: boolean;
  filter_tipo_atto?: string;
  limit?: number;
}

export interface DatasetExportResponse {
  success: boolean;
  download_url?: string;
  format: string;
  records_count: number;
  file_size_mb?: number;
  message: string;
}

// =============================================================================
// REST API
// =============================================================================

/**
 * Lista pipeline runs con filtri opzionali.
 */
export async function listPipelineRuns(params?: {
  status?: PipelineStatus;
  pipeline_type?: PipelineType;
  limit?: number;
}): Promise<PipelineRun[]> {
  const searchParams = new URLSearchParams();

  if (params?.status) {
    searchParams.set('status', params.status);
  }
  if (params?.pipeline_type) {
    searchParams.set('pipeline_type', params.pipeline_type);
  }
  if (params?.limit) {
    searchParams.set('limit', params.limit.toString());
  }

  const query = searchParams.toString();
  const url = `${PREFIX}/pipeline/runs${query ? `?${query}` : ''}`;

  return get(url);
}

/**
 * Recupera dettagli di una singola pipeline run.
 */
export async function getPipelineRun(runId: string): Promise<PipelineRunDetails> {
  return get(`${PREFIX}/pipeline/run/${encodeURIComponent(runId)}`);
}

/**
 * Recupera lista errori per una pipeline run.
 */
export async function getPipelineErrors(runId: string): Promise<PipelineError[]> {
  return get(`${PREFIX}/pipeline/run/${encodeURIComponent(runId)}/errors`);
}

/**
 * Riprova item falliti per una pipeline run.
 */
export async function retryPipelineItems(
  runId: string,
  itemIds?: string[]
): Promise<RetryResponse> {
  const request: RetryRequest = itemIds ? { item_ids: itemIds } : {};
  return post(`${PREFIX}/pipeline/run/${encodeURIComponent(runId)}/retry`, request);
}

// =============================================================================
// START PIPELINE
// =============================================================================

/**
 * Avvia una nuova pipeline di batch ingestion.
 */
export async function startPipeline(
  request: StartPipelineRequest
): Promise<StartPipelineResponse> {
  return post(`${PREFIX}/pipeline/start`, request);
}

// =============================================================================
// DATASET STATS & EXPORT
// =============================================================================

/**
 * Recupera statistiche del dataset.
 */
export async function getDatasetStats(): Promise<DatasetStats> {
  return get(`${PREFIX}/pipeline/dataset/stats`);
}

/**
 * Esporta il dataset in vari formati.
 */
export async function exportDataset(
  request: DatasetExportRequest
): Promise<DatasetExportResponse> {
  return post(`${PREFIX}/pipeline/dataset/export`, request);
}

/**
 * Costruisce URL di download per un export.
 */
export function getExportDownloadUrl(filename: string): string {
  return `/api/${PREFIX}/pipeline/dataset/download/${encodeURIComponent(filename)}`;
}

// =============================================================================
// WEBSOCKET
// =============================================================================

export type WebSocketEventHandler = {
  onInitialState?: (data: ProgressUpdate) => void;
  onProgress?: (data: ProgressUpdate) => void;
  onError?: (data: PipelineError) => void;
  onWarning?: (message: string) => void;
  onConnectionClose?: () => void;
  onConnectionError?: (error: Event) => void;
};

/**
 * Connetti al WebSocket per ricevere progress updates real-time.
 *
 * @param runId - ID della pipeline run da monitorare
 * @param handlers - Callbacks per gestire gli eventi
 * @returns Funzione per chiudere la connessione
 *
 * @example
 * ```ts
 * const close = connectPipelineWebSocket('batch_123', {
 *   onProgress: (data) => console.log(`Progress: ${data.progress}%`),
 *   onError: (error) => console.error(error.error_message),
 * });
 *
 * // Quando vuoi chiudere:
 * close();
 * ```
 */
export function connectPipelineWebSocket(
  runId: string,
  handlers: WebSocketEventHandler
): () => void {
  // Costruisci URL WebSocket
  // NOTA: Usiamo /api/merlt/ per passare attraverso il proxy Vite
  // che riscrive a /api/v1/ e invia a MERL-T (port 8000)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/api/merlt/pipeline/ws/${encodeURIComponent(runId)}`;

  let ws: WebSocket | null = null;
  let pingInterval: number | null = null;
  let isClosedIntentionally = false;

  const connect = () => {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`[PipelineWS] Connected to ${runId}`);

      // Start keep-alive ping every 30 seconds
      pingInterval = window.setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      // Handle pong
      if (event.data === 'pong') {
        return;
      }

      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.event) {
          case 'initial_state':
            handlers.onInitialState?.(message.data as ProgressUpdate);
            break;

          case 'progress_update':
            handlers.onProgress?.(message.data as ProgressUpdate);
            break;

          case 'error':
            handlers.onError?.(message.data as PipelineError);
            break;

          case 'warning':
            handlers.onWarning?.((message.data as { message: string }).message);
            break;

          default:
            console.warn('[PipelineWS] Unknown event type:', message.event);
        }
      } catch (e) {
        console.error('[PipelineWS] Failed to parse message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[PipelineWS] Error:', error);
      handlers.onConnectionError?.(error);
    };

    ws.onclose = () => {
      console.log(`[PipelineWS] Disconnected from ${runId}`);

      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }

      // Auto-reconnect if not closed intentionally
      if (!isClosedIntentionally) {
        console.log('[PipelineWS] Attempting reconnect in 3 seconds...');
        setTimeout(connect, 3000);
      } else {
        handlers.onConnectionClose?.();
      }
    };
  };

  // Start connection
  connect();

  // Return close function
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

/**
 * Calcola tempo rimanente formattato.
 */
export function formatETA(seconds: number): string {
  if (seconds <= 0) return '-';

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
 * Formatta velocita' di processamento.
 */
export function formatSpeed(itemsPerSecond: number): string {
  if (itemsPerSecond <= 0) return '-';

  if (itemsPerSecond >= 1) {
    return `${itemsPerSecond.toFixed(1)}/s`;
  }

  // Items per minute for slow operations
  const itemsPerMinute = itemsPerSecond * 60;
  return `${itemsPerMinute.toFixed(1)}/min`;
}

/**
 * Formatta durata elapsed.
 */
export function formatDuration(seconds: number): string {
  if (seconds <= 0) return '0s';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}
