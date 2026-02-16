/**
 * Circuit Breaker Service
 * =======================
 *
 * Service per gestione circuit breaker del sistema Expert via API MERL-T.
 *
 * Endpoints:
 * - GET /circuit-breaker/status — stato tutti i breaker
 * - GET /circuit-breaker/{expert_type} — stato singolo breaker
 * - PUT /circuit-breaker/{expert_type}/config — aggiorna config
 * - POST /circuit-breaker/{expert_type}/reset — reset manuale
 */

import { get, put, post } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface CircuitBreakerStatus {
  name: string;
  state: 'closed' | 'open' | 'half_open';
  failure_count: number;
  success_count: number;
  last_failure_time: string | null;
  last_success_time: string | null;
  times_opened: number;
  total_failures: number;
  total_successes: number;
  state_changed_at: string | null;
}

export interface CircuitBreakerStatusResponse {
  breakers: Record<string, CircuitBreakerStatus>;
  total_count: number;
  open_count: number;
}

export interface CircuitBreakerConfigUpdate {
  failure_threshold?: number;
  recovery_timeout_seconds?: number;
}

export interface CircuitBreakerResetResponse {
  name: string;
  previous_state: string;
  current_state: string;
  message: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function getCircuitBreakerStatus(): Promise<CircuitBreakerStatusResponse> {
  return get(`${PREFIX}/circuit-breaker/status`);
}

export async function getBreakerStatus(expertType: string): Promise<CircuitBreakerStatus> {
  return get(`${PREFIX}/circuit-breaker/${encodeURIComponent(expertType)}`);
}

export async function updateBreakerConfig(
  expertType: string,
  config: CircuitBreakerConfigUpdate
): Promise<{ expert_type: string; updated: Record<string, unknown> }> {
  return put(`${PREFIX}/circuit-breaker/${encodeURIComponent(expertType)}/config`, config);
}

export async function resetBreaker(
  expertType: string
): Promise<CircuitBreakerResetResponse> {
  return post(`${PREFIX}/circuit-breaker/${encodeURIComponent(expertType)}/reset`);
}
