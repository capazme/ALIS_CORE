/**
 * Schedule Service
 * ================
 *
 * Service per gestione schedule di ingestion automatica via API MERL-T.
 *
 * Endpoints:
 * - GET /ingestion/schedules — lista schedule
 * - POST /ingestion/schedules — crea schedule
 * - PUT /ingestion/schedules/{id} — modifica schedule
 * - DELETE /ingestion/schedules/{id} — elimina schedule
 * - POST /ingestion/schedules/{id}/toggle — pausa/riprendi
 */

import { get, post, put, del } from './api';

const PREFIX = '/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface IngestionSchedule {
  id: number;
  tipo_atto: string;
  cron_expr: string;
  enabled: boolean;
  description: string | null;
  last_run_at: string | null;
  last_run_status: string | null;
  next_run_at: string | null;
  created_at: string | null;
}

export interface ScheduleListResponse {
  schedules: IngestionSchedule[];
  count: number;
}

export interface ScheduleCreateRequest {
  tipo_atto: string;
  cron_expr: string;
  enabled?: boolean;
  description?: string;
}

export interface ScheduleUpdateRequest {
  tipo_atto?: string;
  cron_expr?: string;
  enabled?: boolean;
  description?: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export async function getSchedules(): Promise<ScheduleListResponse> {
  return get(`${PREFIX}/ingestion/schedules`);
}

export async function createSchedule(
  data: ScheduleCreateRequest
): Promise<IngestionSchedule> {
  return post(`${PREFIX}/ingestion/schedules`, data);
}

export async function updateSchedule(
  scheduleId: number,
  data: ScheduleUpdateRequest
): Promise<IngestionSchedule> {
  return put(`${PREFIX}/ingestion/schedules/${scheduleId}`, data);
}

export async function deleteSchedule(scheduleId: number): Promise<{ message: string }> {
  return del(`${PREFIX}/ingestion/schedules/${scheduleId}`);
}

export async function toggleSchedule(scheduleId: number): Promise<IngestionSchedule> {
  return post(`${PREFIX}/ingestion/schedules/${scheduleId}/toggle`);
}
