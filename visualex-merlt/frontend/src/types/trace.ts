/**
 * Trace Types - Types for trace viewer, source resolution, and validity.
 */

import type { ExpertId } from './pipeline';

export interface TraceDetail {
  id: string;
  query: string;
  timestamp: string;
  synthesis: string;
  confidence: number;
  experts: ExpertTraceEntry[];
  metadata?: Record<string, unknown>;
}

export interface ExpertTraceEntry {
  expertId: ExpertId;
  displayName: string;
  interpretation: string;
  confidence: number;
  weight: number;
  sources: string[];
  searchQuery?: string;
  reasoning?: string;
  duration_ms?: number;
}

export interface SourceResolution {
  sourceId: string;
  urn: string;
  label: string;
  chunkText: string;
  score: number;
  expertId: ExpertId;
  metadata?: {
    tipo_atto?: string;
    numero_articolo?: string;
    numero_atto?: string;
    data?: string;
  };
}

export type ValidityStatus = 'vigente' | 'abrogato' | 'modificato' | 'unknown';

export interface ValidityResult {
  urn: string;
  status: ValidityStatus;
  effectiveDate?: string;
  expiryDate?: string;
  modifiedBy?: string;
  notes?: string;
}

export interface ValiditySummary {
  total: number;
  vigente: number;
  abrogato: number;
  modificato: number;
  unknown: number;
  results: ValidityResult[];
}

export interface TraceSourcesResponse {
  traceId: string;
  sources: SourceResolution[];
}

export interface TraceValidityResponse {
  traceId: string;
  validity: ValiditySummary;
}
