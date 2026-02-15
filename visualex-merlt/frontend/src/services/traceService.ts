/**
 * Trace Service - API calls for trace data, sources, and validity.
 *
 * Includes transformation layer to map backend response shapes
 * to the frontend TypeScript types expected by TraceViewer components.
 */

import { get } from './api';
import type {
  TraceDetail,
  ExpertTraceEntry,
  SourceResolution,
  TraceSourcesResponse,
  TraceValidityResponse,
  ValiditySummary,
  ValidityResult,
  ValidityStatus,
} from '../types/trace';
import type { ExpertId } from '../types/pipeline';

const PREFIX = '/merlt';

// ============================================================================
// RAW BACKEND TYPES (internal — not exported)
// ============================================================================

interface RawTrace {
  trace_id: string;
  user_id: string;
  query: string;
  selected_experts: string[];
  synthesis_mode: string;
  synthesis_text: string;
  sources: RawTraceSource[];
  execution_time_ms: number;
  full_trace: RawFullTrace | null;
  consent_level: string;
  query_type: string | null;
  confidence: number | null;
  routing_method: string | null;
  is_archived: boolean;
  archived_at: string | null;
  created_at: string;
}

interface RawTraceSource {
  article_urn: string;
  expert: string;
  relevance: number;
  excerpt: string | null;
}

interface RawFullTrace {
  trace_id: string;
  timestamp: string;
  query_text: string;
  total_time_ms: number;
  total_tokens: number;
  stage_times_ms: Record<string, number>;
  stages: {
    ner?: { entities: Array<{ text: string; type: string }>; query_type: string; time_ms: number };
    gating?: { weights: Record<string, number>; source: string };
    routing?: { method: string; selected_experts: string[]; gating_scores: Record<string, number>; time_ms: number };
    synthesis?: { mode: string; confidence: number; time_ms: number; disagreement_analysis?: Record<string, unknown> };
    expert_executions?: RawExpertExecution[];
  };
}

interface RawExpertExecution {
  expert_type: string;
  success: boolean;
  skipped: boolean;
  error: string | null;
  confidence: number;
  duration_ms: number;
  tokens_used: number;
  started_at: string;
  completed_at: string;
  output: {
    interpretation_preview: string;
    sources_count: number;
  } | null;
  llm_calls: Array<{
    model: string;
    tokens_input: number;
    tokens_output: number;
    duration_ms: number;
    response_summary: string;
  }>;
}

interface RawSourceResolution {
  chunk_id: string;
  graph_node_urn: string;
  node_type: string;
  chunk_text: string | null;
  confidence: number | null;
}

interface RawValiditySummary {
  trace_id: string;
  as_of_date: string | null;
  total_sources: number;
  valid_count: number;
  warning_count: number;
  critical_count: number;
  unknown_count: number;
  results: RawValidityResult[];
  summary_message: string;
}

interface RawValidityResult {
  urn: string;
  status: string;
  effective_date: string | null;
  expiry_date: string | null;
  modified_by: string | null;
  notes: string | null;
}

// ============================================================================
// TRANSFORMERS
// ============================================================================

const EXPERT_DISPLAY_NAMES: Record<string, string> = {
  literal: 'Letterale',
  systemic: 'Sistematico',
  principles: 'Teleologico',
  precedent: 'Giurisprudenziale',
};

/** Shared regex for article ordinal suffixes in URN (bis, ter, quater, etc.) */
const ART_SUFFIXES = 'bis|ter|quater|quinquies|sexies|septies|octies|novies|decies';

function transformExpertExecutions(raw: RawFullTrace | null, selectedExperts: string[]): ExpertTraceEntry[] {
  const executions = raw?.stages?.expert_executions;
  if (!executions || executions.length === 0) {
    // Fallback: create minimal entries from selected_experts
    const gatingWeights = raw?.stages?.gating?.weights ?? {};
    return selectedExperts.map(id => ({
      expertId: id as ExpertId,
      displayName: EXPERT_DISPLAY_NAMES[id] ?? id,
      interpretation: '',
      confidence: 0,
      weight: gatingWeights[id] ?? 0,
      sources: [],
    }));
  }

  return executions
    .filter(e => e.success && !e.skipped)
    .map(e => {
      const gatingWeights = raw?.stages?.gating?.weights ?? {};
      // Try to extract full interpretation from LLM response JSON.
      // The regex handles escaped chars inside the JSON string value.
      // If it fails (e.g. malformed JSON), we fall back to interpretation_preview.
      let interpretation = e.output?.interpretation_preview ?? '';
      if (e.llm_calls?.length > 0) {
        const resp = e.llm_calls[0].response_summary;
        const match = resp?.match(/"interpretation"\s*:\s*"((?:[^"\\]|\\.)*)"/);
        if (match) {
          try {
            interpretation = JSON.parse(`"${match[1]}"`);
          } catch {
            // keep preview
          }
        }
      }

      return {
        expertId: e.expert_type as ExpertId,
        displayName: EXPERT_DISPLAY_NAMES[e.expert_type] ?? e.expert_type,
        interpretation,
        confidence: e.confidence,
        weight: gatingWeights[e.expert_type] ?? 0,
        sources: [],
        reasoning: e.output?.interpretation_preview,
        duration_ms: e.duration_ms,
      } satisfies ExpertTraceEntry;
    });
}

function transformTrace(raw: RawTrace): TraceDetail {
  return {
    id: raw.trace_id,
    query: raw.query,
    timestamp: raw.created_at,
    synthesis: raw.synthesis_text,
    confidence: raw.confidence ?? 0,
    experts: transformExpertExecutions(raw.full_trace, raw.selected_experts),
    metadata: raw.full_trace ? {
      synthesis_mode: raw.synthesis_mode,
      routing_method: raw.routing_method,
      execution_time_ms: raw.execution_time_ms,
      total_tokens: raw.full_trace.total_tokens,
      disagreement_analysis: raw.full_trace.stages?.synthesis?.disagreement_analysis,
    } : undefined,
  };
}

function parseUrnMetadata(urn: string): SourceResolution['metadata'] {
  // Parse URN like "urn:nir:stato:regio.decreto:1942-03-16;262:2~art2043"
  const parts = urn.split(':');
  if (parts.length < 4) return undefined;
  const tipoAtto = parts[3]?.replace(/\./g, ' ');
  const artMatch = urn.match(new RegExp(`~art(\\d+(?:${ART_SUFFIXES})?)`, 'i'));
  const numAtto = urn.match(/;(\d+)/)?.[1];
  return {
    tipo_atto: tipoAtto,
    numero_articolo: artMatch?.[1],
    numero_atto: numAtto,
  };
}

/**
 * Match bridge-resolved sources to trace inline sources by URN to recover expert attribution.
 * Returns the expert id from the inline source that matches the URN, or 'literal' as default.
 */
function findExpertForUrn(urn: string, traceSources: RawTraceSource[]): ExpertId {
  const match = traceSources.find(s => s.article_urn === urn || urn.includes(s.article_urn));
  if (match) {
    return (match.expert === 'combined' ? 'literal' : match.expert) as ExpertId;
  }
  return 'literal' as ExpertId;
}

function transformSources(rawSources: RawSourceResolution[], traceSources: RawTraceSource[]): SourceResolution[] {
  if (rawSources.length > 0) {
    return rawSources.map(s => ({
      sourceId: s.chunk_id,
      urn: s.graph_node_urn,
      label: s.node_type,
      chunkText: s.chunk_text ?? '',
      score: s.confidence ?? 0,
      expertId: findExpertForUrn(s.graph_node_urn, traceSources),
      metadata: parseUrnMetadata(s.graph_node_urn),
    }));
  }

  // Fallback: use inline sources from the trace itself
  return traceSources.map((s, idx) => ({
    sourceId: `source-${idx}`,
    urn: s.article_urn,
    label: extractLabelFromUrn(s.article_urn),
    chunkText: s.excerpt ?? '',
    score: s.relevance,
    expertId: (s.expert === 'combined' ? 'literal' : s.expert) as ExpertId,
    metadata: parseUrnMetadata(s.article_urn),
  }));
}

function extractLabelFromUrn(urn: string): string {
  // "urn:nir:stato:regio.decreto:1942-03-16;262:2~art2043" → "Art. 2043 r.d. 262/1942"
  const artMatch = urn.match(new RegExp(`~art(\\d+(?:${ART_SUFFIXES})?)`, 'i'));
  const numMatch = urn.match(/;(\d+)/);
  const tipoMatch = urn.match(/:([\w.]+):\d{4}/);
  if (artMatch) {
    const parts = [];
    parts.push(`Art. ${artMatch[1]}`);
    if (tipoMatch) {
      const tipo = tipoMatch[1].replace('regio.decreto', 'r.d.').replace('codice.civile', 'c.c.');
      parts.push(tipo);
    }
    if (numMatch) parts.push(numMatch[1]);
    return parts.join(' ');
  }
  return urn.split('/').pop() ?? urn;
}

function mapBackendStatus(status: string): ValidityStatus {
  switch (status) {
    case 'valid':
    case 'vigente':
      return 'vigente';
    case 'warning':
    case 'modificato':
      return 'modificato';
    case 'critical':
    case 'abrogato':
      return 'abrogato';
    default:
      return 'unknown';
  }
}

function transformValidity(raw: RawValiditySummary): ValiditySummary {
  return {
    total: raw.total_sources,
    vigente: raw.valid_count,
    abrogato: raw.critical_count,
    modificato: raw.warning_count,
    unknown: raw.unknown_count,
    results: raw.results.map(r => ({
      urn: r.urn,
      status: mapBackendStatus(r.status),
      effectiveDate: r.effective_date ?? undefined,
      expiryDate: r.expiry_date ?? undefined,
      modifiedBy: r.modified_by ?? undefined,
      notes: r.notes ?? undefined,
    } satisfies ValidityResult)),
  };
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Get full trace detail by ID — transforms backend response to TraceDetail.
 *
 * Also returns raw inline sources for use by getTraceSources (avoids double fetch).
 */
async function fetchRawTrace(traceId: string): Promise<RawTrace> {
  // caller_consent=full is correct here: the trace viewer is only shown after
  // the user has already submitted a query (implying consent to view their own data).
  return get<RawTrace>(`${PREFIX}/traces/${encodeURIComponent(traceId)}?caller_consent=full`);
}

export async function getTrace(traceId: string): Promise<TraceDetail> {
  const raw = await fetchRawTrace(traceId);
  return transformTrace(raw);
}

/**
 * Get resolved sources for a trace — transforms to SourceResolution[].
 *
 * Accepts optional pre-fetched inline sources to avoid a redundant trace fetch.
 * When called from useTraceData, the trace is already fetched in parallel.
 */
export async function getTraceSources(
  traceId: string,
  inlineSources?: RawTraceSource[],
): Promise<TraceSourcesResponse> {
  const raw = await get<{ trace_id: string; sources: RawSourceResolution[] }>(
    `${PREFIX}/traces/${encodeURIComponent(traceId)}/sources`
  );

  // Use provided inline sources, or fetch trace as fallback
  let traceSources: RawTraceSource[] = inlineSources ?? [];
  if (!inlineSources) {
    try {
      const trace = await fetchRawTrace(traceId);
      traceSources = trace.sources ?? [];
    } catch {
      // Fallback: no inline sources
    }
  }

  return {
    traceId: raw.trace_id,
    sources: transformSources(raw.sources, traceSources),
  };
}

/**
 * Get validity information for sources in a trace — transforms to ValiditySummary.
 */
export async function getTraceValidity(traceId: string): Promise<TraceValidityResponse> {
  const raw = await get<RawValiditySummary>(
    `${PREFIX}/traces/${encodeURIComponent(traceId)}/validity`
  );

  return {
    traceId: raw.trace_id,
    validity: transformValidity(raw),
  };
}

/**
 * Fetch trace + sources in a single coordinated call.
 * Avoids the double-fetch problem by sharing inline sources from the trace response.
 */
export async function getTraceWithSources(traceId: string): Promise<{
  trace: TraceDetail;
  sourcesResponse: TraceSourcesResponse;
}> {
  const rawTrace = await fetchRawTrace(traceId);
  const trace = transformTrace(rawTrace);
  const sourcesResponse = await getTraceSources(traceId, rawTrace.sources ?? []);
  return { trace, sourcesResponse };
}
