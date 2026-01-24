/**
 * Dashboard Service
 * =================
 *
 * Service per interagire con l'API Dashboard di MERL-T.
 * Fornisce funzioni per recuperare overview, health, architecture.
 *
 * API Endpoints:
 * - GET /dashboard/overview - KPIs aggregati
 * - GET /dashboard/health - Health check servizi
 * - GET /dashboard/architecture - Dati per diagramma
 * - GET /dashboard/architecture/node/{node_id} - Dettagli nodo
 * - GET /dashboard/activity - Activity feed
 */

import { get } from './api';

const PREFIX = 'merlt';

// =============================================================================
// TYPES
// =============================================================================

export type ServiceStatus = 'online' | 'offline' | 'degraded' | 'unknown';

export type ActivityType =
  | 'pipeline_start'
  | 'pipeline_complete'
  | 'pipeline_error'
  | 'training_start'
  | 'training_epoch'
  | 'training_complete'
  | 'expert_query'
  | 'feedback_received'
  | 'feedback_aggregated';

export interface KnowledgeGraphKPIs {
  total_nodes: number;
  total_edges: number;
  articles_count: number;
  entities_count: number;
  embeddings_count: number;
  bridge_mappings: number;
}

export interface RLCFKPIs {
  total_feedback: number;
  buffer_size: number;
  training_epochs: number;
  avg_authority: number;
  active_users: number;
}

export interface ExpertKPIs {
  total_queries: number;
  avg_latency_ms: number;
  avg_confidence: number;
  agreement_rate: number;
}

export interface ServiceHealth {
  name: string;
  status: ServiceStatus;
  latency_ms?: number;
  details: Record<string, unknown>;
  last_check: string;
}

export interface SystemHealth {
  overall_status: ServiceStatus;
  services: ServiceHealth[];
  uptime_seconds: number;
  last_check: string;
}

export interface ActivityEntry {
  id: string;
  type: ActivityType;
  message: string;
  details: Record<string, unknown>;
  timestamp: string;
  severity: 'info' | 'warning' | 'error';
}

export interface ActivityFeed {
  entries: ActivityEntry[];
  total_count: number;
  has_more: boolean;
}

export interface DashboardOverview {
  knowledge_graph: KnowledgeGraphKPIs;
  rlcf: RLCFKPIs;
  experts: ExpertKPIs;
  health: SystemHealth;
  recent_activity: ActivityFeed;
  last_updated: string;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  type: 'source' | 'pipeline' | 'storage' | 'expert' | 'rlcf';
  metrics: Record<string, unknown>;
  status: ServiceStatus;
}

export interface ArchitectureEdge {
  source: string;
  target: string;
  label?: string;
  animated: boolean;
}

export interface ArchitectureDiagram {
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  selected_node?: string;
}

export interface NodeDetails {
  node_id: string;
  label: string;
  description: string;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
  links: Record<string, string>;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Recupera overview completo della dashboard.
 */
export async function getDashboardOverview(): Promise<DashboardOverview> {
  return get(`${PREFIX}/dashboard/overview`);
}

/**
 * Recupera health check di tutti i servizi.
 */
export async function getSystemHealth(): Promise<SystemHealth> {
  return get(`${PREFIX}/dashboard/health`);
}

/**
 * Recupera dati per diagramma architettura.
 */
export async function getArchitectureDiagram(): Promise<ArchitectureDiagram> {
  return get(`${PREFIX}/dashboard/architecture`);
}

/**
 * Recupera dettagli per un nodo specifico.
 */
export async function getNodeDetails(nodeId: string): Promise<NodeDetails> {
  return get(`${PREFIX}/dashboard/architecture/node/${encodeURIComponent(nodeId)}`);
}

/**
 * Recupera activity feed con paginazione.
 */
export async function getActivityFeed(params?: {
  limit?: number;
  offset?: number;
  activity_type?: ActivityType;
}): Promise<ActivityFeed> {
  const searchParams = new URLSearchParams();

  if (params?.limit) {
    searchParams.set('limit', params.limit.toString());
  }
  if (params?.offset) {
    searchParams.set('offset', params.offset.toString());
  }
  if (params?.activity_type) {
    searchParams.set('activity_type', params.activity_type);
  }

  const query = searchParams.toString();
  return get(`${PREFIX}/dashboard/activity${query ? `?${query}` : ''}`);
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Formatta uptime in formato leggibile.
 */
export function formatUptime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return `${days}d ${remainingHours}h`;
}

/**
 * Ritorna colore per status.
 */
export function getStatusColor(status: ServiceStatus): string {
  switch (status) {
    case 'online':
      return 'text-green-500';
    case 'offline':
      return 'text-red-500';
    case 'degraded':
      return 'text-yellow-500';
    default:
      return 'text-gray-500';
  }
}

/**
 * Ritorna badge color per severity.
 */
export function getSeverityColor(severity: 'info' | 'warning' | 'error'): string {
  switch (severity) {
    case 'info':
      return 'bg-blue-100 text-blue-800';
    case 'warning':
      return 'bg-yellow-100 text-yellow-800';
    case 'error':
      return 'bg-red-100 text-red-800';
  }
}
