/**
 * ArchitectureTab
 * ===============
 *
 * Tab con diagramma architettura interattivo del sistema MERL-T.
 * Mostra visivamente il flusso dati da sources a storage a experts.
 *
 * Features:
 * - Diagramma react-flow interattivo
 * - Click su nodi per vedere dettagli e metriche
 * - Animazione flusso dati
 * - Status indicator per ogni componente
 *
 * @example
 * ```tsx
 * <ArchitectureTab />
 * ```
 */

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  FileText,
  GitBranch,
  Cpu,
  Users,
  Brain,
  Box,
  RefreshCw,
  AlertCircle,
  ExternalLink,
  Server,
  HardDrive,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  getArchitectureDiagram,
  getNodeDetails,
  type ArchitectureDiagram,
  type ArchitectureNode,
  type NodeDetails,
} from '../../../../services/dashboardService';
import {
  getCircuitBreakerStatus,
  resetBreaker,
  type CircuitBreakerStatus,
  type CircuitBreakerStatusResponse,
} from '../../../../services/circuitBreakerService';
import { ApiKeysSection } from './ApiKeysSection';

// =============================================================================
// NODE ICON MAP
// =============================================================================

const NODE_ICONS: Record<string, React.ReactNode> = {
  normattiva: <FileText size={24} className="text-blue-500" />,
  brocardi: <FileText size={24} className="text-green-500" />,
  eurlex: <FileText size={24} className="text-yellow-500" />,
  parser: <GitBranch size={24} className="text-purple-500" />,
  chunker: <Box size={24} className="text-orange-500" />,
  embedder: <Cpu size={24} className="text-cyan-500" />,
  falkordb: <Database size={24} className="text-red-500" />,
  qdrant: <HardDrive size={24} className="text-blue-500" />,
  postgresql: <Server size={24} className="text-indigo-500" />,
  redis: <Database size={24} className="text-red-400" />,
  gating: <Brain size={24} className="text-purple-500" />,
  traversal: <GitBranch size={24} className="text-green-500" />,
  literal: <Users size={24} className="text-blue-500" />,
  systemic: <Users size={24} className="text-green-500" />,
  principles: <Users size={24} className="text-yellow-500" />,
  precedent: <Users size={24} className="text-orange-500" />,
};

// =============================================================================
// ARCHITECTURE NODE
// =============================================================================

interface ArchitectureNodeCardProps {
  node: ArchitectureNode;
  isSelected: boolean;
  onClick: () => void;
  position: { x: number; y: number };
}

function ArchitectureNodeCard({
  node,
  isSelected,
  onClick,
  position,
}: ArchitectureNodeCardProps) {
  const icon = NODE_ICONS[node.id.toLowerCase()] || <Box size={24} />;

  // Border color based on TYPE (not status)
  const typeBorderColors = {
    source: 'border-blue-400',
    pipeline: 'border-purple-400',
    storage: 'border-orange-400',
    expert: 'border-green-400',
    rlcf: 'border-indigo-400',
  };

  // Background slightly tinted by type
  const typeBgColors = {
    source: 'bg-blue-50 dark:bg-blue-900/10',
    pipeline: 'bg-purple-50 dark:bg-purple-900/10',
    storage: 'bg-orange-50 dark:bg-orange-900/10',
    expert: 'bg-green-50 dark:bg-green-900/10',
    rlcf: 'bg-indigo-50 dark:bg-indigo-900/10',
  };

  // Status indicator with icon instead of confusing dot
  const statusConfig = {
    online: { color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30' },
    offline: { color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30' },
    degraded: { color: 'text-yellow-500', bg: 'bg-yellow-100 dark:bg-yellow-900/30' },
    unknown: { color: 'text-slate-500', bg: 'bg-slate-100 dark:bg-slate-900/30' },
  };

  const status = statusConfig[node.status] || statusConfig.unknown;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.05 }}
      className={cn(
        'absolute cursor-pointer transition-all duration-200',
        'w-36 p-3 rounded-xl border-2',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
        isSelected
          ? 'ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-slate-900'
          : '',
        typeBorderColors[node.type],
        typeBgColors[node.type]
      )}
      tabIndex={0}
      role="button"
      aria-label={`${node.label} - ${node.type} - ${node.status}`}
      style={{
        left: position.x,
        top: position.y,
      }}
      onClick={onClick}
    >
      {/* Status badge - clear text instead of confusing dots */}
      <div
        className={cn(
          'absolute -top-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide',
          status.bg,
          status.color
        )}
      >
        {node.status === 'online' ? '●' : node.status === 'offline' ? '○' : '◐'}
      </div>

      <div className="flex flex-col items-center text-center">
        {icon}
        <span className="mt-2 text-xs font-medium text-slate-700 dark:text-slate-300 truncate w-full">
          {node.label}
        </span>
        {/* Type label - small text below */}
        <span className="text-[9px] text-slate-400 dark:text-slate-500 uppercase tracking-wider mt-0.5">
          {node.type}
        </span>
      </div>
    </motion.div>
  );
}

// =============================================================================
// NODE DETAILS PANEL
// =============================================================================

interface NodeDetailsPanelProps {
  details: NodeDetails;
  onClose: () => void;
}

function NodeDetailsPanel({ details, onClose }: NodeDetailsPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 w-80"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          {details.label}
        </h3>
        <button
          onClick={onClose}
          aria-label="Chiudi dettagli nodo"
          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        {details.description}
      </p>

      {/* Metrics */}
      <div className="space-y-3 mb-4">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Metriche
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(details.metrics).map(([key, value]) => (
            <div
              key={key}
              className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-2"
            >
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {key.replace(/_/g, ' ')}
              </p>
              <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
                {typeof value === 'number' ? value.toLocaleString() : String(value)}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Config */}
      {Object.keys(details.config).length > 0 && (
        <div className="space-y-2 mb-4">
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Configurazione
          </h4>
          <div className="text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-700/50 rounded-lg p-2 font-mono">
            {Object.entries(details.config).map(([key, value]) => (
              <div key={key}>
                {key}: {String(value)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Links */}
      {Object.keys(details.links).length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Link utili
          </h4>
          <div className="space-y-1">
            {Object.entries(details.links).map(([label, url]) => (
              <a
                key={label}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
              >
                <ExternalLink size={12} aria-hidden="true" />
                {label}
              </a>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

// =============================================================================
// LEGEND
// =============================================================================

function Legend() {
  const typeItems = [
    { border: 'border-blue-400', bg: 'bg-blue-50', label: 'Source' },
    { border: 'border-purple-400', bg: 'bg-purple-50', label: 'Pipeline' },
    { border: 'border-orange-400', bg: 'bg-orange-50', label: 'Storage' },
    { border: 'border-green-400', bg: 'bg-green-50', label: 'Expert' },
    { border: 'border-indigo-400', bg: 'bg-indigo-50', label: 'RLCF' },
  ];

  const statusItems = [
    { symbol: '●', color: 'text-green-500', label: 'Online' },
    { symbol: '◐', color: 'text-yellow-500', label: 'Degraded' },
    { symbol: '○', color: 'text-red-500', label: 'Offline' },
  ];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
        Legenda
      </h3>

      <div className="flex flex-wrap gap-6">
        {/* Type legend - shows border colors */}
        <div className="space-y-2">
          <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">Tipo (bordo)</p>
          <div className="flex flex-wrap gap-3">
            {typeItems.map((item) => (
              <div key={item.label} className="flex items-center gap-1.5">
                <div
                  className={cn(
                    'w-6 h-4 rounded border-2',
                    item.border,
                    item.bg
                  )}
                />
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Status legend - shows symbols */}
        <div className="space-y-2">
          <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">Status (badge)</p>
          <div className="flex flex-wrap gap-3">
            {statusItems.map((item) => (
              <div key={item.label} className="flex items-center gap-1.5">
                <span className={cn('text-sm font-bold', item.color)}>
                  {item.symbol}
                </span>
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// POSITIONS (Manual layout for now, could use react-flow later)
// =============================================================================

const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  // Sources (left column)
  normattiva: { x: 30, y: 50 },
  brocardi: { x: 30, y: 160 },
  eurlex: { x: 30, y: 270 },

  // Pipeline (center-left) - commented out if not in diagram
  parser: { x: 200, y: 100 },
  chunker: { x: 200, y: 200 },
  embedder: { x: 200, y: 300 },

  // Storage (center)
  falkordb: { x: 380, y: 50 },
  qdrant: { x: 380, y: 160 },
  postgresql: { x: 380, y: 270 },
  redis: { x: 380, y: 380 },

  // Experts (right column)
  literal: { x: 560, y: 50 },
  systemic: { x: 560, y: 140 },
  principles: { x: 560, y: 230 },
  precedent: { x: 560, y: 320 },

  // RLCF (bottom-right)
  gating: { x: 560, y: 420 },
  traversal: { x: 380, y: 490 },
};

// =============================================================================
// CIRCUIT BREAKERS SECTION
// =============================================================================

function CircuitBreakersSection() {
  const [cbData, setCbData] = useState(null as CircuitBreakerStatusResponse | null);
  const [cbLoading, setCbLoading] = useState(true);
  const [resetting, setResetting] = useState(null as string | null);

  const fetchCbStatus = async () => {
    setCbLoading(true);
    try {
      const data = await getCircuitBreakerStatus();
      setCbData(data);
    } catch (err) {
      console.error('Failed to load circuit breaker status:', err);
    } finally {
      setCbLoading(false);
    }
  };

  const handleReset = async (name: string) => {
    setResetting(name);
    try {
      await resetBreaker(name.replace(/_expert$/, ''));
      await fetchCbStatus();
    } catch (err) {
      console.error('Failed to reset circuit breaker:', err);
    } finally {
      setResetting(null);
    }
  };

  useEffect(() => {
    fetchCbStatus();
  }, []);

  const stateColor = (state: string) => {
    if (state === 'closed') return 'text-green-600 dark:text-green-400';
    if (state === 'open') return 'text-red-600 dark:text-red-400';
    return 'text-yellow-600 dark:text-yellow-400';
  };

  const stateBg = (state: string) => {
    if (state === 'closed') return 'bg-green-100 dark:bg-green-900/30';
    if (state === 'open') return 'bg-red-100 dark:bg-red-900/30';
    return 'bg-yellow-100 dark:bg-yellow-900/30';
  };

  if (cbLoading || !cbData) {
    return null;
  }

  const entries = Object.entries(cbData.breakers) as [string, CircuitBreakerStatus][];
  if (entries.length === 0) return null;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Circuit Breakers
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {cbData.total_count} registrati, {cbData.open_count} aperti
          </p>
        </div>
        <button
          onClick={fetchCbStatus}
          className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
          aria-label="Aggiorna stato circuit breaker"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-slate-50 dark:bg-slate-700/50 text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              <th className="px-4 py-3 text-left">Expert</th>
              <th className="px-4 py-3 text-center">State</th>
              <th className="px-4 py-3 text-right">Failures</th>
              <th className="px-4 py-3 text-right">Times Opened</th>
              <th className="px-4 py-3 text-right">Last Failure</th>
              <th className="px-4 py-3 text-center">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
            {entries.map(([name, cb]) => (
              <tr key={name} className="hover:bg-slate-50 dark:hover:bg-slate-700/30">
                <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">
                  {name}
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={cn('px-2 py-1 rounded text-xs font-medium', stateBg(cb.state), stateColor(cb.state))}>
                    {cb.state}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-slate-700 dark:text-slate-300">
                  {cb.failure_count} / {cb.total_failures}
                </td>
                <td className="px-4 py-3 text-right text-slate-700 dark:text-slate-300">
                  {cb.times_opened}
                </td>
                <td className="px-4 py-3 text-right text-xs text-slate-500 dark:text-slate-400">
                  {cb.last_failure_time
                    ? new Date(cb.last_failure_time).toLocaleString('it-IT', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' })
                    : '-'}
                </td>
                <td className="px-4 py-3 text-center">
                  {cb.state !== 'closed' && (
                    <button
                      onClick={() => handleReset(name)}
                      disabled={resetting === name}
                      className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                      {resetting === name ? 'Resetting...' : 'Reset'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ArchitectureTab() {
  const [diagram, setDiagram] = useState(null as ArchitectureDiagram | null);
  const [selectedNode, setSelectedNode] = useState(null as string | null);
  const [nodeDetails, setNodeDetails] = useState(null as NodeDetails | null);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [error, setError] = useState(null as string | null);

  const fetchDiagram = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getArchitectureDiagram();
      setDiagram(data);
    } catch (err) {
      setError('Errore nel caricamento del diagramma');
      console.error('Failed to load architecture diagram:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNodeClick = useCallback(async (nodeId: string) => {
    setSelectedNode(nodeId);
    setDetailsLoading(true);
    try {
      const details = await getNodeDetails(nodeId);
      setNodeDetails(details);
    } catch (err) {
      console.error('Failed to load node details:', err);
    } finally {
      setDetailsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDiagram();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12" role="status">
        <RefreshCw size={24} className="animate-spin text-blue-500" aria-hidden="true" />
        <span className="sr-only">Caricamento diagramma architettura in corso...</span>
      </div>
    );
  }

  if (error || !diagram) {
    return (
      <div className="text-center py-12" role="alert">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" aria-hidden="true" />
        <p className="text-slate-500 dark:text-slate-400">{error}</p>
        <button
          onClick={fetchDiagram}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Legend */}
      <Legend />

      {/* Architecture Diagram */}
      <div className="flex gap-6">
        {/* Diagram area */}
        <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 relative overflow-auto min-h-[600px]">
          {/* Section labels */}
          <div className="absolute top-3 left-6 text-xs font-medium text-slate-400 dark:text-slate-500">
            SOURCES
          </div>
          <div className="absolute top-3 left-[220px] text-xs font-medium text-slate-400 dark:text-slate-500">
            PIPELINE
          </div>
          <div className="absolute top-3 left-[390px] text-xs font-medium text-slate-400 dark:text-slate-500">
            STORAGE
          </div>
          <div className="absolute top-3 left-[560px] text-xs font-medium text-slate-400 dark:text-slate-500">
            EXPERTS / RLCF
          </div>

          {/* Connection lines (SVG) */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
            aria-hidden="true"
          >
            {diagram.edges.map((edge: { source: string; target: string; label?: string; animated?: boolean }, idx: number) => {
              const sourcePos = NODE_POSITIONS[edge.source.toLowerCase()];
              const targetPos = NODE_POSITIONS[edge.target.toLowerCase()];

              if (!sourcePos || !targetPos) return null;

              const startX = sourcePos.x + 64;
              const startY = sourcePos.y + 32;
              const endX = targetPos.x;
              const endY = targetPos.y + 32;

              return (
                <g key={idx}>
                  <line
                    x1={startX}
                    y1={startY}
                    x2={endX}
                    y2={endY}
                    stroke="#94a3b8"
                    strokeWidth="1.5"
                    strokeDasharray={edge.animated ? '4 4' : undefined}
                  />
                  {/* Arrow marker */}
                  <circle cx={endX - 4} cy={endY} r="3" fill="#94a3b8" />
                </g>
              );
            })}
          </svg>

          {/* Nodes */}
          {diagram.nodes.map((node: ArchitectureNode) => {
            const position = NODE_POSITIONS[node.id.toLowerCase()];
            if (!position) return null;

            return (
              <ArchitectureNodeCard
                key={node.id}
                node={node}
                isSelected={selectedNode === node.id}
                onClick={() => handleNodeClick(node.id)}
                position={position}
              />
            );
          })}
        </div>

        {/* Details panel */}
        {(selectedNode || nodeDetails) && (
          <div className="w-80 flex-shrink-0">
            {detailsLoading ? (
              <div className="flex items-center justify-center h-full" role="status">
                <RefreshCw size={24} className="animate-spin text-blue-500" aria-hidden="true" />
                <span className="sr-only">Caricamento dettagli nodo...</span>
              </div>
            ) : nodeDetails ? (
              <NodeDetailsPanel
                details={nodeDetails}
                onClose={() => {
                  setSelectedNode(null);
                  setNodeDetails(null);
                }}
              />
            ) : null}
          </div>
        )}
      </div>

      {/* Circuit Breakers */}
      <CircuitBreakersSection />

      {/* API Keys */}
      <ApiKeysSection />

      {/* Info text */}
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center">
        Clicca su un nodo per visualizzare metriche e configurazione dettagliate
      </p>
    </div>
  );
}

export default ArchitectureTab;
