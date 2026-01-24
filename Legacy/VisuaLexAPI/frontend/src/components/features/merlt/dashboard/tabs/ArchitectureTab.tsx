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
import { cn } from '../../../../../lib/utils';
import {
  getArchitectureDiagram,
  getNodeDetails,
  type ArchitectureDiagram,
  type ArchitectureNode,
  type NodeDetails,
} from '../../../../../services/dashboardService';

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
    unknown: { color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-900/30' },
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
        isSelected
          ? 'ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-gray-900'
          : '',
        typeBorderColors[node.type],
        typeBgColors[node.type]
      )}
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
        <span className="mt-2 text-xs font-medium text-gray-700 dark:text-gray-300 truncate w-full">
          {node.label}
        </span>
        {/* Type label - small text below */}
        <span className="text-[9px] text-gray-400 dark:text-gray-500 uppercase tracking-wider mt-0.5">
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
      className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 w-80"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {details.label}
        </h3>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
        >
          ×
        </button>
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {details.description}
      </p>

      {/* Metrics */}
      <div className="space-y-3 mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Metriche
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(details.metrics).map(([key, value]) => (
            <div
              key={key}
              className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-2"
            >
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {key.replace(/_/g, ' ')}
              </p>
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                {typeof value === 'number' ? value.toLocaleString() : String(value)}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Config */}
      {Object.keys(details.config).length > 0 && (
        <div className="space-y-2 mb-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Configurazione
          </h4>
          <div className="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50 rounded-lg p-2 font-mono">
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
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Link utili
          </h4>
          <div className="space-y-1">
            {Object.entries(details.links).map(([label, url]) => (
              <a
                key={label}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                <ExternalLink size={12} />
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
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
        Legenda
      </h3>

      <div className="flex flex-wrap gap-6">
        {/* Type legend - shows border colors */}
        <div className="space-y-2">
          <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Tipo (bordo)</p>
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
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Status legend - shows symbols */}
        <div className="space-y-2">
          <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Status (badge)</p>
          <div className="flex flex-wrap gap-3">
            {statusItems.map((item) => (
              <div key={item.label} className="flex items-center gap-1.5">
                <span className={cn('text-sm font-bold', item.color)}>
                  {item.symbol}
                </span>
                <span className="text-xs text-gray-600 dark:text-gray-400">
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
// MAIN COMPONENT
// =============================================================================

export function ArchitectureTab() {
  const [diagram, setDiagram] = useState<ArchitectureDiagram | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [nodeDetails, setNodeDetails] = useState<NodeDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      <div className="flex items-center justify-center py-12">
        <RefreshCw size={24} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error || !diagram) {
    return (
      <div className="text-center py-12">
        <AlertCircle size={48} className="mx-auto text-red-400 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">{error}</p>
        <button
          onClick={fetchDiagram}
          className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
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
        <div className="flex-1 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 relative overflow-auto min-h-[600px]">
          {/* Section labels */}
          <div className="absolute top-3 left-6 text-xs font-medium text-gray-400 dark:text-gray-500">
            SOURCES
          </div>
          <div className="absolute top-3 left-[220px] text-xs font-medium text-gray-400 dark:text-gray-500">
            PIPELINE
          </div>
          <div className="absolute top-3 left-[390px] text-xs font-medium text-gray-400 dark:text-gray-500">
            STORAGE
          </div>
          <div className="absolute top-3 left-[560px] text-xs font-medium text-gray-400 dark:text-gray-500">
            EXPERTS / RLCF
          </div>

          {/* Connection lines (SVG) */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
          >
            {diagram.edges.map((edge, idx) => {
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
          {diagram.nodes.map((node) => {
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
              <div className="flex items-center justify-center h-full">
                <RefreshCw size={24} className="animate-spin text-blue-500" />
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

      {/* Info text */}
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
        Clicca su un nodo per visualizzare metriche e configurazione dettagliate
      </p>
    </div>
  );
}

export default ArchitectureTab;
