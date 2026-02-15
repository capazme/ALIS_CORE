/**
 * KnowledgeGraphExplorer
 * ======================
 *
 * Componente per la visualizzazione interattiva del Knowledge Graph MERL-T.
 * Usa Reagraph (WebGL) per performance enterprise-level su grafi di grandi dimensioni.
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  GraphCanvas,
  useSelection,
  lightTheme,
  darkTheme,
} from 'reagraph';
import type { GraphCanvasRef, NodePositionArgs, GraphNode } from 'reagraph';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Filter,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCw,
  Info,
  X,
  GitBranch,
  Loader2,
  Network,
  Sparkles,
  Target,
  AlertTriangle,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { useSubgraph } from '../../../hooks/useSubgraph';
import { getSmartLabel } from '../../../utils/graphLabels';
import { ReportNodeIssueModal } from '../../merlt/ReportNodeIssueModal';
import type { EdgeReportData } from '../../merlt/ReportNodeIssueModal';
import type { SubgraphNode, SubgraphEdge } from '../../../types/merlt';

// =============================================================================
// CONSTANTS & COLORS
// =============================================================================

const NODE_COLORS: Record<string, string> = {
  Norma: '#3B82F6',
  Comma: '#60A5FA',
  Versione: '#93C5FD',
  Entity: '#8B5CF6',
  Concetto: '#A78BFA',
  Principio: '#C4B5FD',
  Dottrina: '#F59E0B',
  Brocardo: '#FBBF24',
  Precedente: '#FCD34D',
  Soggetto: '#10B981',
  Ruolo: '#34D399',
  Unknown: '#94A3B8',
};

const EDGE_COLORS: Record<string, string> = {
  CONTIENE: '#64748B',
  DISCIPLINA: '#3B82F6',
  CITA: '#8B5CF6',
  APPLICA: '#10B981',
  MODIFICA: '#F59E0B',
  ABROGA: '#EF4444',
  default: '#94A3B8',
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

interface NodeDetailsPanelProps {
  node: SubgraphNode;
  neighbors: { nodes: SubgraphNode[]; edges: SubgraphEdge[] };
  onClose: () => void;
  onNavigate: (nodeId: string) => void;
  onReportIssue: (node: SubgraphNode) => void;
  onReportEdge: (edge: SubgraphEdge, sourceNode: SubgraphNode, targetNode: SubgraphNode) => void;
}

function NodeDetailsPanel({ node, neighbors, onClose, onNavigate, onReportIssue, onReportEdge }: NodeDetailsPanelProps) {
  return (
    <motion.div
      initial={{ x: 300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 300, opacity: 0 }}
      className="absolute right-0 top-0 bottom-0 w-80 bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700 shadow-xl z-30 overflow-hidden flex flex-col"
    >
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: NODE_COLORS[node.type] || NODE_COLORS.Unknown }}
            />
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              {node.type}
            </span>
          </div>
          <h3 className="font-semibold text-slate-900 dark:text-white text-sm leading-tight">
            {node.label}
          </h3>
          {node.urn && (
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">
              {node.urn}
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
        >
          <X size={18} />
        </button>
      </div>

      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
        <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
          Proprietà
        </h4>
        <div className="space-y-1.5">
          {Object.entries(node.properties).slice(0, 5).map(([key, value]) => (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-slate-500 dark:text-slate-400">{key}</span>
              <span className="text-slate-700 dark:text-slate-300 truncate max-w-[150px]">
                {String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
          Connessioni ({neighbors.nodes.length})
        </h4>
        <div className="space-y-2">
          {neighbors.edges.map((edge) => {
            const targetId = typeof edge.target === 'string' ? edge.target : edge.target;
            const sourceId = typeof edge.source === 'string' ? edge.source : edge.source;
            const isOutgoing = sourceId === node.id;
            const neighborId = isOutgoing ? targetId : sourceId;
            const neighborNode = neighbors.nodes.find(n => n.id === neighborId);

            if (!neighborNode) return null;

            return (
              <div
                key={edge.id}
                className="p-2 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors group"
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: NODE_COLORS[neighborNode.type] || NODE_COLORS.Unknown }}
                    />
                    <span className="text-[10px] font-medium text-slate-400 uppercase">
                      {edge.type}
                    </span>
                    <GitBranch size={10} className={cn("text-slate-400", isOutgoing ? "" : "rotate-180")} />
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const sourceNode = isOutgoing ? node : neighborNode;
                      const targetNode = isOutgoing ? neighborNode : node;
                      onReportEdge(edge, sourceNode, targetNode);
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-amber-600 hover:text-amber-700"
                    title="Segnala relazione"
                  >
                    <AlertTriangle size={12} />
                  </button>
                </div>
                <button
                  onClick={() => onNavigate(neighborNode.id)}
                  className="text-xs text-slate-700 dark:text-slate-300 hover:text-primary-600 dark:hover:text-primary-400 text-left w-full truncate"
                >
                  {neighborNode.label}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-2">
        <button
          onClick={() => onNavigate(node.id)}
          className="w-full px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors"
        >
          <Target size={14} />
          Esplora da qui
        </button>
        <button
          onClick={() => onReportIssue(node)}
          className="w-full px-3 py-2 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors hover:bg-amber-100 dark:hover:bg-amber-900/50"
        >
          <AlertTriangle size={14} />
          Segnala problema
        </button>
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface KnowledgeGraphExplorerProps {
  initialUrn?: string;
  height?: number | string;
  className?: string;
  onArticleClick?: (urn: string) => void;
  userId?: string;
}

export function KnowledgeGraphExplorer({
  initialUrn,
  height = 600,
  className,
  onArticleClick,
  userId,
}: KnowledgeGraphExplorerProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState([] as string[]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [reportingNode, setReportingNode] = useState(null as SubgraphNode | null);
  const [reportingEdge, setReportingEdge] = useState(null as EdgeReportData | null);
  const graphRef = useRef(null as GraphCanvasRef | null);

  const resolvedUserId = userId || 'anonymous';

  const {
    data,
    loading,
    error,
    selectedNode,
    setSelectedNode,
    expandNode,
    refresh,
    setFilters,
  } = useSubgraph({
    initialUrn,
    depth: 2,
    enabled: true,
    filters: selectedTypes,
  });

  const { selections, onNodeClick, onCanvasClick } = useSelection({
    ref: graphRef,
  });

  useEffect(() => {
    if (!selections.length) {
      setSelectedNode(null);
      return;
    }

    const node = data.nodes.find((n: SubgraphNode) => n.id === selections[0]);
    if (node) {
      setSelectedNode(node);
    }
  }, [selections, data.nodes, setSelectedNode]);

  const filteredNodes = useMemo(() => {
    if (!searchQuery) return data.nodes;
    return data.nodes.filter(node =>
      node.label.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [data.nodes, searchQuery]);

  const graphNodes = useMemo(() => filteredNodes.map((node: SubgraphNode) => ({
    id: node.id,
    label: getSmartLabel(node),
    size: node.type === 'Norma' ? 12 : 8,
    fill: NODE_COLORS[node.type] || NODE_COLORS.Unknown,
    data: node,
  })), [filteredNodes]);

  const graphEdges = useMemo(() => data.edges.map(edge => ({
    id: edge.id,
    source: typeof edge.source === 'string' ? edge.source : edge.source,
    target: typeof edge.target === 'string' ? edge.target : edge.target,
    label: edge.type,
    fill: EDGE_COLORS[edge.type] || EDGE_COLORS.default,
    size: 1,
    data: edge,
  })), [data.edges]);

  const uniqueTypes = useMemo(() => {
    const types = new Set(data.nodes.map(n => n.type));
    return Array.from(types).sort();
  }, [data.nodes]);

  const handleNodeClick = useCallback((graphNode: GraphNode) => {
    onNodeClick?.(graphNode);
    const node = data.nodes.find((n: SubgraphNode) => n.id === graphNode.id);
    if (node?.urn) {
      onArticleClick?.(node.urn);
    }
  }, [onNodeClick, data.nodes, onArticleClick]);


  const handleZoomIn = () => graphRef.current?.zoomIn();
  const handleZoomOut = () => graphRef.current?.zoomOut();
  const handleCenter = () => graphRef.current?.centerGraph();

  const handleReportIssue = (node: SubgraphNode) => {
    setReportingNode(node);
  };

  const handleReportEdge = (edge: SubgraphEdge, sourceNode: SubgraphNode, targetNode: SubgraphNode) => {
    setReportingEdge({ edge, sourceNode, targetNode });
  };

  const handleToggleType = (type: string) => {
    setSelectedTypes((prev: string[]) => {
      const next = prev.includes(type) ? prev.filter((t: string) => t !== type) : [...prev, type];
      setFilters(next);
      return next;
    });
  };

  const containerClasses = cn(
    "relative bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden",
    isFullscreen && "fixed inset-4 z-50",
    className
  );

  return (
    <div className={containerClasses} style={{ height }}>
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Cerca nodo..."
            className="pl-9 pr-3 py-2 rounded-lg text-sm bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={cn(
            "p-2 rounded-lg border text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200",
            showFilters ? "bg-primary-50 dark:bg-primary-900/30 border-primary-200 dark:border-primary-700" : "bg-white/90 dark:bg-slate-800/90 border-slate-200 dark:border-slate-700"
          )}
          title="Filtri"
        >
          <Filter size={16} />
        </button>
      </div>

      <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
        <button onClick={handleZoomIn} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Zoom in">
          <ZoomIn size={16} />
        </button>
        <button onClick={handleZoomOut} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Zoom out">
          <ZoomOut size={16} />
        </button>
        <button onClick={handleCenter} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Centra grafo">
          <Target size={16} />
        </button>
        <button onClick={refresh} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Aggiorna">
          <RefreshCw size={16} />
        </button>
        <button onClick={() => setShowInfo(!showInfo)} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Info">
          <Info size={16} />
        </button>
        <button onClick={() => setIsFullscreen(!isFullscreen)} className="p-2 rounded-lg bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700" title="Fullscreen">
          <Maximize2 size={16} />
        </button>
      </div>

      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-16 left-4 z-20 bg-white dark:bg-slate-900 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 p-3 w-60"
          >
            <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
              Tipi di nodo
            </h4>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {uniqueTypes.map((type: string) => (
                <label key={type} className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedTypes.includes(type)}
                    onChange={() => handleToggleType(type)}
                    className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: NODE_COLORS[type] || NODE_COLORS.Unknown }} />
                  {type}
                </label>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute bottom-4 left-4 z-20 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-1">
          <Network size={12} />
          {data.nodes.length} nodi
        </div>
        <div className="flex items-center gap-1">
          <GitBranch size={12} />
          {data.edges.length} relazioni
        </div>
      </div>

      {loading && (
        <div className="absolute inset-0 bg-white/70 dark:bg-slate-900/70 z-10 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <Loader2 className="animate-spin" size={24} />
            <span className="text-sm">Caricamento grafo...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 bg-white/80 dark:bg-slate-900/80 z-10 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2 text-red-500">
            <AlertTriangle size={24} />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      <div className="w-full h-full">
        <GraphCanvas
          ref={graphRef}
          nodes={graphNodes}
          edges={graphEdges}
          theme={document.documentElement.classList.contains('dark') ? darkTheme : lightTheme}
          onNodeClick={(node: GraphNode) => handleNodeClick(node)}
          onCanvasClick={onCanvasClick}
          draggable
          animated
          layoutType="forceDirected2d"
          sizingType="centrality"
          labelType="nodes"
          edgeLabelType="all"
          nodePosition={(args: NodePositionArgs) => args}
        />
      </div>

      <AnimatePresence>
        {selectedNode && (
          <NodeDetailsPanel
            node={selectedNode}
            neighbors={{
              nodes: data.nodes.filter(n =>
                data.edges.some(e => e.source === n.id || e.target === n.id) && n.id !== selectedNode.id
              ),
              edges: data.edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id),
            }}
            onClose={() => setSelectedNode(null)}
            onNavigate={(nodeId) => expandNode(nodeId)}
            onReportIssue={handleReportIssue}
            onReportEdge={handleReportEdge}
          />
        )}
      </AnimatePresence>

      <ReportNodeIssueModal
        isOpen={!!reportingNode}
        onClose={() => setReportingNode(null)}
        node={reportingNode}
        userId={resolvedUserId}
        onSuccess={() => setReportingNode(null)}
      />
      <ReportNodeIssueModal
        isOpen={!!reportingEdge}
        onClose={() => setReportingEdge(null)}
        edgeData={reportingEdge || undefined}
        userId={resolvedUserId}
        onSuccess={() => setReportingEdge(null)}
      />

      {showInfo && (
        <div className="absolute bottom-4 right-4 z-20 bg-white dark:bg-slate-900 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 p-4 w-64">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={16} className="text-primary-600" />
            <h4 className="text-sm font-semibold text-slate-800 dark:text-white">Knowledge Graph</h4>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Esplora le connessioni tra norme, concetti e precedenti. Seleziona un nodo per vedere i dettagli o espandere il grafo.
          </p>
        </div>
      )}
    </div>
  );
}

export default KnowledgeGraphExplorer;
