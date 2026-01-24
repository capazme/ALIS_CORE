/**
 * KnowledgeGraphExplorer
 * ======================
 *
 * Componente per la visualizzazione interattiva del Knowledge Graph MERL-T.
 * Usa Reagraph (WebGL) per performance enterprise-level su grafi di grandi dimensioni.
 *
 * Features:
 * - WebGL rendering via Reagraph per performance 10x vs Canvas
 * - Clustering automatico per tipo di nodo
 * - Semantic zoom con Level of Detail (LOD)
 * - Click su nodo per espandere i vicini
 * - Hover per vedere dettagli
 * - Filtri per tipo entità
 * - Barra di ricerca per navigare verso un articolo
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  GraphCanvas,
  useSelection,
  lightTheme,
  darkTheme,
} from 'reagraph';
import type { GraphCanvasRef, NodePositionArgs } from 'reagraph';
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
  CircleDot,
  GitBranch,
  Loader2,
  Network,
  Sparkles,
  Target,
  Layers,
  AlertTriangle,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { useSubgraph } from '../../../hooks/useSubgraph';
import { getSmartLabel } from '../../../utils/graphLabels';
import { ReportNodeIssueModal } from '../merlt/ReportNodeIssueModal';
import type { EdgeReportData } from '../merlt/ReportNodeIssueModal';
import { useAuth } from '../../../hooks/useAuth';
import type { SubgraphNode, SubgraphEdge } from '../../../types/merlt';

// =============================================================================
// CONSTANTS & COLORS
// =============================================================================

/**
 * Colori per tipo di nodo - palette vibrante ma professionale
 */
const NODE_COLORS: Record<string, string> = {
  // Norme e struttura
  Norma: '#3B82F6',      // Blue - leggi e articoli
  Comma: '#60A5FA',      // Light blue - comma
  Versione: '#93C5FD',   // Lighter blue - versioni
  // Entità concettuali
  Entity: '#8B5CF6',     // Purple - entità generiche
  Concetto: '#A78BFA',   // Light purple - concetti
  Principio: '#C4B5FD',  // Lighter purple - principi
  // Giurisprudenza
  Dottrina: '#F59E0B',   // Amber - dottrina
  Brocardo: '#FBBF24',   // Yellow - brocardi
  Precedente: '#FCD34D', // Light yellow - precedenti
  // Soggetti e ruoli
  Soggetto: '#10B981',   // Emerald - soggetti
  Ruolo: '#34D399',      // Light green - ruoli
  // Default
  Unknown: '#94A3B8',    // Slate - sconosciuto
};

/**
 * Colori per tipo di relazione
 */
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
      {/* Header */}
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

      {/* Properties */}
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

      {/* Neighbors */}
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
                    <GitBranch size={10} className={cn(
                      "text-slate-400",
                      isOutgoing ? "" : "rotate-180"
                    )} />
                  </div>
                  {/* Bottone segnala relazione */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const sourceNode = isOutgoing ? node : neighborNode;
                      const targetNode = isOutgoing ? neighborNode : node;
                      onReportEdge(edge, sourceNode, targetNode);
                    }}
                    className="p-1 rounded hover:bg-amber-100 dark:hover:bg-amber-900/30 text-slate-400 hover:text-amber-600 dark:hover:text-amber-400 opacity-0 group-hover:opacity-100 transition-all"
                    title="Segnala relazione"
                  >
                    <AlertTriangle size={12} />
                  </button>
                </div>
                <button
                  onClick={() => onNavigate(neighborId)}
                  className="w-full text-left"
                >
                  <p className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate group-hover:text-primary-600">
                    {neighborNode.label}
                  </p>
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer - Report Issue */}
      <div className="p-3 border-t border-slate-200 dark:border-slate-700 flex-shrink-0">
        <button
          onClick={() => onReportIssue(node)}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
        >
          <AlertTriangle size={16} />
          Segnala problema
        </button>
      </div>
    </motion.div>
  );
}

// =============================================================================
// REAGRAPH NODE & EDGE TYPES
// =============================================================================

interface ReagraphNode {
  id: string;
  label: string;
  fill?: string;
  size?: number;
  data?: {
    type: string;
    urn?: string;
    properties: Record<string, unknown>;
    metadata: Record<string, unknown>;
    connectionCount: number;
  };
}

interface ReagraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  fill?: string;
  data?: {
    type: string;
    properties: Record<string, unknown>;
  };
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface KnowledgeGraphExplorerProps {
  /** URN iniziale da caricare (opzionale) */
  initialUrn?: string;
  /** Altezza del componente */
  height?: number | string;
  /** Callback quando si naviga verso un articolo */
  onArticleClick?: (urn: string) => void;
  /** Classe CSS aggiuntiva */
  className?: string;
}

export function KnowledgeGraphExplorer({
  initialUrn,
  height = 600,
  onArticleClick: _onArticleClick,
  className,
}: KnowledgeGraphExplorerProps) {
  const graphRef = useRef<GraphCanvasRef | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<SubgraphNode | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [activeNodeTypes, setActiveNodeTypes] = useState<Set<string>>(new Set());
  const [showOnlyHubs, setShowOnlyHubs] = useState(false);
  const [clusterByType, setClusterByType] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState<Set<string>>(new Set());

  // Issue reporting
  const [reportIssueNode, setReportIssueNode] = useState<SubgraphNode | null>(null);
  const [reportIssueEdge, setReportIssueEdge] = useState<EdgeReportData | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);

  // Detect dark mode
  useEffect(() => {
    const checkDarkMode = () => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // Subgraph hook
  const {
    nodes,
    edges,
    isLoading,
    error,
    loadSubgraph,
    expandNode,
    reset,
    stats,
    getNeighbors,
  } = useSubgraph({
    initialDepth: 2,
    maxNodes: 100,
    includeMetadata: true,
  });

  // Calculate connection counts for each node
  const connectionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    edges.forEach((e: SubgraphEdge) => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source;
      const targetId = typeof e.target === 'string' ? e.target : e.target;
      counts[sourceId] = (counts[sourceId] || 0) + 1;
      counts[targetId] = (counts[targetId] || 0) + 1;
    });
    return counts;
  }, [edges]);

  // Transform data for Reagraph
  const { reagraphNodes, reagraphEdges } = useMemo(() => {
    // Step 1: Filter nodes by active types if any are selected
    let filteredNodes = activeNodeTypes.size > 0
      ? nodes.filter((n: SubgraphNode) => activeNodeTypes.has(n.type))
      : nodes;

    // Step 2: Filter for hub nodes if showOnlyHubs is enabled
    if (showOnlyHubs) {
      filteredNodes = filteredNodes.filter((n: SubgraphNode) =>
        (connectionCounts[n.id] || 0) >= 3
      );
    }

    const nodeIds = new Set(filteredNodes.map((n: SubgraphNode) => n.id));

    // Filter edges
    const filteredEdges = edges.filter((e: SubgraphEdge) => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source;
      const targetId = typeof e.target === 'string' ? e.target : e.target;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });

    // Transform to Reagraph format
    const reagraphNodes: ReagraphNode[] = filteredNodes.map((n: SubgraphNode) => {
      const connCount = connectionCounts[n.id] || 0;
      const isHighlighted = highlightedNodeIds.has(n.id);

      // Node size based on connections (hub nodes are larger)
      // Highlighted nodes get extra size boost
      const baseSize = 5;
      const connectionBonus = connCount * 2;
      const highlightBonus = isHighlighted ? 8 : 0;
      const size = Math.min(25, Math.max(baseSize, baseSize + connectionBonus + highlightBonus));

      // Highlighted nodes get a bright color (green/cyan)
      const fill = isHighlighted
        ? '#10B981' // Emerald green for highlighted
        : (NODE_COLORS[n.type] || NODE_COLORS.Unknown);

      return {
        id: n.id,
        label: getSmartLabel(n),
        fill,
        size,
        data: {
          type: n.type,
          urn: n.urn,
          properties: n.properties,
          metadata: n.metadata,
          connectionCount: connCount,
        },
      };
    });

    const reagraphEdges: ReagraphEdge[] = filteredEdges.map((e: SubgraphEdge) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.type,
      fill: EDGE_COLORS[e.type] || EDGE_COLORS.default,
      data: {
        type: e.type,
        properties: e.properties,
      },
    }));

    return { reagraphNodes, reagraphEdges };
  }, [nodes, edges, activeNodeTypes, showOnlyHubs, connectionCounts, highlightedNodeIds]);

  // Clustering configuration - group by node type
  const clusterAttribute = clusterByType ? 'data.type' : undefined;

  // Selection hook from Reagraph
  const {
    selections,
    actives,
    onNodeClick: onReagraphNodeClick,
    onCanvasClick,
  } = useSelection({
    ref: graphRef,
    nodes: reagraphNodes,
    edges: reagraphEdges,
    pathSelectionType: 'all',
  });

  // Load initial graph
  useEffect(() => {
    if (initialUrn) {
      loadSubgraph(initialUrn);
    }
  }, [initialUrn, loadSubgraph]);

  // Real-time search filter for existing nodes
  useEffect(() => {
    if (!searchQuery.trim()) {
      setHighlightedNodeIds(new Set());
      return;
    }

    const query = searchQuery.toLowerCase().trim();
    const matchedIds = new Set<string>();

    nodes.forEach((node: SubgraphNode) => {
      // Search in label, type, urn, and properties
      const labelMatch = node.label?.toLowerCase().includes(query);
      const typeMatch = node.type?.toLowerCase().includes(query);
      const urnMatch = node.urn?.toLowerCase().includes(query);
      const idMatch = node.id?.toLowerCase().includes(query);

      // Also check for article numbers like "1337" or "art1337"
      const artNum = query.replace(/\D/g, '');
      const artMatch = artNum && (
        node.label?.includes(artNum) ||
        node.urn?.includes(`art${artNum}`) ||
        node.id?.includes(`art${artNum}`)
      );

      if (labelMatch || typeMatch || urnMatch || idMatch || artMatch) {
        matchedIds.add(node.id);
      }
    });

    setHighlightedNodeIds(matchedIds);

    // If exactly one match, select and center on it
    if (matchedIds.size === 1) {
      const matchedId = Array.from(matchedIds)[0];
      const matchedNode = nodes.find((n: SubgraphNode) => n.id === matchedId);
      if (matchedNode && graphRef.current) {
        setSelectedNode(matchedNode);
        graphRef.current.centerGraph([matchedId]);
      }
    }
  }, [searchQuery, nodes]);

  // Handle search submit - load new subgraph
  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const query = searchQuery.trim();
    if (!query) return;

    // If it's a full URL, use it directly
    if (query.includes('normattiva.it') || query.includes('urn:nir:')) {
      loadSubgraph(query);
      setSelectedNode(null);
      return;
    }

    // If matches exist in current graph, just highlight them (already done via useEffect)
    if (highlightedNodeIds.size > 0) {
      // Center on first match if multiple
      const firstMatchId = Array.from(highlightedNodeIds)[0];
      const firstMatch = nodes.find((n: SubgraphNode) => n.id === firstMatchId);
      if (firstMatch && graphRef.current) {
        setSelectedNode(firstMatch);
        graphRef.current.centerGraph([firstMatchId]);
      }
      return;
    }

    // Otherwise, try to construct a URN and load new subgraph
    // Extract article number
    const artNum = query.replace(/\D/g, '');
    if (artNum) {
      // Default to Codice Civile - user can specify others via full URN
      const urn = `https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262:2~art${artNum}`;
      loadSubgraph(urn);
      setSelectedNode(null);
    }
  }, [searchQuery, loadSubgraph, highlightedNodeIds, nodes]);

  // Handle node click
  const handleNodeClick = useCallback((node: NodePositionArgs) => {
    onReagraphNodeClick(node);

    const subgraphNode = nodes.find((n: SubgraphNode) => n.id === node.id);
    if (subgraphNode) {
      setSelectedNode(subgraphNode);
      expandNode(node.id);

      // Center on node
      if (graphRef.current) {
        graphRef.current.centerGraph([node.id]);
      }
    }
  }, [nodes, expandNode, onReagraphNodeClick]);

  // Navigate to node
  const handleNavigateToNode = useCallback((nodeId: string) => {
    const node = nodes.find((n: SubgraphNode) => n.id === nodeId);
    if (node) {
      setSelectedNode(node);
      expandNode(nodeId);

      if (graphRef.current) {
        graphRef.current.centerGraph([nodeId]);
      }
    }
  }, [nodes, expandNode]);

  // Toggle node type filter
  const toggleNodeType = useCallback((type: string) => {
    setActiveNodeTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }, []);

  // Handle report issue for a node
  const handleReportIssue = useCallback((node: SubgraphNode) => {
    setReportIssueNode(node);
    setReportIssueEdge(null); // Clear any edge selection
    setShowReportModal(true);
  }, []);

  // Handle report issue for an edge/relation
  const handleReportEdge = useCallback((edge: SubgraphEdge, sourceNode: SubgraphNode, targetNode: SubgraphNode) => {
    setReportIssueEdge({
      edge,
      sourceNode,
      targetNode,
    });
    setReportIssueNode(null); // Clear any node selection
    setShowReportModal(true);
  }, []);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.zoomIn();
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.zoomOut();
    }
  }, []);

  const handleFitView = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.fitNodesInView();
    }
  }, []);

  // Custom theme based on dark mode
  const graphTheme = useMemo(() => {
    const baseTheme = isDarkMode ? darkTheme : lightTheme;
    return {
      ...baseTheme,
      canvas: {
        ...baseTheme.canvas,
        background: 'transparent',
      },
      node: {
        ...baseTheme.node,
        label: {
          ...baseTheme.node?.label,
          color: isDarkMode ? '#e2e8f0' : '#1e293b',
          fontSize: 10,
          maxWidth: 100,
        },
      },
      edge: {
        ...baseTheme.edge,
        label: {
          ...baseTheme.edge?.label,
          fontSize: 8,
          color: isDarkMode ? '#94a3b8' : '#64748b',
        },
      },
      cluster: {
        ...baseTheme.cluster,
        label: {
          ...baseTheme.cluster?.label,
          fontSize: 12,
          color: isDarkMode ? '#f8fafc' : '#0f172a',
        },
      },
    };
  }, [isDarkMode]);

  return (
    <div
      className={cn("relative bg-slate-950 rounded-xl overflow-hidden", className)}
      ref={containerRef}
      style={{ height, minHeight: 400 }}
    >
      {/* Gradient background effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-900/20 via-transparent to-purple-900/20 pointer-events-none" />

      {/* Header / Controls */}
      <div className="absolute top-0 left-0 right-0 z-20 p-4 bg-gradient-to-b from-slate-900/90 to-transparent">
        <div className="flex items-center gap-3">
          {/* Search */}
          <form onSubmit={handleSearch} className="flex-1 max-w-md relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Cerca nel grafo o carica articolo..."
              className={cn(
                "w-full pl-9 pr-28 py-2 bg-slate-800/80 backdrop-blur border rounded-lg text-white text-sm placeholder:text-slate-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent",
                highlightedNodeIds.size > 0 ? "border-green-500" : "border-slate-700"
              )}
            />
            {/* Search results indicator and clear button */}
            {searchQuery.trim() && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                {highlightedNodeIds.size > 0 ? (
                  <span className="text-xs text-green-400 font-medium">
                    {highlightedNodeIds.size} {highlightedNodeIds.size === 1 ? 'trovato' : 'trovati'}
                  </span>
                ) : (
                  <span className="text-xs text-slate-500">
                    Invio per caricare
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setHighlightedNodeIds(new Set());
                  }}
                  className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                  title="Cancella ricerca"
                >
                  <X size={14} />
                </button>
              </div>
            )}
          </form>

          {/* Filter toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "p-2 rounded-lg border transition-colors",
              showFilters
                ? "bg-primary-600 border-primary-500 text-white"
                : "bg-slate-800/80 border-slate-700 text-slate-400 hover:text-white"
            )}
            title="Filtra per tipo"
          >
            <Filter size={18} />
          </button>

          {/* Solo Hub toggle */}
          <button
            onClick={() => setShowOnlyHubs(!showOnlyHubs)}
            className={cn(
              "p-2 rounded-lg border transition-colors flex items-center gap-1.5",
              showOnlyHubs
                ? "bg-amber-600 border-amber-500 text-white"
                : "bg-slate-800/80 border-slate-700 text-slate-400 hover:text-white"
            )}
            title="Mostra solo nodi hub (>=3 connessioni)"
          >
            <Target size={16} />
            <span className="text-xs font-medium hidden sm:inline">Hub</span>
          </button>

          {/* Cluster toggle */}
          <button
            onClick={() => setClusterByType(!clusterByType)}
            className={cn(
              "p-2 rounded-lg border transition-colors flex items-center gap-1.5",
              clusterByType
                ? "bg-purple-600 border-purple-500 text-white"
                : "bg-slate-800/80 border-slate-700 text-slate-400 hover:text-white"
            )}
            title="Raggruppa per tipo (clustering)"
          >
            <Layers size={16} />
            <span className="text-xs font-medium hidden sm:inline">Cluster</span>
          </button>

          {/* Stats */}
          <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <CircleDot size={12} />
              {stats.nodeCount} nodi
            </span>
            <span className="flex items-center gap-1">
              <GitBranch size={12} />
              {stats.edgeCount} relazioni
            </span>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 bg-slate-800/80 rounded-lg border border-slate-700">
            <button
              onClick={handleZoomOut}
              className="p-2 text-slate-400 hover:text-white transition-colors"
              title="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <button
              onClick={handleZoomIn}
              className="p-2 text-slate-400 hover:text-white transition-colors"
              title="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
            <button
              onClick={handleFitView}
              className="p-2 text-slate-400 hover:text-white transition-colors"
              title="Fit view"
            >
              <Maximize2 size={16} />
            </button>
          </div>

          {/* Refresh */}
          <button
            onClick={reset}
            className="p-2 bg-slate-800/80 border border-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
            title="Reset"
          >
            <RefreshCw size={16} />
          </button>
        </div>

        {/* Filter panel */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="mt-3 p-3 bg-slate-800/80 backdrop-blur rounded-lg border border-slate-700">
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
                  Filtra per tipo
                </p>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(stats.nodeTypes).map((type) => (
                    <button
                      key={type}
                      onClick={() => toggleNodeType(type)}
                      className={cn(
                        "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
                        activeNodeTypes.size === 0 || activeNodeTypes.has(type)
                          ? "bg-slate-700 text-white"
                          : "bg-slate-800 text-slate-500"
                      )}
                    >
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: NODE_COLORS[type] || NODE_COLORS.Unknown }}
                      />
                      {type}
                      <span className="text-slate-400">({stats.nodeTypes[type]})</span>
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Loading overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-40 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm"
          >
            <div className="text-center">
              <Loader2 size={40} className="animate-spin text-primary-500 mx-auto mb-3" />
              <p className="text-white font-medium">Caricamento Knowledge Graph...</p>
              <p className="text-slate-400 text-sm mt-1">Esplorando le connessioni normative</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!isLoading && nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center max-w-md px-4">
            <Network size={64} className="mx-auto text-slate-600 mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">
              Esplora il Knowledge Graph
            </h3>
            <p className="text-slate-400 mb-6">
              Cerca un articolo per iniziare a esplorare la rete di norme, concetti e relazioni giuridiche.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {['1218', '2043', '832', '1337'].map((art) => (
                <button
                  key={art}
                  onClick={() => {
                    setSearchQuery(art);
                    loadSubgraph(`https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262:2~art${art}`);
                  }}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-white text-sm transition-colors"
                >
                  Art. {art} c.c.
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-red-400">
            <Info size={48} className="mx-auto mb-4" />
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Graph - Reagraph WebGL */}
      {nodes.length > 0 && (
        <GraphCanvas
          ref={graphRef}
          nodes={reagraphNodes}
          edges={reagraphEdges}
          selections={selections}
          actives={actives}
          onNodeClick={handleNodeClick}
          onCanvasClick={(e) => {
            onCanvasClick(e);
            setSelectedNode(null);
          }}
          theme={graphTheme}
          layoutType="forceDirected2d"
          clusterAttribute={clusterAttribute}
          edgeLabelPosition="natural"
          labelType="all"
          sizingType="attribute"
          sizingAttribute="size"
          draggable
          animated
          cameraMode="pan"
          layoutOverrides={{
            nodeStrength: -150,
            linkDistance: 80,
          }}
        />
      )}

      {/* Details panel */}
      <AnimatePresence>
        {selectedNode && (
          <NodeDetailsPanel
            node={selectedNode}
            neighbors={getNeighbors(selectedNode.id)}
            onClose={() => setSelectedNode(null)}
            onNavigate={handleNavigateToNode}
            onReportIssue={handleReportIssue}
            onReportEdge={handleReportEdge}
          />
        )}
      </AnimatePresence>

      {/* Report Issue Modal - supporta nodi e relazioni */}
      <ReportNodeIssueModal
        isOpen={showReportModal}
        onClose={() => {
          setShowReportModal(false);
          setReportIssueNode(null);
          setReportIssueEdge(null);
        }}
        node={reportIssueNode}
        edgeData={reportIssueEdge}
        userId={user?.id || 'anonymous'}
        onSuccess={() => {
          // Optionally refresh or show notification
        }}
      />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-20">
        <div className="bg-slate-900/90 backdrop-blur rounded-lg border border-slate-700 p-3">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
            Legenda
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {Object.entries(NODE_COLORS).slice(0, 6).map(([type, color]) => (
              <div key={type} className="flex items-center gap-1.5">
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-slate-400">{type}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Branding */}
      <div className="absolute bottom-4 right-4 z-20">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Sparkles size={12} />
          <span>MERL-T Knowledge Graph • WebGL</span>
        </div>
      </div>
    </div>
  );
}

export default KnowledgeGraphExplorer;
