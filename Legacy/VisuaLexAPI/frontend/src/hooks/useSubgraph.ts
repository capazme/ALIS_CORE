/**
 * useSubgraph Hook
 * ================
 *
 * Hook per il caricamento e gestione del subgraph dal Knowledge Graph MERL-T.
 * Usato per la visualizzazione interattiva del grafo nella Bacheca.
 *
 * Features:
 * - Caricamento subgraph da un nodo radice
 * - Espansione dinamica dei nodi
 * - Filtro per tipo entità/relazione
 * - Caching delle query
 * - Stato loading/error
 */

import { useState, useCallback, useRef, useMemo } from 'react';
import { merltService } from '../services/merltService';
import type {
  SubgraphRequest,
  SubgraphNode,
  SubgraphEdge,
} from '../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface SubgraphState {
  /** Tutti i nodi caricati (accumulati dalle espansioni) */
  nodes: SubgraphNode[];
  /** Tutti gli archi caricati */
  edges: SubgraphEdge[];
  /** URN del nodo radice corrente */
  rootUrn: string | null;
  /** Loading state */
  isLoading: boolean;
  /** Espansione in corso per un nodo specifico */
  expandingNodeId: string | null;
  /** Error message */
  error: string | null;
  /** Metadati ultima query */
  lastQueryTime: number | null;
  /** Nodi già espansi (per evitare ricaricamenti) */
  expandedNodeIds: Set<string>;
}

interface UseSubgraphOptions {
  /** Profondità iniziale di esplorazione (default 2) */
  initialDepth?: number;
  /** Max nodi per query (default 50) */
  maxNodes?: number;
  /** Include metadati extra nei nodi */
  includeMetadata?: boolean;
  /** Carica automaticamente al mount se rootUrn è specificato */
  autoLoad?: boolean;
}

interface UseSubgraphReturn extends SubgraphState {
  /** Carica subgraph da un nodo radice */
  loadSubgraph: (rootUrn: string, options?: Partial<SubgraphRequest>) => Promise<void>;
  /** Espandi un nodo (carica i suoi vicini) */
  expandNode: (nodeId: string) => Promise<void>;
  /** Reset completo dello stato */
  reset: () => void;
  /** Filtra nodi per tipo */
  filterNodesByType: (types: string[]) => SubgraphNode[];
  /** Filtra archi per tipo */
  filterEdgesByType: (types: string[]) => SubgraphEdge[];
  /** Statistiche grafo */
  stats: {
    nodeCount: number;
    edgeCount: number;
    nodeTypes: Record<string, number>;
    edgeTypes: Record<string, number>;
  };
  /** Trova nodo per ID */
  getNodeById: (id: string) => SubgraphNode | undefined;
  /** Trova vicini di un nodo */
  getNeighbors: (nodeId: string) => { nodes: SubgraphNode[]; edges: SubgraphEdge[] };
}

// =============================================================================
// INITIAL STATE
// =============================================================================

const initialState: SubgraphState = {
  nodes: [],
  edges: [],
  rootUrn: null,
  isLoading: false,
  expandingNodeId: null,
  error: null,
  lastQueryTime: null,
  expandedNodeIds: new Set(),
};

// =============================================================================
// HOOK
// =============================================================================

export function useSubgraph(options: UseSubgraphOptions = {}): UseSubgraphReturn {
  const {
    initialDepth = 2,
    maxNodes = 50,
    includeMetadata = true,
  } = options;

  const [state, setState] = useState<SubgraphState>(initialState);

  // Refs per evitare stale closures
  const nodesMapRef = useRef<Map<string, SubgraphNode>>(new Map());
  const edgesMapRef = useRef<Map<string, SubgraphEdge>>(new Map());

  // ==========================================================================
  // LOAD SUBGRAPH
  // ==========================================================================

  const loadSubgraph = useCallback(async (
    rootUrn: string,
    requestOptions?: Partial<SubgraphRequest>
  ) => {
    setState(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      rootUrn,
    }));

    try {
      const response = await merltService.getSubgraph({
        root_urn: rootUrn,
        depth: requestOptions?.depth ?? initialDepth,
        max_nodes: requestOptions?.max_nodes ?? maxNodes,
        include_metadata: requestOptions?.include_metadata ?? includeMetadata,
        entity_types: requestOptions?.entity_types,
        relation_types: requestOptions?.relation_types,
      });

      // Reset maps e popola con nuovi dati
      nodesMapRef.current.clear();
      edgesMapRef.current.clear();

      response.nodes.forEach(node => {
        nodesMapRef.current.set(node.id, node);
      });

      response.edges.forEach(edge => {
        edgesMapRef.current.set(edge.id, edge);
      });

      setState(prev => ({
        ...prev,
        nodes: response.nodes,
        edges: response.edges,
        isLoading: false,
        lastQueryTime: response.metadata?.query_time_ms ?? null,
        expandedNodeIds: new Set([rootUrn]),
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Errore nel caricamento del grafo',
      }));
    }
  }, [initialDepth, maxNodes, includeMetadata]);

  // ==========================================================================
  // EXPAND NODE
  // ==========================================================================

  const expandNode = useCallback(async (nodeId: string) => {
    // Se già espanso, skip
    if (state.expandedNodeIds.has(nodeId)) {
      return;
    }

    setState(prev => ({
      ...prev,
      expandingNodeId: nodeId,
      error: null,
    }));

    try {
      const response = await merltService.getSubgraph({
        root_urn: nodeId,
        depth: 1,
        max_nodes: maxNodes,
        include_metadata: includeMetadata,
      });

      // Merge nuovi nodi/archi (evita duplicati)
      response.nodes.forEach(node => {
        if (!nodesMapRef.current.has(node.id)) {
          nodesMapRef.current.set(node.id, node);
        }
      });

      response.edges.forEach(edge => {
        if (!edgesMapRef.current.has(edge.id)) {
          edgesMapRef.current.set(edge.id, edge);
        }
      });

      setState(prev => ({
        ...prev,
        nodes: Array.from(nodesMapRef.current.values()),
        edges: Array.from(edgesMapRef.current.values()),
        expandingNodeId: null,
        expandedNodeIds: new Set([...prev.expandedNodeIds, nodeId]),
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        expandingNodeId: null,
        error: err instanceof Error ? err.message : 'Errore nell\'espansione del nodo',
      }));
    }
  }, [state.expandedNodeIds, maxNodes, includeMetadata]);

  // ==========================================================================
  // RESET
  // ==========================================================================

  const reset = useCallback(() => {
    nodesMapRef.current.clear();
    edgesMapRef.current.clear();
    setState(initialState);
  }, []);

  // ==========================================================================
  // FILTERS
  // ==========================================================================

  const filterNodesByType = useCallback((types: string[]): SubgraphNode[] => {
    if (!types.length) return state.nodes;
    const typeSet = new Set(types.map(t => t.toLowerCase()));
    return state.nodes.filter(n => typeSet.has(n.type.toLowerCase()));
  }, [state.nodes]);

  const filterEdgesByType = useCallback((types: string[]): SubgraphEdge[] => {
    if (!types.length) return state.edges;
    const typeSet = new Set(types.map(t => t.toUpperCase()));
    return state.edges.filter(e => typeSet.has(e.type.toUpperCase()));
  }, [state.edges]);

  // ==========================================================================
  // HELPERS
  // ==========================================================================

  const getNodeById = useCallback((id: string): SubgraphNode | undefined => {
    return nodesMapRef.current.get(id);
  }, []);

  const getNeighbors = useCallback((nodeId: string): { nodes: SubgraphNode[]; edges: SubgraphEdge[] } => {
    const neighborEdges = state.edges.filter(
      e => e.source === nodeId || e.target === nodeId
    );
    const neighborIds = new Set<string>();
    neighborEdges.forEach(e => {
      if (e.source !== nodeId) neighborIds.add(e.source);
      if (e.target !== nodeId) neighborIds.add(e.target);
    });
    const neighborNodes = state.nodes.filter(n => neighborIds.has(n.id));
    return { nodes: neighborNodes, edges: neighborEdges };
  }, [state.nodes, state.edges]);

  // ==========================================================================
  // STATS (memoized)
  // ==========================================================================

  const stats = useMemo(() => {
    const nodeTypes: Record<string, number> = {};
    const edgeTypes: Record<string, number> = {};

    state.nodes.forEach(n => {
      nodeTypes[n.type] = (nodeTypes[n.type] || 0) + 1;
    });

    state.edges.forEach(e => {
      edgeTypes[e.type] = (edgeTypes[e.type] || 0) + 1;
    });

    return {
      nodeCount: state.nodes.length,
      edgeCount: state.edges.length,
      nodeTypes,
      edgeTypes,
    };
  }, [state.nodes, state.edges]);

  // ==========================================================================
  // RETURN
  // ==========================================================================

  return {
    ...state,
    loadSubgraph,
    expandNode,
    reset,
    filterNodesByType,
    filterEdgesByType,
    stats,
    getNodeById,
    getNeighbors,
  };
}

export default useSubgraph;
