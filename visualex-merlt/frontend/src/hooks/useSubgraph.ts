/**
 * useSubgraph Hook
 * ===============
 *
 * Hook per caricare e gestire un subgraph dal Knowledge Graph MERL-T.
 * Supporta:
 * - Caricamento iniziale di un nodo connesso
 * - Espansione dinamica dei nodi
 * - Filtri per tipo
 * - Cache locale per performance
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { merltService } from '../services/merltService';
import type { SubgraphNode, SubgraphEdge } from '../types/merlt';

export interface UseSubgraphOptions {
  initialUrn?: string;
  depth?: number;
  enabled?: boolean;
  filters?: string[];
}

export interface SubgraphData {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
}

export interface UseSubgraphReturn {
  data: SubgraphData;
  loading: boolean;
  error: string | null;
  selectedNode: SubgraphNode | null;
  setSelectedNode: (node: SubgraphNode | null) => void;
  expandNode: (nodeId: string) => Promise<void>;
  refresh: () => Promise<void>;
  setFilters: (filters: string[]) => void;
  filters: string[];
}

export function useSubgraph(options: UseSubgraphOptions = {}): UseSubgraphReturn {
  const { initialUrn, depth = 2, enabled = true, filters: initialFilters = [] } = options;

  const [data, setData] = useState({
    nodes: [] as SubgraphNode[],
    edges: [] as SubgraphEdge[],
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [selectedNode, setSelectedNode] = useState(null as SubgraphNode | null);
  const [filters, setFilters] = useState(initialFilters as string[]);

  const refresh = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);

    try {
      let response;
      if (initialUrn) {
        response = await merltService.getSubgraph({
          root_urn: initialUrn,
          depth,
          entity_types: filters.length ? filters : undefined,
        });
      } else {
        // No root URN: load graph overview (most-connected nodes)
        response = await merltService.getGraphOverview(50);
      }
      setData(response);
    } catch (err) {
      console.error('Error fetching subgraph:', err);
      setError(err instanceof Error ? err.message : 'Errore nel caricamento del grafo');
    } finally {
      setLoading(false);
    }
  }, [enabled, initialUrn, depth, filters]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const expandNode = useCallback(async (nodeId: string) => {
    try {
      setLoading(true);
      const response = await merltService.getSubgraph({
        root_urn: nodeId,
        depth: 1,
        entity_types: filters.length ? filters : undefined,
      });

      setData((prev: SubgraphData) => {
        const existingNodeIds = new Set(prev.nodes.map(n => n.id));
        const existingEdgeIds = new Set(prev.edges.map(e => e.id));

        const newNodes = response.nodes.filter((n: SubgraphNode) => !existingNodeIds.has(n.id));
        const newEdges = response.edges.filter((e: SubgraphEdge) => !existingEdgeIds.has(e.id));

        return {
          nodes: [...prev.nodes, ...newNodes],
          edges: [...prev.edges, ...newEdges],
        };
      });
    } catch (err) {
      console.error('Error expanding node:', err);
      setError(err instanceof Error ? err.message : 'Errore nell\'espansione del nodo');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const filteredData = useMemo(() => {
    if (!filters.length) return data;

    const nodes = data.nodes.filter((n: SubgraphNode) => filters.includes(n.type));
    const nodeIds = new Set(nodes.map((n: SubgraphNode) => n.id));
    const edges = data.edges.filter((e: SubgraphEdge) => nodeIds.has(e.source as string) && nodeIds.has(e.target as string));

    return { nodes, edges };
  }, [data, filters]);

  return {
    data: filteredData,
    loading,
    error,
    selectedNode,
    setSelectedNode,
    expandNode,
    refresh,
    setFilters,
    filters,
  };
}

export default useSubgraph;
