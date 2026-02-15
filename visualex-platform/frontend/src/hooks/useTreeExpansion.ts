/**
 * useTreeExpansion - Manages expand/collapse state for tree nodes.
 * Persists state in sessionStorage keyed by tree URN.
 */

import { useState, useCallback, useMemo } from 'react';

const STORAGE_PREFIX = 'tree-expansion:';

function getStorageKey(urn: string): string {
  return `${STORAGE_PREFIX}${urn}`;
}

function loadExpandedSet(urn: string): Set<string> {
  try {
    const stored = sessionStorage.getItem(getStorageKey(urn));
    if (stored) {
      return new Set(JSON.parse(stored));
    }
  } catch {
    // Ignore parse errors
  }
  return new Set<string>();
}

function saveExpandedSet(urn: string, expanded: Set<string>): void {
  try {
    sessionStorage.setItem(getStorageKey(urn), JSON.stringify([...expanded]));
  } catch {
    // Ignore storage errors
  }
}

export interface UseTreeExpansionReturn {
  /** Set of currently expanded node IDs */
  expandedIds: Set<string>;
  /** Check if a node is expanded */
  isExpanded: (id: string) => boolean;
  /** Toggle a single node */
  toggle: (id: string) => void;
  /** Expand all nodes */
  expandAll: (allIds: string[]) => void;
  /** Collapse all nodes */
  collapseAll: () => void;
}

export function useTreeExpansion(urn: string): UseTreeExpansionReturn {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => loadExpandedSet(urn));

  const isExpanded = useCallback((id: string) => expandedIds.has(id), [expandedIds]);

  const toggle = useCallback((id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      saveExpandedSet(urn, next);
      return next;
    });
  }, [urn]);

  const expandAll = useCallback((allIds: string[]) => {
    const next = new Set(allIds);
    saveExpandedSet(urn, next);
    setExpandedIds(next);
  }, [urn]);

  const collapseAll = useCallback(() => {
    const next = new Set<string>();
    saveExpandedSet(urn, next);
    setExpandedIds(next);
  }, [urn]);

  return useMemo(() => ({
    expandedIds,
    isExpanded,
    toggle,
    expandAll,
    collapseAll,
  }), [expandedIds, isExpanded, toggle, expandAll, collapseAll]);
}
