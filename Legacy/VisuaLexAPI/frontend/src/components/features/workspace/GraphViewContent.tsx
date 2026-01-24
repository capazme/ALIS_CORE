/**
 * GraphViewContent
 * ================
 *
 * Componente wrapper per Knowledge Graph Explorer nel Workspace.
 * Integra il grafo come content type delle Workspace Tab.
 *
 * Features:
 * - Header con URN display e controlli depth
 * - Riutilizza KnowledgeGraphExplorer esistente
 * - Callback per selezione nodi
 * - Refresh e controlli depth integrati
 */

import { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Layers, Network, AlertCircle } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { KnowledgeGraphExplorer } from '../bulletin/KnowledgeGraphExplorer';
import type { SubgraphNode } from '../../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

export interface GraphViewContentProps {
  /** URN del nodo radice */
  rootUrn: string;
  /** Profondità di esplorazione (1-3) */
  depth?: number;
  /** Callback quando un nodo viene selezionato */
  onNodeSelect?: (node: SubgraphNode | null) => void;
  /** Classe CSS aggiuntiva */
  className?: string;
  /** Altezza personalizzata (default: 100%) */
  height?: number | string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function GraphViewContent({
  rootUrn,
  depth: initialDepth = 2,
  onNodeSelect,
  className,
  height = '100%',
}: GraphViewContentProps) {
  const [depth, setDepth] = useState(initialDepth);
  const [selectedNode, setSelectedNode] = useState<SubgraphNode | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Handle node selection
  const handleNodeClick = useCallback((urn: string) => {
    // This callback receives URN from KnowledgeGraphExplorer
    // We can pass it to parent or store locally
    // For now, we just update state (KnowledgeGraphExplorer already has its own selection UI)
    console.log('[GraphViewContent] Node clicked:', urn);
  }, []);

  // Notify parent when selection changes
  useEffect(() => {
    if (onNodeSelect) {
      onNodeSelect(selectedNode);
    }
  }, [selectedNode, onNodeSelect]);

  // Refresh graph
  const handleRefresh = useCallback(() => {
    setRefreshKey(prev => prev + 1);
  }, []);

  // Cycle depth (1 -> 2 -> 3 -> 1)
  const handleCycleDepth = useCallback(() => {
    setDepth(prev => {
      const next = prev === 3 ? 1 : prev + 1;
      setRefreshKey(k => k + 1); // Force refresh with new depth
      return next;
    });
  }, []);

  // Extract display label from URN
  const getDisplayLabel = (urn: string): string => {
    try {
      // Extract article number from URN
      // Example: https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262:2~art1218
      const match = urn.match(/art(\d+)/i);
      if (match) {
        return `Art. ${match[1]}`;
      }
      // If no article match, just show last segment
      const segments = urn.split(/[/:~]/);
      return segments[segments.length - 1];
    } catch {
      return 'Nodo radice';
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-50/50 dark:bg-slate-900/50 border-b border-slate-200/60 dark:border-slate-800 shrink-0">
        {/* Left: Root URN display */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="flex items-center justify-center w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-lg shrink-0">
            <Network size={16} className="text-primary-600 dark:text-primary-400" />
          </div>
          <div className="flex flex-col min-w-0">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              Knowledge Graph
            </span>
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-200 truncate">
              {getDisplayLabel(rootUrn)}
            </span>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Depth Selector */}
          <button
            onClick={handleCycleDepth}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all",
              "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-700",
              "hover:bg-slate-200 dark:hover:bg-slate-700",
              "text-slate-700 dark:text-slate-300 text-xs font-medium"
            )}
            title="Cambia profondità (1-3)"
          >
            <Layers size={14} />
            <span>Depth: {depth}</span>
          </button>

          {/* Refresh Button */}
          <button
            onClick={handleRefresh}
            className={cn(
              "p-1.5 rounded-lg border transition-all",
              "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-700",
              "hover:bg-slate-200 dark:hover:bg-slate-700",
              "text-slate-600 dark:text-slate-400"
            )}
            title="Ricarica grafo"
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {/* Graph Explorer */}
      <div className="flex-1 relative overflow-hidden">
        {rootUrn ? (
          <KnowledgeGraphExplorer
            key={`${rootUrn}-${depth}-${refreshKey}`}
            initialUrn={rootUrn}
            height={height}
            onArticleClick={handleNodeClick}
            className="w-full h-full"
          />
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <div className="w-16 h-16 bg-slate-50 dark:bg-slate-800/50 rounded-full flex items-center justify-center mb-3">
              <AlertCircle size={24} className="opacity-50" />
            </div>
            <p className="text-sm font-medium">Nessun URN specificato</p>
            <p className="text-xs text-slate-500 mt-1">
              Impossibile caricare il Knowledge Graph
            </p>
          </div>
        )}
      </div>

      {/* Footer info (optional) */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          className="shrink-0 px-4 py-2 bg-primary-50/50 dark:bg-primary-950/50 border-t border-primary-100 dark:border-primary-900"
        >
          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium text-primary-700 dark:text-primary-300">
              Selezione:
            </span>
            <span className="text-primary-600 dark:text-primary-400">
              {selectedNode.label}
            </span>
            <span className="px-1.5 py-0.5 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 rounded-full font-medium">
              {selectedNode.type}
            </span>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default GraphViewContent;
