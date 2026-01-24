/**
 * GraphViewContent
 * ================
 *
 * Wrapper per Knowledge Graph Explorer nel Workspace.
 */

import { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Layers, Network, AlertCircle } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { KnowledgeGraphExplorer } from '../bulletin/KnowledgeGraphExplorer';
import type { SubgraphNode } from '../../../types/merlt';

export interface GraphViewContentProps {
  rootUrn: string;
  depth?: number;
  onNodeSelect?: (node: SubgraphNode | null) => void;
  className?: string;
  height?: number | string;
  userId?: string;
}

export function GraphViewContent({
  rootUrn,
  depth: initialDepth = 2,
  onNodeSelect,
  className,
  height = '100%',
  userId,
}: GraphViewContentProps) {
  const [depth, setDepth] = useState(initialDepth);
  const [selectedNode, setSelectedNode] = useState<SubgraphNode | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleNodeClick = useCallback((urn: string) => {
    console.log('[GraphViewContent] Node clicked:', urn);
  }, []);

  useEffect(() => {
    if (onNodeSelect) {
      onNodeSelect(selectedNode);
    }
  }, [selectedNode, onNodeSelect]);

  const handleRefresh = useCallback(() => {
    setRefreshKey(prev => prev + 1);
  }, []);

  const handleCycleDepth = useCallback(() => {
    setDepth(prev => {
      const next = prev === 3 ? 1 : prev + 1;
      setRefreshKey(k => k + 1);
      return next;
    });
  }, []);

  const getDisplayLabel = (urn: string): string => {
    try {
      const match = urn.match(/art(\d+)/i);
      if (match) {
        return `Art. ${match[1]}`;
      }
      const segments = urn.split(/[/:~]/);
      return segments[segments.length - 1];
    } catch {
      return 'Nodo radice';
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      <div className="flex items-center justify-between px-4 py-3 bg-slate-50/50 dark:bg-slate-900/50 border-b border-slate-200/60 dark:border-slate-800 shrink-0">
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

        <div className="flex items-center gap-2 shrink-0">
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

      <div className="flex-1 relative overflow-hidden">
        {rootUrn ? (
          <KnowledgeGraphExplorer
            key={`${rootUrn}-${depth}-${refreshKey}`}
            initialUrn={rootUrn}
            height={height}
            onArticleClick={handleNodeClick}
            className="w-full h-full"
            userId={userId}
          />
        ) : (
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
