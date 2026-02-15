import { useMemo, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Check, FileText, FolderOpen, List, Layers, ChevronsDown, ChevronsUp, Search } from 'lucide-react';
import { cn } from '../../../lib/utils';
import type { AnnexMetadata } from '../../../types';
import { useTour } from '../../../hooks/useTour';
import { buildHierarchicalTree, type HierarchicalNode } from '../../../utils/treeTransform';
import { useTreeExpansion } from '../../../hooks/useTreeExpansion';
import { TreeBranch } from './tree/TreeBranch';

// Normalize article ID for comparison (handles "3 bis" vs "3-bis")
function normalizeArticleId(id: string): string {
  if (!id) return id;
  return id.trim().toLowerCase().replace(/\s+/g, '-');
}

export interface TreeViewPanelProps {
  isOpen: boolean;
  onClose: () => void;
  treeData: unknown[];
  urn: string;
  title?: string;
  /** Callback when article is selected - includes target annex (null for dispositivo) */
  onArticleSelect?: (articleNumber: string, targetAnnex: string | null) => void;
  loadedArticles?: string[];
  /** Annex metadata for tabs - if provided, shows annex tabs */
  annexes?: AnnexMetadata[];
  /** Currently selected annex (from loaded articles) */
  currentAnnex?: string | null;
}

export function TreeViewPanel({
  isOpen,
  onClose,
  treeData,
  urn,
  title = 'Struttura Atto',
  onArticleSelect,
  loadedArticles = [],
  annexes,
  currentAnnex
}: TreeViewPanelProps) {
  const [selectedAnnex, setSelectedAnnex] = useState<string | null | undefined>(undefined);
  const [filterText, setFilterText] = useState('');
  const { tryStartTour } = useTour();

  // Tree expansion state
  const treeExpansion = useTreeExpansion(urn);

  // Start tree view tour on first open
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => tryStartTour('treeView'), 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, tryStartTour]);

  // Reset selected annex when panel opens or when currentAnnex changes externally
  useEffect(() => {
    setSelectedAnnex(undefined);
    setFilterText('');
  }, [currentAnnex, isOpen]);

  const effectiveAnnex = selectedAnnex !== undefined ? selectedAnnex : currentAnnex;
  const showAnnexes = annexes && annexes.length > 1;

  const handleAnnexTabClick = (annexNumber: string | null) => {
    if (annexNumber === effectiveAnnex) return;
    setSelectedAnnex(annexNumber);
  };

  // Build hierarchical tree from data
  const hierarchicalTree = useMemo(() => {
    if (treeData && treeData.length > 0) {
      return buildHierarchicalTree(treeData, effectiveAnnex ?? null);
    }
    return [];
  }, [treeData, effectiveAnnex]);

  // Filter tree nodes by search text
  const filteredTree = useMemo(() => {
    if (!filterText.trim()) return hierarchicalTree;

    const query = filterText.toLowerCase().trim();

    function filterNode(node: HierarchicalNode): HierarchicalNode | null {
      // Article node - check if number matches
      if (node.type === 'articolo') {
        const matches =
          (node.numero && node.numero.toLowerCase().includes(query)) ||
          node.label.toLowerCase().includes(query);
        return matches ? node : null;
      }

      // Section node - filter children recursively
      const filteredChildren = node.children
        .map(filterNode)
        .filter((n): n is HierarchicalNode => n !== null);

      // Include section if it has matching children OR its label matches
      if (filteredChildren.length > 0 || node.label.toLowerCase().includes(query)) {
        return {
          ...node,
          children: filteredChildren.length > 0 ? filteredChildren : node.children,
          articleCount: filteredChildren.reduce((sum, c) => sum + c.articleCount, 0),
        };
      }

      return null;
    }

    return hierarchicalTree
      .map(filterNode)
      .filter((n): n is HierarchicalNode => n !== null);
  }, [hierarchicalTree, filterText]);

  // Collect all section node IDs for expand/collapse all
  const allSectionIds = useMemo(() => {
    const ids: string[] = [];
    function collect(nodes: HierarchicalNode[]) {
      for (const node of nodes) {
        if (node.type !== 'articolo' && node.children.length > 0) {
          ids.push(node.id);
          collect(node.children);
        }
      }
    }
    collect(hierarchicalTree);
    return ids;
  }, [hierarchicalTree]);

  // Check if tree has section structure (not just flat articles)
  const hasHierarchy = useMemo(() => {
    return hierarchicalTree.some(n => n.type !== 'articolo');
  }, [hierarchicalTree]);

  // Stats
  const stats = useMemo(() => {
    function countArticles(nodes: HierarchicalNode[]): number {
      return nodes.reduce((sum, n) => {
        if (n.type === 'articolo') return sum + 1;
        return sum + countArticles(n.children);
      }, 0);
    }
    return { total: countArticles(hierarchicalTree), loaded: loadedArticles.length };
  }, [hierarchicalTree, loadedArticles]);

  // Create normalized set for comparison
  const loadedSetNormalized = useMemo(
    () => new Set(loadedArticles.map(normalizeArticleId)),
    [loadedArticles]
  );

  const isArticleLoaded = useCallback((articleNum: string) => {
    const uniqueIdForContext = effectiveAnnex
      ? `all${effectiveAnnex}:${articleNum}`
      : articleNum;
    return loadedSetNormalized.has(normalizeArticleId(uniqueIdForContext));
  }, [loadedSetNormalized, effectiveAnnex]);

  const handleArticleSelect = useCallback((articleNumber: string) => {
    onArticleSelect?.(articleNumber, effectiveAnnex ?? null);
  }, [onArticleSelect, effectiveAnnex]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/40 backdrop-blur-sm z-[100]"
          />

          {/* Sidebar Panel */}
          <motion.aside
            initial={{ x: '100%', opacity: 0.9 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0.9 }}
            transition={{ type: 'spring', damping: 35, stiffness: 400 }}
            className="fixed right-0 top-0 bottom-0 w-full sm:w-[450px] bg-white dark:bg-slate-900 shadow-2xl z-[100] flex flex-col border-l border-slate-200 dark:border-slate-800"
          >
            {/* Header */}
            <div className="sticky top-0 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200 dark:border-slate-800 px-6 py-5 z-10">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 shadow-sm border border-primary-200/50 dark:border-primary-800/50">
                    <List size={22} />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-slate-900 dark:text-white leading-tight uppercase tracking-tight">{title}</h3>
                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mt-0.5">Indice Strutturale</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-all border border-transparent hover:border-slate-200 dark:hover:border-slate-700 shadow-sm"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Stats */}
              {stats.total > 0 && (
                <div id="tour-tree-stats" className="flex items-center gap-4 mb-3">
                  <div className="px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center gap-2">
                    <FileText size={14} className="text-slate-500" />
                    <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                      {stats.total} <span className="opacity-60 font-medium">Articoli totali</span>
                    </span>
                  </div>
                  <div className="px-3 py-1.5 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg flex items-center gap-2 border border-emerald-100/50 dark:border-emerald-800/30">
                    <Check size={14} className="text-emerald-500" />
                    <span className="text-xs font-bold text-emerald-700 dark:text-emerald-400">
                      {stats.loaded} <span className="opacity-60 font-medium uppercase text-[10px]">Caricati</span>
                    </span>
                  </div>
                </div>
              )}

              {/* Filter + Expand/Collapse controls */}
              {hasHierarchy && (
                <div className="flex items-center gap-2">
                  <div className="flex-1 relative">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                      type="text"
                      value={filterText}
                      onChange={(e) => setFilterText(e.target.value)}
                      placeholder="Filtra articoli..."
                      className="w-full pl-9 pr-3 py-1.5 text-sm rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 placeholder-slate-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                    />
                  </div>
                  <button
                    onClick={() => treeExpansion.expandAll(allSectionIds)}
                    className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 transition-colors"
                    title="Espandi tutto"
                  >
                    <ChevronsDown size={16} />
                  </button>
                  <button
                    onClick={() => treeExpansion.collapseAll()}
                    className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 transition-colors"
                    title="Comprimi tutto"
                  >
                    <ChevronsUp size={16} />
                  </button>
                </div>
              )}
            </div>

            {/* Annex Tabs */}
            {showAnnexes && (
              <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-gradient-to-b from-slate-50/80 to-white dark:from-slate-800/50 dark:to-slate-900">
                <div className="flex items-center gap-2 mb-3">
                  <Layers size={14} className="text-primary-500" />
                  <span className="text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider">
                    Sezioni documento
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {annexes!.map((annex) => {
                    const isActive = effectiveAnnex === annex.number ||
                      (effectiveAnnex === null && annex.number === null);

                    return (
                      <motion.button
                        key={annex.number ?? 'main'}
                        onClick={() => handleAnnexTabClick(annex.number)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        layout
                        className={cn(
                          "relative px-3 py-2 text-xs font-semibold rounded-xl transition-colors border overflow-hidden",
                          isActive
                            ? "bg-gradient-to-br from-primary-500 to-primary-600 text-white border-primary-500 shadow-lg shadow-primary-500/25"
                            : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-primary-300 dark:hover:border-primary-600 hover:text-primary-600 dark:hover:text-primary-400 hover:shadow-md"
                        )}
                      >
                        {isActive && (
                          <motion.div
                            layoutId="annexActiveGlow"
                            className="absolute inset-0 bg-gradient-to-br from-primary-400/20 to-transparent"
                            transition={{ type: 'spring', damping: 30, stiffness: 400 }}
                          />
                        )}
                        <span className="relative block truncate max-w-[140px]">{annex.label}</span>
                        <span className={cn(
                          "relative text-[10px] block mt-0.5 font-medium",
                          isActive ? "text-primary-100" : "text-slate-400 dark:text-slate-500"
                        )}>
                          {annex.article_count} articoli
                        </span>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Scrollable Content */}
            <div id="tour-tree-structure" className="flex-1 overflow-y-auto p-4 custom-scrollbar">
              {filteredTree.length > 0 ? (
                <motion.div
                  key={effectiveAnnex ?? 'dispositivo'}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="pb-10"
                >
                  <div role="tree" aria-label="Struttura normativa">
                    <TreeBranch
                      nodes={filteredTree}
                      isExpanded={treeExpansion.isExpanded}
                      onToggle={treeExpansion.toggle}
                      onArticleSelect={handleArticleSelect}
                      isArticleLoaded={isArticleLoaded}
                    />
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center py-20 text-slate-400"
                >
                  <List size={48} className="opacity-20 mb-4" />
                  <p className="text-sm font-medium">
                    {filterText ? 'Nessun risultato trovato' : 'Struttura non ancora disponibile'}
                  </p>
                </motion.div>
              )}
            </div>

            {/* Footer shadow fade */}
            <div className="h-10 bg-gradient-to-t from-white dark:from-slate-900 to-transparent pointer-events-none sticky bottom-0" />
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
