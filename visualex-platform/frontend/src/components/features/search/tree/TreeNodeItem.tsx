/**
 * TreeNodeItem - Single collapsible tree node with chevron, label, and article count badge.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, FolderOpen, FileText, Check } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import type { HierarchicalNode } from '../../../../utils/treeTransform';

export interface TreeNodeItemProps {
  node: HierarchicalNode;
  isExpanded: boolean;
  onToggle: () => void;
  onArticleSelect?: (articleNumber: string) => void;
  isArticleLoaded?: (articleNum: string) => boolean;
  depth: number;
}

const TYPE_COLORS: Record<string, { bg: string; text: string; icon: string }> = {
  libro: { bg: 'bg-indigo-100 dark:bg-indigo-900/30', text: 'text-indigo-700 dark:text-indigo-400', icon: 'text-indigo-600 dark:text-indigo-400' },
  titolo: { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400', icon: 'text-blue-600 dark:text-blue-400' },
  capo: { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400', icon: 'text-amber-600 dark:text-amber-400' },
  sezione: { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-400', icon: 'text-emerald-600 dark:text-emerald-400' },
  group: { bg: 'bg-slate-100 dark:bg-slate-800', text: 'text-slate-700 dark:text-slate-300', icon: 'text-slate-600 dark:text-slate-400' },
};

export function TreeNodeItem({
  node,
  isExpanded,
  onToggle,
  onArticleSelect,
  isArticleLoaded,
  depth,
}: TreeNodeItemProps) {
  // Article leaf node
  if (node.type === 'articolo') {
    const loaded = isArticleLoaded?.(node.numero || '') ?? false;
    const isClickable = onArticleSelect && !loaded;

    return (
      <motion.button
        onClick={() => isClickable && node.numero && onArticleSelect(node.numero)}
        disabled={!isClickable}
        role="treeitem"
        aria-label={`Articolo ${node.numero || node.label}`}
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.15 }}
        whileHover={isClickable ? { x: 2 } : {}}
        className={cn(
          "flex items-center gap-2 w-full px-3 py-1.5 text-left rounded-lg transition-colors text-sm",
          loaded
            ? "bg-emerald-50/80 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400"
            : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer",
          !isClickable && !loaded && "opacity-50 cursor-default"
        )}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
      >
        <FileText size={14} className={cn(
          loaded ? "text-emerald-500" : "text-slate-400 dark:text-slate-500"
        )} />
        <span className="font-medium">{node.numero || node.label}</span>
        {loaded && (
          <div className="ml-auto w-4 h-4 bg-emerald-500 rounded-full flex items-center justify-center">
            <Check size={10} className="text-white" strokeWidth={3} />
          </div>
        )}
      </motion.button>
    );
  }

  // Section node (collapsible)
  const colors = TYPE_COLORS[node.type] || TYPE_COLORS.group;
  const hasChildren = node.children.length > 0;

  return (
    <button
      onClick={onToggle}
      role="treeitem"
      aria-expanded={isExpanded}
      className={cn(
        "flex items-center gap-2 w-full text-left rounded-lg transition-colors group",
        "px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-800/50",
      )}
      style={{ paddingLeft: `${depth * 16 + 4}px` }}
    >
      {/* Chevron */}
      <motion.div
        animate={{ rotate: isExpanded ? 90 : 0 }}
        transition={{ duration: 0.15 }}
        className={cn(
          "w-5 h-5 flex items-center justify-center rounded",
          hasChildren ? "text-slate-500 dark:text-slate-400" : "invisible"
        )}
      >
        <ChevronRight size={14} />
      </motion.div>

      {/* Icon */}
      <div className={cn("w-6 h-6 rounded flex items-center justify-center flex-shrink-0", colors.bg)}>
        <FolderOpen size={14} className={colors.icon} />
      </div>

      {/* Label */}
      <div className="flex-1 min-w-0">
        <span className={cn("text-sm font-semibold leading-tight line-clamp-2", colors.text)}>
          {node.label}
        </span>
      </div>

      {/* Article count badge */}
      <AnimatePresence>
        {node.articleCount > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="ml-auto text-[10px] font-bold text-slate-400 dark:text-slate-500 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-full flex-shrink-0"
          >
            {node.articleCount}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
