/**
 * ExpertAccordion - Multi-open accordion with compare grid for expert traces.
 *
 * Click toggles individual experts without closing others.
 * "Confronta tutti" expands all 4.
 * When >=2 open and panel >=480px: 2-column grid.
 * Debug info (searchQuery, duration_ms) in collapsible "Dettagli tecnici".
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, BookOpen, TrendingUp, Scale, Gavel, Columns2, ChevronDown } from 'lucide-react';
import { cn } from '../../lib/utils';
import { ConfidenceMeter } from './ConfidenceMeter';
import { ExpertFeedbackPanel } from './ExpertFeedbackPanel';
import type { ExpertTraceEntry } from '../../types/trace';
import { EXPERT_CONFIG, type ExpertId } from '../../types/pipeline';

export interface ExpertAccordionProps {
  experts: ExpertTraceEntry[];
  traceId?: string;
  className?: string;
}

const ICON_MAP: Record<string, React.ComponentType<{ size?: number; className?: string; style?: React.CSSProperties }>> = {
  BookOpen,
  TrendingUp,
  Scale,
  Gavel,
};

export function ExpertAccordion({ experts, traceId, className }: ExpertAccordionProps) {
  const [expandedIds, setExpandedIds] = useState(new Set() as Set<ExpertId>);
  const [showTechDetails, setShowTechDetails] = useState(false);
  const containerRef = useRef(null as HTMLDivElement | null);
  const [isWide, setIsWide] = useState(false);

  // Track container width for grid layout
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width >= 480);
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const toggle = useCallback((id: ExpertId) => {
    setExpandedIds((prev: Set<ExpertId>) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedIds(new Set(experts.map((e) => e.expertId)));
  }, [experts]);

  const collapseAll = useCallback(() => {
    setExpandedIds(new Set());
  }, []);

  if (experts.length === 0) {
    return (
      <div className={cn("text-sm text-slate-500 text-center py-6", className)}>
        Nessuna analisi esperta disponibile
      </div>
    );
  }

  const allExpanded = expandedIds.size === experts.length;
  const multiOpen = expandedIds.size >= 2;
  const useGrid = multiOpen && isWide;

  return (
    <div ref={containerRef} className={cn("space-y-2", className)}>
      {/* Compare all / Collapse all button */}
      <div className="flex justify-end">
        <button
          onClick={allExpanded ? collapseAll : expandAll}
          className={cn(
            'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
            'text-slate-500 hover:text-blue-600 hover:bg-blue-50',
            'dark:hover:text-blue-400 dark:hover:bg-blue-900/20',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          )}
        >
          <Columns2 size={12} aria-hidden="true" />
          {allExpanded ? 'Chiudi tutti' : 'Confronta tutti'}
        </button>
      </div>

      {/* Expert cards - grid when >=2 open and wide */}
      <div className={cn(useGrid && 'grid grid-cols-2 auto-rows-fr gap-2', !useGrid && 'space-y-2')}>
        {experts.map((expert) => {
          const config = EXPERT_CONFIG[expert.expertId];
          const IconComponent = ICON_MAP[config?.icon || 'BookOpen'] || BookOpen;
          const isExpanded = expandedIds.has(expert.expertId);

          return (
            <div
              key={expert.expertId}
              className={cn(
                'rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden',
                useGrid && 'flex flex-col',
              )}
            >
              {/* Header */}
              <button
                onClick={() => toggle(expert.expertId)}
                aria-expanded={isExpanded}
                aria-controls={`expert-panel-${expert.expertId}`}
                className={cn(
                  'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset',
                  isExpanded
                    ? 'bg-slate-50 dark:bg-slate-800/50'
                    : 'bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800/30',
                )}
              >
                <motion.div
                  animate={{ rotate: isExpanded ? 90 : 0 }}
                  transition={{ duration: 0.15 }}
                >
                  <ChevronRight size={14} className="text-slate-400" aria-hidden="true" />
                </motion.div>

                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                  style={{ backgroundColor: `${config?.color || '#64748b'}15` }}
                >
                  <IconComponent size={16} style={{ color: config?.color || '#64748b' }} aria-hidden="true" />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                      {expert.displayName}
                    </span>
                    <span className="text-[10px] font-medium text-slate-400">
                      peso: {Math.round(expert.weight * 100)}%
                    </span>
                  </div>
                  <ConfidenceMeter value={expert.confidence} size="sm" showLabel={false} className="mt-1 max-w-[120px]" />
                </div>

                <span
                  className="text-sm font-bold shrink-0"
                  style={{ color: config?.color || '#64748b' }}
                >
                  {Math.round(expert.confidence * 100)}%
                </span>
              </button>

              {/* Expanded content */}
              <AnimatePresence initial={false}>
                {isExpanded && (
                  <motion.div
                    id={`expert-panel-${expert.expertId}`}
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className={cn('overflow-hidden', useGrid && 'flex-1')}
                    role="region"
                    aria-label={`Dettagli ${expert.displayName}`}
                  >
                    <div className="px-4 py-3 border-t border-slate-100 dark:border-slate-800 space-y-3">
                      {/* Interpretation */}
                      <div>
                        <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">Interpretazione</h4>
                        <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                          {expert.interpretation}
                        </p>
                      </div>

                      {/* Reasoning */}
                      {expert.reasoning && (
                        <div>
                          <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">Ragionamento</h4>
                          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                            {expert.reasoning}
                          </p>
                        </div>
                      )}

                      {/* Sources used */}
                      {expert.sources.length > 0 && (
                        <div>
                          <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">
                            Fonti ({expert.sources.length})
                          </h4>
                          <div className="flex flex-wrap gap-1">
                            {expert.sources.map((s) => (
                              <span key={s} className="text-[10px] px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 rounded text-slate-600 dark:text-slate-400">
                                {s}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Expert Feedback */}
                      {traceId && (
                        <ExpertFeedbackPanel
                          traceId={traceId}
                          expertId={expert.expertId}
                        />
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* Tech details — collapsed by default, dev-only info */}
      {experts.some((e) => e.searchQuery || e.duration_ms !== undefined) && (
        <div className="border-t border-slate-100 dark:border-slate-800 pt-2">
          <button
            onClick={() => setShowTechDetails(!showTechDetails)}
            className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
          >
            {showTechDetails ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            Dettagli tecnici
          </button>
          {showTechDetails && (
            <div className="mt-2 space-y-2 text-[10px] text-slate-400">
              {experts.map((expert) => (
                <div key={`tech-${expert.expertId}`} className="space-y-0.5">
                  <span className="font-medium">{expert.displayName}:</span>
                  {expert.searchQuery && (
                    <code className="block pl-2 text-slate-500 bg-slate-50 dark:bg-slate-800 px-2 py-0.5 rounded">
                      {expert.searchQuery}
                    </code>
                  )}
                  {expert.duration_ms !== undefined && (
                    <span className="pl-2 block">{expert.duration_ms}ms</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
