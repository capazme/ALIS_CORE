/**
 * ExpertAccordion - Accordion with 4 sections for expert trace details.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, BookOpen, TrendingUp, Scale, Gavel, Clock } from 'lucide-react';
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
  const [expandedId, setExpandedId] = useState(null as ExpertId | null);

  const toggle = (id: ExpertId) => {
    setExpandedId((prev: ExpertId | null) => prev === id ? null : id);
  };

  if (experts.length === 0) {
    return (
      <div className={cn("text-sm text-slate-500 text-center py-6", className)}>
        Nessuna analisi esperta disponibile
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {experts.map(expert => {
        const config = EXPERT_CONFIG[expert.expertId];
        const IconComponent = ICON_MAP[config?.icon || 'BookOpen'] || BookOpen;
        const isExpanded = expandedId === expert.expertId;

        return (
          <div key={expert.expertId} className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
            {/* Header */}
            <button
              onClick={() => toggle(expert.expertId)}
              aria-expanded={isExpanded}
              aria-controls={`expert-panel-${expert.expertId}`}
              className={cn(
                "w-full flex items-center gap-3 px-4 py-3 text-left transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset",
                isExpanded
                  ? "bg-slate-50 dark:bg-slate-800/50"
                  : "bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800/30"
              )}
            >
              {/* Chevron */}
              <motion.div
                animate={{ rotate: isExpanded ? 90 : 0 }}
                transition={{ duration: 0.15 }}
              >
                <ChevronRight size={14} className="text-slate-400" aria-hidden="true" />
              </motion.div>

              {/* Expert icon */}
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                style={{ backgroundColor: `${config?.color || '#64748b'}15` }}
              >
                <IconComponent size={16} style={{ color: config?.color || '#64748b' }} aria-hidden="true" />
              </div>

              {/* Name + confidence */}
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

              {/* Confidence percentage */}
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
                  className="overflow-hidden"
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

                    {/* Search query */}
                    {expert.searchQuery && (
                      <div>
                        <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">Query ricerca</h4>
                        <code className="text-xs text-slate-500 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded block">
                          {expert.searchQuery}
                        </code>
                      </div>
                    )}

                    {/* Sources used */}
                    {expert.sources.length > 0 && (
                      <div>
                        <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">
                          Fonti ({expert.sources.length})
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {expert.sources.map(s => (
                            <span key={s} className="text-[10px] px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 rounded text-slate-600 dark:text-slate-400">
                              {s}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Duration */}
                    {expert.duration_ms !== undefined && (
                      <div className="flex items-center gap-1 text-[10px] text-slate-400">
                        <Clock size={10} aria-hidden="true" />
                        {expert.duration_ms}ms
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
  );
}
