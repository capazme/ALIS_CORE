/**
 * SourcesUsedPanel — Shows all sources used by experts in a trace.
 *
 * Groups sources by expert, shows source type badges, validity status,
 * and per-source rating capability.
 *
 * Story 6-6: Fonti Usate Panel
 */

import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BookOpen,
  Scale,
  TrendingUp,
  Gavel,
  Star,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  FileText,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  MessageSquarePlus,
} from 'lucide-react';
import type { SourceResolution, ExpertTraceEntry, ValiditySummary, ValidityResult } from '../../types/trace';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface SourcesUsedPanelProps {
  sources: SourceResolution[];
  experts: ExpertTraceEntry[];
  validity: ValiditySummary | null;
  onRateSource?: (sourceId: string, rating: number) => void;
  onSuggestMissing?: (urn: string) => void;
}

type SourceType = 'norma' | 'giurisprudenza' | 'dottrina' | 'unknown';

interface GroupedSource {
  source: SourceResolution;
  validityStatus?: ValidityResult;
  sourceType: SourceType;
}

// ─── Config ─────────────────────────────────────────────────────────────────

const EXPERT_CONFIG: Record<string, { label: string; icon: typeof BookOpen; color: string }> = {
  literal: {
    label: 'Letterale',
    icon: BookOpen,
    color: 'text-blue-600 dark:text-blue-400',
  },
  systemic: {
    label: 'Sistematico',
    icon: TrendingUp,
    color: 'text-purple-600 dark:text-purple-400',
  },
  principles: {
    label: 'Principi',
    icon: Scale,
    color: 'text-amber-600 dark:text-amber-400',
  },
  precedent: {
    label: 'Giurisprudenza',
    icon: Gavel,
    color: 'text-emerald-600 dark:text-emerald-400',
  },
};

const SOURCE_TYPE_BADGES: Record<SourceType, { label: string; bg: string; text: string }> = {
  norma: {
    label: 'Norma',
    bg: 'bg-blue-100 dark:bg-blue-900/30',
    text: 'text-blue-700 dark:text-blue-300',
  },
  giurisprudenza: {
    label: 'Giurisprudenza',
    bg: 'bg-emerald-100 dark:bg-emerald-900/30',
    text: 'text-emerald-700 dark:text-emerald-300',
  },
  dottrina: {
    label: 'Dottrina',
    bg: 'bg-purple-100 dark:bg-purple-900/30',
    text: 'text-purple-700 dark:text-purple-300',
  },
  unknown: {
    label: 'Altro',
    bg: 'bg-slate-100 dark:bg-slate-800',
    text: 'text-slate-600 dark:text-slate-400',
  },
};

const VALIDITY_CONFIG: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
  vigente: { icon: CheckCircle2, color: 'text-emerald-500', label: 'Vigente' },
  abrogato: { icon: AlertCircle, color: 'text-red-500', label: 'Abrogato' },
  modificato: { icon: AlertTriangle, color: 'text-amber-500', label: 'Modificato' },
  unknown: { icon: HelpCircle, color: 'text-slate-400', label: 'Sconosciuto' },
};

// ─── Helpers ────────────────────────────────────────────────────────────────

function detectSourceType(urn: string): SourceType {
  const lower = urn.toLowerCase();
  if (lower.includes('cass') || lower.includes('sent') || lower.includes('corte')) {
    return 'giurisprudenza';
  }
  if (lower.includes('dottrina') || lower.includes('commento')) {
    return 'dottrina';
  }
  if (lower.includes('urn:nir') || lower.includes('art') || lower.includes('legge') || lower.includes('decreto')) {
    return 'norma';
  }
  return 'unknown';
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function StarRating({
  onRate,
  sourceId,
}: {
  onRate: (sourceId: string, rating: number) => void;
  sourceId: string;
}) {
  const [hovered, setHovered] = useState(0);
  const [selected, setSelected] = useState(0);

  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          onClick={() => {
            setSelected(n);
            onRate(sourceId, n);
          }}
          onMouseEnter={() => setHovered(n)}
          onMouseLeave={() => setHovered(0)}
          className="p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
          aria-label={`${n} stelle`}
        >
          <Star
            size={14}
            className={
              n <= (hovered || selected)
                ? 'fill-amber-400 text-amber-400'
                : 'text-slate-300 dark:text-slate-600'
            }
          />
        </button>
      ))}
    </div>
  );
}

function SourceCard({
  grouped,
  onRate,
}: {
  grouped: GroupedSource;
  onRate?: (sourceId: string, rating: number) => void;
}) {
  const { source, validityStatus, sourceType } = grouped;
  const badge = SOURCE_TYPE_BADGES[sourceType];
  const validity = validityStatus
    ? VALIDITY_CONFIG[validityStatus.status] || VALIDITY_CONFIG.unknown
    : null;
  const ValidityIcon = validity?.icon || HelpCircle;

  return (
    <div className="flex items-start gap-3 p-2.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 transition-colors duration-150">
      <div className="flex-1 min-w-0">
        {/* URN + type badge */}
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${badge.bg} ${badge.text}`}>
            {badge.label}
          </span>
          {validity && (
            <span className={`flex items-center gap-1 text-[10px] ${validity.color}`}>
              <ValidityIcon size={10} />
              {validity.label}
            </span>
          )}
        </div>

        {/* Source label / URN */}
        <p className="text-xs font-medium text-slate-800 dark:text-slate-200 truncate" title={source.urn}>
          {source.label || source.urn}
        </p>

        {/* Relevance bar */}
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.round(source.score * 100)}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">
            {Math.round(source.score * 100)}%
          </span>
        </div>
      </div>

      {/* Rating widget */}
      {onRate && (
        <div className="flex-shrink-0 pt-1">
          <StarRating onRate={onRate} sourceId={source.sourceId} />
        </div>
      )}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export function SourcesUsedPanel({
  sources,
  experts,
  validity,
  onRateSource,
  onSuggestMissing,
}: SourcesUsedPanelProps) {
  const [expandedExperts, setExpandedExperts] = useState<Set<string>>(new Set(['literal', 'systemic', 'principles', 'precedent']));
  const [showMissingInput, setShowMissingInput] = useState(false);
  const [missingUrn, setMissingUrn] = useState('');

  // Build validity lookup
  const validityMap = useMemo(() => {
    const map = new Map<string, ValidityResult>();
    if (validity?.results) {
      for (const r of validity.results) {
        map.set(r.urn, r);
      }
    }
    return map;
  }, [validity]);

  // Group sources by expert
  const groupedByExpert = useMemo(() => {
    const groups: Record<string, GroupedSource[]> = {};

    for (const expert of experts) {
      const expertSources: GroupedSource[] = [];
      for (const sourceId of expert.sources) {
        const source = sources.find((s) => s.sourceId === sourceId);
        if (source) {
          expertSources.push({
            source,
            validityStatus: validityMap.get(source.urn),
            sourceType: detectSourceType(source.urn),
          });
        }
      }
      if (expertSources.length > 0) {
        groups[expert.expertId] = expertSources;
      }
    }

    return groups;
  }, [sources, experts, validityMap]);

  const toggleExpert = useCallback((expertId: string) => {
    setExpandedExperts((prev) => {
      const next = new Set(prev);
      if (next.has(expertId)) {
        next.delete(expertId);
      } else {
        next.add(expertId);
      }
      return next;
    });
  }, []);

  const handleSuggestMissing = useCallback(() => {
    if (missingUrn.trim() && onSuggestMissing) {
      onSuggestMissing(missingUrn.trim());
      setMissingUrn('');
      setShowMissingInput(false);
    }
  }, [missingUrn, onSuggestMissing]);

  const totalSources = sources.length;
  const expertIds = Object.keys(groupedByExpert);

  if (totalSources === 0) {
    return (
      <div className="text-center py-6" role="status">
        <FileText size={24} className="mx-auto text-slate-300 dark:text-slate-600 mb-2" />
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Nessuna fonte utilizzata
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Stats summary */}
      <div className="flex items-center justify-between px-1">
        <span className="text-xs font-medium text-slate-700 dark:text-slate-300">
          {totalSources} fonti da {expertIds.length} esperti
        </span>
        {validity && (
          <span className="text-[10px] text-slate-500 dark:text-slate-400">
            {validity.vigente} vigenti · {validity.abrogato + validity.modificato} da verificare
          </span>
        )}
      </div>

      {/* Sources grouped by expert */}
      {expertIds.map((expertId) => {
        const config = EXPERT_CONFIG[expertId] || {
          label: expertId,
          icon: FileText,
          color: 'text-slate-600',
        };
        const ExpertIcon = config.icon;
        const expertSources = groupedByExpert[expertId];
        const isExpanded = expandedExperts.has(expertId);

        return (
          <div key={expertId}>
            <button
              onClick={() => toggleExpert(expertId)}
              aria-expanded={isExpanded}
              className="flex items-center gap-2 w-full text-left py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            >
              {isExpanded ? (
                <ChevronDown size={12} className="text-slate-400" />
              ) : (
                <ChevronRight size={12} className="text-slate-400" />
              )}
              <ExpertIcon size={14} className={config.color} />
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                {config.label}
              </span>
              <span className="text-[10px] text-slate-400 ml-auto">
                {expertSources.length}
              </span>
            </button>

            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="space-y-1.5 pl-6 pt-1">
                    {expertSources.map((gs) => (
                      <SourceCard
                        key={gs.source.sourceId}
                        grouped={gs}
                        onRate={onRateSource}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}

      {/* "Manca fonte migliore" button */}
      {onSuggestMissing && (
        <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
          {!showMissingInput ? (
            <button
              onClick={() => setShowMissingInput(true)}
              className="flex items-center gap-1.5 text-xs text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              <MessageSquarePlus size={14} />
              Manca una fonte migliore?
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={missingUrn}
                onChange={(e) => setMissingUrn(e.target.value)}
                placeholder="URN fonte (es. urn:nir:stato:...)"
                className="flex-1 text-xs px-2 py-1.5 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                onKeyDown={(e) => e.key === 'Enter' && handleSuggestMissing()}
              />
              <button
                onClick={handleSuggestMissing}
                disabled={!missingUrn.trim()}
                className="text-xs px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Invia
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
