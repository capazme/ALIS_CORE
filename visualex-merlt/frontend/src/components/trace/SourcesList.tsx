/**
 * SourcesList - List of sources with validity indicators (green/yellow/red).
 */

import { CheckCircle2, AlertTriangle, XCircle, HelpCircle } from 'lucide-react';
import { cn } from '../../lib/utils';
import { SourceRatingWidget } from './SourceRatingWidget';
import type { SourceResolution, ValidityResult, ValiditySummary, ValidityStatus } from '../../types/trace';
import { EXPERT_CONFIG } from '../../types/pipeline';

export interface SourcesListProps {
  sources: SourceResolution[];
  validity: ValiditySummary | null;
  traceId?: string;
  onSourceClick?: (source: SourceResolution) => void;
  className?: string;
}

const VALIDITY_ICON: Record<ValidityStatus, React.ReactNode> = {
  vigente: <CheckCircle2 size={14} className="text-emerald-500" aria-hidden="true" />,
  modificato: <AlertTriangle size={14} className="text-amber-500" aria-hidden="true" />,
  abrogato: <XCircle size={14} className="text-red-500" aria-hidden="true" />,
  unknown: <HelpCircle size={14} className="text-slate-400" aria-hidden="true" />,
};

const VALIDITY_LABEL: Record<ValidityStatus, string> = {
  vigente: 'Vigente',
  modificato: 'Modificato',
  abrogato: 'Abrogato',
  unknown: 'Non verificato',
};

const VALIDITY_STYLE: Record<ValidityStatus, string> = {
  vigente: 'border-emerald-200 dark:border-emerald-800 bg-emerald-50/30 dark:bg-emerald-900/10',
  modificato: 'border-amber-200 dark:border-amber-800 bg-amber-50/30 dark:bg-amber-900/10',
  abrogato: 'border-red-200 dark:border-red-800 bg-red-50/30 dark:bg-red-900/10',
  unknown: 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800',
};

function getValidityForSource(urn: string, validity: ValiditySummary | null): ValidityResult | undefined {
  return validity?.results.find(v => v.urn === urn);
}

export function SourcesList({ sources, validity, traceId, onSourceClick, className }: SourcesListProps) {
  if (sources.length === 0) {
    return (
      <div className={cn("text-sm text-slate-500 text-center py-6", className)}>
        Nessuna fonte disponibile
      </div>
    );
  }

  // Validity summary bar
  const validitySummary = validity && (
    <div className="flex items-center gap-3 mb-3 text-xs">
      {validity.vigente > 0 && (
        <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
          <CheckCircle2 size={12} aria-hidden="true" /> {validity.vigente} vigenti
        </span>
      )}
      {validity.modificato > 0 && (
        <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
          <AlertTriangle size={12} aria-hidden="true" /> {validity.modificato} modificati
        </span>
      )}
      {validity.abrogato > 0 && (
        <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
          <XCircle size={12} aria-hidden="true" /> {validity.abrogato} abrogati
        </span>
      )}
    </div>
  );

  return (
    <div className={cn("space-y-2", className)}>
      {validitySummary}

      {sources.map((source, i) => {
        const validityInfo = getValidityForSource(source.urn, validity);
        const status: ValidityStatus = validityInfo?.status || 'unknown';
        const expertConfig = EXPERT_CONFIG[source.expertId];

        return (
          <button
            key={source.sourceId || i}
            onClick={() => onSourceClick?.(source)}
            aria-label={`Fonte ${i + 1}: ${source.label} - ${VALIDITY_LABEL[status]}`}
            className={cn(
              "w-full text-left p-3 rounded-lg border transition-all",
              "hover:shadow-md hover:scale-[1.01]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
              VALIDITY_STYLE[status]
            )}
          >
            <div className="flex items-start gap-2">
              {/* Validity indicator */}
              <div className="mt-0.5 shrink-0">
                {VALIDITY_ICON[status]}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 truncate">
                    {source.label}
                  </span>
                  {/* Expert badge */}
                  <span
                    className="text-[9px] font-bold px-1.5 py-0.5 rounded-full shrink-0"
                    style={{
                      backgroundColor: `${expertConfig?.color || '#64748b'}20`,
                      color: expertConfig?.color || '#64748b',
                    }}
                  >
                    {expertConfig?.displayName || source.expertId}
                  </span>
                </div>

                {/* Chunk text preview */}
                <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2 leading-relaxed">
                  {source.chunkText}
                </p>

                {/* Footer: validity label + score + rating */}
                <div className="flex items-center gap-2 mt-1.5">
                  <span className={cn(
                    "text-[10px] font-medium",
                    status === 'vigente' && "text-emerald-600 dark:text-emerald-400",
                    status === 'modificato' && "text-amber-600 dark:text-amber-400",
                    status === 'abrogato' && "text-red-600 dark:text-red-400",
                    status === 'unknown' && "text-slate-400",
                  )}>
                    {VALIDITY_LABEL[status]}
                  </span>
                  <span className="text-[10px] text-slate-400">
                    Score: {Math.round(source.score * 100)}%
                  </span>
                  {traceId && (
                    <SourceRatingWidget
                      traceId={traceId}
                      sourceUrn={source.urn}
                      className="ml-auto"
                    />
                  )}
                </div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
