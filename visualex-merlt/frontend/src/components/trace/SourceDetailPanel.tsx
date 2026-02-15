/**
 * SourceDetailPanel - Panel showing source detail: URN, chunk text, validity, "Apri articolo" button.
 */

import { ExternalLink, FileText, CheckCircle2, AlertTriangle, XCircle, X } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { SourceResolution, ValiditySummary, ValidityStatus } from '../../types/trace';
import { EXPERT_CONFIG } from '../../types/pipeline';

export interface SourceDetailPanelProps {
  source: SourceResolution;
  validity: ValiditySummary | null;
  onClose: () => void;
  onOpenArticle?: (source: SourceResolution) => void;
  className?: string;
}

const VALIDITY_BADGE: Record<ValidityStatus, { icon: React.ReactNode; label: string; className: string }> = {
  vigente: {
    icon: <CheckCircle2 size={14} aria-hidden="true" />,
    label: 'Vigente',
    className: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  },
  modificato: {
    icon: <AlertTriangle size={14} aria-hidden="true" />,
    label: 'Modificato',
    className: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  },
  abrogato: {
    icon: <XCircle size={14} aria-hidden="true" />,
    label: 'Abrogato',
    className: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
  },
  unknown: {
    icon: <FileText size={14} aria-hidden="true" />,
    label: 'Non verificato',
    className: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  },
};

export function SourceDetailPanel({ source, validity, onClose, onOpenArticle, className }: SourceDetailPanelProps) {
  const validityResult = validity?.results.find(v => v.urn === source.urn);
  const status: ValidityStatus = validityResult?.status || 'unknown';
  const badge = VALIDITY_BADGE[status];
  const expertConfig = EXPERT_CONFIG[source.expertId];

  return (
    <div className={cn("flex flex-col h-full bg-white dark:bg-slate-900 border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-700", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
        <div className="flex items-center gap-2 min-w-0">
          <FileText size={16} className="text-slate-500 shrink-0" aria-hidden="true" />
          <span className="text-sm font-semibold text-slate-700 dark:text-slate-300 truncate">
            {source.label}
          </span>
        </div>
        <button
          onClick={onClose}
          aria-label="Chiudi dettaglio fonte"
          className="p-1 rounded text-slate-400 transition-colors shrink-0 hover:bg-slate-200 dark:hover:bg-slate-700 focus-visible:ring-2 focus-visible:ring-blue-500 outline-none"
        >
          <X size={16} aria-hidden="true" />
        </button>
      </div>

      {/* Metadata */}
      <div className="px-4 py-3 space-y-3 border-b border-slate-100 dark:border-slate-800">
        {/* URN */}
        <div>
          <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-0.5">URN</h4>
          <code className="text-xs text-slate-600 dark:text-slate-400 font-mono break-all">
            {source.urn}
          </code>
        </div>

        {/* Expert + Score row */}
        <div className="flex items-center gap-3">
          <span
            className="text-[10px] font-bold px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: `${expertConfig?.color || '#64748b'}20`,
              color: expertConfig?.color || '#64748b',
            }}
          >
            {expertConfig?.displayName || source.expertId}
          </span>
          <span className="text-[10px] text-slate-400">
            Relevance: {Math.round(source.score * 100)}%
          </span>
        </div>

        {/* Validity badge */}
        <div className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium", badge.className)}>
          {badge.icon}
          {badge.label}
          {validityResult?.effectiveDate && (
            <span className="opacity-70 text-[10px] ml-1">
              dal {validityResult.effectiveDate}
            </span>
          )}
        </div>
      </div>

      {/* Chunk text */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-2">Testo fonte</h4>
        <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap bg-slate-50 dark:bg-slate-800/30 rounded-lg p-3 border border-slate-100 dark:border-slate-700">
          {source.chunkText}
        </div>

        {/* Modification info */}
        {validityResult?.modifiedBy && (
          <div className="mt-3">
            <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">Modificato da</h4>
            <p className="text-xs text-slate-600 dark:text-slate-400">{validityResult.modifiedBy}</p>
          </div>
        )}
        {validityResult?.notes && (
          <div className="mt-2">
            <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-1">Note</h4>
            <p className="text-xs text-slate-600 dark:text-slate-400">{validityResult.notes}</p>
          </div>
        )}
      </div>

      {/* Footer: Open article button */}
      {onOpenArticle && (
        <div className="px-4 py-3 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={() => onOpenArticle(source)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 text-white transition-colors hover:bg-blue-700 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 outline-none"
          >
            <ExternalLink size={14} aria-hidden="true" />
            Apri articolo
          </button>
        </div>
      )}
    </div>
  );
}
