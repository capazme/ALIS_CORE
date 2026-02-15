/**
 * CitationExportButton - Dropdown button to export citations in various formats.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { Download, Loader2, ChevronDown, Check, FileText, Code, BookOpen, Braces } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { cn } from '../../lib/utils';
import { exportCitations, triggerFileDownload } from '../../services/merltService';
import type { CitationFormat } from '../../types/merlt';

interface FormatOption {
  value: CitationFormat;
  label: string;
  description: string;
  Icon: LucideIcon;
}

// L1 fix: use component references instead of JSX instances
const CITATION_FORMATS: FormatOption[] = [
  {
    value: 'italian_legal',
    label: 'Citazione Italiana',
    description: 'Stile giuridico standard (Art. 1453 c.c.)',
    Icon: BookOpen,
  },
  {
    value: 'bibtex',
    label: 'BibTeX',
    description: 'Per paper accademici e LaTeX',
    Icon: Code,
  },
  {
    value: 'plain_text',
    label: 'Testo',
    description: 'Lista numerata semplice',
    Icon: FileText,
  },
  {
    value: 'json',
    label: 'JSON',
    description: 'Dati strutturati con metadati URN',
    Icon: Braces,
  },
];

export interface CitationExportButtonProps {
  traceId: string;
  sourcesCount: number;
  className?: string;
}

export function CitationExportButton({ traceId, sourcesCount, className }: CitationExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [lastExported, setLastExported] = useState(null as CitationFormat | null);
  const [error, setError] = useState(null as string | null);
  const dropdownRef = useRef(null as HTMLDivElement | null);

  // Close dropdown on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // M3 fix: stopPropagation on Escape to prevent parent drawer from also closing
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        setIsOpen(false);
      }
    };
    // Use capture phase to intercept before parent handler
    document.addEventListener('keydown', handleKeyDown, true);
    return () => document.removeEventListener('keydown', handleKeyDown, true);
  }, [isOpen]);

  const handleExport = useCallback(async (format: CitationFormat) => {
    setIsExporting(true);
    setError(null);
    setIsOpen(false);

    try {
      const result = await exportCitations({
        trace_id: traceId,
        format,
        include_query_summary: true,
        include_attribution: true,
      });

      if (result.success && result.download_url) {
        // H2 fix: build correct URL and use hidden <a> for download
        const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
        const downloadPath = result.download_url.replace('/api/v1/', '/merlt/');
        triggerFileDownload(`${API_BASE_URL}${downloadPath}`, result.filename);
        setLastExported(format);
      }
    } catch (err: unknown) {
      setError((err as { message?: string })?.message || 'Errore durante l\'esportazione');
    } finally {
      setIsExporting(false);
    }
  }, [traceId]);

  // H4 fix: clear error when opening dropdown
  const handleToggle = useCallback(() => {
    setIsOpen((prev: boolean) => {
      if (!prev) setError(null);
      return !prev;
    });
  }, []);

  if (sourcesCount === 0) return null;

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      <button
        type="button"
        onClick={handleToggle}
        disabled={isExporting}
        className={cn(
          'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border',
          'bg-white dark:bg-slate-800',
          'text-slate-700 dark:text-slate-300',
          'border-slate-200 dark:border-slate-700',
          'transition-colors',
          'hover:bg-slate-50 dark:hover:bg-slate-700',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          'disabled:opacity-50 disabled:cursor-not-allowed',
        )}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-label={`Esporta ${sourcesCount} citazioni`}
      >
        {isExporting ? (
          <Loader2 size={12} className="animate-spin" aria-hidden="true" />
        ) : (
          <Download size={12} aria-hidden="true" />
        )}
        Esporta citazioni
        <ChevronDown size={12} className={cn("transition-transform", isOpen && "rotate-180")} aria-hidden="true" />
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          className={cn(
            'absolute right-0 top-full mt-1 z-10 w-64',
            'bg-white dark:bg-slate-800',
            'border border-slate-200 dark:border-slate-700',
            'rounded-lg shadow-lg',
          )}
          role="listbox"
          aria-label="Formati di esportazione"
        >
          {CITATION_FORMATS.map((format) => (
            <button
              key={format.value}
              type="button"
              role="option"
              aria-selected={lastExported === format.value}
              onClick={() => handleExport(format.value)}
              className={cn(
                'w-full flex items-start gap-2.5 px-3 py-2.5 text-left',
                'transition-colors',
                'first:rounded-t-lg last:rounded-b-lg',
                'hover:bg-slate-50 dark:hover:bg-slate-700',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500',
              )}
            >
              <span className="mt-0.5 text-slate-500 dark:text-slate-400">
                <format.Icon size={14} aria-hidden="true" />
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-slate-700 dark:text-slate-300">
                    {format.label}
                  </span>
                  {lastExported === format.value && (
                    <Check size={12} className="text-emerald-500" aria-label="Ultimo formato usato" />
                  )}
                </div>
                <span className="text-[10px] text-slate-500 dark:text-slate-400">
                  {format.description}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Error feedback — only shown when dropdown is closed */}
      {error && !isOpen && (
        <div className="absolute right-0 top-full mt-1 z-10 w-64 p-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}
