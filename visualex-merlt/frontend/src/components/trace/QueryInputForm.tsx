/**
 * QueryInputForm - Enhanced query input for the sidebar Chiedi tab.
 *
 * Features:
 * - Auto-resize textarea (80px-200px)
 * - Article context chip when articleUrn present
 * - Query history from localStorage (last 5)
 * - Rotating placeholder examples
 * - Prefill support from store
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { Search, Loader2, AlertCircle, Clock, X } from 'lucide-react';
import { cn } from '../../lib/utils';
import { queryExperts } from '../../services/merltService';
import { useMerltPanelStore } from '../../store/useMerltSidebarStore';

const MIN_QUERY_LENGTH = 10;
const MAX_QUERY_LENGTH = 2000;
const HISTORY_KEY = 'merlt-query-history';
const MAX_HISTORY = 5;

const EXAMPLE_QUERIES = [
  'Quali sono i presupposti della responsabilità extracontrattuale ex art. 2043 c.c.?',
  'Come si configura il diritto di recesso nel contratto di locazione?',
  'Qual è il rapporto tra buona fede e abuso del diritto?',
  'Come si applica il principio di proporzionalità nelle sanzioni amministrative?',
];

function getQueryHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveQueryToHistory(query: string) {
  const history = getQueryHistory().filter((q) => q !== query);
  history.unshift(query);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

function formatUrnShort(urn: string): string {
  const match = urn.match(/art[._](\d+[a-z-]*).*?(cod[._]civ|cod[._]pen|cod[._]proc|[a-z.]+)/i);
  if (match) return `Art. ${match[1]} ${match[2].replace(/[._]/g, '. ')}`;
  return urn.length > 30 ? urn.slice(0, 30) + '...' : urn;
}

export interface QueryInputFormProps {
  articleUrn?: string;
  userId: string;
  onTraceCreated: (traceId: string) => void;
  disabled?: boolean;
  className?: string;
}

export function QueryInputForm({
  articleUrn,
  userId,
  onTraceCreated,
  disabled = false,
  className,
}: QueryInputFormProps) {
  const [query, setQuery] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [showHistory, setShowHistory] = useState(false);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);

  const textareaRef = useRef(null as HTMLTextAreaElement | null);
  const historyRef = useRef(null as HTMLDivElement | null);

  const prefillQuery = useMerltPanelStore((s) => s.prefillQuery);
  const clearPrefill = useMerltPanelStore((s) => s.clearPrefill);

  // Handle prefill from store (e.g., "Chiedi a MERL-T" from SelectionPopup)
  useEffect(() => {
    if (prefillQuery) {
      setQuery(prefillQuery);
      clearPrefill();
      textareaRef.current?.focus();
    }
  }, [prefillQuery, clearPrefill]);

  // Rotate placeholder every 5s
  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIdx((prev: number) => (prev + 1) % EXAMPLE_QUERIES.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Auto-resize textarea
  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(200, Math.max(80, el.scrollHeight))}px`;
  }, []);

  useEffect(() => {
    autoResize();
  }, [query, autoResize]);

  // Close history dropdown on outside click
  useEffect(() => {
    if (!showHistory) return;
    const handleClick = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showHistory]);

  const isValid = query.trim().length >= MIN_QUERY_LENGTH;
  const history = getQueryHistory();

  const handleSubmit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isValid || isSubmitting || disabled) return;

    setError(null);
    setIsSubmitting(true);

    try {
      const trimmed = query.trim();
      saveQueryToHistory(trimmed);

      const response = await queryExperts({
        query: trimmed,
        user_id: userId,
        include_trace: true,
        consent_level: 'basic',
        ...(articleUrn && {
          context: { article_urn: articleUrn },
        }),
      });

      onTraceCreated(response.trace_id);
    } catch (err: unknown) {
      const message = (err as { message?: string })?.message || 'Errore durante la query. Riprova.';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [query, isValid, isSubmitting, disabled, userId, articleUrn, onTraceCreated]);

  const selectFromHistory = (q: string) => {
    setQuery(q);
    setShowHistory(false);
    textareaRef.current?.focus();
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={cn('space-y-3', className)}
      role="form"
      aria-label="Query analisi esperti"
    >
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label
            htmlFor="merlt-query-input"
            className="block text-xs font-semibold text-slate-600 dark:text-slate-400"
          >
            Domanda giuridica
          </label>

          {/* History button */}
          {history.length > 0 && (
            <div className="relative" ref={historyRef}>
              <button
                type="button"
                onClick={() => setShowHistory(!showHistory)}
                className={cn(
                  'p-1 rounded-md transition-colors',
                  'text-slate-400 hover:text-slate-600 hover:bg-slate-100',
                  'dark:hover:text-slate-300 dark:hover:bg-slate-800',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                )}
                aria-label="Cronologia query"
                title="Cronologia query"
              >
                <Clock size={14} />
              </button>
              {showHistory && (
                <div className="absolute right-0 top-full mt-1 w-64 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-10 py-1">
                  {history.map((q, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => selectFromHistory(q)}
                      className="w-full text-left px-3 py-2 text-xs text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700 truncate transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Article context chip */}
        {articleUrn && (
          <div className="flex items-center gap-1.5 mb-2">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-full border border-blue-200 dark:border-blue-800">
              {formatUrnShort(articleUrn)}
            </span>
          </div>
        )}

        <textarea
          ref={textareaRef}
          id="merlt-query-input"
          value={query}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
            setQuery(e.target.value);
            if (error) setError(null);
          }}
          placeholder={`Es. ${EXAMPLE_QUERIES[placeholderIdx]}`}
          maxLength={MAX_QUERY_LENGTH}
          disabled={isSubmitting || disabled}
          className={cn(
            'w-full px-3 py-2 text-sm rounded-lg border resize-none',
            'min-h-[80px] max-h-[200px]',
            'bg-white dark:bg-slate-800',
            'text-slate-900 dark:text-slate-100',
            'placeholder:text-slate-400 dark:placeholder:text-slate-500',
            'border-slate-200 dark:border-slate-700',
            'transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500',
            'disabled:opacity-50 disabled:cursor-not-allowed',
          )}
        />
        <div className="flex items-center justify-between mt-1">
          <span className="text-[10px] text-slate-400">
            Min {MIN_QUERY_LENGTH} caratteri
          </span>
          <span className={cn(
            'text-[10px]',
            query.length >= MIN_QUERY_LENGTH ? 'text-emerald-500' : 'text-slate-400',
          )}>
            {query.length} / {MAX_QUERY_LENGTH}
          </span>
        </div>
      </div>

      {error && (
        <div
          className="flex items-start gap-2 p-2.5 text-xs text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800"
          role="alert"
        >
          <AlertCircle size={14} className="mt-0.5 shrink-0" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      <button
        type="submit"
        disabled={!isValid || isSubmitting || disabled}
        className={cn(
          'w-full flex items-center justify-center gap-2',
          'px-4 py-2 rounded-lg text-sm font-medium',
          'bg-blue-600 text-white',
          'transition-colors',
          'hover:bg-blue-700',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed',
        )}
      >
        {isSubmitting ? (
          <>
            <Loader2 size={14} className="animate-spin" aria-hidden="true" />
            Analisi in corso...
          </>
        ) : (
          <>
            <Search size={14} aria-hidden="true" />
            Analizza con MERL-T
          </>
        )}
      </button>
    </form>
  );
}
