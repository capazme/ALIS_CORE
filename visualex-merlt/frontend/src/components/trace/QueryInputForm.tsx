/**
 * QueryInputForm - Compact query input for the sidebar Analisi tab.
 *
 * Captures user query, submits to merltService.queryExperts(),
 * and returns the trace_id on success.
 */

import { useState, useCallback } from 'react';
import { Search, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '../../lib/utils';
import { queryExperts } from '../../services/merltService';

const MIN_QUERY_LENGTH = 10;
const MAX_QUERY_LENGTH = 2000;

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

  const isValid = query.trim().length >= MIN_QUERY_LENGTH;

  const handleSubmit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isValid || isSubmitting || disabled) return;

    setError(null);
    setIsSubmitting(true);

    try {
      const response = await queryExperts({
        query: query.trim(),
        user_id: userId,
        include_trace: true,
        consent_level: 'basic',
        ...(articleUrn && {
          context: { article_urn: articleUrn },
        }),
      });

      onTraceCreated(response.trace_id);
    } catch (err: unknown) {
      const message =
        (err as { message?: string })?.message ||
        'Errore durante la query. Riprova.';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [query, isValid, isSubmitting, disabled, userId, articleUrn, onTraceCreated]);

  return (
    <form
      onSubmit={handleSubmit}
      className={cn('space-y-3', className)}
      role="form"
      aria-label="Query analisi esperti"
    >
      <div>
        <label
          htmlFor="merlt-query-input"
          className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5"
        >
          Domanda giuridica
        </label>
        <textarea
          id="merlt-query-input"
          value={query}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
            setQuery(e.target.value);
            if (error) setError(null);
          }}
          placeholder="Es. Quali sono i presupposti della responsabilità extracontrattuale ex art. 2043 c.c.?"
          maxLength={MAX_QUERY_LENGTH}
          rows={3}
          disabled={isSubmitting || disabled}
          className={cn(
            'w-full px-3 py-2 text-sm rounded-lg border resize-none',
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
