/**
 * MerltToolbar
 *
 * Rendered in the article-toolbar slot when MERLT plugin is active.
 * Shows Brain button with status badge and quick actions for MERLT features.
 */

import { useMemo } from 'react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { Brain, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '../lib/utils';
import { useMerltArticleStatus } from '../hooks/useMerltArticleStatus';

type Props = SlotProps['article-toolbar'];

export function MerltToolbar({ urn, articleId }: Props): React.ReactElement {
  // TODO: Parse article metadata from urn or articleId
  // For now, using placeholder values
  const articleMeta = useMemo(() => {
    // In real implementation, parse from urn like "lex:art2~cc" -> tipo_atto: "codice civile", articolo: "2"
    return {
      tipo_atto: 'codice civile',
      articolo: '1',
      numero_atto: undefined,
      data: undefined,
      user_id: 'anonymous', // TODO: Get from plugin context
    };
  }, [urn, articleId]);

  const {
    isLoading,
    isEnriching,
    hasBeenProcessed,
    pendingCount,
    validatedCount,
    requestEnrichment,
  } = useMerltArticleStatus({
    ...articleMeta,
    enabled: true,
  });

  const handleAnalyze = () => {
    requestEnrichment();
  };

  // Determine status badge
  const statusBadge = useMemo(() => {
    if (isEnriching) {
      return (
        <div className="flex items-center gap-1 text-blue-600">
          <Loader2 size={12} className="animate-spin" />
          <span className="text-[10px] font-medium">Analisi...</span>
        </div>
      );
    }

    if (pendingCount > 0) {
      return (
        <div className="flex items-center gap-1 text-amber-600">
          <AlertCircle size={12} />
          <span className="text-[10px] font-medium">{pendingCount} da validare</span>
        </div>
      );
    }

    if (hasBeenProcessed && validatedCount > 0) {
      return (
        <div className="flex items-center gap-1 text-emerald-600">
          <CheckCircle2 size={12} />
          <span className="text-[10px] font-medium">{validatedCount} entità</span>
        </div>
      );
    }

    return null;
  }, [isEnriching, pendingCount, hasBeenProcessed, validatedCount]);

  return (
    <div className="flex items-center gap-3">
      {/* Brain Button */}
      <button
        onClick={handleAnalyze}
        disabled={isEnriching || isLoading}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all',
          'bg-gradient-to-r from-blue-50 to-indigo-50',
          'border border-blue-200',
          'hover:from-blue-100 hover:to-indigo-100 hover:border-blue-300',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
          'dark:from-blue-900/30 dark:to-indigo-900/30',
          'dark:border-blue-700',
          'dark:hover:from-blue-900/40 dark:hover:to-indigo-900/40'
        )}
        title="Analizza articolo con AI"
      >
        <Brain
          size={16}
          className={cn(
            'text-blue-600 dark:text-blue-400',
            isEnriching && 'animate-pulse'
          )}
        />
        <span className="text-xs font-medium text-blue-700 dark:text-blue-300">
          {isEnriching ? 'Analisi...' : hasBeenProcessed ? 'Rianalizza' : 'Analizza'}
        </span>
      </button>

      {/* Status Badge */}
      {statusBadge && (
        <div className="px-2 py-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
          {statusBadge}
        </div>
      )}
    </div>
  );
}
