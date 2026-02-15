/**
 * MerltToolbar
 *
 * Rendered in the article-toolbar slot when MERLT plugin is active.
 * Shows Brain button that opens the MERLT inspector drawer.
 * Pattern: single icon button with pending-count badge overlay (Legacy style).
 */

import { useMemo } from 'react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { Brain } from 'lucide-react';
import { cn } from '../lib/utils';
import { useMerltArticleStatus } from '../hooks/useMerltArticleStatus';
import { useMerltPanelStore } from '../store/useMerltSidebarStore';

type Props = SlotProps['article-toolbar'];

export function MerltToolbar({ urn, articleId }: Props): React.ReactElement {
  const articleMeta = useMemo(() => ({
    tipo_atto: 'codice civile',
    articolo: '1',
    numero_atto: undefined,
    data: undefined,
    user_id: 'anonymous',
  }), [urn, articleId]);

  const open = useMerltPanelStore((s) => s.open);
  const isOpen = useMerltPanelStore((s) => s.isOpen);

  const {
    isEnriching,
    hasBeenProcessed,
    pendingCount,
  } = useMerltArticleStatus({
    ...articleMeta,
    enabled: true,
  });

  return (
    <div className="flex items-center">
      <button
        onClick={open}
        className={cn(
          'relative p-1.5 rounded-md transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
          isOpen
            ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
            : hasBeenProcessed
              ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-900/20 dark:text-emerald-400'
              : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-blue-500'
        )}
        title={hasBeenProcessed ? 'Nel Knowledge Graph' : 'Contribuisci al Knowledge Graph'}
        aria-label="Apri pannello MERLT"
      >
        <Brain
          size={16}
          className={cn(isEnriching && 'animate-pulse')}
        />
        {pendingCount > 0 && (
          <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-amber-500 text-white text-[9px] rounded-full flex items-center justify-center font-medium">
            {pendingCount > 9 ? '9+' : pendingCount}
          </span>
        )}
      </button>
    </div>
  );
}
