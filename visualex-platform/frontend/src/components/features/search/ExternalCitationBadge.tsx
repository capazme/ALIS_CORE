/**
 * ExternalCitationBadge - Badge icon for EUR-Lex and other external citations.
 */

import { ExternalLink, Globe } from 'lucide-react';
import { cn } from '../../../lib/utils';
import type { CitationType } from '../../../utils/citationNavigator';

export interface ExternalCitationBadgeProps {
  type: CitationType;
  className?: string;
}

export function ExternalCitationBadge({ type, className }: ExternalCitationBadgeProps) {
  if (type === 'internal') return null;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 ml-0.5 px-1 py-0.5 rounded text-[9px] font-bold align-middle",
        type === 'external-eu'
          ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
          : "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
        className
      )}
      title={type === 'external-eu' ? 'EUR-Lex' : 'Fonte esterna'}
    >
      {type === 'external-eu' ? (
        <>
          <Globe size={9} />
          EU
        </>
      ) : (
        <ExternalLink size={9} />
      )}
    </span>
  );
}
