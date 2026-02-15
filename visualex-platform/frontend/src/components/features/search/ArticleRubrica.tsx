/**
 * ArticleRubrica - Displays article rubrica (title), URN, and vigency badge.
 * Meant to be sticky at the top of the article viewer.
 */

import { Scale, Clock, Link2 } from 'lucide-react';
import { cn } from '../../../lib/utils';
import type { ArticleData } from '../../../types';

export interface ArticleRubricaProps {
  data: ArticleData;
}

export function ArticleRubrica({ data }: ArticleRubricaProps) {
  const { norma_data, brocardi_info, versionInfo } = data;
  const rubrica = brocardi_info?.Rubrica ?? undefined;

  const isHistorical = versionInfo?.isHistorical ?? false;

  // Build URN display
  const urnParts: string[] = [];
  if (norma_data.tipo_atto) urnParts.push(norma_data.tipo_atto);
  if (norma_data.numero_atto) urnParts.push(`n. ${norma_data.numero_atto}`);
  if (norma_data.data) urnParts.push(norma_data.data);
  const urnDisplay = urnParts.join(' ');

  return (
    <div className="px-4 py-3 bg-gradient-to-r from-slate-50 to-white border-b border-slate-200 dark:from-slate-800/50 dark:to-slate-900/50 dark:border-slate-700">
      {/* Article number + rubrica */}
      <div className="flex items-start gap-3">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white shrink-0">
            Art. {norma_data.numero_articolo}
            {norma_data.allegato && (
              <span className="text-sm font-medium text-indigo-600 dark:text-indigo-400 ml-1.5">
                (All. {norma_data.allegato})
              </span>
            )}
          </h3>
          {rubrica && (
            <span className="text-sm text-slate-600 dark:text-slate-400 italic truncate">
              — {rubrica}
            </span>
          )}
        </div>

        {/* Vigency badge */}
        <div className={cn(
          "flex items-center gap-1 px-2 py-1 rounded-md text-xs font-semibold shrink-0",
          isHistorical
            ? "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
            : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
        )}>
          {isHistorical ? <Clock size={12} aria-hidden="true" /> : <Scale size={12} aria-hidden="true" />}
          {isHistorical ? 'Storica' : 'Vigente'}
        </div>
      </div>

      {/* URN line */}
      <div className="flex items-center gap-2 mt-1.5">
        <Link2 size={12} className="text-slate-400 shrink-0" aria-hidden="true" />
        <span className="text-xs text-slate-500 dark:text-slate-400 font-mono truncate">
          {norma_data.urn || urnDisplay}
        </span>
        {norma_data.data_versione && (
          <>
            <span className="text-slate-300 dark:text-slate-600">|</span>
            <span className="text-xs text-slate-400 dark:text-slate-500">
              Aggiornato al: {norma_data.data_versione}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
