/**
 * CitationDefinitionSnippet - Compact definition snippet for citation popups.
 * Shows first sentence of article text + rubrica if available.
 */

import type { ArticleData } from '../../../types';
import { cn } from '../../../lib/utils';

export interface CitationDefinitionSnippetProps {
  article: ArticleData | null;
  className?: string;
}

/**
 * Extract the first meaningful sentence from article HTML text.
 */
function extractFirstSentence(html: string): string {
  // Strip HTML tags
  const plain = html
    .replace(/<br\s*\/?>/gi, ' ')
    .replace(/<\/p>/gi, ' ')
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!plain) return '';

  // Find first sentence (up to period followed by space or end)
  const match = plain.match(/^(.+?[.!?])(?:\s|$)/);
  if (match) {
    const sentence = match[1];
    // Cap at 200 chars
    return sentence.length > 200 ? sentence.slice(0, 197) + '...' : sentence;
  }

  // No period found - take first 200 chars
  return plain.length > 200 ? plain.slice(0, 197) + '...' : plain;
}

export function CitationDefinitionSnippet({ article, className }: CitationDefinitionSnippetProps) {
  if (!article) return null;

  const rubrica = article.brocardi_info?.Rubrica ?? undefined;
  const snippet = article.article_text ? extractFirstSentence(article.article_text) : null;

  if (!rubrica && !snippet) return null;

  return (
    <div className={cn("space-y-1.5", className)}>
      {rubrica && (
        <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 italic">
          {rubrica}
        </div>
      )}
      {snippet && (
        <div className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
          {snippet}
        </div>
      )}
    </div>
  );
}
