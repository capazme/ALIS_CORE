/**
 * MerltContentOverlay
 *
 * Rendered in the article-content-overlay slot when MERLT plugin is active.
 * Shows CitationCorrectionCard when text is selected and citation detected.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { CheckCircle2, Loader2 } from 'lucide-react';
import { EventBus } from '@visualex/platform/lib/plugins';
import type { SlotProps, PluginEventMap } from '@visualex/platform/lib/plugins';
import { CitationCorrectionCard } from './merlt/CitationCorrectionCard';
import { confirmCitation } from '../services/merltService';
import { getCurrentUserId } from '../services/merltInit';
import { cn } from '../lib/utils';
import type { ParsedCitationData } from '../types/merlt';

type Props = SlotProps['article-content-overlay'];

interface CitationSelection {
  text: string;
  startOffset: number;
  endOffset: number;
  position: { top: number; left: number };
  parsed?: ParsedCitationData;
  confidence?: number;
}

export function MerltContentOverlay({ urn, articleId, contentRef }: Props): React.ReactElement | null {
  const [selection, setSelection] = useState(null as CitationSelection | null);
  const [confirmState, setConfirmState] = useState('idle' as 'idle' | 'submitting' | 'confirmed' | 'error');
  const userId = useMemo(() => getCurrentUserId(), []);

  // Extract context window from content DOM
  const getContextWindow = useCallback((): string => {
    if (!contentRef.current) return '';

    const text = contentRef.current.innerText || contentRef.current.textContent || '';

    // Return surrounding text (up to 500 chars before and after selection)
    // In a real implementation, you'd use startOffset/endOffset to find exact position
    return text.substring(0, 1000);
  }, [contentRef]);

  // Listen to article:text-selected events
  useEffect(() => {
    const handleTextSelected = (data: PluginEventMap['article:text-selected']) => {
      // Only handle events for current article
      if (data.urn !== urn) return;

      // Check if text is long enough to be a citation
      if (data.text.length < 5) {
        setSelection(null);
        return;
      }

      // Simple heuristic: check if text contains citation keywords
      const citationKeywords = /\b(art\.?|articolo|comma|legge|decreto|codice)\b/i;
      if (!citationKeywords.test(data.text)) {
        setSelection(null);
        return;
      }

      // Calculate position for card (relative to content container)
      // In a real implementation, you'd use Range API to get precise selection position
      const position = {
        top: 100, // Placeholder - should calculate from selection
        left: 200, // Placeholder - should calculate from selection
      };

      setSelection({
        text: data.text,
        startOffset: data.startOffset,
        endOffset: data.endOffset,
        position,
      });
      setConfirmState('idle');
    };

    const unsubscribe = EventBus.on('article:text-selected', handleTextSelected);
    return unsubscribe;
  }, [urn]);

  // Listen to citation:detected events (from citation hover/preview)
  useEffect(() => {
    const handleCitationDetected = (data: PluginEventMap['citation:detected']) => {
      // Only handle events for current article
      if (data.urn !== urn) return;

      // Enhance selection with parsed citation data
      setSelection((prev: CitationSelection | null) => {
        if (!prev || prev.text !== data.text) {
          // New citation detected from hover preview — reset confirm state
          setConfirmState('idle');
          return {
            text: data.text,
            startOffset: 0, // Unknown from hover
            endOffset: data.text.length,
            position: { top: 100, left: 200 }, // Placeholder
            parsed: data.parsed as unknown as ParsedCitationData,
            confidence: (data.parsed as unknown as { confidence?: number })?.confidence,
          };
        }

        // Enhance existing selection with parsed data
        return {
          ...prev,
          parsed: data.parsed as unknown as ParsedCitationData,
          confidence: (data.parsed as unknown as { confidence?: number })?.confidence,
        };
      });
    };

    const unsubscribe = EventBus.on('citation:detected', handleCitationDetected);
    return unsubscribe;
  }, [urn]);

  // Handle successful citation correction/annotation
  const handleSuccess = useCallback(() => {
    // Emit event to notify platform/other plugins
    EventBus.emit('enrichment:requested', { urn, userId });

    // Clear selection
    setSelection(null);
  }, [urn, userId]);

  // Close handler
  const handleClose = useCallback(() => {
    setSelection(null);
    setConfirmState('idle');
  }, []);

  // Confirm high-confidence citation
  const handleConfirm = useCallback(async () => {
    if (!selection?.parsed) return;
    setConfirmState('submitting');
    try {
      await confirmCitation({
        article_urn: urn,
        text: selection.text,
        parsed: selection.parsed,
        user_id: userId,
      });
      setConfirmState('confirmed');
      setTimeout(() => {
        setSelection(null);
        setConfirmState('idle');
      }, 1500);
    } catch {
      setConfirmState('error');
      setTimeout(() => setConfirmState('idle'), 2000);
    }
  }, [selection, urn, userId]);

  // Don't render anything if no selection
  if (!selection) {
    return null;
  }

  // High-confidence citation: show quick "Corretto" button
  const isHighConfidence = selection.parsed && (selection.confidence ?? 0) > 0.8;

  return (
    <>
      {isHighConfidence && (
        <div
          className={cn(
            "absolute z-40 flex items-center gap-1.5",
            "bg-white dark:bg-slate-800 shadow-md rounded-md border border-slate-200 dark:border-slate-700",
            "px-2 py-1",
          )}
          style={{ top: selection.position.top - 36, left: selection.position.left }}
        >
          {confirmState === 'confirmed' ? (
            <span className="flex items-center gap-1 text-[10px] text-emerald-600">
              <CheckCircle2 size={12} aria-hidden="true" /> Confermata
            </span>
          ) : confirmState === 'error' ? (
            <span className="text-[10px] text-red-500">Errore</span>
          ) : (
            <button
              onClick={handleConfirm}
              disabled={confirmState === 'submitting'}
              className={cn(
                "flex items-center gap-1 text-[10px] font-medium text-emerald-600",
                "hover:text-emerald-700 transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded",
                "disabled:opacity-50",
              )}
            >
              {confirmState === 'submitting' ? (
                <Loader2 size={12} className="animate-spin" aria-hidden="true" />
              ) : (
                <CheckCircle2 size={12} aria-hidden="true" />
              )}
              Corretto
            </button>
          )}
        </div>
      )}

      <CitationCorrectionCard
        isOpen={!!selection}
        onClose={handleClose}
        anchorPosition={selection.position}
        containerRef={contentRef}
        selectedText={selection.text}
        articleUrn={urn}
        originalParsed={selection.parsed}
        confidenceBefore={selection.confidence}
        source={selection.parsed ? 'citation_preview' : 'selection_popup'}
        userId={userId}
        getContextWindow={getContextWindow}
        onSuccess={handleSuccess}
      />
    </>
  );
}
