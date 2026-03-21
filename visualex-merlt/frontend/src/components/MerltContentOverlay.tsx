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
    const startOff = selection?.startOffset ?? 0;
    const endOff = selection?.endOffset ?? 0;

    // Return up to 500 chars before and after the selection
    const contextStart = Math.max(0, startOff - 500);
    const contextEnd = Math.min(text.length, endOff + 500);
    return text.substring(contextStart, contextEnd);
  }, [contentRef, selection?.startOffset, selection?.endOffset]);

  // Listen to article:text-selected events
  useEffect(() => {
    const handleTextSelected = (data: PluginEventMap['article:text-selected']) => {
      // Only handle events for current article
      if (data.urn !== urn) return;

      // Minimum length + multi-pattern confidence scoring
      if (data.text.length < 15) {
        setSelection(null);
        return;
      }

      // Multi-pattern confidence scoring instead of single regex
      let confidence = 0;
      const patterns: [RegExp, number][] = [
        [/\b(art\.?\s*\d+|articolo\s+\d+)/i, 0.4],
        [/\b(comma\s+\d+|lett\.\s*[a-z])/i, 0.3],
        [/\b(legge|decreto|d\.?l\.?g?s?\.?|d\.?p\.?r\.?|codice)\b/i, 0.3],
        [/\b(c\.c\.|c\.p\.|c\.p\.c\.|c\.p\.p\.)/i, 0.3],
        [/\b(n\.\s*\d+\/\d{4}|\d+\/\d{4})/i, 0.2],
        [/\b(direttiva|regolamento)\s+(UE|CE)/i, 0.2],
      ];

      for (const [pattern, score] of patterns) {
        if (pattern.test(data.text)) confidence += score;
      }

      if (confidence < 0.3) {
        setSelection(null);
        return;
      }

      // Calculate position from event data or Range API
      let position = { top: 100, left: 200 };
      const pos = data.position as { top?: number; left?: number } | undefined;
      if (pos && typeof pos.top === 'number' && typeof pos.left === 'number') {
        position = { top: pos.top, left: pos.left };
      } else if (contentRef.current) {
        const sel = window.getSelection();
        if (sel && sel.rangeCount > 0) {
          const range = sel.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          const containerRect = contentRef.current.getBoundingClientRect();
          position = {
            top: rect.bottom - containerRect.top + 8,
            left: rect.left - containerRect.left,
          };
        }
      }

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
            position: (data as unknown as { position?: { top: number; left: number } }).position || prev?.position || { top: 100, left: 200 },
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

  // When no selection, show discrete activity dot
  if (!selection) {
    return (
      <div
        className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse opacity-60"
        title="MERL-T attivo"
        aria-label="MERL-T è attivo su questo articolo"
      />
    );
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
