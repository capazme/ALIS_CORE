/**
 * MerltContentOverlay
 *
 * Rendered in the article-content-overlay slot when MERLT plugin is active.
 * Shows CitationCorrectionCard when text is selected and citation detected.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { EventBus } from '@visualex/platform/lib/plugins';
import type { SlotProps, PluginEvents } from '@visualex/platform/lib/plugins';
import { CitationCorrectionCard } from './merlt/CitationCorrectionCard';
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
  const [selection, setSelection] = useState<CitationSelection | null>(null);
  const [userId, setUserId] = useState<string>(''); // TODO: Get from plugin context

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
    const handleTextSelected = (data: PluginEvents['article:text-selected']) => {
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
    };

    const unsubscribe = EventBus.on('article:text-selected', handleTextSelected);
    return unsubscribe;
  }, [urn]);

  // Listen to citation:detected events (from citation hover/preview)
  useEffect(() => {
    const handleCitationDetected = (data: PluginEvents['citation:detected']) => {
      // Only handle events for current article
      if (data.urn !== urn) return;

      // Enhance selection with parsed citation data
      setSelection(prev => {
        if (!prev || prev.text !== data.text) {
          // New citation detected from hover preview
          return {
            text: data.text,
            startOffset: 0, // Unknown from hover
            endOffset: data.text.length,
            position: { top: 100, left: 200 }, // Placeholder
            parsed: data.parsed as ParsedCitationData,
            confidence: (data.parsed as { confidence?: number })?.confidence,
          };
        }

        // Enhance existing selection with parsed data
        return {
          ...prev,
          parsed: data.parsed as ParsedCitationData,
          confidence: (data.parsed as { confidence?: number })?.confidence,
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
  }, []);

  // Don't render anything if no selection
  if (!selection) {
    return null;
  }

  return (
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
      userId={userId || 'anonymous'} // TODO: Get from plugin context
      getContextWindow={getContextWindow}
      onSuccess={handleSuccess}
    />
  );
}
