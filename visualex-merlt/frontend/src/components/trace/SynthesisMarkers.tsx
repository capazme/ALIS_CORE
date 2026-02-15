/**
 * SynthesisMarkers - Background markers showing which expert contributed which part.
 *
 * Each segment of the synthesis can be attributed to an expert via source references.
 * This component renders subtle background highlights based on expert colors.
 */

import { useMemo } from 'react';
import { cn } from '../../lib/utils';
import { parseSynthesisWithSources } from '../../utils/synthesisParser';
import type { SourceResolution } from '../../types/trace';
import { EXPERT_CONFIG } from '../../types/pipeline';

export interface SynthesisMarkersProps {
  text: string;
  sources: SourceResolution[];
  className?: string;
}

interface MarkedSegment {
  content: string;
  expertColor: string | null;
  expertName: string | null;
}

export function SynthesisMarkers({ text, sources, className }: SynthesisMarkersProps) {
  const markedSegments = useMemo(() => {
    const segments = parseSynthesisWithSources(text);
    const result: MarkedSegment[] = [];

    let currentExpertColor: string | null = null;
    let currentExpertName: string | null = null;

    for (const segment of segments) {
      if (segment.type === 'source-ref') {
        const idx = segment.sourceIndex ?? 0;
        const source = idx >= 0 && idx < sources.length ? sources[idx] : undefined;
        if (source) {
          const config = EXPERT_CONFIG[source.expertId];
          currentExpertColor = config?.color || null;
          currentExpertName = config?.displayName || null;
        }
        // Don't render the [N] reference in this view, it's handled by SynthesisWithSources
        continue;
      }

      result.push({
        content: segment.content,
        expertColor: currentExpertColor,
        expertName: currentExpertName,
      });
    }

    return result;
  }, [text, sources]);

  if (markedSegments.length === 0) {
    return (
      <div className={cn("text-sm text-slate-500 text-center py-6", className)}>
        Nessun marcatore disponibile
      </div>
    );
  }

  return (
    <div className={cn("text-sm leading-relaxed", className)}>
      {markedSegments.map((segment: MarkedSegment, i: number) => (
        <span
          key={`seg-${i}`}
          className={cn(
            "transition-colors duration-200",
            segment.expertColor && "rounded px-0.5"
          )}
          style={segment.expertColor ? {
            backgroundColor: `${segment.expertColor}10`,
            borderBottom: `2px solid ${segment.expertColor}40`,
          } : undefined}
          title={segment.expertName ? `Contributo: ${segment.expertName}` : undefined}
        >
          {segment.content}
        </span>
      ))}
    </div>
  );
}
