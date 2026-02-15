/**
 * SynthesisWithSources - Synthesis text with clickable superscript source references [1], [2].
 */

import { useMemo } from 'react';
import { cn } from '../../lib/utils';
import { parseSynthesisWithSources, type SynthesisSegment } from '../../utils/synthesisParser';
import type { SourceResolution } from '../../types/trace';
import { EXPERT_CONFIG, type ExpertId } from '../../types/pipeline';

export interface SynthesisWithSourcesProps {
  text: string;
  sources: SourceResolution[];
  selectedSourceIndex: number | null;
  onSourceClick: (sourceIndex: number) => void;
  className?: string;
}

export function SynthesisWithSources({
  text,
  sources,
  selectedSourceIndex,
  onSourceClick,
  className,
}: SynthesisWithSourcesProps) {
  const segments = useMemo(() => parseSynthesisWithSources(text), [text]);

  if (!text || segments.length === 0) {
    return (
      <div className={cn("text-sm text-slate-500 text-center py-6", className)}>
        Nessuna sintesi disponibile
      </div>
    );
  }

  return (
    <div className={cn("text-sm text-slate-700 dark:text-slate-300 leading-relaxed", className)}>
      {segments.map((segment: SynthesisSegment, i: number) => {
        if (segment.type === 'text') {
          return <span key={`t-${i}`}>{segment.content}</span>;
        }

        // Source reference
        const idx = segment.sourceIndex ?? 0;
        if (idx < 0 || idx >= sources.length) {
          return <span key={`r-${i}`} className="text-[10px] text-slate-400">[{idx + 1}]</span>;
        }
        const source = sources[idx];
        const isSelected = selectedSourceIndex === idx;
        const expertColor = EXPERT_CONFIG[source.expertId]?.color || '#64748b';

        return (
          <button
            key={`r-${i}`}
            onClick={() => onSourceClick(idx)}
            aria-label={`Fonte ${idx + 1}: ${source.label}`}
            className={cn(
              "inline-flex items-center justify-center",
              "w-5 h-5 rounded-full text-[10px] font-bold",
              "align-super -translate-y-0.5",
              "transition-all duration-150 cursor-pointer",
              "hover:scale-110 hover:shadow-md",
              "focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 outline-none",
              isSelected
                ? "ring-2 ring-offset-1 ring-blue-500"
                : ""
            )}
            style={{
              backgroundColor: `${expertColor}20`,
              color: expertColor,
              borderColor: expertColor,
            }}
            title={source ? `${source.label} (${EXPERT_CONFIG[source.expertId]?.displayName || source.expertId})` : `Fonte ${idx + 1}`}
          >
            {idx + 1}
          </button>
        );
      })}
    </div>
  );
}
