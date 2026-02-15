/**
 * SourceRatingWidget — 5-star rating for individual sources.
 */

import { useState, useCallback, useRef } from 'react';
import { Star, Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { submitSourceFeedback } from '../../services/merltService';
import { getCurrentUserId } from '../../services/merltInit';

export interface SourceRatingWidgetProps {
  traceId: string;
  sourceUrn: string;
  className?: string;
}

type RatingState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'submitted'; rating: number }
  | { status: 'error' };

export function SourceRatingWidget({ traceId, sourceUrn, className }: SourceRatingWidgetProps) {
  const [state, setState] = useState({ status: 'idle' } as RatingState);
  const [hoverIndex, setHoverIndex] = useState(-1);
  const busyRef = useRef(false);

  const handleClick = useCallback(async (rating: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (busyRef.current) return;
    busyRef.current = true;

    setState({ status: 'submitting' });
    try {
      const userId = getCurrentUserId();
      await submitSourceFeedback({
        trace_id: traceId,
        user_id: userId,
        source_urn: sourceUrn,
        rating: rating as 1 | 2 | 3 | 4 | 5,
      });
      setState({ status: 'submitted', rating });
    } catch {
      setState({ status: 'error' });
      busyRef.current = false;
    }
  }, [traceId, sourceUrn]);

  const isSubmitted = state.status === 'submitted';
  const submittedRating = isSubmitted ? state.rating : 0;

  return (
    <div
      className={cn("flex items-center gap-0.5", className)}
      onClick={(e: React.MouseEvent) => e.stopPropagation()}
      role="group"
      aria-label={`Valuta fonte${isSubmitted ? `: ${submittedRating} stelle` : ''}`}
    >
      {state.status === 'submitting' && (
        <Loader2 size={12} className="animate-spin text-amber-400 mr-1" aria-hidden="true" />
      )}

      {[1, 2, 3, 4, 5].map((star) => {
        const isFilled = isSubmitted
          ? star <= submittedRating
          : star <= hoverIndex;

        return (
          <button
            key={star}
            onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleClick(star, e)}
            onMouseEnter={() => { if (!isSubmitted) setHoverIndex(star); }}
            onMouseLeave={() => { if (!isSubmitted) setHoverIndex(-1); }}
            disabled={state.status === 'submitting' || isSubmitted}
            aria-label={`${star} stell${star === 1 ? 'a' : 'e'}`}
            className={cn(
              "p-0 transition-colors",
              !isSubmitted && "cursor-pointer hover:scale-110",
              isSubmitted && "cursor-default",
              "disabled:cursor-default",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-400 rounded-sm",
            )}
          >
            <Star
              size={12}
              className={cn(
                "transition-colors",
                isFilled
                  ? "text-amber-400 fill-amber-400"
                  : "text-slate-300 dark:text-slate-600",
              )}
              aria-hidden="true"
            />
          </button>
        );
      })}

      {state.status === 'error' && (
        <span className="text-[9px] text-red-500 ml-1">Errore</span>
      )}
    </div>
  );
}
