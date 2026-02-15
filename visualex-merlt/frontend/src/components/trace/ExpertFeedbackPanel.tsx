/**
 * ExpertFeedbackPanel — Per-expert rating with stars, reason and optional comment.
 */

import { useState, useCallback } from 'react';
import { Star, Loader2, CheckCircle2, ChevronDown } from 'lucide-react';
import { cn } from '../../lib/utils';
import { submitDetailedFeedback } from '../../services/merltService';
import { getCurrentUserId } from '../../services/merltInit';

export interface ExpertFeedbackPanelProps {
  traceId: string;
  expertId: string;
  className?: string;
}

type PanelState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'submitted' }
  | { status: 'error'; message: string };

const REASON_OPTIONS = [
  { value: 'accurate', label: 'Accurata' },
  { value: 'incomplete', label: 'Incompleta' },
  { value: 'irrelevant', label: 'Non pertinente' },
  { value: 'other', label: 'Altro' },
] as const;

type ReasonValue = typeof REASON_OPTIONS[number]['value'];

export function ExpertFeedbackPanel({ traceId, expertId, className }: ExpertFeedbackPanelProps) {
  const [state, setState] = useState({ status: 'idle' } as PanelState);
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(-1);
  const [reason, setReason] = useState('accurate' as ReasonValue);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');

  const handleSubmit = useCallback(async () => {
    if (rating === 0) return;
    setState({ status: 'submitting' });

    try {
      const userId = getCurrentUserId();
      const starRating = rating as 1 | 2 | 3 | 4 | 5;

      // Map reason to differentiated dimension scores:
      // - "accurate": all dimensions equal (positive)
      // - "incomplete": penalize completeness
      // - "irrelevant": penalize relevance
      // - "other": all dimensions equal
      let accuracy = starRating;
      let completeness = starRating;
      let relevance = starRating;

      if (reason === 'incomplete') {
        completeness = Math.max(1, starRating - 1) as 1 | 2 | 3 | 4 | 5;
      } else if (reason === 'irrelevant') {
        relevance = Math.max(1, starRating - 1) as 1 | 2 | 3 | 4 | 5;
      }

      const feedbackComment = comment
        ? `[expert:${expertId}][${reason}] ${comment}`
        : `[expert:${expertId}][${reason}]`;

      await submitDetailedFeedback({
        trace_id: traceId,
        user_id: userId,
        accuracy,
        completeness,
        relevance,
        comment: feedbackComment,
      });
      setState({ status: 'submitted' });
    } catch (err) {
      setState({
        status: 'error',
        message: err instanceof Error ? err.message : 'Errore invio feedback',
      });
    }
  }, [traceId, expertId, rating, reason, comment]);

  if (state.status === 'submitted') {
    return (
      <div className={cn(
        "flex items-center gap-2 p-2 rounded-md bg-emerald-50/50 dark:bg-emerald-900/10",
        className,
      )}>
        <CheckCircle2 size={14} className="text-emerald-500 shrink-0" aria-hidden="true" />
        <span className="text-[10px] text-emerald-600 dark:text-emerald-400">Feedback inviato</span>
      </div>
    );
  }

  return (
    <div className={cn(
      "mt-2 pt-2 border-t border-slate-100 dark:border-slate-800 space-y-2",
      className,
    )}>
      <h4 className="text-[10px] font-bold uppercase text-slate-500">Valuta esperto</h4>

      {/* Star rating */}
      <div className="flex items-center gap-1" role="group" aria-label="Valutazione stelle">
        {[1, 2, 3, 4, 5].map((star) => {
          const isFilled = star <= (hoverRating > 0 ? hoverRating : rating);
          return (
            <button
              key={star}
              onClick={() => setRating(star)}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(-1)}
              aria-label={`${star} stell${star === 1 ? 'a' : 'e'}`}
              className={cn(
                "p-0.5 transition-transform hover:scale-110",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-400 rounded-sm",
              )}
            >
              <Star
                size={14}
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
        {rating > 0 && (
          <span className="text-[10px] text-slate-400 ml-1">{rating}/5</span>
        )}
      </div>

      {/* Reason select */}
      <div className="relative">
        <select
          value={reason}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setReason(e.target.value as ReasonValue)}
          className={cn(
            "w-full text-xs rounded-md border border-slate-200 dark:border-slate-700",
            "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300",
            "px-2 py-1.5 pr-7 appearance-none",
            "focus:outline-none focus:ring-2 focus:ring-blue-500",
          )}
        >
          {REASON_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <ChevronDown
          size={12}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
          aria-hidden="true"
        />
      </div>

      {/* Collapsible comment */}
      {!showComment ? (
        <button
          onClick={() => setShowComment(true)}
          className="text-[10px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
        >
          + Aggiungi commento
        </button>
      ) : (
        <textarea
          value={comment}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setComment(e.target.value)}
          placeholder="Commento opzionale..."
          maxLength={500}
          rows={2}
          className={cn(
            "w-full text-xs rounded-md border border-slate-200 dark:border-slate-700",
            "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300",
            "p-2 resize-none",
            "focus:outline-none focus:ring-2 focus:ring-blue-500",
          )}
        />
      )}

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={rating === 0 || state.status === 'submitting'}
        className={cn(
          "w-full text-xs font-medium px-3 py-1.5 rounded-md transition-colors",
          "bg-blue-600 text-white hover:bg-blue-700",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2",
          "disabled:opacity-50 disabled:cursor-not-allowed",
        )}
      >
        {state.status === 'submitting' ? (
          <span className="flex items-center justify-center gap-1">
            <Loader2 size={12} className="animate-spin" aria-hidden="true" />
            Invio...
          </span>
        ) : (
          'Invia'
        )}
      </button>

      {state.status === 'error' && (
        <p className="text-[10px] text-red-500">{state.message}</p>
      )}
    </div>
  );
}
