/**
 * InlineFeedbackPanel — Thumbs up/down feedback under synthesis.
 */

import { useState, useCallback } from 'react';
import { ThumbsUp, ThumbsDown, Loader2, CheckCircle2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { submitInlineFeedback } from '../../services/merltService';
import { getCurrentUserId } from '../../services/merltInit';

export interface InlineFeedbackPanelProps {
  traceId: string;
  className?: string;
}

type FeedbackState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'submitted'; positive: boolean }
  | { status: 'error'; message: string };

export function InlineFeedbackPanel({ traceId, className }: InlineFeedbackPanelProps) {
  const [state, setState] = useState({ status: 'idle' } as FeedbackState);

  const handleVote = useCallback(async (positive: boolean) => {
    setState({ status: 'submitting' });
    try {
      const userId = getCurrentUserId();
      await submitInlineFeedback(traceId, userId, positive ? 5 : 1);
      setState({ status: 'submitted', positive });
    } catch (err) {
      setState({
        status: 'error',
        message: err instanceof Error ? err.message : 'Errore invio feedback',
      });
    }
  }, [traceId]);

  if (state.status === 'submitted') {
    return (
      <div className={cn(
        "flex items-center gap-2 bg-slate-50 dark:bg-slate-800/30 rounded-lg border border-slate-200 dark:border-slate-700 p-3",
        className,
      )}>
        <CheckCircle2 size={16} className="text-emerald-500 shrink-0" aria-hidden="true" />
        <span className="text-xs text-slate-500">
          Grazie per il feedback!
        </span>
      </div>
    );
  }

  return (
    <div className={cn(
      "bg-slate-50 dark:bg-slate-800/30 rounded-lg border border-slate-200 dark:border-slate-700 p-3",
      className,
    )}>
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500 font-medium">Utile?</span>

        <button
          onClick={() => handleVote(true)}
          disabled={state.status === 'submitting'}
          aria-label="Risposta utile"
          className={cn(
            "p-1.5 rounded-md transition-colors",
            "hover:bg-emerald-100 dark:hover:bg-emerald-900/30 hover:text-emerald-600",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500",
            "text-slate-400",
            "disabled:opacity-50 disabled:cursor-not-allowed",
          )}
        >
          {state.status === 'submitting' ? (
            <Loader2 size={16} className="animate-spin" aria-hidden="true" />
          ) : (
            <ThumbsUp size={16} aria-hidden="true" />
          )}
        </button>

        <button
          onClick={() => handleVote(false)}
          disabled={state.status === 'submitting'}
          aria-label="Risposta non utile"
          className={cn(
            "p-1.5 rounded-md transition-colors",
            "hover:bg-red-100 dark:hover:bg-red-900/30 hover:text-red-600",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500",
            "text-slate-400",
            "disabled:opacity-50 disabled:cursor-not-allowed",
          )}
        >
          <ThumbsDown size={16} aria-hidden="true" />
        </button>
      </div>

      {state.status === 'error' && (
        <p className="text-[10px] text-red-500 mt-1">{state.message}</p>
      )}
    </div>
  );
}
