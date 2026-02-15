/**
 * RouterFeedbackPanel — Feedback for high-authority users on routing decisions.
 *
 * Shows "Was the routing correct?" with expert multi-select for alternative routing.
 * Only visible for users with authority >= 0.7.
 *
 * TD-4: Router feedback frontend
 */

import { useState, useCallback } from 'react';
import {
  ArrowRightLeft,
  Check,
  X,
  Send,
  Shield,
  Loader2,
} from 'lucide-react';

interface RouterFeedbackPanelProps {
  traceId: string;
  routedExperts: string[];
  userAuthority: number;
  onSubmit: (feedback: RouterFeedbackData) => void | Promise<void>;
  isSubmitting?: boolean;
}

export interface RouterFeedbackData {
  trace_id: string;
  user_id: string;
  routing_correct: boolean;
  suggested_weights?: Record<string, number>;
  suggested_query_type?: string;
  comment?: string;
}

const EXPERT_OPTIONS = [
  { id: 'literal', label: 'Letterale', description: 'Interpretazione del testo' },
  { id: 'systemic', label: 'Sistematico', description: 'Relazioni e sistema normativo' },
  { id: 'principles', label: 'Principi', description: 'Ratio legis e principi costituzionali' },
  { id: 'precedent', label: 'Giurisprudenza', description: 'Precedenti e massime' },
] as const;

const AUTHORITY_THRESHOLD = 0.7;

export function RouterFeedbackPanel({
  traceId,
  routedExperts,
  userAuthority,
  onSubmit,
  isSubmitting = false,
}: RouterFeedbackPanelProps) {
  const [routingCorrect, setRoutingCorrect] = useState<boolean | null>(null);
  const [selectedExperts, setSelectedExperts] = useState<Set<string>>(new Set(routedExperts));
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const isAuthorized = userAuthority >= AUTHORITY_THRESHOLD;

  const toggleExpert = useCallback((expertId: string) => {
    setSelectedExperts((prev) => {
      const next = new Set(prev);
      if (next.has(expertId)) {
        next.delete(expertId);
      } else {
        next.add(expertId);
      }
      return next;
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (routingCorrect === null) return;

    const weights: Record<string, number> = {};
    if (!routingCorrect) {
      const selected = Array.from(selectedExperts);
      const w = selected.length > 0 ? 1 / selected.length : 0;
      for (const e of selected) {
        weights[e] = Math.round(w * 100) / 100;
      }
    }

    await onSubmit({
      trace_id: traceId,
      user_id: '', // filled by caller
      routing_correct: routingCorrect,
      suggested_weights: !routingCorrect ? weights : undefined,
      comment: comment.trim() || undefined,
    });
    setSubmitted(true);
  }, [routingCorrect, selectedExperts, comment, traceId, onSubmit]);

  if (!isAuthorized) {
    return (
      <div className="flex items-center gap-2 p-3 rounded-lg bg-slate-100 dark:bg-slate-800/50">
        <Shield size={14} className="text-slate-400" />
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Feedback sul routing disponibile per utenti con autorità ≥ {AUTHORITY_THRESHOLD}
          <span className="ml-1 text-slate-400">
            (tua: {userAuthority.toFixed(2)})
          </span>
        </p>
      </div>
    );
  }

  if (submitted) {
    return (
      <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30">
        <Check size={14} className="text-emerald-600 dark:text-emerald-400" />
        <p className="text-xs text-emerald-700 dark:text-emerald-300">
          Feedback sul routing registrato. Grazie!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="flex items-center gap-2">
        <ArrowRightLeft size={14} className="text-blue-600 dark:text-blue-400" />
        <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
          Valutazione Routing
        </h4>
        <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
          Alta autorità
        </span>
      </div>

      {/* Current routing display */}
      <div className="text-xs text-slate-600 dark:text-slate-400">
        Expert attivati:{' '}
        {routedExperts.map((e) => (
          <span
            key={e}
            className="inline-block px-1.5 py-0.5 mr-1 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-[10px]"
          >
            {e}
          </span>
        ))}
      </div>

      {/* Correct/Incorrect buttons */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-600 dark:text-slate-400">
          Il routing è corretto?
        </span>
        <button
          onClick={() => setRoutingCorrect(true)}
          className={`flex items-center gap-1 px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
            routingCorrect === true
              ? 'bg-emerald-600 text-white'
              : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600 hover:border-emerald-400'
          }`}
        >
          <Check size={12} />
          Sì
        </button>
        <button
          onClick={() => setRoutingCorrect(false)}
          className={`flex items-center gap-1 px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
            routingCorrect === false
              ? 'bg-red-600 text-white'
              : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600 hover:border-red-400'
          }`}
        >
          <X size={12} />
          No
        </button>
      </div>

      {/* Expert selection (only when routing incorrect) */}
      {routingCorrect === false && (
        <div className="space-y-2">
          <p className="text-xs text-slate-600 dark:text-slate-400">
            Quali expert andrebbero attivati?
          </p>
          <div className="grid grid-cols-2 gap-1.5">
            {EXPERT_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                onClick={() => toggleExpert(opt.id)}
                className={`flex items-center gap-2 p-2 rounded text-left text-xs transition-colors ${
                  selectedExperts.has(opt.id)
                    ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-400 dark:border-blue-600 border'
                    : 'bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 hover:border-blue-300'
                }`}
              >
                <div
                  className={`w-3 h-3 rounded-sm border flex items-center justify-center ${
                    selectedExperts.has(opt.id)
                      ? 'bg-blue-600 border-blue-600'
                      : 'border-slate-300 dark:border-slate-500'
                  }`}
                >
                  {selectedExperts.has(opt.id) && <Check size={8} className="text-white" />}
                </div>
                <div>
                  <div className="font-medium text-slate-700 dark:text-slate-300">{opt.label}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Comment */}
      {routingCorrect !== null && (
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Commento opzionale..."
          rows={2}
          className="w-full text-xs px-2.5 py-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
      )}

      {/* Submit */}
      {routingCorrect !== null && (
        <button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-blue-600 text-white text-xs font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? (
            <Loader2 size={12} className="animate-spin" />
          ) : (
            <Send size={12} />
          )}
          Invia feedback routing
        </button>
      )}
    </div>
  );
}
