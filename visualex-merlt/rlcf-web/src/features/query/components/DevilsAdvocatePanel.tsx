/**
 * Devil's Advocate Panel
 *
 * Collapsible panel shown when expert disagreement is low (< 0.1),
 * presenting a critical perspective to challenge consensus and collect
 * feedback on the critical prompt.
 */

import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
import { Collapsible } from '@components/ui/Collapsible';
import { Textarea } from '@components/ui/Textarea';
import { AlertTriangle, Send, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

const ORCHESTRATION_URL = import.meta.env.VITE_ORCHESTRATION_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: `${ORCHESTRATION_URL}/api/v1/devils-advocate`,
  timeout: 15000,
});

export interface DevilsAdvocatePanelProps {
  traceId: string;
  disagreementScore: number;
  criticalPrompt: string | null;
}

type FeedbackAssessment = 'valid' | 'weak' | 'interesting';

const ASSESSMENT_OPTIONS: { value: FeedbackAssessment; label: string }[] = [
  { value: 'valid', label: 'Valida' },
  { value: 'weak', label: 'Debole' },
  { value: 'interesting', label: 'Interessante' },
];

export function DevilsAdvocatePanel({
  traceId,
  disagreementScore,
  criticalPrompt,
}: DevilsAdvocatePanelProps) {
  const [feedbackText, setFeedbackText] = useState('');
  const [assessment, setAssessment] = useState<FeedbackAssessment>('interesting');

  const shouldShow = disagreementScore < 0.1 && criticalPrompt !== null;

  const submitMutation = useMutation({
    mutationFn: (payload: { trace_id: string; feedback_text: string; assessment: string }) =>
      api.post('/feedback', payload).then((r) => r.data),
  });

  const handleSubmit = useCallback(() => {
    if (!feedbackText.trim()) return;
    submitMutation.mutate({
      trace_id: traceId,
      feedback_text: feedbackText,
      assessment,
    });
  }, [feedbackText, assessment, traceId, submitMutation]);

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFeedbackText(e.target.value);
  }, []);

  if (!shouldShow) return null;

  return (
    <Collapsible
      title="Prospettiva Critica (Devil's Advocate)"
      icon={<AlertTriangle className="h-4 w-4 text-amber-400" />}
      badge={
        <span className="px-2 py-0.5 text-xs font-medium bg-amber-900/50 text-amber-300 rounded-full">
          Basso disaccordo
        </span>
      }
      defaultOpen={false}
      className="border-amber-700/50"
    >
      <div className="space-y-4">
        {/* Critical prompt */}
        <div className="p-3 bg-amber-900/20 border border-amber-800/40 rounded-lg">
          <p className="text-sm text-amber-200 leading-relaxed whitespace-pre-wrap">
            {criticalPrompt}
          </p>
        </div>

        {/* Feedback form */}
        {submitMutation.isSuccess ? (
          <div className="flex items-center gap-2 p-3 bg-green-900/20 border border-green-800/40 rounded-lg">
            <CheckCircle className="h-4 w-4 text-green-400 shrink-0" />
            <p className="text-sm text-green-300">Feedback inviato. Grazie per il contributo!</p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Assessment select */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Valutazione:</span>
              <div className="flex gap-1.5">
                {ASSESSMENT_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setAssessment(opt.value)}
                    className={cn(
                      'px-3 py-1 text-xs font-medium rounded-full transition-colors',
                      'focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-slate-900',
                      assessment === opt.value
                        ? 'bg-amber-600 text-white'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Feedback text */}
            <Textarea
              placeholder="La tua risposta alla prospettiva critica..."
              value={feedbackText}
              onChange={handleTextChange}
              rows={3}
              showCharCount
              maxLength={2000}
            />

            {/* Submit */}
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!feedbackText.trim() || submitMutation.isPending}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-slate-900',
                  'bg-amber-600 text-white hover:bg-amber-700',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                <Send className="h-3.5 w-3.5" />
                {submitMutation.isPending ? 'Invio...' : 'Invia Feedback'}
              </button>
            </div>

            {submitMutation.isError && (
              <p className="text-sm text-red-400">
                Errore nell&apos;invio del feedback. Riprova.
              </p>
            )}
          </div>
        )}
      </div>
    </Collapsible>
  );
}
