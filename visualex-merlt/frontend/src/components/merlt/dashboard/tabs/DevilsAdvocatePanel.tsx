/**
 * DevilsAdvocatePanel
 * ====================
 *
 * Pannello per il sistema Devil's Advocate (RLCF Pillar 4).
 * Mostra metriche di efficacia e consente di testare il trigger manualmente.
 *
 * Features:
 * - Effectiveness KPIs (triggers, feedbacks, engagement, keywords)
 * - Manual trigger test per un trace_id
 * - Feedback submission form
 *
 * @example
 * ```tsx
 * <DevilsAdvocatePanel />
 * ```
 */

import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  AlertTriangle,
  MessageSquare,
  Zap,
  Send,
  AlertCircle,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  getDAEffectiveness,
  checkDevilsAdvocate,
  submitDAFeedback,
} from '../../../../services/devilsAdvocateService';
import type {
  DAEffectivenessResponse,
  DACheckResponse,
} from '../../../../services/devilsAdvocateService';

// =============================================================================
// KPI CARD
// =============================================================================

interface KpiProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}

function KpiCard({ label, value, icon, color }: KpiProps) {
  return (
    <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        <span className={cn('text-sm', color)}>{icon}</span>
        <span className="text-xs text-slate-500 dark:text-slate-400">{label}</span>
      </div>
      <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">{value}</p>
    </div>
  );
}

// =============================================================================
// TRIGGER TEST
// =============================================================================

interface TriggerTestProps {
  onTriggered: (resp: DACheckResponse) => void;
}

function TriggerTestForm({ onTriggered }: TriggerTestProps) {
  const [traceId, setTraceId] = useState('');
  const [disagreement, setDisagreement] = useState('0.05');
  const [checking, setChecking] = useState(false);

  const handleCheck = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!traceId.trim()) return;

    setChecking(true);
    try {
      const resp = await checkDevilsAdvocate(traceId.trim(), parseFloat(disagreement));
      onTriggered(resp);
    } catch (err) {
      console.error('DA check failed:', err);
    } finally {
      setChecking(false);
    }
  };

  return (
    <form onSubmit={handleCheck} className="flex items-end gap-2">
      <div className="flex-1">
        <label htmlFor="da-trace" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
          Trace ID
        </label>
        <input
          id="da-trace"
          type="text"
          value={traceId}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTraceId(e.target.value)}
          placeholder="trace_abc123"
          className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          required
        />
      </div>
      <div className="w-24">
        <label htmlFor="da-disagreement" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
          Disagreement
        </label>
        <input
          id="da-disagreement"
          type="number"
          step="0.01"
          min="0"
          max="1"
          value={disagreement}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDisagreement(e.target.value)}
          className="w-full px-3 py-1.5 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        />
      </div>
      <button
        type="submit"
        disabled={checking}
        className="px-4 py-1.5 text-sm bg-amber-600 text-white rounded-md hover:bg-amber-700 disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-2"
      >
        {checking ? 'Check...' : 'Test Trigger'}
      </button>
    </form>
  );
}

// =============================================================================
// FEEDBACK FORM
// =============================================================================

interface FeedbackFormProps {
  traceId: string;
  onSubmitted: () => void;
}

function DAFeedbackForm({ traceId, onSubmitted }: FeedbackFormProps) {
  const [text, setText] = useState('');
  const [assessment, setAssessment] = useState('interesting' as 'valid' | 'weak' | 'interesting');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!text.trim()) return;

    setSubmitting(true);
    try {
      await submitDAFeedback({
        trace_id: traceId,
        feedback_text: text.trim(),
        assessment,
      });
      setText('');
      onSubmitted();
    } catch (err) {
      console.error('DA feedback submission failed:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <textarea
        value={text}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
        placeholder="Scrivi la tua risposta critica..."
        rows={3}
        className="w-full px-3 py-2 text-sm rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 resize-none"
        required
      />
      <div className="flex items-center gap-2">
        <select
          value={assessment}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
            setAssessment(e.target.value as 'valid' | 'weak' | 'interesting')
          }
          className="px-2 py-1.5 text-xs rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          <option value="valid">Valid</option>
          <option value="interesting">Interesting</option>
          <option value="weak">Weak</option>
        </select>
        <button
          type="submit"
          disabled={submitting}
          className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          <Send size={12} aria-hidden="true" />
          {submitting ? 'Invio...' : 'Invia Feedback'}
        </button>
      </div>
    </form>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function DevilsAdvocatePanel() {
  const [effectiveness, setEffectiveness] = useState(null as DAEffectivenessResponse | null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null as string | null);
  const [triggerResult, setTriggerResult] = useState(null as DACheckResponse | null);

  const fetchEffectiveness = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getDAEffectiveness();
      setEffectiveness(data);
    } catch (err) {
      setError("Errore caricamento metriche Devil's Advocate");
      console.error('Failed to load DA effectiveness:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEffectiveness();
  }, [fetchEffectiveness]);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle size={20} className="text-amber-500" aria-hidden="true" />
          <div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Devil's Advocate
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              RLCF Pillar 4 — Anti-conformismo
            </p>
          </div>
        </div>
        <button
          onClick={fetchEffectiveness}
          className="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
          aria-label="Aggiorna"
        >
          <RefreshCw size={16} aria-hidden="true" />
        </button>
      </div>

      <div className="p-4 space-y-5">
        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-6" role="status">
            <RefreshCw size={20} className="animate-spin text-blue-500" aria-hidden="true" />
            <span className="sr-only">Caricamento...</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-center py-4" role="alert">
            <AlertCircle size={24} className="mx-auto text-red-400 mb-2" aria-hidden="true" />
            <p className="text-sm text-slate-500">{error}</p>
          </div>
        )}

        {/* KPIs */}
        {!loading && effectiveness && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <KpiCard
                label="Triggers"
                value={String(effectiveness.total_triggers)}
                icon={<Zap size={14} />}
                color="text-amber-500"
              />
              <KpiCard
                label="Feedbacks"
                value={String(effectiveness.total_feedbacks)}
                icon={<MessageSquare size={14} />}
                color="text-blue-500"
              />
              <KpiCard
                label="Avg Engagement"
                value={effectiveness.avg_engagement.toFixed(2)}
                icon={<TrendingUpIcon />}
                color="text-green-500"
              />
              <KpiCard
                label="Avg Keywords"
                value={effectiveness.avg_keywords.toFixed(1)}
                icon={<AlertTriangle size={14} />}
                color="text-purple-500"
              />
            </div>

            {/* Description */}
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Il Devil's Advocate si attiva quando il disagreement tra expert scende sotto 0.1
              (alto consenso → rischio groupthink). Genera prompt critici per sfidare l'analisi.
            </p>
          </>
        )}

        {/* Trigger Test */}
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Test Trigger
          </h4>
          <TriggerTestForm onTriggered={setTriggerResult} />
        </div>

        {/* Trigger result */}
        {triggerResult && (
          <div
            className={cn(
              'p-3 rounded-lg border text-sm',
              triggerResult.triggered
                ? 'bg-amber-50 border-amber-200 dark:bg-amber-900/20 dark:border-amber-700'
                : 'bg-slate-50 border-slate-200 dark:bg-slate-700/50 dark:border-slate-600'
            )}
          >
            <p
              className={cn(
                'font-medium mb-1',
                triggerResult.triggered
                  ? 'text-amber-800 dark:text-amber-200'
                  : 'text-slate-700 dark:text-slate-300'
              )}
            >
              {triggerResult.triggered ? 'DA Triggered!' : 'Non triggered'}
            </p>
            <p className="text-xs text-slate-600 dark:text-slate-400">{triggerResult.message}</p>
            {triggerResult.critical_prompt && (
              <blockquote className="mt-2 pl-3 border-l-2 border-amber-400 text-xs italic text-slate-600 dark:text-slate-400">
                {triggerResult.critical_prompt}
              </blockquote>
            )}

            {/* Feedback form if triggered */}
            {triggerResult.triggered && triggerResult.critical_prompt && (
              <div className="mt-3">
                <h5 className="text-xs font-medium text-slate-600 dark:text-slate-300 mb-2">
                  Rispondi al prompt critico:
                </h5>
                <DAFeedbackForm
                  traceId="test"
                  onSubmitted={fetchEffectiveness}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Small inline icon to avoid importing from lucide (TrendingUp already used in PolicyEvolutionChart)
function TrendingUpIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
      <polyline points="16 7 22 7 22 13" />
    </svg>
  );
}

export default DevilsAdvocatePanel;
