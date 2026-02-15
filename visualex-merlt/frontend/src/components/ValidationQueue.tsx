/**
 * ValidationQueue Component
 *
 * Shows pending validations for the current article.
 * Users can approve or reject proposed entities/relations.
 */

import { useState } from 'react';
import { usePendingValidations } from '../hooks/usePendingValidations';

interface ValidationQueueProps {
  articleUrn: string;
  userId?: string;
}

export function ValidationQueue({ articleUrn, userId }: ValidationQueueProps): React.ReactElement {
  const { validations, isLoading, submitDecision } = usePendingValidations(articleUrn, userId);
  const [submitting, setSubmitting] = useState(null as string | null);

  const handleDecision = async (validationId: string, decision: 'approve' | 'reject', type: 'entity' | 'relation') => {
    setSubmitting(validationId);
    try {
      await submitDecision(validationId, decision, type);
    } finally {
      setSubmitting(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32" role="status">
        <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        <span className="sr-only">Caricamento validazioni...</span>
      </div>
    );
  }

  if (!validations || validations.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-slate-500 dark:text-slate-400 text-sm">Nessuna validazione in attesa.</p>
        <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">Le nuove proposte appariranno qui.</p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-2">
      {validations.map((validation) => (
        <div key={validation.id} className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-start justify-between">
            <div>
              <span
                className={`
                  text-xs px-1.5 py-0.5 rounded
                  ${validation.type === 'entity' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'}
                `}
              >
                {validation.type}
              </span>
              <p className="mt-1 text-sm font-medium text-slate-900 dark:text-slate-100">{validation.content.name}</p>
              {validation.content.description && (
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{validation.content.description}</p>
              )}
            </div>
            <span className="text-xs text-slate-400 dark:text-slate-500">
              {Math.round(validation.content.confidence * 100)}%
            </span>
          </div>

          <div className="flex gap-2 mt-3">
            <button
              onClick={() => handleDecision(validation.id, 'approve', validation.type)}
              disabled={submitting === validation.id}
              aria-label={`Approva ${validation.content.name}`}
              className="
                flex-1 px-2 py-1.5 text-xs font-medium min-h-[44px]
                bg-green-600 text-white rounded
                hover:bg-green-700 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
              "
            >
              {submitting === validation.id ? (
                <span role="status">
                  <span aria-hidden="true">...</span>
                  <span className="sr-only">Invio in corso...</span>
                </span>
              ) : 'Approva'}
            </button>
            <button
              onClick={() => handleDecision(validation.id, 'reject', validation.type)}
              disabled={submitting === validation.id}
              aria-label={`Rifiuta ${validation.content.name}`}
              className="
                flex-1 px-2 py-1.5 text-xs font-medium min-h-[44px]
                bg-red-600 text-white rounded
                hover:bg-red-700 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
              "
            >
              {submitting === validation.id ? (
                <span role="status">
                  <span aria-hidden="true">...</span>
                  <span className="sr-only">Invio in corso...</span>
                </span>
              ) : 'Rifiuta'}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
