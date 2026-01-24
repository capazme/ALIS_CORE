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
}

export function ValidationQueue({ articleUrn }: ValidationQueueProps): React.ReactElement {
  const { validations, isLoading, submitDecision } = usePendingValidations(articleUrn);
  const [submitting, setSubmitting] = useState<string | null>(null);

  const handleDecision = async (validationId: string, decision: 'approve' | 'reject') => {
    setSubmitting(validationId);
    try {
      await submitDecision(validationId, decision);
    } finally {
      setSubmitting(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!validations || validations.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-gray-500 text-sm">Nessuna validazione in attesa.</p>
        <p className="text-gray-400 text-xs mt-1">Le nuove proposte appariranno qui.</p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-2">
      {validations.map((validation) => (
        <div key={validation.id} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex items-start justify-between">
            <div>
              <span
                className={`
                  text-xs px-1.5 py-0.5 rounded
                  ${validation.type === 'entity' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}
                `}
              >
                {validation.type}
              </span>
              <p className="mt-1 text-sm font-medium text-gray-900">{validation.content.name}</p>
              {validation.content.description && (
                <p className="text-xs text-gray-500 mt-0.5">{validation.content.description}</p>
              )}
            </div>
            <span className="text-xs text-gray-400">
              {Math.round(validation.content.confidence * 100)}%
            </span>
          </div>

          <div className="flex gap-2 mt-3">
            <button
              onClick={() => handleDecision(validation.id, 'approve')}
              disabled={submitting === validation.id}
              className="
                flex-1 px-2 py-1.5 text-xs font-medium
                bg-green-600 text-white rounded
                hover:bg-green-700 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            >
              {submitting === validation.id ? '...' : 'Approva'}
            </button>
            <button
              onClick={() => handleDecision(validation.id, 'reject')}
              disabled={submitting === validation.id}
              className="
                flex-1 px-2 py-1.5 text-xs font-medium
                bg-red-600 text-white rounded
                hover:bg-red-700 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            >
              {submitting === validation.id ? '...' : 'Rifiuta'}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
