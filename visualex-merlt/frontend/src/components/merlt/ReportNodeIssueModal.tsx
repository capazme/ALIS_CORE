/**
 * ReportNodeIssueModal
 * ====================
 *
 * Modal per segnalare un problema su un nodo O una relazione del Knowledge Graph.
 * Fa parte del ciclo RLCF: se le segnalazioni raggiungono una soglia,
 * l'entità torna in validazione (needs_revision).
 */

import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { AlertTriangle, Loader2, CheckCircle2, AlertCircle, Info, ArrowRight, Link2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { reportNodeIssue } from '../../services/merltService';
import type { SubgraphNode, SubgraphEdge, IssueType, IssueSeverity } from '../../types/merlt';
import { ISSUE_TYPE_LABELS, ISSUE_SEVERITY_LABELS } from '../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

/** Dati per segnalare una relazione, include info sui nodi source/target */
export type EdgeReportData = {
  edge: SubgraphEdge;
  sourceNode?: SubgraphNode;
  targetNode?: SubgraphNode;
};

interface ReportNodeIssueModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  /** Nodo da segnalare (mutualmente esclusivo con edgeData) */
  node?: SubgraphNode | null;
  /** Relazione da segnalare (mutualmente esclusivo con node) */
  edgeData?: EdgeReportData | null;
  userId: string;
}

// Group issue types by category
const ISSUE_TYPES_GROUPED = {
  errors: [
    'factual_error',
    'wrong_relation',
    'wrong_type',
    'duplicate',
    'outdated',
  ] as IssueType[],
  suggestions: [
    'missing_relation',
    'incomplete',
    'improve_label',
    'other',
  ] as IssueType[],
};

// =============================================================================
// COMPONENT
// =============================================================================

export function ReportNodeIssueModal({
  isOpen,
  onClose,
  onSuccess,
  node,
  edgeData,
  userId,
}: ReportNodeIssueModalProps) {
  // Determine if we're reporting a node or an edge
  const isEdgeReport = !!edgeData?.edge;
  const entityId = isEdgeReport
    ? `rel_${edgeData.edge.source}_${edgeData.edge.type}_${edgeData.edge.target}`
    : node?.id;

  // Form state - default to wrong_relation for edges
  const [issueType, setIssueType] = useState(isEdgeReport ? 'wrong_relation' : 'factual_error' as IssueType);
  const [severity, setSeverity] = useState('medium' as IssueSeverity);
  const [description, setDescription] = useState('');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null as string | null);
  const [success, setSuccess] = useState(false);
  const [mergedWith, setMergedWith] = useState(null as string | null);

  // Reset form
  const resetForm = () => {
    setIssueType(isEdgeReport ? 'wrong_relation' : 'factual_error');
    setSeverity('medium');
    setDescription('');
    setError(null);
    setSuccess(false);
    setMergedWith(null);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!entityId) {
      setError('Nessuna entità selezionata');
      return;
    }

    if (!description.trim() || description.trim().length < 10) {
      setError('La descrizione deve essere di almeno 10 caratteri');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await reportNodeIssue({
        entity_id: entityId,
        issue_type: issueType,
        severity,
        description: description.trim(),
        user_id: userId,
      });

      if (result.success) {
        setSuccess(true);
        if (result.status === 'merged') {
          setMergedWith(result.merged_with || null);
        }

        // Callback after short delay
        setTimeout(() => {
          onSuccess?.();
          handleClose();
        }, 2000);
      } else {
        setError('Errore nella creazione della segnalazione');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Errore nella segnalazione';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!node && !edgeData) return null;

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Segnala Problema" size="md">
      {success ? (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-amber-100 dark:bg-amber-900/30 text-amber-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 size={28} aria-hidden="true" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2" role="status">
            Segnalazione inviata!
          </h3>
          <p className="text-slate-500 dark:text-slate-400">
            {mergedWith
              ? 'La tua segnalazione e\' stata unita a una esistente.'
              : 'La community votera\' sulla validita\' della segnalazione.'}
          </p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Entity preview - Node or Edge */}
          <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            {isEdgeReport && edgeData ? (
              // Edge/Relation preview
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                  <Link2 size={14} />
                  <span>Relazione da segnalare</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  {/* Source Node */}
                  <div className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600">
                    <div className="w-6 h-6 rounded bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                      <span className="text-purple-600 dark:text-purple-400 font-medium text-xs">
                        {edgeData.sourceNode?.type?.charAt(0) || 'S'}
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900 dark:text-white">
                        {edgeData.sourceNode?.label || edgeData.edge.source}
                      </p>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400">
                        {edgeData.sourceNode?.type || 'Nodo'}
                      </p>
                    </div>
                  </div>

                  {/* Arrow with relation type */}
                  <div className="flex items-center gap-1 px-2 py-1 bg-amber-100 dark:bg-amber-900/30 rounded text-amber-700 dark:text-amber-300">
                    <ArrowRight size={14} />
                    <span className="text-xs font-bold">{edgeData.edge.type}</span>
                  </div>

                  {/* Target Node */}
                  <div className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600">
                    <div className="w-6 h-6 rounded bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                      <span className="text-blue-600 dark:text-blue-400 font-medium text-xs">
                        {edgeData.targetNode?.type?.charAt(0) || 'T'}
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900 dark:text-white">
                        {edgeData.targetNode?.label || edgeData.edge.target}
                      </p>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400">
                        {edgeData.targetNode?.type || 'Nodo'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : node ? (
              // Node preview
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-primary-600 dark:text-primary-400 font-medium text-sm">
                    {node.type?.charAt(0).toUpperCase() || 'N'}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-900 dark:text-white truncate">
                    {node.label}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                    {node.type} {node.urn && `• ${node.urn.slice(-30)}...`}
                  </p>
                </div>
              </div>
            ) : null}
          </div>

          {/* Info banner */}
          <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/30 rounded-lg">
            <Info size={18} className="text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Se la segnalazione riceve abbastanza consenso, {isEdgeReport ? 'la relazione' : 'il nodo'} tornera' in validazione per essere {isEdgeReport ? 'corretta' : 'corretto'}.
            </p>
          </div>

          {/* Issue Type */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
              Tipo di problema *
            </label>

            {/* Errors section */}
            <p className="text-xs font-medium text-red-600 dark:text-red-400 mb-2 flex items-center gap-1.5">
              <AlertTriangle size={12} />
              Errori
            </p>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {ISSUE_TYPES_GROUPED.errors.map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => setIssueType(type)}
                  className={cn(
                    'px-3 py-2.5 rounded-lg border-2 text-left transition-all text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                    issueType === type
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 text-slate-700 dark:text-slate-300'
                  )}
                >
                  {ISSUE_TYPE_LABELS[type].label}
                </button>
              ))}
            </div>

            {/* Suggestions section */}
            <p className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-2 flex items-center gap-1.5">
              <Info size={12} />
              Suggerimenti
            </p>
            <div className="grid grid-cols-2 gap-2">
              {ISSUE_TYPES_GROUPED.suggestions.map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => setIssueType(type)}
                  className={cn(
                    'px-3 py-2.5 rounded-lg border-2 text-left transition-all text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                    issueType === type
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 text-slate-700 dark:text-slate-300'
                  )}
                >
                  {ISSUE_TYPE_LABELS[type].label}
                </button>
              ))}
            </div>
          </div>

          {/* Severity */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Gravita' *
            </label>
            <div className="flex gap-2">
              {(['low', 'medium', 'high'] as IssueSeverity[]).map((sev) => (
                <button
                  key={sev}
                  type="button"
                  onClick={() => setSeverity(sev)}
                  className={cn(
                    'flex-1 px-4 py-2.5 rounded-lg border-2 transition-all text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                    severity === sev
                      ? sev === 'high'
                        ? 'border-red-500 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                        : sev === 'medium'
                        ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300'
                        : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 text-slate-600 dark:text-slate-400'
                  )}
                >
                  {ISSUE_SEVERITY_LABELS[sev].label}
                </button>
              ))}
            </div>
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Descrizione *
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Descrivi il problema in dettaglio. Cosa e' errato? Quale sarebbe la correzione?"
              rows={4}
              className={cn(
                'w-full px-4 py-3 rounded-xl border transition-all resize-none',
                'bg-slate-50 dark:bg-slate-800',
                'border-slate-200 dark:border-slate-700',
                'text-slate-900 dark:text-white placeholder-slate-400',
                'focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none'
              )}
            />
            <p className="mt-1 text-xs text-slate-400">
              Minimo 10 caratteri ({description.length}/10)
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400" role="alert">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2.5 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              Annulla
            </button>
            <button
              type="submit"
              disabled={isSubmitting || description.trim().length < 10}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all',
                'bg-amber-600 hover:bg-amber-700 text-white',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'
              )}
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                  <span>Invio...</span>
                  <span className="sr-only">Invio in corso</span>
                </>
              ) : (
                <>
                  <AlertTriangle size={18} aria-hidden="true" />
                  Segnala
                </>
              )}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}
