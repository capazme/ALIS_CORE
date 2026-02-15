/**
 * ValidationBulkActionsBar
 * ========================
 *
 * Smart bulk actions bar for validation panel.
 * Shows contextual actions based on available items:
 * - "Approva tutti >90%" when high-confidence items exist
 * - "Rifiuta tutti <50%" when low-confidence items exist
 * - Selection mode for manual bulk operations
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2,
  XCircle,
  Sparkles,
  AlertTriangle,
  ChevronDown,
  X,
  Zap,
  Filter,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { BulkActionCandidate } from '../../hooks/useValidationState';

interface ValidationBulkActionsBarProps {
  /** High confidence items (>90%) */
  highConfidenceItems: BulkActionCandidate[];
  /** Low confidence items (<50%) */
  lowConfidenceItems: BulkActionCandidate[];
  /** Currently selected item IDs */
  selectedItems: Set<string>;
  /** Total visible items count */
  totalItems: number;
  /** Is bulk action in progress */
  isProcessing: boolean;
  /** Callbacks */
  onApproveAll: (itemIds: string[]) => Promise<void>;
  onRejectAll: (itemIds: string[]) => Promise<void>;
  onSelectHighConfidence: () => void;
  onSelectLowConfidence: () => void;
  onClearSelection: () => void;
  className?: string;
}

export function ValidationBulkActionsBar({
  highConfidenceItems,
  lowConfidenceItems,
  selectedItems,
  totalItems,
  isProcessing,
  onApproveAll,
  onRejectAll,
  onSelectHighConfidence,
  onSelectLowConfidence,
  onClearSelection,
  className,
}: ValidationBulkActionsBarProps) {
  const [showConfirmDialog, setShowConfirmDialog] = useState(null as 'approve' | 'reject' | null);
  const [processingAction, setProcessingAction] = useState(null as 'approve' | 'reject' | null);

  const hasHighConfidence = highConfidenceItems.length > 0;
  const hasLowConfidence = lowConfidenceItems.length > 0;
  const hasSelection = selectedItems.size > 0;

  // Don't show bar if no bulk actions available
  if (!hasHighConfidence && !hasLowConfidence && !hasSelection) {
    return null;
  }

  const handleApproveHighConfidence = async () => {
    setProcessingAction('approve');
    try {
      const ids = highConfidenceItems.map((c) => c.item.id);
      await onApproveAll(ids);
    } finally {
      setProcessingAction(null);
      setShowConfirmDialog(null);
    }
  };

  const handleRejectLowConfidence = async () => {
    setProcessingAction('reject');
    try {
      const ids = lowConfidenceItems.map((c) => c.item.id);
      await onRejectAll(ids);
    } finally {
      setProcessingAction(null);
      setShowConfirmDialog(null);
    }
  };

  const handleApproveSelected = async () => {
    setProcessingAction('approve');
    try {
      await onApproveAll(Array.from(selectedItems));
    } finally {
      setProcessingAction(null);
      onClearSelection();
    }
  };

  const handleRejectSelected = async () => {
    setProcessingAction('reject');
    try {
      await onRejectAll(Array.from(selectedItems));
    } finally {
      setProcessingAction(null);
      onClearSelection();
    }
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={cn(
          'flex items-center gap-2 p-2 rounded-lg',
          'bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700',
          className
        )}
      >
        <Zap size={14} className="text-amber-500 flex-shrink-0" aria-hidden="true" />

        <div className="flex-1 flex items-center gap-2 overflow-x-auto">
          {/* Smart suggestions */}
          {!hasSelection && (
            <>
              {/* Approve high confidence */}
              {hasHighConfidence && (
                <button
                  onClick={() => setShowConfirmDialog('approve')}
                  disabled={isProcessing}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium min-h-[44px]',
                    'bg-emerald-100 hover:bg-emerald-200 text-emerald-700',
                    'dark:bg-emerald-900/30 dark:hover:bg-emerald-900/50 dark:text-emerald-300',
                    'transition-colors whitespace-nowrap',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                  )}
                >
                  <Sparkles size={12} aria-hidden="true" />
                  Approva {highConfidenceItems.length} ({'>'}90%)
                </button>
              )}

              {/* Reject low confidence */}
              {hasLowConfidence && (
                <button
                  onClick={() => setShowConfirmDialog('reject')}
                  disabled={isProcessing}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium min-h-[44px]',
                    'bg-red-100 hover:bg-red-200 text-red-700',
                    'dark:bg-red-900/30 dark:hover:bg-red-900/50 dark:text-red-300',
                    'transition-colors whitespace-nowrap',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                  )}
                >
                  <AlertTriangle size={12} aria-hidden="true" />
                  Rifiuta {lowConfidenceItems.length} ({'<'}50%)
                </button>
              )}

              {/* Filter buttons */}
              <div className="h-4 w-px bg-slate-300 dark:bg-slate-600 mx-1" aria-hidden="true" />
              <button
                onClick={onSelectHighConfidence}
                disabled={!hasHighConfidence}
                className={cn(
                  'p-1.5 rounded-md text-slate-500 hover:text-emerald-600 hover:bg-emerald-50 min-w-[44px] min-h-[44px] flex items-center justify-center',
                  'dark:hover:bg-emerald-900/20 dark:hover:text-emerald-400',
                  'transition-colors disabled:opacity-30 disabled:cursor-not-allowed',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                )}
                aria-label="Seleziona alta confidenza"
                title="Seleziona alta confidenza"
              >
                <Filter size={12} aria-hidden="true" />
              </button>
            </>
          )}

          {/* Selection mode */}
          {hasSelection && (
            <>
              <span className="text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                {selectedItems.size} selezionati
              </span>

              <button
                onClick={handleApproveSelected}
                disabled={isProcessing}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 rounded text-xs font-medium min-h-[44px]',
                  'bg-emerald-500 hover:bg-emerald-600 text-white',
                  'transition-colors disabled:opacity-50',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                )}
              >
                <CheckCircle2 size={10} aria-hidden="true" />
                Approva
              </button>

              <button
                onClick={handleRejectSelected}
                disabled={isProcessing}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 rounded text-xs font-medium min-h-[44px]',
                  'bg-red-500 hover:bg-red-600 text-white',
                  'transition-colors disabled:opacity-50',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                )}
              >
                <XCircle size={10} aria-hidden="true" />
                Rifiuta
              </button>

              <button
                onClick={onClearSelection}
                aria-label="Annulla selezione"
                className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 min-w-[44px] min-h-[44px] flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                title="Annulla selezione"
              >
                <X size={12} aria-hidden="true" />
              </button>
            </>
          )}
        </div>
      </motion.div>

      {/* Confirmation Dialog */}
      <AnimatePresence>
        {showConfirmDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[70] bg-black/50 flex items-center justify-center p-4"
            onClick={() => setShowConfirmDialog(null)}
          >
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-label={showConfirmDialog === 'approve' ? 'Conferma approvazione' : 'Conferma rifiuto'}
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
              className={cn(
                'bg-white dark:bg-slate-900 rounded-xl shadow-xl',
                'border border-slate-200 dark:border-slate-800',
                'p-6 max-w-sm w-full'
              )}
            >
              <div className="flex items-start gap-4">
                <div
                  className={cn(
                    'w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0',
                    showConfirmDialog === 'approve'
                      ? 'bg-emerald-100 dark:bg-emerald-900/30'
                      : 'bg-red-100 dark:bg-red-900/30'
                  )}
                >
                  {showConfirmDialog === 'approve' ? (
                    <CheckCircle2
                      size={24}
                      className="text-emerald-600 dark:text-emerald-400"
                    />
                  ) : (
                    <XCircle size={24} className="text-red-600 dark:text-red-400" />
                  )}
                </div>

                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-1">
                    {showConfirmDialog === 'approve'
                      ? 'Approva elementi'
                      : 'Rifiuta elementi'}
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                    {showConfirmDialog === 'approve'
                      ? `Stai per approvare ${highConfidenceItems.length} elementi con confidenza >90%. Questa azione non può essere annullata.`
                      : `Stai per rifiutare ${lowConfidenceItems.length} elementi con confidenza <50%. Questa azione non può essere annullata.`}
                  </p>

                  {/* Preview items */}
                  <div className="max-h-32 overflow-y-auto mb-4 space-y-1">
                    {(showConfirmDialog === 'approve'
                      ? highConfidenceItems
                      : lowConfidenceItems
                    )
                      .slice(0, 5)
                      .map((candidate) => (
                        <div
                          key={candidate.item.id}
                          className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400"
                        >
                          <span className="w-8 text-right font-mono">
                            {Math.round(candidate.confidence * 100)}%
                          </span>
                          <span className="truncate">{candidate.item.name}</span>
                        </div>
                      ))}
                    {(showConfirmDialog === 'approve'
                      ? highConfidenceItems
                      : lowConfidenceItems
                    ).length > 5 && (
                      <div className="text-xs text-slate-400 italic">
                        ...e altri{' '}
                        {(showConfirmDialog === 'approve'
                          ? highConfidenceItems
                          : lowConfidenceItems
                        ).length - 5}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowConfirmDialog(null)}
                      disabled={processingAction !== null}
                      className={cn(
                        'flex-1 px-4 py-2 rounded-lg text-sm font-medium min-h-[44px]',
                        'bg-slate-100 hover:bg-slate-200 text-slate-700',
                        'dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-300',
                        'transition-colors disabled:opacity-50',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                      )}
                    >
                      Annulla
                    </button>
                    <button
                      onClick={
                        showConfirmDialog === 'approve'
                          ? handleApproveHighConfidence
                          : handleRejectLowConfidence
                      }
                      disabled={processingAction !== null}
                      className={cn(
                        'flex-1 px-4 py-2 rounded-lg text-sm font-medium text-white min-h-[44px]',
                        'transition-colors disabled:opacity-50',
                        showConfirmDialog === 'approve'
                          ? 'bg-emerald-600 hover:bg-emerald-700'
                          : 'bg-red-600 hover:bg-red-700',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                      )}
                    >
                      {processingAction !== null
                        ? 'Elaborazione...'
                        : showConfirmDialog === 'approve'
                        ? 'Approva tutti'
                        : 'Rifiuta tutti'}
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
