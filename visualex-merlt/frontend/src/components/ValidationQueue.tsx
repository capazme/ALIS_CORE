/**
 * ValidationQueue Component
 *
 * Refactored: uses ValidationCard + ValidationBulkActionsBar
 * instead of basic approve/reject buttons.
 */

import { useState, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';
import { usePendingValidations } from '../hooks/usePendingValidations';
import { useValidationState } from '../hooks/useValidationState';
import { ValidationCard } from './merlt/ValidationCard';
import { ValidationBulkActionsBar } from './merlt/ValidationBulkActionsBar';
import { getCurrentUserId } from '../services/merltInit';
import { merltService } from '../services/merltService';
import type { VoteType } from '../types/merlt';

interface ValidationQueueProps {
  articleUrn: string;
  userId?: string;
}

export function ValidationQueue({ articleUrn, userId }: ValidationQueueProps): React.ReactElement {
  const effectiveUserId = userId || getCurrentUserId();
  const { validations: rawValidations, isLoading, submitDecision, refetch } = usePendingValidations(articleUrn, effectiveUserId);

  // Map raw validations to pending entities/relations for useValidationState
  const { entities, relations } = usePendingEntityRelationSplit(rawValidations);

  const validationState = useValidationState({
    entities,
    relations,
  });

  const {
    visibleItems,
    currentIndex,
    selectedItems,
    toggleSelection,
    clearSelection,
    selectHighConfidence,
    selectLowConfidence,
    bulkCandidates,
    skipItem,
    recordVote,
    voteHistory,
    navigateToNext,
  } = validationState;

  const [validatingId, setValidatingId] = useState(null as string | null);
  const [undoToast, setUndoToast] = useState(null as { id: string; name: string; vote: VoteType } | null);
  const [undoTimeout, setUndoTimeout] = useState(null as ReturnType<typeof setTimeout> | null);
  const [bulkProcessing, setBulkProcessing] = useState(false);

  const handleVote = useCallback(async (item: { id: string; name: string; type: 'entity' | 'relation' }, vote: VoteType) => {
    setValidatingId(item.id);
    recordVote(item.id, item.name, item.type, vote);

    // Show undo toast
    const toastData = { id: item.id, name: item.name, vote };
    setUndoToast(toastData);

    // Clear previous timeout
    if (undoTimeout) clearTimeout(undoTimeout);

    // Submit after 3s unless undone
    const timeout = setTimeout(async () => {
      try {
        const decision = vote === 'approve' ? 'approve' : 'reject';
        await submitDecision(item.id, decision, item.type);
      } finally {
        setValidatingId(null);
        setUndoToast(null);
      }
    }, 3000);

    setUndoTimeout(timeout);
    setValidatingId(null);
  }, [submitDecision, recordVote, undoTimeout]);

  const handleUndo = useCallback(() => {
    if (undoTimeout) {
      clearTimeout(undoTimeout);
      setUndoTimeout(null);
    }
    setUndoToast(null);
    refetch();
  }, [undoTimeout, refetch]);

  const handleBulkApprove = useCallback(async (itemIds: string[]) => {
    setBulkProcessing(true);
    try {
      for (const id of itemIds) {
        const item = visibleItems.find((v: { id: string }) => v.id === id) as { id: string; type: 'entity' | 'relation' } | undefined;
        if (item) {
          await submitDecision(id, 'approve', item.type);
        }
      }
    } finally {
      setBulkProcessing(false);
    }
  }, [visibleItems, submitDecision]);

  const handleBulkReject = useCallback(async (itemIds: string[]) => {
    setBulkProcessing(true);
    try {
      for (const id of itemIds) {
        const item = visibleItems.find((v: { id: string }) => v.id === id) as { id: string; type: 'entity' | 'relation' } | undefined;
        if (item) {
          await submitDecision(id, 'reject', item.type);
        }
      }
    } finally {
      setBulkProcessing(false);
    }
  }, [visibleItems, submitDecision]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32" role="status">
        <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        <span className="sr-only">Caricamento validazioni...</span>
      </div>
    );
  }

  if (visibleItems.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-slate-500 dark:text-slate-400 text-sm">Nessuna validazione in attesa.</p>
        <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">Le nuove proposte appariranno qui.</p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-2">
      {/* Bulk actions bar */}
      <ValidationBulkActionsBar
        highConfidenceItems={bulkCandidates.highConfidence}
        lowConfidenceItems={bulkCandidates.lowConfidence}
        selectedItems={selectedItems}
        totalItems={visibleItems.length}
        isProcessing={bulkProcessing}
        onApproveAll={handleBulkApprove}
        onRejectAll={handleBulkReject}
        onSelectHighConfidence={selectHighConfidence}
        onSelectLowConfidence={selectLowConfidence}
        onClearSelection={clearSelection}
      />

      {/* Validation cards */}
      <AnimatePresence mode="popLayout">
        {visibleItems.map((item: { id: string; name: string; type: 'entity' | 'relation' }, index: number) => (
          <ValidationCard
            key={item.id}
            item={item as import('../hooks/useValidationState').ValidationItem}
            isValidating={validatingId === item.id}
            isFocused={currentIndex === index}
            isSelected={selectedItems.has(item.id)}
            hasVoted={voteHistory.some((v: { itemId: string }) => v.itemId === item.id)}
            onVote={(vote: import('../types/merlt').VoteType) => handleVote(item, vote)}
            onEdit={() => { /* TODO: edit flow */ }}
            onSkip={() => skipItem(item.id, item.type)}
            onToggleSelect={() => toggleSelection(item.id)}
            selectionMode={selectedItems.size > 0}
          />
        ))}
      </AnimatePresence>

      {/* Undo toast */}
      {undoToast && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2.5 bg-slate-900 dark:bg-slate-700 text-white rounded-lg shadow-xl text-sm animate-in slide-in-from-bottom-4 duration-200">
          <span>
            {undoToast.vote === 'approve' ? 'Approvato' : 'Rifiutato'}: {undoToast.name}
          </span>
          <button
            onClick={handleUndo}
            className="px-2 py-0.5 text-xs font-medium bg-white/20 hover:bg-white/30 rounded transition-colors"
          >
            Annulla
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Split raw validation items into entities/relations for useValidationState.
 */
function usePendingEntityRelationSplit(rawValidations: Array<{
  id: string;
  type: 'entity' | 'relation';
  content: { name: string; description?: string; confidence: number };
  articleUrn: string;
  createdAt: string;
}> | undefined) {
  if (!rawValidations) return { entities: [], relations: [] };

  // Create minimal PendingEntity/PendingRelation objects that useValidationState needs
  const entities = rawValidations
    .filter((v) => v.type === 'entity')
    .map((v) => ({
      id: v.id,
      nome: v.content.name,
      tipo: 'concetto' as const,
      descrizione: v.content.description || '',
      articoli_correlati: [v.articleUrn],
      ambito: '',
      fonte: 'llm',
      llm_confidence: v.content.confidence,
      raw_context: '',
      validation_status: 'pending' as const,
      approval_score: 0,
      rejection_score: 0,
      votes_count: 0,
      contributed_by: '',
      contributor_authority: 0,
      created_at: v.createdAt,
    }));

  const relations = rawValidations
    .filter((v) => v.type === 'relation')
    .map((v) => ({
      id: v.id,
      source_urn: v.content.name.split(' → ')[0] || '',
      target_urn: v.content.name.split(' → ')[1] || '',
      relation_type: 'CORRELATO' as const,
      fonte: 'llm',
      llm_confidence: v.content.confidence,
      evidence: v.content.description || '',
      validation_status: 'pending' as const,
      approval_score: 0,
      rejection_score: 0,
      votes_count: 0,
      contributed_by: '',
      contributor_authority: 0,
      created_at: v.createdAt,
    }));

  return { entities, relations };
}
