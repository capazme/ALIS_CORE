/**
 * MERL-T Inspector Panel
 * ======================
 *
 * Industrial-grade validation panel with:
 * - Keyboard shortcuts (A/R/E/J/K/S/?)
 * - Inline description preview
 * - Vote weight indicator
 * - Bulk actions bar
 * - Skip queue management
 * - Vote history timeline
 * - Vote confirmation animations
 * - Undo action toast
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Brain,
  CheckCircle2,
  Clock,
  Plus,
  Sparkles,
  AlertCircle,
  Loader2,
  Network,
  Bot,
  Users,
  Send,
  Timer,
  History,
  Keyboard,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { EditEntityDrawer } from './EditEntityDrawer';
import { EditRelationDrawer } from './EditRelationDrawer';
import { ValidationCard } from './ValidationCard';
import { ValidationBulkActionsBar } from './ValidationBulkActionsBar';
import { SkipQueuePanel } from './SkipQueuePanel';
import { VoteHistoryModal } from './VoteHistoryModal';
import { ValidationKeyboardHelpModal } from './ValidationKeyboardHelpModal';
import { useValidationState, type ValidationItem, type VoteHistoryEntry } from '../../hooks/useValidationState';
import { useValidationKeyboard } from '../../hooks/useValidationKeyboard';
import { showUndoToast } from '../../hooks/useUndoableAction';
import type {
  PendingEntity,
  PendingRelation,
  VoteType,
} from '../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface MerltInspectorPanelProps {
  isOpen: boolean;
  onClose: () => void;

  // User info
  userId: string;
  /** User authority score A_u [0-1] from RLCF */
  userAuthorityScore?: number;

  // Article info
  tipoAtto: string;
  articolo: string;

  // Status
  isLoading: boolean;
  isValidating: boolean;
  isEnriching?: boolean;
  progressMessage?: string;
  waitingForOther?: boolean;
  waitProgress?: number;
  waitElapsed?: number;
  error: string | null;

  // Data
  inGraph: boolean;
  entityCount: number;
  pendingEntities: PendingEntity[];
  pendingRelations: PendingRelation[];

  // Vote tracking
  hasVotedEntity?: (entityId: string) => boolean;
  hasVotedRelation?: (relationId: string) => boolean;
  uncommittedVoteCount?: number;

  // Actions
  onValidateEntity: (entityId: string, vote: VoteType, comment?: string) => Promise<void>;
  onValidateRelation: (relationId: string, vote: VoteType, comment?: string) => Promise<void>;
  onRequestEnrichment: () => void;
  onCommitFeedback?: () => void;
  onProposeEntity?: () => void;
  onProposeRelation?: () => void;
  onRefresh?: () => void;
}

// =============================================================================
// SUBCOMPONENTS
// =============================================================================

function StatusBadge({ inGraph, entityCount }: { inGraph: boolean; entityCount: number }) {
  if (inGraph && entityCount > 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30 rounded-lg">
        <CheckCircle2 size={16} className="text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
        <div className="text-sm">
          <span className="font-medium text-emerald-700 dark:text-emerald-300">Nel Knowledge Graph</span>
          <span className="text-emerald-600 dark:text-emerald-400 ml-2">
            {entityCount} entità
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg">
      <Network size={16} className="text-slate-400" aria-hidden="true" />
      <span className="text-sm text-slate-600 dark:text-slate-400">Articolo non ancora analizzato</span>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function MerltInspectorPanel({
  isOpen,
  onClose,
  userId,
  userAuthorityScore = 1.0,
  tipoAtto,
  articolo,
  isLoading,
  isValidating,
  isEnriching = false,
  progressMessage,
  waitingForOther = false,
  waitProgress = 0,
  waitElapsed = 0,
  error,
  inGraph,
  entityCount,
  pendingEntities,
  pendingRelations,
  hasVotedEntity = () => false,
  hasVotedRelation = () => false,
  uncommittedVoteCount = 0,
  onValidateEntity,
  onValidateRelation,
  onRequestEnrichment,
  onCommitFeedback,
  onProposeEntity,
  onProposeRelation,
  onRefresh,
}: MerltInspectorPanelProps) {
  const [showValidated, setShowValidated] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [showVoteHistory, setShowVoteHistory] = useState(false);

  // Edit drawer state
  const [editingEntity, setEditingEntity] = useState(null as PendingEntity | null);
  const [editingRelation, setEditingRelation] = useState(null as PendingRelation | null);

  // Card refs for focus management
  const cardRefs = useRef(new Map() as Map<string, HTMLDivElement>);

  // Use validation state hook
  const {
    allItems,
    visibleItems,
    currentItem,
    currentIndex,
    navigateToNext,
    navigateToPrev,
    navigateToIndex,
    skipQueue,
    skipItem,
    unSkipItem,
    clearSkipQueue,
    voteHistory,
    recordVote,
    markVotesAsCommitted,
    removeVoteFromHistory,
    selectedItems,
    toggleSelection,
    selectAll,
    clearSelection,
    selectHighConfidence,
    selectLowConfidence,
    bulkCandidates,
  } = useValidationState({
    entities: pendingEntities,
    relations: pendingRelations,
    hasVotedEntity,
    hasVotedRelation,
  });

  const pendingCount = pendingEntities.length + pendingRelations.length;
  const showEnrichmentCTA = entityCount === 0 && pendingCount === 0 && !isEnriching;

  // Scroll to focused card
  useEffect(() => {
    if (currentItem) {
      const ref = cardRefs.current.get(currentItem.id);
      ref?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [currentItem]);

  // Handle vote with undo toast
  const handleVote = useCallback(
    async (itemId: string, itemType: 'entity' | 'relation', vote: VoteType) => {
      const item = allItems.find((i: ValidationItem) => i.id === itemId);
      if (!item) return;

      // Record in history immediately
      recordVote(itemId, item.name, itemType, vote);

      // Show undo toast and execute vote
      const completed = await showUndoToast({
        action: async () => {
          if (itemType === 'entity') {
            await onValidateEntity(itemId, vote);
          } else {
            await onValidateRelation(itemId, vote);
          }
          return { itemId, vote };
        },
        undo: () => {
          // Remove from history on undo
          const entry = voteHistory.find((h: VoteHistoryEntry) => h.itemId === itemId);
          if (entry) {
            removeVoteFromHistory(entry.id);
          }
          // Note: Backend undo would require an API call
        },
        message:
          vote === 'approve'
            ? `${item.name} approvato`
            : vote === 'reject'
            ? `${item.name} rifiutato`
            : `${item.name} modificato`,
        duration: 5000,
      });

      // Navigate to next item after vote
      if (completed) {
        navigateToNext();
      }
    },
    [
      allItems,
      recordVote,
      onValidateEntity,
      onValidateRelation,
      voteHistory,
      removeVoteFromHistory,
      navigateToNext,
    ]
  );

  // Handle edit
  const handleEdit = useCallback(
    (itemId: string, itemType: 'entity' | 'relation') => {
      if (itemType === 'entity') {
        const entity = pendingEntities.find((e) => e.id === itemId);
        if (entity) setEditingEntity(entity);
      } else {
        const relation = pendingRelations.find((r) => r.id === itemId);
        if (relation) setEditingRelation(relation);
      }
    },
    [pendingEntities, pendingRelations]
  );

  // Handle skip
  const handleSkip = useCallback(
    (itemId: string, itemType: 'entity' | 'relation') => {
      skipItem(itemId, itemType);
    },
    [skipItem]
  );

  // Get item name for skip queue
  const getItemName = useCallback(
    (itemId: string) => {
      const item = allItems.find((i: ValidationItem) => i.id === itemId);
      return item?.name ?? itemId;
    },
    [allItems]
  );

  // Bulk action handlers
  const handleBulkApprove = useCallback(
    async (itemIds: string[]) => {
      for (const itemId of itemIds) {
        const item = allItems.find((i: ValidationItem) => i.id === itemId);
        if (item) {
          if (item.type === 'entity') {
            await onValidateEntity(itemId, 'approve');
          } else {
            await onValidateRelation(itemId, 'approve');
          }
          recordVote(itemId, item.name, item.type, 'approve');
        }
      }
    },
    [allItems, onValidateEntity, onValidateRelation, recordVote]
  );

  const handleBulkReject = useCallback(
    async (itemIds: string[]) => {
      for (const itemId of itemIds) {
        const item = allItems.find((i: ValidationItem) => i.id === itemId);
        if (item) {
          if (item.type === 'entity') {
            await onValidateEntity(itemId, 'reject');
          } else {
            await onValidateRelation(itemId, 'reject');
          }
          recordVote(itemId, item.name, item.type, 'reject');
        }
      }
    },
    [allItems, onValidateEntity, onValidateRelation, recordVote]
  );

  // Commit feedback handler
  const handleCommitFeedback = useCallback(() => {
    onCommitFeedback?.();
    markVotesAsCommitted();
  }, [onCommitFeedback, markVotesAsCommitted]);

  // Keyboard shortcuts
  useValidationKeyboard({
    isActive: isOpen && !editingEntity && !editingRelation && !showKeyboardHelp && !showVoteHistory,
    isValidating,
    itemCount: visibleItems.length,
    currentIndex,
    onApprove: () => {
      if (currentItem) {
        handleVote(currentItem.id, currentItem.type, 'approve');
      }
    },
    onReject: () => {
      if (currentItem) {
        handleVote(currentItem.id, currentItem.type, 'reject');
      }
    },
    onEdit: () => {
      if (currentItem) {
        handleEdit(currentItem.id, currentItem.type);
      }
    },
    onSkip: () => {
      if (currentItem) {
        handleSkip(currentItem.id, currentItem.type);
      }
    },
    onNavigate: (direction) => {
      if (direction === 'next') navigateToNext();
      else navigateToPrev();
    },
    onShowHelp: () => setShowKeyboardHelp(true),
    onClose,
  });

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="fixed inset-0 bg-transparent z-40"
            />

            {/* Panel */}
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-label="Knowledge Graph Inspector"
              initial={{ x: '100%', opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '100%', opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className={cn(
                'fixed top-0 right-0 h-full w-full max-w-sm z-50',
                'bg-slate-50 dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800',
                'shadow-2xl flex flex-col'
              )}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50">
                <div className="flex items-center gap-2">
                  <Brain size={20} className="text-primary-600 dark:text-primary-400" aria-hidden="true" />
                  <div>
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Knowledge Graph
                    </h2>
                    <p className="text-xs text-slate-500">
                      Art. {articolo} {tipoAtto}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {/* History button */}
                  <button
                    onClick={() => setShowVoteHistory(true)}
                    className={cn(
                      'p-1.5 rounded-lg transition-colors',
                      'hover:bg-slate-100 dark:hover:bg-slate-700',
                      'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                      voteHistory.length > 0 && 'text-primary-500'
                    )}
                    aria-label="Cronologia voti"
                    title="Cronologia voti"
                  >
                    <History size={16} aria-hidden="true" />
                    {voteHistory.filter((h: VoteHistoryEntry) => !h.isCommitted).length > 0 && (
                      <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary-500 text-white text-[9px] font-bold rounded-full flex items-center justify-center">
                        {voteHistory.filter((h: VoteHistoryEntry) => !h.isCommitted).length}
                      </span>
                    )}
                  </button>

                  {/* Keyboard help button */}
                  <button
                    onClick={() => setShowKeyboardHelp(true)}
                    className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    aria-label="Scorciatoie tastiera"
                    title="Scorciatoie tastiera (?)"
                  >
                    <Keyboard size={16} aria-hidden="true" />
                  </button>

                  {/* Close button */}
                  <button
                    onClick={onClose}
                    className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    aria-label="Chiudi pannello"
                  >
                    <X size={18} aria-hidden="true" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Error */}
                {error && (
                  <div role="alert" className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30 rounded-lg text-sm text-red-700 dark:text-red-400">
                    <AlertCircle size={16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
                    {error}
                  </div>
                )}

                {/* Loading */}
                {isLoading && (
                  <div className="flex items-center justify-center py-8" role="status">
                    <Loader2 size={24} className="animate-spin text-primary-500" aria-hidden="true" />
                    <span className="sr-only">Caricamento in corso...</span>
                  </div>
                )}

                {/* AI Enrichment in progress */}
                {isEnriching && !waitingForOther && (
                  <div className="bg-gradient-to-br from-violet-50 to-violet-100 dark:from-violet-900/20 dark:to-violet-800/20 border border-violet-200 dark:border-violet-800/30 rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                        <Bot size={20} className="text-violet-600 dark:text-violet-400 animate-pulse" aria-hidden="true" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm font-semibold text-violet-900 dark:text-violet-100 mb-1 flex items-center gap-2">
                          <Loader2 size={14} className="animate-spin" />
                          Estrazione in corso...
                        </h3>
                        <p className="text-xs text-violet-700 dark:text-violet-300">
                          {progressMessage || "L'intelligenza artificiale sta analizzando l'articolo..."}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Waiting for another user */}
                {isEnriching && waitingForOther && (
                  <div className="bg-gradient-to-br from-amber-50 to-orange-100 dark:from-amber-900/20 dark:to-orange-800/20 border border-amber-200 dark:border-amber-800/30 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                        <Users size={20} className="text-amber-600 dark:text-amber-400" aria-hidden="true" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-1 flex items-center gap-2">
                          <Timer size={14} className="text-amber-600" />
                          Estrazione già in corso
                        </h3>
                        <p className="text-xs text-amber-700 dark:text-amber-300 mb-3">
                          {progressMessage || 'Un altro utente sta già estraendo le entità. Attendere...'}
                        </p>

                        {/* Progress bar */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-xs text-amber-600 dark:text-amber-400">
                            <span>Progresso stimato</span>
                            <span>{waitProgress}%</span>
                          </div>
                          <div className="w-full h-2 bg-amber-200 dark:bg-amber-900/50 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-500 dark:bg-amber-400 transition-all duration-500 ease-out"
                              style={{ width: `${waitProgress}%` }}
                            />
                          </div>
                          <div className="flex items-center justify-between text-xs text-amber-500">
                            <span className="flex items-center gap-1">
                              <Loader2 size={10} className="animate-spin" />
                              In attesa...
                            </span>
                            <span>
                              {Math.floor(waitElapsed / 60)}:{String(waitElapsed % 60).padStart(2, '0')} trascorsi
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {!isLoading && !isEnriching && (
                  <>
                    {/* Status Badge */}
                    <StatusBadge inGraph={inGraph} entityCount={entityCount} />

                    {/* CTA Estrazione */}
                    {showEnrichmentCTA && (
                      <div className="bg-gradient-to-br from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 border border-primary-200 dark:border-primary-800/30 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                            <Sparkles size={20} className="text-primary-600 dark:text-primary-400" />
                          </div>
                          <div className="flex-1">
                            <h3 className="text-sm font-semibold text-primary-900 dark:text-primary-100 mb-1">
                              Arricchisci questo articolo
                            </h3>
                            <p className="text-xs text-primary-700 dark:text-primary-300 mb-3">
                              L'IA estrarrà concetti, principi e relazioni per il Knowledge Graph.
                            </p>
                            <button
                              onClick={onRequestEnrichment}
                              disabled={isEnriching}
                              className={cn(
                                'w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium',
                                'bg-primary-600 hover:bg-primary-700 text-white shadow-sm',
                                'disabled:opacity-50 disabled:cursor-not-allowed transition-colors',
                                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                              )}
                            >
                              <Sparkles size={14} aria-hidden="true" />
                              Avvia Estrazione
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Bulk Actions Bar */}
                    {visibleItems.length > 0 && (
                      <ValidationBulkActionsBar
                        highConfidenceItems={bulkCandidates.highConfidence}
                        lowConfidenceItems={bulkCandidates.lowConfidence}
                        selectedItems={selectedItems}
                        totalItems={visibleItems.length}
                        isProcessing={isValidating}
                        onApproveAll={handleBulkApprove}
                        onRejectAll={handleBulkReject}
                        onSelectHighConfidence={selectHighConfidence}
                        onSelectLowConfidence={selectLowConfidence}
                        onClearSelection={clearSelection}
                      />
                    )}

                    {/* Skip Queue */}
                    <SkipQueuePanel
                      skipQueue={skipQueue}
                      onUnSkip={unSkipItem}
                      onClearAll={clearSkipQueue}
                      getItemName={getItemName}
                    />

                    {/* Pending Items List */}
                    {visibleItems.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <Clock size={14} className="text-primary-500" />
                          <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                            In Attesa ({visibleItems.length})
                          </h3>
                          <div className="flex-1" />
                          <span className="text-[10px] text-slate-400">
                            Usa J/K per navigare
                          </span>
                        </div>
                        <div className="space-y-2">
                          <AnimatePresence mode="popLayout">
                            {visibleItems.map((item: ValidationItem, index: number) => (
                              <ValidationCard
                                key={item.id}
                                ref={(el: HTMLDivElement | null) => {
                                  if (el) cardRefs.current.set(item.id, el);
                                  else cardRefs.current.delete(item.id);
                                }}
                                item={item}
                                isValidating={isValidating}
                                isFocused={index === currentIndex}
                                isSelected={selectedItems.has(item.id)}
                                hasVoted={
                                  item.type === 'entity'
                                    ? hasVotedEntity(item.id)
                                    : hasVotedRelation(item.id)
                                }
                                userAuthorityScore={userAuthorityScore}
                                onVote={(vote: VoteType) => handleVote(item.id, item.type, vote)}
                                onEdit={() => handleEdit(item.id, item.type)}
                                onSkip={() => handleSkip(item.id, item.type)}
                                onToggleSelect={() => toggleSelection(item.id)}
                                selectionMode={selectedItems.size > 0}
                              />
                            ))}
                          </AnimatePresence>
                        </div>
                      </div>
                    )}

                    {/* All validated message */}
                    {pendingCount === 0 && entityCount > 0 && visibleItems.length === 0 && skipQueue.length === 0 && (
                      <div className="text-center py-6">
                        <CheckCircle2 size={32} className="mx-auto text-emerald-400 mb-2" />
                        <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">
                          Tutto validato!
                        </p>
                        <p className="text-xs text-slate-400">
                          Proponi nuove entità o relazioni
                        </p>
                      </div>
                    )}

                    {/* Validated items (collapsible) */}
                    {inGraph && entityCount > 0 && (
                      <div>
                        <button
                          onClick={() => setShowValidated(!showValidated)}
                          aria-expanded={showValidated}
                          className="flex items-center gap-2 w-full text-left py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-lg"
                        >
                          {showValidated ? (
                            <ChevronDown size={14} className="text-slate-400" aria-hidden="true" />
                          ) : (
                            <ChevronRight size={14} className="text-slate-400" aria-hidden="true" />
                          )}
                          <CheckCircle2 size={14} className="text-emerald-500" aria-hidden="true" />
                          <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                            Nel Grafo ({entityCount})
                          </h3>
                        </button>

                        <AnimatePresence>
                          {showValidated && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="overflow-hidden"
                            >
                              <p className="text-xs text-slate-500 dark:text-slate-400 py-2">
                                {entityCount} entità e relazioni validate.
                              </p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Footer Actions */}
              <div className="border-t border-slate-200 dark:border-slate-800 p-4 bg-white dark:bg-slate-800/50 space-y-3">
                {/* Commit feedback button */}
                {uncommittedVoteCount > 0 && onCommitFeedback && (
                  <button
                    onClick={handleCommitFeedback}
                    className={cn(
                      'w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium',
                      'bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm',
                      'transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                    )}
                  >
                    <Send size={14} aria-hidden="true" />
                    Conferma Feedback ({uncommittedVoteCount})
                  </button>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={onProposeEntity}
                    className={cn(
                      'flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium min-h-[44px]',
                      'bg-slate-100 hover:bg-slate-200 text-slate-700',
                      'dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-300',
                      'transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                    )}
                  >
                    <Plus size={12} aria-hidden="true" />
                    Proponi Entità
                  </button>
                  <button
                    onClick={onProposeRelation}
                    className={cn(
                      'flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium min-h-[44px]',
                      'bg-slate-100 hover:bg-slate-200 text-slate-700',
                      'dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-300',
                      'transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
                    )}
                  >
                    <Plus size={12} aria-hidden="true" />
                    Proponi Relazione
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Edit Entity Drawer */}
      {editingEntity && (
        <EditEntityDrawer
          isOpen={!!editingEntity}
          onClose={() => setEditingEntity(null)}
          onSuccess={() => {
            setEditingEntity(null);
            onRefresh?.();
          }}
          entity={editingEntity}
          userId={userId}
        />
      )}

      {/* Edit Relation Drawer */}
      {editingRelation && (
        <EditRelationDrawer
          isOpen={!!editingRelation}
          onClose={() => setEditingRelation(null)}
          onSuccess={() => {
            setEditingRelation(null);
            onRefresh?.();
          }}
          relation={editingRelation}
          userId={userId}
        />
      )}

      {/* Keyboard Help Modal */}
      <ValidationKeyboardHelpModal
        isOpen={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />

      {/* Vote History Modal */}
      <VoteHistoryModal
        isOpen={showVoteHistory}
        onClose={() => setShowVoteHistory(false)}
        history={voteHistory}
        onUndoVote={removeVoteFromHistory}
      />
    </>
  );
}
