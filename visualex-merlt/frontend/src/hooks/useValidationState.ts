/**
 * useValidationState
 * ==================
 *
 * State management hook for validation panel.
 * Handles:
 * - Current item selection
 * - Skip queue management
 * - Vote history tracking
 * - Bulk action candidates
 */

import { useState, useCallback, useMemo } from 'react';
import type { PendingEntity, PendingRelation, VoteType } from '../types/merlt';
import { formatUrnToReadable } from '../utils/normattivaParser';

export type ValidationItemType = 'entity' | 'relation';

export interface ValidationItem {
  id: string;
  type: ValidationItemType;
  name: string;
  description?: string;
  confidence: number;
  approvalScore: number;
  rejectionScore: number;
  votesCount: number;
  isAiGenerated: boolean;
  raw: PendingEntity | PendingRelation;
}

export interface VoteHistoryEntry {
  id: string;
  itemId: string;
  itemName: string;
  itemType: ValidationItemType;
  vote: VoteType;
  timestamp: Date;
  isCommitted: boolean;
}

export interface SkipQueueItem {
  itemId: string;
  itemType: ValidationItemType;
  skippedAt: Date;
  reason?: string;
}

export interface BulkActionCandidate {
  item: ValidationItem;
  reason: 'high_confidence' | 'low_confidence' | 'user_selected';
  confidence: number;
}

interface UseValidationStateOptions {
  entities: PendingEntity[];
  relations: PendingRelation[];
  hasVotedEntity?: (id: string) => boolean;
  hasVotedRelation?: (id: string) => boolean;
}

export function useValidationState({
  entities,
  relations,
  hasVotedEntity = () => false,
  hasVotedRelation = () => false,
}: UseValidationStateOptions) {
  // Current focused item index
  const [currentIndex, setCurrentIndex] = useState(0);

  // Skip queue
  const [skipQueue, setSkipQueue] = useState<SkipQueueItem[]>([]);

  // Vote history (local session only)
  const [voteHistory, setVoteHistory] = useState<VoteHistoryEntry[]>([]);

  // Selected items for bulk actions
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());

  // Convert entities and relations to unified items
  const allItems = useMemo<ValidationItem[]>(() => {
    const entityItems: ValidationItem[] = entities.map((e) => ({
      id: e.id,
      type: 'entity' as const,
      name: e.nome,
      description: e.descrizione,
      confidence: e.llm_confidence,
      approvalScore: e.approval_score,
      rejectionScore: e.rejection_score,
      votesCount: e.votes_count,
      isAiGenerated: e.llm_confidence > 0,
      raw: e,
    }));

    const relationItems: ValidationItem[] = relations.map((r) => ({
      id: r.id,
      type: 'relation' as const,
      name: `${r.relation_type}`,
      description: `${formatUrnToReadable(r.source_urn)} → ${formatUrnToReadable(r.target_urn)}`,
      confidence: r.llm_confidence,
      approvalScore: r.approval_score,
      rejectionScore: r.rejection_score,
      votesCount: r.votes_count,
      isAiGenerated: r.llm_confidence > 0,
      raw: r,
    }));

    return [...entityItems, ...relationItems];
  }, [entities, relations]);

  // Items not in skip queue and not voted
  const visibleItems = useMemo(() => {
    const skippedIds = new Set(skipQueue.map((s) => s.itemId));
    return allItems.filter((item) => {
      if (skippedIds.has(item.id)) return false;
      if (item.type === 'entity' && hasVotedEntity(item.id)) return false;
      if (item.type === 'relation' && hasVotedRelation(item.id)) return false;
      return true;
    });
  }, [allItems, skipQueue, hasVotedEntity, hasVotedRelation]);

  // Current item
  const currentItem = visibleItems[currentIndex] ?? null;

  // Bulk action candidates
  const bulkCandidates = useMemo<{
    highConfidence: BulkActionCandidate[];
    lowConfidence: BulkActionCandidate[];
  }>(() => {
    const high: BulkActionCandidate[] = [];
    const low: BulkActionCandidate[] = [];

    for (const item of visibleItems) {
      if (item.confidence >= 0.9) {
        high.push({ item, reason: 'high_confidence', confidence: item.confidence });
      } else if (item.confidence < 0.5 && item.confidence > 0) {
        low.push({ item, reason: 'low_confidence', confidence: item.confidence });
      }
    }

    return { highConfidence: high, lowConfidence: low };
  }, [visibleItems]);

  // Navigation
  const navigateToNext = useCallback(() => {
    setCurrentIndex((prev) => Math.min(prev + 1, visibleItems.length - 1));
  }, [visibleItems.length]);

  const navigateToPrev = useCallback(() => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const navigateToIndex = useCallback((index: number) => {
    setCurrentIndex(Math.max(0, Math.min(index, visibleItems.length - 1)));
  }, [visibleItems.length]);

  // Skip management
  const skipItem = useCallback((itemId: string, itemType: ValidationItemType, reason?: string) => {
    setSkipQueue((prev) => [
      ...prev,
      { itemId, itemType, skippedAt: new Date(), reason },
    ]);
    // Keep index in bounds
    setCurrentIndex((prev) => Math.min(prev, Math.max(0, visibleItems.length - 2)));
  }, [visibleItems.length]);

  const unSkipItem = useCallback((itemId: string) => {
    setSkipQueue((prev) => prev.filter((s) => s.itemId !== itemId));
  }, []);

  const clearSkipQueue = useCallback(() => {
    setSkipQueue([]);
  }, []);

  // Vote history
  const recordVote = useCallback((
    itemId: string,
    itemName: string,
    itemType: ValidationItemType,
    vote: VoteType
  ) => {
    const entry: VoteHistoryEntry = {
      id: crypto.randomUUID(),
      itemId,
      itemName,
      itemType,
      vote,
      timestamp: new Date(),
      isCommitted: false,
    };
    setVoteHistory((prev) => [entry, ...prev]);
  }, []);

  const markVotesAsCommitted = useCallback(() => {
    setVoteHistory((prev) =>
      prev.map((entry) => ({ ...entry, isCommitted: true }))
    );
  }, []);

  const removeVoteFromHistory = useCallback((id: string) => {
    setVoteHistory((prev) => prev.filter((entry) => entry.id !== id));
  }, []);

  // Selection for bulk actions
  const toggleSelection = useCallback((itemId: string) => {
    setSelectedItems((prev) => {
      const next = new Set(prev);
      if (next.has(itemId)) {
        next.delete(itemId);
      } else {
        next.add(itemId);
      }
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedItems(new Set(visibleItems.map((item) => item.id)));
  }, [visibleItems]);

  const clearSelection = useCallback(() => {
    setSelectedItems(new Set());
  }, []);

  const selectHighConfidence = useCallback(() => {
    const ids = bulkCandidates.highConfidence.map((c) => c.item.id);
    setSelectedItems(new Set(ids));
  }, [bulkCandidates.highConfidence]);

  const selectLowConfidence = useCallback(() => {
    const ids = bulkCandidates.lowConfidence.map((c) => c.item.id);
    setSelectedItems(new Set(ids));
  }, [bulkCandidates.lowConfidence]);

  return {
    // Items
    allItems,
    visibleItems,
    currentItem,
    currentIndex,

    // Navigation
    navigateToNext,
    navigateToPrev,
    navigateToIndex,

    // Skip queue
    skipQueue,
    skipItem,
    unSkipItem,
    clearSkipQueue,

    // Vote history
    voteHistory,
    recordVote,
    markVotesAsCommitted,
    removeVoteFromHistory,

    // Selection
    selectedItems,
    toggleSelection,
    selectAll,
    clearSelection,
    selectHighConfidence,
    selectLowConfidence,

    // Bulk candidates
    bulkCandidates,
  };
}
