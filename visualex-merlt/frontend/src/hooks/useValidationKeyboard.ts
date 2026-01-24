/**
 * useValidationKeyboard
 * ====================
 *
 * Industrial-grade keyboard shortcuts hook for validation panel.
 * Supports:
 * - A: Approve current item
 * - R: Reject current item
 * - E: Edit current item
 * - J/K or ↓/↑: Navigate between items
 * - S: Skip current item
 * - ?: Show keyboard shortcuts help
 * - Esc: Close panel
 */

import { useEffect, useCallback, useRef } from 'react';

export interface ValidationKeyboardConfig {
  /** Is the validation panel open/active */
  isActive: boolean;
  /** Is there any pending validation in progress */
  isValidating: boolean;
  /** Total number of items */
  itemCount: number;
  /** Currently focused item index */
  currentIndex: number;
  /** Callbacks */
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
  onSkip: () => void;
  onNavigate: (direction: 'next' | 'prev') => void;
  onShowHelp: () => void;
  onClose: () => void;
}

export interface KeyboardShortcut {
  key: string;
  label: string;
  description: string;
  category: 'voting' | 'navigation' | 'other';
}

export const VALIDATION_SHORTCUTS: KeyboardShortcut[] = [
  { key: 'A', label: 'A', description: 'Approva elemento corrente', category: 'voting' },
  { key: 'R', label: 'R', description: 'Rifiuta elemento corrente', category: 'voting' },
  { key: 'E', label: 'E', description: 'Modifica elemento corrente', category: 'voting' },
  { key: 'S', label: 'S', description: 'Salta (rivedi dopo)', category: 'voting' },
  { key: 'J', label: 'J / ↓', description: 'Prossimo elemento', category: 'navigation' },
  { key: 'K', label: 'K / ↑', description: 'Elemento precedente', category: 'navigation' },
  { key: '?', label: '?', description: 'Mostra scorciatoie', category: 'other' },
  { key: 'Escape', label: 'Esc', description: 'Chiudi pannello', category: 'other' },
];

export function useValidationKeyboard(config: ValidationKeyboardConfig) {
  const {
    isActive,
    isValidating,
    itemCount,
    currentIndex,
    onApprove,
    onReject,
    onEdit,
    onSkip,
    onNavigate,
    onShowHelp,
    onClose,
  } = config;

  // Use refs for callbacks to avoid recreating the listener
  const callbacksRef = useRef({
    onApprove,
    onReject,
    onEdit,
    onSkip,
    onNavigate,
    onShowHelp,
    onClose,
  });

  // Keep refs updated
  useEffect(() => {
    callbacksRef.current = {
      onApprove,
      onReject,
      onEdit,
      onSkip,
      onNavigate,
      onShowHelp,
      onClose,
    };
  }, [onApprove, onReject, onEdit, onSkip, onNavigate, onShowHelp, onClose]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't capture if not active
      if (!isActive) return;

      // Don't capture if user is typing in an input
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      // Don't capture if modal is open (handled elsewhere)
      if (document.querySelector('[role="dialog"]')) {
        // Only allow Escape in modals
        if (event.key !== 'Escape') return;
      }

      const { key, shiftKey } = event;
      const callbacks = callbacksRef.current;

      // Help shortcut (Shift + /)
      if (key === '?' || (key === '/' && shiftKey)) {
        event.preventDefault();
        callbacks.onShowHelp();
        return;
      }

      // Escape to close
      if (key === 'Escape') {
        event.preventDefault();
        callbacks.onClose();
        return;
      }

      // Don't allow other shortcuts while validating
      if (isValidating) return;

      // Voting shortcuts (only if there are items)
      if (itemCount > 0) {
        switch (key.toLowerCase()) {
          case 'a':
            event.preventDefault();
            callbacks.onApprove();
            return;
          case 'r':
            event.preventDefault();
            callbacks.onReject();
            return;
          case 'e':
            event.preventDefault();
            callbacks.onEdit();
            return;
          case 's':
            event.preventDefault();
            callbacks.onSkip();
            return;
        }
      }

      // Navigation shortcuts
      if (key === 'j' || key === 'ArrowDown') {
        event.preventDefault();
        if (currentIndex < itemCount - 1) {
          callbacks.onNavigate('next');
        }
        return;
      }

      if (key === 'k' || key === 'ArrowUp') {
        event.preventDefault();
        if (currentIndex > 0) {
          callbacks.onNavigate('prev');
        }
        return;
      }
    },
    [isActive, isValidating, itemCount, currentIndex]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return {
    shortcuts: VALIDATION_SHORTCUTS,
  };
}
