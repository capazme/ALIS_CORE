/**
 * useBrocardiSidebar - Manages Brocardi sidebar state.
 * Persists open/tab state in localStorage.
 */

import { useState, useCallback, useMemo } from 'react';

export type BrocardiTab = 'brocardi' | 'ratio' | 'spiegazione' | 'massime' | 'relazioni';

const STORAGE_KEY = 'brocardi-sidebar-state';

interface StoredState {
  isOpen: boolean;
  activeTab: BrocardiTab;
}

const VALID_TABS: readonly BrocardiTab[] = ['brocardi', 'ratio', 'spiegazione', 'massime', 'relazioni'];

function loadState(): StoredState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed: unknown = JSON.parse(stored);
      if (
        typeof parsed === 'object' && parsed !== null &&
        'isOpen' in parsed && typeof (parsed as Record<string, unknown>).isOpen === 'boolean' &&
        'activeTab' in parsed && VALID_TABS.includes((parsed as Record<string, unknown>).activeTab as BrocardiTab)
      ) {
        return parsed as StoredState;
      }
    }
  } catch {
    // Ignore
  }
  return { isOpen: false, activeTab: 'brocardi' };
}

function saveState(state: StoredState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore
  }
}

export interface UseBrocardiSidebarReturn {
  isOpen: boolean;
  activeTab: BrocardiTab;
  toggle: () => void;
  open: () => void;
  close: () => void;
  setTab: (tab: BrocardiTab) => void;
}

export function useBrocardiSidebar(): UseBrocardiSidebarReturn {
  const [state, setState] = useState<StoredState>(loadState);

  const toggle = useCallback(() => {
    setState(prev => {
      const next = { ...prev, isOpen: !prev.isOpen };
      saveState(next);
      return next;
    });
  }, []);

  const open = useCallback(() => {
    setState(prev => {
      const next = { ...prev, isOpen: true };
      saveState(next);
      return next;
    });
  }, []);

  const close = useCallback(() => {
    setState(prev => {
      const next = { ...prev, isOpen: false };
      saveState(next);
      return next;
    });
  }, []);

  const setTab = useCallback((tab: BrocardiTab) => {
    setState(prev => {
      const next = { ...prev, activeTab: tab, isOpen: true };
      saveState(next);
      return next;
    });
  }, []);

  return useMemo(() => ({
    isOpen: state.isOpen,
    activeTab: state.activeTab,
    toggle,
    open,
    close,
    setTab,
  }), [state, toggle, open, close, setTab]);
}
