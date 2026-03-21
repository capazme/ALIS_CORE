/**
 * MERLT Panel Store (Zustand)
 *
 * Controls sidebar state: open/close, active tab, prefill query.
 * Shared between MerltToolbar (trigger), MerltSidebarPanel (drawer),
 * and external triggers (SelectionPopup "Chiedi a MERL-T").
 */

import { create } from 'zustand';

export type SidebarTab = 'analysis' | 'entities' | 'validate' | 'contribute';

export interface MerltPanelState {
  isOpen: boolean;
  activeTab: SidebarTab;
  prefillQuery: string | null;
  open: () => void;
  close: () => void;
  toggle: () => void;
  setActiveTab: (tab: SidebarTab) => void;
  openWithQuery: (query: string) => void;
  clearPrefill: () => void;
}

export const useMerltPanelStore = create<MerltPanelState>((set) => ({
  isOpen: false,
  activeTab: 'analysis',
  prefillQuery: null,

  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
  toggle: () => set((s) => ({ isOpen: !s.isOpen })),
  setActiveTab: (tab) => set({ activeTab: tab }),
  openWithQuery: (query) => set({ isOpen: true, activeTab: 'analysis', prefillQuery: query }),
  clearPrefill: () => set({ prefillQuery: null }),
}));
