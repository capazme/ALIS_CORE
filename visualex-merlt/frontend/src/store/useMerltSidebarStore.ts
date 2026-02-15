/**
 * MERLT Panel Store (Zustand)
 *
 * Controls whether the MerltSidebarPanel drawer is open.
 * Shared between MerltToolbar (trigger) and MerltSidebarPanel (drawer)
 * since they live in separate plugin slots.
 */

import { create } from 'zustand';

export interface MerltPanelState {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

export const useMerltPanelStore = create<MerltPanelState>((set) => ({
  isOpen: false,

  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
  toggle: () => set((s) => ({ isOpen: !s.isOpen })),
}));
