/**
 * useSourceNavigation - Manages source selection state and split view toggle.
 */

import { useState, useCallback, useMemo } from 'react';
import type { SourceResolution } from '../types/trace';

export interface UseSourceNavigationReturn {
  selectedSource: SourceResolution | null;
  selectedIndex: number | null;
  isSplitView: boolean;
  selectSource: (source: SourceResolution, index: number) => void;
  clearSelection: () => void;
  toggleSplitView: () => void;
  setSplitView: (open: boolean) => void;
}

export function useSourceNavigation(): UseSourceNavigationReturn {
  const [selectedSource, setSelectedSource] = useState(null as SourceResolution | null);
  const [selectedIndex, setSelectedIndex] = useState(null as number | null);
  const [isSplitView, setIsSplitView] = useState(false);

  const selectSource = useCallback((source: SourceResolution, index: number) => {
    setSelectedSource(source);
    setSelectedIndex(index);
    setIsSplitView(true);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedSource(null);
    setSelectedIndex(null);
  }, []);

  const toggleSplitView = useCallback(() => {
    setIsSplitView((prev: boolean) => !prev);
  }, []);

  const setSplitViewOpen = useCallback((open: boolean) => {
    setIsSplitView(open);
    if (!open) {
      setSelectedSource(null);
      setSelectedIndex(null);
    }
  }, []);

  return useMemo(() => ({
    selectedSource,
    selectedIndex,
    isSplitView,
    selectSource,
    clearSelection,
    toggleSplitView,
    setSplitView: setSplitViewOpen,
  }), [selectedSource, selectedIndex, isSplitView, selectSource, clearSelection, toggleSplitView, setSplitViewOpen]);
}
