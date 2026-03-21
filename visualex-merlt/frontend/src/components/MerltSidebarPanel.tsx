/**
 * MerltSidebarPanel
 *
 * Rendered in the article-sidebar slot. Layout adapts to viewport:
 * - Desktop (>=1280px): docked inline panel, no backdrop, 380px width
 * - Tablet (768-1279px): drawer from right, 420px, semi-transparent overlay
 * - Mobile (<768px): bottom sheet 90vh
 *
 * Tab order: Chiedi | Grafo | Valida (N) | Proponi
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { EventBus } from '@visualex/platform/lib/plugins';
import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import { cn } from '../lib/utils';
import { useMerltArticleAnalysis } from '../hooks/useMerltArticleAnalysis';
import { useMerltPanelStore } from '../store/useMerltSidebarStore';
import type { SidebarTab } from '../store/useMerltSidebarStore';
import { EntityList } from './EntityList';
import { ValidationQueue } from './ValidationQueue';
import { ContributionPanel } from './ContributionPanel';
import { ExpertProgressIndicator } from './pipeline/ExpertProgressIndicator';
import { useExpertPipelineStatus } from '../hooks/useExpertPipelineStatus';
import { TraceViewer } from './trace/TraceViewer';
import { QueryInputForm } from './trace/QueryInputForm';
import { getCurrentUserId } from '../services/merltInit';
import { usePendingValidations } from '../hooks/usePendingValidations';
import type { SourceResolution } from '../types/trace';

type Props = SlotProps['article-sidebar'];

type Viewport = 'mobile' | 'tablet' | 'desktop';

const SIDEBAR_WIDTH_KEY = 'merlt-sidebar-width';
const MIN_WIDTH = 320;
const MAX_WIDTH = 600;
const DEFAULT_WIDTH = 380;

function useViewport(): Viewport {
  const [viewport, setViewport] = useState('desktop' as Viewport);

  useEffect(() => {
    const check = () => {
      const w = window.innerWidth;
      if (w >= 1280) setViewport('desktop');
      else if (w >= 768) setViewport('tablet');
      else setViewport('mobile');
    };
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  return viewport;
}

function useSavedWidth(): [number, (w: number) => void] {
  const [width, setWidth] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_WIDTH_KEY);
    return saved ? Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, Number(saved))) : DEFAULT_WIDTH;
  });

  const save = useCallback((w: number) => {
    const clamped = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, w));
    setWidth(clamped);
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(clamped));
  }, []);

  return [width, save];
}

export function MerltSidebarPanel({ urn }: Props): React.ReactElement | null {
  const isOpen = useMerltPanelStore((s) => s.isOpen);
  const close = useMerltPanelStore((s) => s.close);
  const activeTab = useMerltPanelStore((s) => s.activeTab);
  const setActiveTab = useMerltPanelStore((s) => s.setActiveTab);

  const viewport = useViewport();
  const [sidebarWidth, setSidebarWidth] = useSavedWidth();
  const [isResizing, setIsResizing] = useState(false);

  const [activeQueryId, setActiveQueryId] = useState(null as string | null);
  const { entities, relations, isLoading, error } = useMerltArticleAnalysis(urn);
  const { status: pipelineStatus, isActive: pipelineActive } = useExpertPipelineStatus(activeQueryId);
  const { validations: pendingValidations } = usePendingValidations(urn);
  const pendingCount = pendingValidations?.length ?? 0;

  const panelRef = useRef(null as HTMLDivElement | null);
  const previousFocusRef = useRef(null as HTMLElement | null);

  // Reset query when article changes
  useEffect(() => {
    setActiveQueryId(null);
  }, [urn]);

  // Focus management
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      requestAnimationFrame(() => {
        const firstFocusable = panelRef.current?.querySelector(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        ) as HTMLElement | null;
        firstFocusable?.focus();
      });
    } else if (previousFocusRef.current) {
      previousFocusRef.current.focus();
      previousFocusRef.current = null;
    }
  }, [isOpen]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, close]);

  // Focus trap (tablet/mobile only — desktop is docked inline)
  useEffect(() => {
    if (!isOpen || viewport === 'desktop') return;

    const handleTab = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !panelRef.current) return;

      const focusable = panelRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      ) as NodeListOf<HTMLElement>;
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleTab);
    return () => document.removeEventListener('keydown', handleTab);
  }, [isOpen, viewport]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+Shift+M: toggle panel
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'M') {
        e.preventDefault();
        useMerltPanelStore.getState().toggle();
      }
      // Cmd+Shift+Q: focus query textarea
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'Q') {
        e.preventDefault();
        const store = useMerltPanelStore.getState();
        if (!store.isOpen) store.open();
        store.setActiveTab('analysis');
        requestAnimationFrame(() => {
          document.getElementById('merlt-query-input')?.focus();
        });
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Subscribe to article events
  useEffect(() => {
    const handleArticleViewed = (data: { urn: string }) => {
      if (data.urn === urn) { /* could refresh */ }
    };
    const handleTextSelected = (data: { urn: string; text: string }) => {
      if (data.urn === urn) { /* could highlight related entities */ }
    };
    const unsub1 = EventBus.on('article:viewed', handleArticleViewed) as unknown as (() => void) | void;
    const unsub2 = EventBus.on('article:text-selected', handleTextSelected) as unknown as (() => void) | void;
    return () => {
      if (typeof unsub1 === 'function') unsub1();
      if (typeof unsub2 === 'function') unsub2();
    };
  }, [urn]);

  const handleSourceNavigate = useCallback((source: SourceResolution) => {
    EventBus.emit('merlt:source-navigate', {
      urn: source.urn,
      articleId: source.label,
      sourceId: source.sourceId,
    });
  }, []);

  // Resize handle
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    if (viewport !== 'desktop') return;
    e.preventDefault();
    setIsResizing(true);
    const startX = e.clientX;
    const startWidth = sidebarWidth;

    const handleMove = (moveEvent: MouseEvent) => {
      const delta = startX - moveEvent.clientX;
      setSidebarWidth(startWidth + delta);
    };

    const handleUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
    };

    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
  }, [viewport, sidebarWidth, setSidebarWidth]);

  // Tab config
  const tabs: { id: SidebarTab; label: string }[] = [
    { id: 'analysis', label: 'Chiedi' },
    { id: 'entities', label: 'Grafo' },
    { id: 'validate', label: pendingCount > 0 ? `Valida (${pendingCount})` : 'Valida' },
    { id: 'contribute', label: 'Proponi' },
  ];

  // Motion variants by viewport
  const getMotionProps = () => {
    if (viewport === 'mobile') {
      return {
        initial: { y: '100%', opacity: 0 },
        animate: { y: 0, opacity: 1 },
        exit: { y: '100%', opacity: 0 },
        transition: { type: 'spring' as const, damping: 25, stiffness: 300 },
      };
    }
    return {
      initial: { x: '100%', opacity: 0 },
      animate: { x: 0, opacity: 1 },
      exit: { x: '100%', opacity: 0 },
      transition: { type: 'spring' as const, damping: 25, stiffness: 300 },
    };
  };

  const panelClasses = cn(
    'bg-white dark:bg-slate-900',
    'border-slate-200 dark:border-slate-800',
    'shadow-2xl flex flex-col',
    viewport === 'mobile' && 'fixed bottom-0 left-0 right-0 h-[90vh] rounded-t-2xl border-t z-50',
    viewport === 'tablet' && 'fixed top-0 right-0 h-full w-[420px] border-l z-50',
    viewport === 'desktop' && 'fixed top-0 right-0 h-full border-l z-50',
  );

  const showBackdrop = viewport !== 'desktop';

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop — tablet/mobile only */}
          {showBackdrop && (
            <motion.div
              key="merlt-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={close}
              className="fixed inset-0 bg-black/30 z-40"
              aria-hidden="true"
            />
          )}

          {/* Panel */}
          <motion.div
            key="merlt-drawer"
            ref={panelRef}
            {...getMotionProps()}
            className={panelClasses}
            style={viewport === 'desktop' ? { width: sidebarWidth } : undefined}
            role="dialog"
            aria-label="MERL-T"
            aria-modal={viewport !== 'desktop'}
          >
            {/* Resize handle — desktop only */}
            {viewport === 'desktop' && (
              <div
                onMouseDown={handleResizeStart}
                className={cn(
                  'absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-10',
                  'hover:bg-blue-500/20 transition-colors',
                  isResizing && 'bg-blue-500/30',
                )}
                role="separator"
                aria-orientation="vertical"
                aria-label="Ridimensiona pannello"
              >
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 rounded-full bg-slate-400/50 opacity-0 hover:opacity-100 transition-opacity" />
              </div>
            )}

            {/* Mobile drag handle */}
            {viewport === 'mobile' && (
              <div className="flex justify-center pt-2 pb-1">
                <div className="w-10 h-1 rounded-full bg-slate-300 dark:bg-slate-600" />
              </div>
            )}

            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                MERL-T
              </h2>
              <button
                onClick={close}
                className={cn(
                  'p-1.5 rounded-md transition-colors',
                  'text-slate-400 hover:text-slate-600 hover:bg-slate-100',
                  'dark:hover:text-slate-300 dark:hover:bg-slate-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500',
                )}
                aria-label="Chiudi pannello MERL-T"
              >
                <X size={16} />
              </button>
            </div>

            {error ? (
              <div className="p-4 m-4 text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800" role="alert">
                <p className="font-medium">Errore MERLT</p>
                <p className="text-sm mt-1">{error.message}</p>
              </div>
            ) : (
              <>
                {/* Pipeline Progress */}
                {pipelineActive && pipelineStatus && (
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                    <ExpertProgressIndicator status={pipelineStatus} />
                  </div>
                )}

                {/* Tabs */}
                <div className="flex border-b border-slate-200 dark:border-slate-700" role="tablist" aria-label="Sezioni MERL-T">
                  {tabs.map((tab) => (
                    <TabButton
                      key={tab.id}
                      active={activeTab === tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      id={tab.id}
                      hasBadge={tab.id === 'validate' && pendingCount > 0}
                    >
                      {tab.label}
                    </TabButton>
                  ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto">
                  {activeTab !== 'analysis' && isLoading ? (
                    <div className="flex items-center justify-center h-32" aria-label="Caricamento in corso">
                      <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" role="status">
                        <span className="sr-only">Caricamento...</span>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div
                        id="panel-analysis"
                        role="tabpanel"
                        aria-labelledby="tab-analysis"
                        hidden={activeTab !== 'analysis'}
                      >
                        {activeTab === 'analysis' && (
                          <div className="p-4 space-y-4">
                            <QueryInputForm
                              articleUrn={urn}
                              userId={getCurrentUserId()}
                              onTraceCreated={setActiveQueryId}
                              disabled={pipelineActive}
                            />
                            {activeQueryId && (
                              <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
                                <TraceViewer
                                  traceId={activeQueryId}
                                  onSourceNavigate={handleSourceNavigate}
                                />
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <div
                        id="panel-entities"
                        role="tabpanel"
                        aria-labelledby="tab-entities"
                        hidden={activeTab !== 'entities'}
                      >
                        {activeTab === 'entities' && <EntityList entities={entities} relations={relations} />}
                      </div>
                      <div
                        id="panel-validate"
                        role="tabpanel"
                        aria-labelledby="tab-validate"
                        hidden={activeTab !== 'validate'}
                      >
                        {activeTab === 'validate' && <ValidationQueue articleUrn={urn} />}
                      </div>
                      <div
                        id="panel-contribute"
                        role="tabpanel"
                        aria-labelledby="tab-contribute"
                        hidden={activeTab !== 'contribute'}
                      >
                        {activeTab === 'contribute' && <ContributionPanel articleUrn={urn} />}
                      </div>
                    </>
                  )}
                </div>
              </>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  id: string;
  hasBadge?: boolean;
}

function TabButton({ active, onClick, children, id, hasBadge }: TabButtonProps): React.ReactElement {
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${id}`}
      aria-selected={active}
      aria-controls={`panel-${id}`}
      onClick={onClick}
      className={cn(
        'flex-1 px-3 py-2 text-xs font-medium transition-colors relative',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset',
        active
          ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20'
          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50',
      )}
    >
      {children}
      {hasBadge && (
        <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-blue-500" aria-hidden="true" />
      )}
    </button>
  );
}
