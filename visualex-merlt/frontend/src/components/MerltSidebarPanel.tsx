/**
 * MerltSidebarPanel
 *
 * Rendered in the article-sidebar slot but displayed as a fixed right-side drawer
 * (Legacy MerltInspectorPanel pattern). Opens/closes via useMerltPanelStore.
 *
 * Layout: fixed right-0 top-0 h-full max-w-sm z-50
 * Animation: spring slide from right (damping=25, stiffness=300)
 * Backdrop: transparent click-to-close
 */

import { useState, useEffect, useCallback } from 'react';
import type { SlotProps } from '@visualex/platform/lib/plugins';
import { EventBus } from '@visualex/platform/lib/plugins';
import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import { cn } from '../lib/utils';
import { useMerltArticleAnalysis } from '../hooks/useMerltArticleAnalysis';
import { useMerltPanelStore } from '../store/useMerltSidebarStore';
import { EntityList } from './EntityList';
import { ValidationQueue } from './ValidationQueue';
import { ContributionPanel } from './ContributionPanel';
import { ExpertProgressIndicator } from './pipeline/ExpertProgressIndicator';
import { useExpertPipelineStatus } from '../hooks/useExpertPipelineStatus';
import { TraceViewer } from './trace/TraceViewer';
import { QueryInputForm } from './trace/QueryInputForm';
import { getCurrentUserId } from '../services/merltInit';
import type { SourceResolution } from '../types/trace';

type Props = SlotProps['article-sidebar'];

type Tab = 'entities' | 'validate' | 'contribute' | 'analysis';

export function MerltSidebarPanel({ urn }: Props): React.ReactElement | null {
  const isOpen = useMerltPanelStore((s) => s.isOpen);
  const close = useMerltPanelStore((s) => s.close);

  const [activeTab, setActiveTab] = useState('entities' as Tab);
  const [activeQueryId, setActiveQueryId] = useState(null as string | null);
  const { entities, relations, isLoading, error } = useMerltArticleAnalysis(urn);
  const { status: pipelineStatus, isActive: pipelineActive } = useExpertPipelineStatus(activeQueryId);

  // Reset query when article changes (H3 fix)
  useEffect(() => {
    setActiveQueryId(null);
  }, [urn]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, close]);

  // Subscribe to article:viewed event to update context
  useEffect(() => {
    const handleArticleViewed = (data: { urn: string; articleId: string; userId?: string }) => {
      if (data.urn === urn) {
        // Could trigger refresh or update internal state here
      }
    };

    const unsubscribe = EventBus.on('article:viewed', handleArticleViewed);
    return unsubscribe;
  }, [urn]);

  // Subscribe to article:text-selected for entity proposal flows
  useEffect(() => {
    const handleTextSelected = (data: { urn: string; text: string; startOffset: number; endOffset: number }) => {
      if (data.urn === urn) {
        // Could highlight related entities in sidebar or show suggestion UI
      }
    };

    const unsubscribe = EventBus.on('article:text-selected', handleTextSelected);
    return unsubscribe;
  }, [urn]);

  const handleSourceNavigate = useCallback((source: SourceResolution) => {
    EventBus.emit('merlt:source-navigate', {
      urn: source.urn,
      articleId: source.label,
      sourceId: source.sourceId,
    });
  }, []);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop — transparent, click to close */}
          <motion.div
            key="merlt-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={close}
            className="fixed inset-0 bg-transparent z-40"
            aria-hidden="true"
          />

          {/* Drawer panel */}
          <motion.div
            key="merlt-drawer"
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={cn(
              'fixed top-0 right-0 h-full w-full max-w-sm z-50',
              'bg-white dark:bg-slate-900',
              'border-l border-slate-200 dark:border-slate-800',
              'shadow-2xl flex flex-col'
            )}
            role="dialog"
            aria-label="MERLT Research"
            aria-modal="true"
          >
            {/* Header with close button */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700">
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  MERLT Research
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Contribuisci alla ricerca
                </p>
              </div>
              <button
                onClick={close}
                className={cn(
                  'p-1.5 rounded-md transition-colors',
                  'text-slate-400 hover:text-slate-600 hover:bg-slate-100',
                  'dark:hover:text-slate-300 dark:hover:bg-slate-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                aria-label="Chiudi pannello MERLT"
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
                {/* Pipeline Progress — shown when active */}
                {pipelineActive && pipelineStatus && (
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                    <ExpertProgressIndicator status={pipelineStatus} />
                  </div>
                )}

                {/* Tabs */}
                <div className="flex border-b border-slate-200 dark:border-slate-700" role="tablist" aria-label="Sezioni MERLT">
                  <TabButton active={activeTab === 'entities'} onClick={() => setActiveTab('entities')} id="entities">
                    Entità ({entities?.length ?? 0})
                  </TabButton>
                  <TabButton active={activeTab === 'validate'} onClick={() => setActiveTab('validate')} id="validate">
                    Valida
                  </TabButton>
                  <TabButton active={activeTab === 'contribute'} onClick={() => setActiveTab('contribute')} id="contribute">
                    Proponi
                  </TabButton>
                  <TabButton active={activeTab === 'analysis'} onClick={() => setActiveTab('analysis')} id="analysis">
                    Analisi
                  </TabButton>
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
                      {activeTab === 'entities' && <EntityList entities={entities} relations={relations} />}
                      {activeTab === 'validate' && <ValidationQueue articleUrn={urn} />}
                      {activeTab === 'contribute' && <ContributionPanel articleUrn={urn} />}
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
}

function TabButton({ active, onClick, children, id }: TabButtonProps): React.ReactElement {
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${id}`}
      aria-selected={active}
      aria-controls={`panel-${id}`}
      onClick={onClick}
      className={cn(
        'flex-1 px-3 py-2 text-xs font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset',
        active
          ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20'
          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50'
      )}
    >
      {children}
    </button>
  );
}
