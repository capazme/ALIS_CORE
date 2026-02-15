/**
 * TraceViewer - Main container for trace data: loads trace, renders sub-components.
 */

import { useCallback, useMemo } from 'react';
import { Loader2, AlertCircle, FileSearch } from 'lucide-react';
import { cn } from '../../lib/utils';
import { useTraceData } from '../../hooks/useTraceData';
import { useSourceNavigation } from '../../hooks/useSourceNavigation';
import { ExpertAccordion } from './ExpertAccordion';
import { ConfidenceGauge } from './ConfidenceMeter';
import { SourcesList } from './SourcesList';
import { SynthesisWithSources } from './SynthesisWithSources';
import { SourceDetailPanel } from './SourceDetailPanel';
import { SourceSplitView } from './SourceSplitView';
import { CitationExportButton } from './CitationExportButton';
import { InlineFeedbackPanel } from './InlineFeedbackPanel';
import type { SourceResolution } from '../../types/trace';

export interface TraceViewerProps {
  traceId: string | null;
  onSourceClick?: (source: SourceResolution) => void;
  onSourceNavigate?: (source: SourceResolution) => void;
  className?: string;
}

export function TraceViewer({ traceId, onSourceClick, onSourceNavigate, className }: TraceViewerProps) {
  const { trace, sources, validity, isLoading, error } = useTraceData(traceId);
  const sourceNav = useSourceNavigation();

  const { selectSource, selectedSource, selectedIndex, isSplitView, setSplitView } = sourceNav;

  const handleSourceRefClick = useCallback((sourceIndex: number) => {
    const source = sources[sourceIndex];
    if (source) {
      selectSource(source, sourceIndex);
    }
  }, [sources, selectSource]);

  const closeSplitView = useCallback(() => setSplitView(false), [setSplitView]);

  const handleOpenArticle = useCallback((source: SourceResolution) => {
    onSourceNavigate?.(source);
  }, [onSourceNavigate]);

  const handleSourceListClick = useCallback((source: SourceResolution) => {
    let index = sources.indexOf(source);
    if (index < 0) {
      index = sources.findIndex(s => s.sourceId === source.sourceId);
    }
    selectSource(source, index >= 0 ? index : 0);
    onSourceClick?.(source);
  }, [sources, selectSource, onSourceClick]);

  const formattedTimestamp = useMemo(
    () => trace ? new Date(trace.timestamp).toLocaleString('it-IT') : '',
    [trace?.timestamp]
  );

  if (!traceId) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-12 text-slate-400", className)}>
        <FileSearch size={40} className="opacity-30 mb-3" aria-hidden="true" />
        <p className="text-sm font-medium">Nessun trace selezionato</p>
        <p className="text-xs mt-1">Avvia una query per vedere l'analisi dettagliata</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center py-12", className)} role="status">
        <Loader2 size={24} className="animate-spin text-blue-500" aria-hidden="true" />
        <span className="ml-2 text-sm text-slate-500">Caricamento trace...</span>
        <span className="sr-only">Caricamento in corso</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("flex items-center gap-2 p-4 text-red-600 bg-red-50 dark:bg-red-900/20 rounded-lg", className)} role="alert">
        <AlertCircle size={18} aria-hidden="true" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  if (!trace) return null;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header: Query + overall confidence */}
      <div className="flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Query</h3>
          <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
            {trace.query}
          </p>
          <div className="text-[10px] text-slate-400 mt-1">
            {formattedTimestamp}
          </div>
        </div>
        <div className="shrink-0 flex flex-col items-center">
          <ConfidenceGauge value={trace.confidence} size={56} />
          <span className="text-[10px] font-medium text-slate-500 mt-1">Confidenza</span>
        </div>
      </div>

      {/* Synthesis with source navigation */}
      <div>
        <h3 className="text-xs font-bold uppercase text-slate-500 mb-2">Sintesi</h3>
        <SourceSplitView
          isOpen={isSplitView}
          className="bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 min-h-[200px]"
          leftContent={
            <div className="p-4">
              <SynthesisWithSources
                text={trace.synthesis}
                sources={sources}
                selectedSourceIndex={selectedIndex}
                onSourceClick={handleSourceRefClick}
              />
            </div>
          }
          rightContent={
            selectedSource ? (
              <SourceDetailPanel
                source={selectedSource}
                validity={validity}
                onClose={closeSplitView}
                onOpenArticle={onSourceNavigate ? handleOpenArticle : undefined}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-sm text-slate-400 p-4">
                Clicca su un riferimento [N] per vedere il dettaglio
              </div>
            )
          }
        />
      </div>

      {/* Inline Feedback (thumbs) */}
      <InlineFeedbackPanel traceId={traceId} />

      {/* Expert Accordion */}
      <div>
        <h3 className="text-xs font-bold uppercase text-slate-500 mb-2">
          Analisi Esperti ({trace.experts.length})
        </h3>
        <ExpertAccordion experts={trace.experts} traceId={traceId} />
      </div>

      {/* Sources List */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-bold uppercase text-slate-500">
            Fonti ({sources.length})
          </h3>
          <CitationExportButton
            traceId={traceId}
            sourcesCount={sources.length}
          />
        </div>
        <SourcesList
          sources={sources}
          validity={validity}
          traceId={traceId}
          onSourceClick={handleSourceListClick}
        />
      </div>
    </div>
  );
}
