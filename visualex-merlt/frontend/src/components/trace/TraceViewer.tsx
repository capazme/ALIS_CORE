/**
 * TraceViewer - Main trace container with redesigned visual hierarchy.
 *
 * Section order:
 * 1. ConfidenceGauge centered (80px) + DisagreementBadge
 * 2. Synthesis (serif font, leading-loose, dominant weight)
 * 3. Expert Grid (multi-open accordion)
 * 4. Sources (collapsed summary, expand on demand)
 * 5. InlineFeedback (after everything — user gives feedback after reading)
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { Loader2, AlertCircle, FileSearch, ChevronDown, ChevronRight } from 'lucide-react';
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

interface DisagreementBadgeProps {
  intensity: number;
  onCompare?: () => void;
}

function DisagreementBadge({ intensity, onCompare }: DisagreementBadgeProps) {
  if (intensity <= 0.3) return null;

  const isHigh = intensity > 0.7;

  return (
    <button
      onClick={onCompare}
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
        isHigh
          ? 'bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50'
          : 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:hover:bg-amber-900/50',
      )}
    >
      <span className={cn(
        'w-2 h-2 rounded-full',
        isHigh ? 'bg-red-500 animate-pulse' : 'bg-amber-500',
      )} />
      {isHigh ? 'Forte disaccordo' : 'Posizioni divergenti'}
    </button>
  );
}

function computeDisagreementIntensity(experts: Array<{ confidence: number }>): number {
  if (experts.length < 2) return 0;
  const confidences = experts.map((e) => e.confidence);
  const max = Math.max(...confidences);
  const min = Math.min(...confidences);
  return max - min;
}

export function TraceViewer({ traceId, onSourceClick, onSourceNavigate, className }: TraceViewerProps) {
  const { trace, sources, validity, isLoading, error } = useTraceData(traceId);
  const sourceNav = useSourceNavigation();
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  const { selectSource, selectedSource, selectedIndex, isSplitView, setSplitView } = sourceNav;

  useEffect(() => {
    setSplitView(false);
    setSourcesExpanded(false);
  }, [traceId, setSplitView]);

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
      index = sources.findIndex((s) => s.sourceId === source.sourceId);
    }
    selectSource(source, index >= 0 ? index : 0);
    onSourceClick?.(source);
  }, [sources, selectSource, onSourceClick]);

  const formattedTimestamp = useMemo(
    () => (trace ? new Date(trace.timestamp).toLocaleString('it-IT') : ''),
    [trace?.timestamp],
  );

  const disagreementIntensity = useMemo(
    () => (trace ? computeDisagreementIntensity(trace.experts) : 0),
    [trace],
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
      {/* 1. Confidence Gauge — centered, prominent */}
      <div className="flex flex-col items-center gap-2">
        <ConfidenceGauge value={trace.confidence} size={80} />
        <span className="text-[10px] font-medium text-slate-500">Confidenza</span>
        <DisagreementBadge intensity={disagreementIntensity} />
      </div>

      {/* Query + timestamp — compact */}
      <div>
        <h3 className="text-xs font-bold uppercase text-slate-500 mb-1">Query</h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
          {trace.query}
        </p>
        <div className="text-[10px] text-slate-400 mt-1">
          {formattedTimestamp}
        </div>
      </div>

      {/* 2. Synthesis — dominant visual weight */}
      <div>
        <h3 className="text-xs font-bold uppercase text-slate-500 mb-2">Sintesi</h3>
        <SourceSplitView
          isOpen={isSplitView}
          hasSelectedSource={!!selectedSource}
          className="bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 min-h-[200px]"
          leftContent={
            <div className="p-4">
              <SynthesisWithSources
                text={trace.synthesis}
                sources={sources}
                selectedSourceIndex={selectedIndex}
                onSourceClick={handleSourceRefClick}
                className="font-serif leading-loose"
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

      {/* 3. Expert Grid — multi-open accordion */}
      <div>
        <h3 className="text-xs font-bold uppercase text-slate-500 mb-2">
          Analisi Esperti ({trace.experts.length})
        </h3>
        <ExpertAccordion experts={trace.experts} traceId={traceId} />
      </div>

      {/* 4. Sources — collapsed summary, expand on demand */}
      <div>
        <button
          onClick={() => setSourcesExpanded(!sourcesExpanded)}
          className="flex items-center justify-between w-full mb-2 group"
        >
          <h3 className="text-xs font-bold uppercase text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-300 transition-colors">
            Fonti ({sources.length})
          </h3>
          <div className="flex items-center gap-2">
            <CitationExportButton
              traceId={traceId}
              sourcesCount={sources.length}
            />
            {sourcesExpanded ? (
              <ChevronDown size={14} className="text-slate-400" />
            ) : (
              <ChevronRight size={14} className="text-slate-400" />
            )}
          </div>
        </button>
        {sourcesExpanded && (
          <SourcesList
            sources={sources}
            validity={validity}
            traceId={traceId}
            onSourceClick={handleSourceListClick}
          />
        )}
        {!sourcesExpanded && sources.length > 0 && (
          <p className="text-xs text-slate-400">
            {sources.slice(0, 3).map((s) => s.label).join(', ')}
            {sources.length > 3 && ` e altre ${sources.length - 3}`}
          </p>
        )}
      </div>

      {/* 5. Inline Feedback — last, after user has read everything */}
      <InlineFeedbackPanel traceId={traceId} />
    </div>
  );
}
