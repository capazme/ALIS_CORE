/**
 * ExpertCompareView
 * =================
 *
 * Componente per visualizzare interpretazioni divergenti degli Expert MERL-T.
 *
 * Quando gli Expert hanno posizioni significativamente diverse (mode=divergent),
 * questo componente presenta le alternative side-by-side con:
 * - Tipo di ragionamento di ogni expert
 * - Confidenza e fonti giuridiche
 * - Spiegazione della divergenza
 * - Feedback per RLCF (quale interpretazione preferita)
 */

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Scale,
  BookOpen,
  Gavel,
  Building2,
  ThumbsUp,
  ThumbsDown,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react';
import { cn } from '../../lib/utils';

// =============================================================================
// TYPES
// =============================================================================

interface ExpertAlternative {
  expert: string;
  position: string;
  confidence: number;
  legal_basis: string[];
  reasoning_type: string;
}

interface DisagreementInfo {
  type?: string;
  intensity?: number;
  resolvability?: number;
  explanation?: string;
}

interface ExpertCompareViewProps {
  /** Query originale dell'utente */
  query: string;
  /** Alternative divergenti dagli Expert */
  alternatives: ExpertAlternative[];
  /** Sintesi principale (overview) */
  synthesis?: string;
  /** Informazioni sulla divergenza */
  disagreement?: DisagreementInfo;
  /** Trace ID per feedback RLCF */
  traceId?: string;
  /** User ID per feedback */
  userId?: string;
  /** Callback quando utente seleziona interpretazione preferita */
  onPreferenceSelected?: (expertType: string, traceId: string) => void;
  /** Callback per feedback dettagliato */
  onDetailedFeedback?: (data: {
    traceId: string;
    userId: string;
    preferredExpert: string;
    comment?: string;
  }) => void;
  /** Classe CSS aggiuntiva */
  className?: string;
}

// =============================================================================
// EXPERT CONFIG
// =============================================================================

const EXPERT_CONFIG: Record<
  string,
  { icon: typeof Scale; color: string; bgColor: string; borderColor: string; label: string }
> = {
  literal: {
    icon: BookOpen,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
    borderColor: 'border-blue-200 dark:border-blue-700',
    label: 'Letterale',
  },
  systemic: {
    icon: Building2,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
    borderColor: 'border-purple-200 dark:border-purple-700',
    label: 'Sistematico',
  },
  principles: {
    icon: Scale,
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-50 dark:bg-amber-900/30',
    borderColor: 'border-amber-200 dark:border-amber-700',
    label: 'Teleologico',
  },
  precedent: {
    icon: Gavel,
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
    borderColor: 'border-emerald-200 dark:border-emerald-700',
    label: 'Giurisprudenziale',
  },
};

const getExpertConfig = (expertType: string) => {
  return (
    EXPERT_CONFIG[expertType.toLowerCase()] || {
      icon: Scale,
      color: 'text-slate-600 dark:text-slate-400',
      bgColor: 'bg-slate-50 dark:bg-slate-900/30',
      borderColor: 'border-slate-200 dark:border-slate-700',
      label: expertType,
    }
  );
};

// =============================================================================
// SUBCOMPONENTS
// =============================================================================

interface ExpertCardProps {
  alternative: ExpertAlternative;
  isSelected: boolean;
  onSelect: () => void;
  showFeedback: boolean;
}

function ExpertCard({ alternative, isSelected, onSelect, showFeedback }: ExpertCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const config = getExpertConfig(alternative.expert);
  const Icon = config.icon;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'rounded-xl border-2 transition-all duration-200',
        config.borderColor,
        isSelected
          ? 'ring-2 ring-offset-2 ring-blue-500 dark:ring-blue-400'
          : 'hover:shadow-md',
        config.bgColor
      )}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between p-4 cursor-pointer',
          'border-b',
          config.borderColor
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <div className={cn('p-2 rounded-lg', config.bgColor)}>
            <Icon className={cn('w-5 h-5', config.color)} />
          </div>
          <div>
            <h3 className={cn('font-semibold', config.color)}>{config.label}</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {alternative.reasoning_type}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Confidence Badge */}
          <div
            className={cn(
              'px-2 py-1 rounded-full text-xs font-medium',
              alternative.confidence >= 0.8
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                : alternative.confidence >= 0.6
                  ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                  : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
            )}
          >
            {(alternative.confidence * 100).toFixed(0)}%
          </div>

          {/* Expand/Collapse */}
          <button
            className="p-1 hover:bg-white/50 dark:hover:bg-black/20 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-label={isExpanded ? 'Comprimi dettagli' : 'Espandi dettagli'}
            aria-expanded={isExpanded}
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-slate-500" aria-hidden="true" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500" aria-hidden="true" />
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="p-4 space-y-4">
              {/* Interpretation */}
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                  {alternative.position}
                </p>
              </div>

              {/* Legal Basis */}
              {alternative.legal_basis.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Fonti Giuridiche
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {alternative.legal_basis.map((source, idx) => (
                      <span
                        key={idx}
                        className={cn(
                          'inline-flex items-center gap-1 px-2 py-1 rounded text-xs',
                          'bg-white/50 dark:bg-black/20',
                          'border border-slate-200 dark:border-slate-600',
                          'text-slate-600 dark:text-slate-400'
                        )}
                      >
                        <ExternalLink className="w-3 h-3" />
                        {source}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Feedback Button */}
              {showFeedback && (
                <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect();
                    }}
                    className={cn(
                      'w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg',
                      'text-sm font-medium transition-all',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
                      isSelected
                        ? 'bg-blue-600 text-white'
                        : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
                    )}
                  >
                    {isSelected ? (
                      <>
                        <CheckCircle2 className="w-4 h-4" />
                        Selezionata come preferita
                      </>
                    ) : (
                      <>
                        <ThumbsUp className="w-4 h-4" />
                        Preferisco questa interpretazione
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

interface DisagreementBannerProps {
  disagreement?: DisagreementInfo;
}

function DisagreementBanner({ disagreement }: DisagreementBannerProps) {
  const [showDetails, setShowDetails] = useState(false);

  if (!disagreement) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg p-4"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-amber-800 dark:text-amber-300">
            Interpretazioni Divergenti Rilevate
          </h4>
          <p className="text-sm text-amber-700 dark:text-amber-400 mt-1">
            Gli esperti hanno posizioni significativamente diverse su questa questione.
            Valuta criticamente le alternative presentate.
          </p>

          {/* Metrics */}
          {(disagreement.intensity !== undefined || disagreement.resolvability !== undefined) && (
            <div className="flex gap-4 mt-3">
              {disagreement.intensity !== undefined && (
                <div className="text-xs">
                  <span className="text-amber-600 dark:text-amber-500">Intensita': </span>
                  <span className="font-medium text-amber-800 dark:text-amber-300">
                    {(disagreement.intensity * 100).toFixed(0)}%
                  </span>
                </div>
              )}
              {disagreement.resolvability !== undefined && (
                <div className="text-xs">
                  <span className="text-amber-600 dark:text-amber-500">Risolvibilita': </span>
                  <span className="font-medium text-amber-800 dark:text-amber-300">
                    {(disagreement.resolvability * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Explanation Toggle */}
          {disagreement.explanation && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 mt-3 text-xs text-amber-600 dark:text-amber-400 hover:underline"
            >
              <Info className="w-3 h-3" />
              {showDetails ? 'Nascondi dettagli' : 'Mostra dettagli divergenza'}
            </button>
          )}
        </div>
      </div>

      {/* Detailed Explanation */}
      <AnimatePresence>
        {showDetails && disagreement.explanation && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4 pt-4 border-t border-amber-200 dark:border-amber-700"
          >
            <div className="prose prose-sm dark:prose-invert max-w-none text-amber-800 dark:text-amber-300">
              <div dangerouslySetInnerHTML={{ __html: disagreement.explanation.replace(/\n/g, '<br/>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/_(.*?)_/g, '<em>$1</em>') }} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExpertCompareView({
  query,
  alternatives,
  synthesis,
  disagreement,
  traceId,
  userId,
  onPreferenceSelected,
  onDetailedFeedback,
  className,
}: ExpertCompareViewProps) {
  const [selectedExpert, setSelectedExpert] = useState(null as string | null);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);

  const handleSelectExpert = (expertType: string) => {
    setSelectedExpert(expertType);
    if (traceId && onPreferenceSelected) {
      onPreferenceSelected(expertType, traceId);
    }
  };

  const handleSubmitFeedback = () => {
    if (selectedExpert && traceId && userId && onDetailedFeedback) {
      onDetailedFeedback({
        traceId,
        userId,
        preferredExpert: selectedExpert,
        comment: feedbackComment || undefined,
      });
      setFeedbackSent(true);
    }
  };

  // Sort alternatives by confidence
  const sortedAlternatives = useMemo(
    () => [...alternatives].sort((a, b) => b.confidence - a.confidence),
    [alternatives]
  );

  if (alternatives.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Confronto Interpretazioni
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {alternatives.length} posizioni alternative identificate
          </p>
        </div>
      </div>

      {/* Disagreement Banner */}
      <DisagreementBanner disagreement={disagreement} />

      {/* Query Context */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4">
        <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
          Domanda Analizzata
        </h4>
        <p className="text-slate-700 dark:text-slate-300 italic">&ldquo;{query}&rdquo;</p>
      </div>

      {/* Expert Cards Grid */}
      <div
        className={cn(
          'grid gap-4',
          alternatives.length === 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
        )}
      >
        {sortedAlternatives.map((alt: ExpertAlternative) => (
          <ExpertCard
            key={alt.expert}
            alternative={alt}
            isSelected={selectedExpert === alt.expert}
            onSelect={() => handleSelectExpert(alt.expert)}
            showFeedback={!!traceId && !feedbackSent}
          />
        ))}
      </div>

      {/* Feedback Section */}
      {traceId && selectedExpert && !feedbackSent && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-4"
        >
          <h4 className="font-medium text-slate-900 dark:text-slate-100">
            Feedback sulla tua scelta
          </h4>

          <div>
            <label className="block text-sm text-slate-600 dark:text-slate-400 mb-2">
              Commento opzionale (aiuta a migliorare il sistema)
            </label>
            <textarea
              value={feedbackComment}
              onChange={(e) => setFeedbackComment(e.target.value)}
              placeholder="Perche' preferisci questa interpretazione?"
              rows={3}
              className={cn(
                'w-full px-3 py-2 rounded-lg border',
                'border-slate-300 dark:border-slate-600',
                'bg-white dark:bg-slate-900',
                'text-slate-900 dark:text-slate-100',
                'placeholder:text-slate-400 dark:placeholder:text-slate-500',
                'focus:ring-2 focus:ring-blue-500 focus:border-transparent',
                'resize-none'
              )}
            />
          </div>

          <div className="flex justify-end gap-3">
            <button
              onClick={() => {
                setSelectedExpert(null);
                setFeedbackComment('');
              }}
              className="px-4 py-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              Annulla
            </button>
            <button
              onClick={handleSubmitFeedback}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium',
                'bg-blue-600 text-white hover:bg-blue-700',
                'transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'
              )}
            >
              Invia Feedback
            </button>
          </div>
        </motion.div>
      )}

      {/* Feedback Sent Confirmation */}
      {feedbackSent && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700 rounded-lg"
        >
          <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
          <div>
            <p className="font-medium text-green-800 dark:text-green-300">
              Grazie per il tuo feedback!
            </p>
            <p className="text-sm text-green-600 dark:text-green-400">
              La tua preferenza aiuta a migliorare il sistema di interpretazione.
            </p>
          </div>
        </motion.div>
      )}

      {/* Synthesis Overview (if provided) */}
      {synthesis && (
        <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
            Sintesi Complessiva
          </h4>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="text-slate-700 dark:text-slate-300">{synthesis}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default ExpertCompareView;
