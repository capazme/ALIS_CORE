/**
 * ValidationCard
 * ==============
 *
 * Industrial-grade validation card with:
 * - Always-visible description preview (truncated with expand)
 * - Vote confirmation ripple animation
 * - Slide-out animation on vote
 * - Skip button
 * - Selection checkbox for bulk actions
 * - Focus indicator for keyboard navigation
 * - Vote weight preview
 */

import { useState, useRef, forwardRef } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import {
  ThumbsUp,
  ThumbsDown,
  Edit3,
  SkipForward,
  ChevronDown,
  ChevronRight,
  Bot,
  User,
  Square,
  CheckSquare,
  Sparkles,
  ArrowRight,
  Link2,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { VoteWeightIndicator } from './VoteWeightIndicator';
import { formatUrnToReadable } from '../../../utils/normattivaParser';
import type { VoteType, PendingRelation } from '../../../types/merlt';
import type { ValidationItem } from '../../../hooks/useValidationState';

interface ValidationCardProps {
  item: ValidationItem;
  isValidating: boolean;
  isFocused: boolean;
  isSelected: boolean;
  hasVoted: boolean;
  userAuthorityScore?: number;
  /** Callbacks */
  onVote: (vote: VoteType) => void;
  onEdit: () => void;
  onSkip: () => void;
  onToggleSelect: () => void;
  /** Show selection checkbox */
  selectionMode?: boolean;
  /** Custom class */
  className?: string;
}

// Ripple animation component
function VoteRipple({
  color,
  isVisible,
}: {
  color: 'emerald' | 'red' | 'amber';
  isVisible: boolean;
}) {
  const colors = {
    emerald: 'bg-emerald-500',
    red: 'bg-red-500',
    amber: 'bg-amber-500',
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ scale: 0, opacity: 0.8 }}
          animate={{ scale: 4, opacity: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={cn(
            'absolute inset-0 rounded-full z-0 pointer-events-none',
            colors[color]
          )}
          style={{ transformOrigin: 'center' }}
        />
      )}
    </AnimatePresence>
  );
}

const ENTITY_TYPE_LABELS: Record<string, string> = {
  norma: 'Norma',
  versione: 'Versione',
  concetto: 'Concetto',
  principio: 'Principio',
  definizione_legale: 'Definizione',
  soggetto_giuridico: 'Soggetto',
  ruolo_giuridico: 'Ruolo',
  organo: 'Organo',
  fatto_giuridico: 'Fatto',
  procedura: 'Procedura',
  sanzione: 'Sanzione',
  termine: 'Termine',
  responsabilita: 'Responsabilità',
  diritto_soggettivo: 'Diritto',
  caso: 'Caso',
  dottrina: 'Dottrina',
  atto_giudiziario: 'Atto Giudiziario',
};

const RELATION_TYPE_LABELS: Record<string, string> = {
  DISCIPLINA: 'disciplina',
  DEFINISCE: 'definisce',
  PRESUPPONE: 'presuppone',
  CITA: 'cita',
  DEROGA_A: 'deroga a',
  INTEGRA: 'integra',
  APPLICA_A: 'si applica a',
  SPECIES: 'è tipo di',
  PRODUCE_EFFETTO: 'produce effetto',
  TUTELA: 'tutela',
  LIMITA: 'limita',
  INTERPRETA: 'interpreta',
  CONTIENE: 'contiene',
  PARTE_DI: 'è parte di',
  SOSTITUISCE: 'sostituisce',
  ABROGA_TOTALMENTE: 'abroga',
  ABROGA_PARZIALMENTE: 'abroga parz.',
  ATTUA: 'attua',
  RECEPISCE: 'recepisce',
  EMESSO_DA: 'emesso da',
};

/**
 * Visual component for displaying relation source → target
 */
function RelationDisplay({ relation }: { relation: PendingRelation }) {
  const sourceLabel = formatUrnToReadable(relation.source_urn);
  const targetLabel = formatUrnToReadable(relation.target_urn);
  const relationLabel = RELATION_TYPE_LABELS[relation.relation_type] || relation.relation_type.toLowerCase().replace(/_/g, ' ');

  return (
    <div className="flex flex-col gap-1.5">
      {/* Source node */}
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
        <span className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
          {sourceLabel}
        </span>
      </div>

      {/* Relation arrow with label */}
      <div className="flex items-center gap-2 pl-3">
        <div className="flex items-center gap-1 px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/20 rounded border border-indigo-200 dark:border-indigo-800/30">
          <ArrowRight size={10} className="text-indigo-500" />
          <span className="text-[10px] font-medium text-indigo-600 dark:text-indigo-400 uppercase tracking-wide">
            {relationLabel}
          </span>
        </div>
      </div>

      {/* Target node */}
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
        <span className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
          {targetLabel}
        </span>
      </div>
    </div>
  );
}

export const ValidationCard = forwardRef<HTMLDivElement, ValidationCardProps>(
  function ValidationCard(
    {
      item,
      isValidating,
      isFocused,
      isSelected,
      hasVoted,
      userAuthorityScore = 1.0,
      onVote,
      onEdit,
      onSkip,
      onToggleSelect,
      selectionMode = false,
      className,
    },
    ref
  ) {
    const [expanded, setExpanded] = useState(false);
    const [voteAnimation, setVoteAnimation] = useState<'approve' | 'reject' | 'edit' | null>(null);
    const [isExiting, setIsExiting] = useState(false);
    const reducedMotion = useReducedMotion();

    const cardRef = useRef<HTMLDivElement>(null);

    // Get labels based on item type
    const getTypeLabel = () => {
      if (item.type === 'entity') {
        const entity = item.raw as { tipo: string };
        return ENTITY_TYPE_LABELS[entity.tipo] || entity.tipo;
      } else {
        return RELATION_TYPE_LABELS[item.name] || item.name;
      }
    };

    const confidencePercent = Math.round(item.confidence * 100);

    // Handle vote with animation
    const handleVote = async (vote: VoteType) => {
      if (isValidating || isExiting) return;

      // Trigger ripple animation
      const rippleColor = vote === 'approve' ? 'emerald' : vote === 'reject' ? 'red' : 'amber';
      setVoteAnimation(vote);

      // Start exit animation after ripple
      if (!reducedMotion) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setIsExiting(true);
        await new Promise((resolve) => setTimeout(resolve, 300));
      }

      onVote(vote);

      // Reset state
      setVoteAnimation(null);
      setIsExiting(false);
    };

    // Truncate description for preview
    const descriptionPreview = item.description
      ? item.description.length > 100
        ? item.description.slice(0, 100) + '...'
        : item.description
      : 'Nessuna descrizione disponibile.';

    return (
      <motion.div
        ref={ref}
        layout={!reducedMotion}
        initial={{ opacity: 0, y: 10 }}
        animate={{
          opacity: isExiting ? 0 : 1,
          y: isExiting ? -20 : 0,
          x: isExiting ? (voteAnimation === 'approve' ? 50 : voteAnimation === 'reject' ? -50 : 0) : 0,
          scale: isExiting ? 0.95 : 1,
        }}
        exit={{ opacity: 0, height: 0, marginBottom: 0 }}
        transition={{
          layout: { duration: 0.2 },
          opacity: { duration: 0.2 },
          y: { duration: 0.2 },
        }}
        className={cn(
          'relative bg-white dark:bg-slate-800 border rounded-lg overflow-hidden transition-all',
          isFocused && 'ring-2 ring-primary-500 ring-offset-2 dark:ring-offset-slate-900',
          hasVoted
            ? 'border-emerald-300 dark:border-emerald-700 opacity-60'
            : 'border-slate-200 dark:border-slate-700',
          className
        )}
      >
        {/* Ripple animation overlay */}
        <div className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none">
          <VoteRipple
            color={voteAnimation === 'approve' ? 'emerald' : voteAnimation === 'reject' ? 'red' : 'amber'}
            isVisible={voteAnimation !== null}
          />
        </div>

        {/* Header */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="relative z-10 w-full flex items-start gap-3 p-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
        >
          {/* Selection checkbox */}
          {selectionMode && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleSelect();
              }}
              className="flex-shrink-0 mt-0.5"
            >
              {isSelected ? (
                <CheckSquare size={16} className="text-primary-500" />
              ) : (
                <Square size={16} className="text-slate-400" />
              )}
            </button>
          )}

          <div className="flex-1 min-w-0">
            {/* Badges row */}
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              {/* Type badge - show entity type or "Relazione" for relations */}
              <span
                className={cn(
                  'inline-flex items-center gap-1 text-xs font-medium px-1.5 py-0.5 rounded',
                  item.type === 'entity'
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300'
                )}
              >
                {item.type === 'relation' && <Link2 size={10} />}
                {item.type === 'entity' ? getTypeLabel() : 'Relazione'}
              </span>

              {/* Origin badge - AI or Community */}
              {item.isAiGenerated ? (
                <span className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 rounded">
                  <Bot size={10} />
                  AI {confidencePercent}%
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded">
                  <User size={10} />
                  Community
                </span>
              )}

              {/* High confidence indicator */}
              {item.confidence >= 0.9 && (
                <span className="inline-flex items-center gap-0.5 text-[10px] px-1 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded">
                  <Sparkles size={8} />
                  Alta
                </span>
              )}
            </div>

            {/* Content: Entity name or Relation display */}
            {item.type === 'entity' ? (
              <>
                <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                  {item.name}
                </p>
                {item.description && (
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                    {descriptionPreview}
                  </p>
                )}
              </>
            ) : (
              <RelationDisplay relation={item.raw as PendingRelation} />
            )}
          </div>

          {expanded ? (
            <ChevronDown size={16} className="text-slate-400 mt-1 flex-shrink-0" />
          ) : (
            <ChevronRight size={16} className="text-slate-400 mt-1 flex-shrink-0" />
          )}
        </button>

        {/* Expanded content */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden relative z-10"
            >
              <div className="px-3 pb-3 border-t border-slate-100 dark:border-slate-700 pt-2 space-y-3">
                {/* Full description */}
                {item.description && (
                  <p className="text-xs text-slate-600 dark:text-slate-400">
                    {item.description}
                  </p>
                )}

                {/* Vote progress */}
                <div className="flex items-center gap-2 text-xs">
                  <div className="flex-1 h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{
                        width: `${Math.min((item.approvalScore / 2) * 100, 100)}%`,
                      }}
                      className="h-full bg-emerald-500 rounded-full"
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                    />
                  </div>
                  <span className="text-slate-500 whitespace-nowrap">
                    {item.approvalScore.toFixed(1)} / 2.0
                  </span>
                </div>

                {/* Vote weight preview */}
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500">Il tuo impatto:</span>
                  <VoteWeightIndicator
                    authorityScore={userAuthorityScore}
                    compact
                    showTooltip
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Actions - always visible */}
        <div className="relative z-10 flex items-center gap-1 px-3 pb-3 pt-1">
          {/* Approve */}
          <button
            onClick={() => handleVote('approve')}
            disabled={isValidating || isExiting}
            className={cn(
              'flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
              'bg-emerald-50 hover:bg-emerald-100 text-emerald-700',
              'dark:bg-emerald-900/20 dark:hover:bg-emerald-900/30 dark:text-emerald-400',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'active:scale-95'
            )}
          >
            <ThumbsUp size={12} />
            <span className="hidden sm:inline">Approva</span>
            <kbd className="hidden sm:inline text-[10px] opacity-50 ml-1">A</kbd>
          </button>

          {/* Reject */}
          <button
            onClick={() => handleVote('reject')}
            disabled={isValidating || isExiting}
            className={cn(
              'flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
              'bg-red-50 hover:bg-red-100 text-red-700',
              'dark:bg-red-900/20 dark:hover:bg-red-900/30 dark:text-red-400',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'active:scale-95'
            )}
          >
            <ThumbsDown size={12} />
            <span className="hidden sm:inline">Rifiuta</span>
            <kbd className="hidden sm:inline text-[10px] opacity-50 ml-1">R</kbd>
          </button>

          {/* Skip */}
          <button
            onClick={onSkip}
            disabled={isValidating || isExiting}
            className={cn(
              'p-1.5 rounded-md transition-colors',
              'text-slate-400 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-900/20 dark:hover:text-amber-400',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title="Salta (rivedi dopo) [S]"
          >
            <SkipForward size={14} />
          </button>

          {/* Edit */}
          <button
            onClick={onEdit}
            disabled={isValidating || isExiting}
            className={cn(
              'p-1.5 rounded-md transition-colors',
              'text-slate-400 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 dark:hover:text-blue-400',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title="Modifica [E]"
          >
            <Edit3 size={14} />
          </button>
        </div>
      </motion.div>
    );
  }
);
