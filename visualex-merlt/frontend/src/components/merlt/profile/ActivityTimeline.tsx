/**
 * ActivityTimeline
 * ================
 *
 * Mostra la timeline delle attività recenti dell'utente con:
 * - Tipo di attività (voto, proposta, edit)
 * - Outcome (approvato, rigettato, pending)
 * - Impatto sul track record (delta)
 * - Timestamp relativo
 * - Dominio legale coinvolto
 *
 * Permette di capire DA DOVE derivano le statistiche.
 */

import { motion } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  Clock,
  ThumbsUp,
  Plus,
  Edit3,
  ArrowUp,
  ArrowDown,
  Minus,
  Activity,
  Scale,
  Briefcase,
  Building2,
  Gavel,
  BookOpen,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';
import type { ProfileActivityEntry, LegalDomain } from '../../../../types/merlt';

// =============================================================================
// CONSTANTS
// =============================================================================

const ACTIVITY_ICONS = {
  vote: ThumbsUp,
  proposal: Plus,
  edit: Edit3,
  ner_feedback: BookOpen,
} as const;

const ACTIVITY_LABELS = {
  vote: 'Voto',
  proposal: 'Proposta',
  edit: 'Modifica',
  ner_feedback: 'Citazione NER',
} as const;

const OUTCOME_CONFIG = {
  approved: {
    icon: CheckCircle,
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
    borderColor: 'border-emerald-200 dark:border-emerald-800/30',
    label: 'Approvato',
  },
  rejected: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800/30',
    label: 'Rigettato',
  },
  pending: {
    icon: Clock,
    color: 'text-amber-500',
    bgColor: 'bg-amber-50 dark:bg-amber-900/20',
    borderColor: 'border-amber-200 dark:border-amber-800/30',
    label: 'In attesa',
  },
} as const;

const DOMAIN_ICONS: Partial<Record<LegalDomain, typeof Scale>> = {
  civile: Scale,
  penale: Gavel,
  lavoro: Briefcase,
  amministrativo: Building2,
};

// =============================================================================
// HELPERS
// =============================================================================

function formatRelativeTime(isoString: string | undefined | null): string {
  if (!isoString) return 'N/A';

  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return 'N/A';

    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'ora';
    if (diffMins < 60) return `${diffMins}m fa`;
    if (diffHours < 24) return `${diffHours}h fa`;
    if (diffDays < 7) return `${diffDays}g fa`;
    return date.toLocaleDateString('it-IT', { day: 'numeric', month: 'short' });
  } catch {
    return 'N/A';
  }
}

function formatDelta(delta: number | undefined | null): {
  text: string;
  icon: typeof ArrowUp;
  color: string;
} | null {
  // Handle null, undefined, and zero - all return null (no delta to display)
  if (delta == null || delta === 0) return null;

  if (delta > 0) {
    return {
      text: `+${delta.toFixed(3)}`,
      icon: ArrowUp,
      color: 'text-emerald-500',
    };
  } else {
    return {
      text: delta.toFixed(3),
      icon: ArrowDown,
      color: 'text-red-500',
    };
  }
}

// =============================================================================
// ACTIVITY ITEM
// =============================================================================

interface ActivityItemProps {
  activity: ProfileActivityEntry;
  index: number;
  isLast: boolean;
}

function ActivityItem({ activity, index, isLast }: ActivityItemProps) {
  // Defensive: use defaults if data is missing or unexpected
  const outcomeConfig = OUTCOME_CONFIG[activity.outcome] ?? OUTCOME_CONFIG.pending;
  const OutcomeIcon = outcomeConfig.icon;
  const ActivityIcon = ACTIVITY_ICONS[activity.type] ?? ACTIVITY_ICONS.proposal;
  const activityLabel = ACTIVITY_LABELS[activity.type] ?? 'Attività';
  const delta = formatDelta(activity.track_record_delta);
  const DomainIcon = activity.domain ? DOMAIN_ICONS[activity.domain as LegalDomain] : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="relative flex gap-3"
    >
      {/* Timeline line */}
      {!isLast && (
        <div className="absolute left-[15px] top-8 bottom-0 w-0.5 bg-slate-200 dark:bg-slate-700" />
      )}

      {/* Icon */}
      <div
        className={cn(
          'relative z-10 w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          outcomeConfig.bgColor,
          'border',
          outcomeConfig.borderColor
        )}
      >
        <OutcomeIcon size={14} className={outcomeConfig.color} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 pb-4">
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            {/* Activity type + item name */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400 flex items-center gap-1">
                <ActivityIcon size={10} />
                {activityLabel}
              </span>
              <span className="text-xs text-slate-400">su</span>
              <span className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate">
                {activity.item_name}
              </span>
            </div>

            {/* Metadata row */}
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              {/* Outcome badge */}
              <span
                className={cn(
                  'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
                  outcomeConfig.bgColor,
                  outcomeConfig.color
                )}
              >
                {outcomeConfig.label}
              </span>

              {/* Item type */}
              <span className="text-[10px] text-slate-400">
                {activity.item_type === 'entity' ? 'Entità' : activity.item_type === 'citation' ? 'Citazione' : 'Relazione'}
              </span>

              {/* Domain */}
              {activity.domain && DomainIcon && (
                <span className="text-[10px] text-slate-400 flex items-center gap-0.5 capitalize">
                  <DomainIcon size={10} />
                  {activity.domain}
                </span>
              )}

              {/* Timestamp */}
              <span className="text-[10px] text-slate-400">
                {formatRelativeTime(activity.timestamp)}
              </span>
            </div>
          </div>

          {/* Track record delta */}
          {delta && (
            <div
              className={cn(
                'flex items-center gap-0.5 px-2 py-1 rounded-lg',
                'bg-slate-100 dark:bg-slate-800'
              )}
            >
              <delta.icon size={12} className={delta.color} />
              <span className={cn('text-xs font-mono font-medium', delta.color)}>
                {delta.text}
              </span>
              <span className="text-[9px] text-slate-400 ml-0.5">T_u</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface ActivityTimelineProps {
  activities: ProfileActivityEntry[];
  className?: string;
  maxItems?: number;
}

export function ActivityTimeline({
  activities,
  className,
  maxItems = 10,
}: ActivityTimelineProps) {
  const displayedActivities = activities.slice(0, maxItems);

  // Calculate summary stats
  const stats = {
    total: activities.length,
    approved: activities.filter((a) => a.outcome === 'approved').length,
    rejected: activities.filter((a) => a.outcome === 'rejected').length,
    pending: activities.filter((a) => a.outcome === 'pending').length,
    totalDelta: activities.reduce((sum, a) => sum + (a.track_record_delta || 0), 0),
  };

  if (activities.length === 0) {
    return (
      <div
        className={cn(
          'rounded-xl border border-slate-200 dark:border-slate-700',
          'bg-white dark:bg-slate-900 p-5',
          className
        )}
      >
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Attività Recente
          </h3>
        </div>
        <div className="text-center py-8 text-slate-400">
          <Clock size={32} className="mx-auto mb-2 opacity-50" />
          <p className="text-sm">Nessuna attività recente</p>
          <p className="text-xs mt-1">Inizia a validare per vedere la tua cronologia</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-200 dark:border-slate-700',
        'bg-white dark:bg-slate-900 p-5',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity size={18} className="text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Attività Recente
          </h3>
        </div>

        {/* Summary */}
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-emerald-500 flex items-center gap-1">
            <CheckCircle size={10} />
            {stats.approved}
          </span>
          <span className="text-red-500 flex items-center gap-1">
            <XCircle size={10} />
            {stats.rejected}
          </span>
          <span className="text-amber-500 flex items-center gap-1">
            <Clock size={10} />
            {stats.pending}
          </span>
        </div>
      </div>

      {/* Track record impact summary */}
      {stats.totalDelta !== 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={cn(
            'mb-4 p-2.5 rounded-lg flex items-center justify-between',
            stats.totalDelta > 0
              ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30'
              : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30'
          )}
        >
          <span className="text-xs text-slate-600 dark:text-slate-400">
            Impatto netto sul Track Record (T_u)
          </span>
          <span
            className={cn(
              'text-sm font-mono font-bold flex items-center gap-1',
              stats.totalDelta > 0 ? 'text-emerald-600' : 'text-red-600'
            )}
          >
            {stats.totalDelta > 0 ? (
              <ArrowUp size={14} />
            ) : (
              <ArrowDown size={14} />
            )}
            {stats.totalDelta > 0 ? '+' : ''}
            {stats.totalDelta.toFixed(3)}
          </span>
        </motion.div>
      )}

      {/* Timeline */}
      <div className="space-y-0">
        {displayedActivities.map((activity, index) => (
          <ActivityItem
            key={activity.id || `${activity.timestamp}-${index}`}
            activity={activity}
            index={index}
            isLast={index === displayedActivities.length - 1}
          />
        ))}
      </div>

      {/* Show more indicator */}
      {activities.length > maxItems && (
        <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 text-center">
          <span className="text-xs text-slate-400">
            +{activities.length - maxItems} altre attività
          </span>
        </div>
      )}

      {/* Tracing explanation */}
      <div className="mt-4 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50">
        <p className="text-[10px] text-slate-500 dark:text-slate-400">
          <strong className="text-slate-600 dark:text-slate-300">Come leggere i delta:</strong>{' '}
          Ogni attività con esito influenza il tuo Track Record (T_u).
          Voti corretti e proposte approvate aumentano T_u, mentre rigetti lo riducono.
          Il peso dell'impatto dipende dalla tua authority attuale e dal dominio.
        </p>
      </div>
    </div>
  );
}

export default ActivityTimeline;
