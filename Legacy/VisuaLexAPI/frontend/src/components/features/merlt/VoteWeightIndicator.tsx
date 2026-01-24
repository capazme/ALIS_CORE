/**
 * VoteWeightIndicator
 * ===================
 *
 * Shows the user's voting weight/authority based on RLCF framework.
 *
 * Formula RLCF: A_u(t) = 0.3·B_u + 0.5·T_u + 0.2·P_u
 *
 * Where:
 * - B_u = Baseline credentials (qualifiche professionali)
 * - T_u = Track record (storico performance esponenzialmente pesato)
 * - P_u = Performance recente (level authority)
 *
 * Il voto è pesato direttamente per A_u:
 * - Voto di utente con authority 0.8 vale 0.8 verso threshold
 * - Threshold approvazione: Σ(weighted_votes) >= 2.0
 */

import { motion } from 'framer-motion';
import { Scale, Award, Shield, User, GraduationCap, Gavel, Info } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { Tooltip } from '../../ui/Tooltip';
import type { AuthorityBreakdown } from '../../../types/merlt';

interface VoteWeightIndicatorProps {
  /** User's authority score A_u [0-1] */
  authorityScore: number;
  /** Optional breakdown components */
  breakdown?: AuthorityBreakdown;
  /** Compact mode for inline display */
  compact?: boolean;
  /** Show tooltip with explanation */
  showTooltip?: boolean;
  className?: string;
}

interface AuthorityTier {
  tier: 'novizio' | 'contributore' | 'esperto' | 'autorita';
  label: string;
  icon: typeof User;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  thresholdMin: number;
  thresholdMax: number;
}

/**
 * Determine user tier based on RLCF authority score.
 * Tiers aligned with backend thresholds.
 */
function getTier(authority: number): AuthorityTier {
  if (authority >= 0.8) {
    return {
      tier: 'autorita',
      label: 'Autorità',
      icon: Gavel,
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20',
      borderColor: 'border-purple-200 dark:border-purple-800/30',
      description: 'Il tuo voto pesa molto. Sei riconosciuto come esperto nel dominio legale.',
      thresholdMin: 0.8,
      thresholdMax: 1.0,
    };
  } else if (authority >= 0.6) {
    return {
      tier: 'esperto',
      label: 'Esperto',
      icon: Award,
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20',
      borderColor: 'border-blue-200 dark:border-blue-800/30',
      description: 'Hai dimostrato competenza. I tuoi voti hanno peso significativo.',
      thresholdMin: 0.6,
      thresholdMax: 0.8,
    };
  } else if (authority >= 0.4) {
    return {
      tier: 'contributore',
      label: 'Contributore',
      icon: GraduationCap,
      color: 'text-emerald-600 dark:text-emerald-400',
      bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
      borderColor: 'border-emerald-200 dark:border-emerald-800/30',
      description: 'Stai costruendo il tuo track record. Ogni voto corretto aumenta la tua authority.',
      thresholdMin: 0.4,
      thresholdMax: 0.6,
    };
  } else {
    return {
      tier: 'novizio',
      label: 'Novizio',
      icon: User,
      color: 'text-slate-500 dark:text-slate-400',
      bgColor: 'bg-slate-50 dark:bg-slate-800/50',
      borderColor: 'border-slate-200 dark:border-slate-700',
      description: 'Benvenuto! Inizia a validare per costruire la tua reputazione nel sistema.',
      thresholdMin: 0.0,
      thresholdMax: 0.4,
    };
  }
}

export function VoteWeightIndicator({
  authorityScore,
  breakdown,
  compact = false,
  showTooltip = true,
  className,
}: VoteWeightIndicatorProps) {
  const tier = getTier(authorityScore);
  const Icon = tier.icon;

  // Format authority as percentage
  const authorityPercent = Math.round(authorityScore * 100);

  const content = (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full',
        compact ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs',
        tier.bgColor,
        'border',
        tier.borderColor,
        className
      )}
    >
      <Icon size={compact ? 10 : 12} className={tier.color} />
      <span className={cn('font-medium', tier.color)}>
        {compact ? `${authorityPercent}%` : `${tier.label} ${authorityPercent}%`}
      </span>
    </motion.div>
  );

  if (!showTooltip) {
    return content;
  }

  const tooltipContent = (
    <div className="max-w-xs space-y-2">
      <p className="font-medium">{tier.label}</p>
      <p className="text-xs opacity-90">{tier.description}</p>
      <div className="text-xs opacity-75 border-t border-white/20 pt-2 mt-2">
        <p>Peso voto: {authorityPercent}% del valore</p>
        <p className="text-[10px] mt-1">
          Servono ≥2.0 punti pesati per approvare
        </p>
      </div>
    </div>
  );

  return (
    <Tooltip content={tooltipContent} placement="top">
      {content}
    </Tooltip>
  );
}

/**
 * Expanded vote weight display with RLCF breakdown
 */
export function VoteWeightCard({
  authorityScore,
  breakdown,
  totalContributions = 0,
  approvedContributions = 0,
  className,
}: VoteWeightIndicatorProps & {
  totalContributions?: number;
  approvedContributions?: number;
}) {
  const tier = getTier(authorityScore);
  const Icon = tier.icon;
  const authorityPercent = Math.round(authorityScore * 100);

  // Progress to next tier
  const getProgressToNextTier = () => {
    const nextThreshold = tier.thresholdMax;
    if (nextThreshold >= 1.0) return { progress: 100, nextLabel: null };

    const tierRange = tier.thresholdMax - tier.thresholdMin;
    const positionInTier = authorityScore - tier.thresholdMin;
    const progress = Math.min(100, Math.max(0, (positionInTier / tierRange) * 100));

    const nextTier = getTier(tier.thresholdMax);
    return { progress, nextLabel: nextTier.label };
  };

  const { progress, nextLabel } = getProgressToNextTier();

  // RLCF weights
  const WEIGHT_BASELINE = 0.3;
  const WEIGHT_TRACK_RECORD = 0.5;
  const WEIGHT_LEVEL_AUTH = 0.2;

  return (
    <div
      className={cn(
        'rounded-lg p-3 border',
        tier.bgColor,
        tier.borderColor,
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            'w-10 h-10 rounded-lg flex items-center justify-center',
            'bg-white dark:bg-slate-800 shadow-sm'
          )}
        >
          <Icon size={20} className={tier.color} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={cn('font-semibold text-sm', tier.color)}>
              {tier.label}
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Authority: {authorityPercent}%
            </span>
          </div>

          <p className="text-xs text-slate-600 dark:text-slate-400 mb-2">
            {tier.description}
          </p>

          {/* RLCF Breakdown */}
          {breakdown && (
            <div className="grid grid-cols-3 gap-2 mb-3 text-[10px]">
              <div className="bg-white/50 dark:bg-slate-800/30 rounded p-1.5">
                <div className="text-slate-500 mb-0.5">Qualifiche (B_u)</div>
                <div className="font-medium text-slate-700 dark:text-slate-300">
                  {Math.round(breakdown.baseline * 100)}%
                </div>
                <div className="text-[8px] text-slate-400">×{WEIGHT_BASELINE}</div>
              </div>
              <div className="bg-white/50 dark:bg-slate-800/30 rounded p-1.5">
                <div className="text-slate-500 mb-0.5">Track Record (T_u)</div>
                <div className="font-medium text-slate-700 dark:text-slate-300">
                  {Math.round(breakdown.track_record * 100)}%
                </div>
                <div className="text-[8px] text-slate-400">×{WEIGHT_TRACK_RECORD}</div>
              </div>
              <div className="bg-white/50 dark:bg-slate-800/30 rounded p-1.5">
                <div className="text-slate-500 mb-0.5">Performance (P_u)</div>
                <div className="font-medium text-slate-700 dark:text-slate-300">
                  {Math.round(breakdown.level_authority * 100)}%
                </div>
                <div className="text-[8px] text-slate-400">×{WEIGHT_LEVEL_AUTH}</div>
              </div>
            </div>
          )}

          {/* Progress to next tier */}
          {nextLabel && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-slate-500">Verso {nextLabel}</span>
                <span className="text-slate-400">{Math.round(progress)}%</span>
              </div>
              <div className="h-1.5 bg-white dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  className="h-full bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full"
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>
            </div>
          )}

          {/* Stats */}
          {totalContributions > 0 && (
            <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500">
              <span>
                {approvedContributions}/{totalContributions} contributi approvati
              </span>
              <span>
                {totalContributions > 0
                  ? Math.round((approvedContributions / totalContributions) * 100)
                  : 0}
                % successo
              </span>
            </div>
          )}

          {/* Vote impact example */}
          <div className="mt-2 pt-2 border-t border-slate-200/50 dark:border-slate-700/50">
            <div className="flex items-center gap-1 text-[10px] text-slate-500">
              <Scale size={10} />
              <span>
                Il tuo voto vale <strong className={tier.color}>{(authorityScore).toFixed(2)}</strong> punti
              </span>
            </div>
            <p className="text-[9px] text-slate-400 mt-0.5">
              Threshold approvazione: 2.0 punti totali
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
