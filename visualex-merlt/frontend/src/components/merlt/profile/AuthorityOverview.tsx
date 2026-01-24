/**
 * AuthorityOverview
 * =================
 *
 * Mostra l'authority RLCF dell'utente con:
 * - Radial progress animato
 * - Tier badge con colore
 * - Progress bar verso il prossimo tier
 * - Descrizione educativa del tier
 *
 * Formula RLCF: A_u(t) = 0.3·B_u + 0.5·T_u + 0.2·P_u
 */

import { motion } from 'framer-motion';
import { User, GraduationCap, Award, Gavel, TrendingUp } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import type { AuthorityTier, AuthorityBreakdown } from '../../../../types/merlt';

// =============================================================================
// TIER CONFIG
// =============================================================================

interface TierConfig {
  label: string;
  icon: typeof User;
  color: string;
  bgColor: string;
  borderColor: string;
  gradientFrom: string;
  gradientTo: string;
  description: string;
  thresholdMin: number;
  thresholdMax: number;
}

const TIER_CONFIGS: Record<AuthorityTier, TierConfig> = {
  novizio: {
    label: 'Novizio',
    icon: User,
    color: 'text-slate-600 dark:text-slate-400',
    bgColor: 'bg-slate-100 dark:bg-slate-800/50',
    borderColor: 'border-slate-200 dark:border-slate-700',
    gradientFrom: 'from-slate-400',
    gradientTo: 'to-slate-500',
    description: 'Benvenuto! Inizia a validare entità e relazioni per costruire la tua reputazione nel sistema.',
    thresholdMin: 0,
    thresholdMax: 0.4,
  },
  contributore: {
    label: 'Contributore',
    icon: GraduationCap,
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
    borderColor: 'border-emerald-200 dark:border-emerald-800/30',
    gradientFrom: 'from-emerald-400',
    gradientTo: 'to-emerald-500',
    description: 'Stai costruendo il tuo track record. Ogni voto corretto aumenta la tua authority nel sistema.',
    thresholdMin: 0.4,
    thresholdMax: 0.6,
  },
  esperto: {
    label: 'Esperto',
    icon: Award,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    borderColor: 'border-blue-200 dark:border-blue-800/30',
    gradientFrom: 'from-blue-400',
    gradientTo: 'to-blue-500',
    description: 'Hai dimostrato competenza. I tuoi voti hanno peso significativo nelle decisioni del sistema.',
    thresholdMin: 0.6,
    thresholdMax: 0.8,
  },
  autorita: {
    label: 'Autorità',
    icon: Gavel,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    borderColor: 'border-purple-200 dark:border-purple-800/30',
    gradientFrom: 'from-purple-400',
    gradientTo: 'to-purple-500',
    description: 'Sei riconosciuto come esperto nel dominio giuridico. Il tuo voto pesa molto nelle validazioni.',
    thresholdMin: 0.8,
    thresholdMax: 1.0,
  },
};

// =============================================================================
// RADIAL PROGRESS
// =============================================================================

interface RadialProgressProps {
  value: number; // 0-1
  size?: number;
  strokeWidth?: number;
  tier: AuthorityTier;
  className?: string;
}

function RadialProgress({
  value,
  size = 160,
  strokeWidth = 12,
  tier,
  className,
}: RadialProgressProps) {
  const config = TIER_CONFIGS[tier];
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - value);

  return (
    <div className={cn('relative', className)} style={{ width: size, height: size }}>
      {/* Background circle */}
      <svg className="w-full h-full -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-200 dark:text-slate-700"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="url(#progressGradient)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop
              offset="0%"
              className={config.gradientFrom.replace('from-', 'stop-')}
              stopColor="currentColor"
            />
            <stop
              offset="100%"
              className={config.gradientTo.replace('to-', 'stop-')}
              stopColor="currentColor"
            />
          </linearGradient>
        </defs>
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className={cn('text-3xl font-bold', config.color)}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          {value.toFixed(2)}
        </motion.span>
        <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Authority Score
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

interface AuthorityOverviewProps {
  authorityScore: number;
  tier: AuthorityTier;
  breakdown?: AuthorityBreakdown;
  nextTierThreshold: number;
  progressToNext: number;
  className?: string;
}

export function AuthorityOverview({
  authorityScore,
  tier,
  nextTierThreshold,
  progressToNext,
  className,
}: AuthorityOverviewProps) {
  const config = TIER_CONFIGS[tier];
  const Icon = config.icon;

  // Determina il prossimo tier
  const tierOrder: AuthorityTier[] = ['novizio', 'contributore', 'esperto', 'autorita'];
  const currentIndex = tierOrder.indexOf(tier);
  const nextTier = currentIndex < tierOrder.length - 1 ? tierOrder[currentIndex + 1] : null;
  const nextTierConfig = nextTier ? TIER_CONFIGS[nextTier] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'rounded-xl border p-6',
        config.bgColor,
        config.borderColor,
        className
      )}
    >
      <div className="flex flex-col md:flex-row items-center gap-6">
        {/* Radial Progress */}
        <RadialProgress value={authorityScore} tier={tier} />

        {/* Info */}
        <div className="flex-1 text-center md:text-left">
          {/* Tier Badge */}
          <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
            <div
              className={cn(
                'w-10 h-10 rounded-lg flex items-center justify-center',
                'bg-white dark:bg-slate-800 shadow-sm'
              )}
            >
              <Icon size={22} className={config.color} />
            </div>
            <div>
              <h2 className={cn('text-xl font-bold', config.color)}>
                {config.label}
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Livello di reputazione
              </p>
            </div>
          </div>

          {/* Description */}
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-4 max-w-md">
            {config.description}
          </p>

          {/* Progress to next tier */}
          {nextTier && nextTierConfig && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500 dark:text-slate-400 flex items-center gap-1">
                  <TrendingUp size={12} />
                  Verso {nextTierConfig.label}
                </span>
                <span className={nextTierConfig.color}>
                  {authorityScore.toFixed(2)} / {nextTierThreshold.toFixed(2)}
                </span>
              </div>
              <div className="h-2 bg-white dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progressToNext}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut', delay: 0.5 }}
                  className={cn(
                    'h-full rounded-full bg-gradient-to-r',
                    nextTierConfig.gradientFrom,
                    nextTierConfig.gradientTo
                  )}
                />
              </div>
              <p className="text-[10px] text-slate-400 dark:text-slate-500">
                Raggiungi {nextTierThreshold.toFixed(2)} di authority per diventare {nextTierConfig.label}
              </p>
            </div>
          )}

          {/* Max tier message */}
          {!nextTier && (
            <div className="flex items-center gap-2 text-sm text-purple-600 dark:text-purple-400">
              <Gavel size={14} />
              <span>Hai raggiunto il massimo livello di authority!</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default AuthorityOverview;
