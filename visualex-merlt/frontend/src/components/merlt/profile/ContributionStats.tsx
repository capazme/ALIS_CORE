/**
 * ContributionStats
 * =================
 *
 * Card con le statistiche dei contributi dell'utente:
 * - Totale contributi
 * - Approvati / Rigettati / Pending
 * - Success rate
 * - Peso voto attuale
 */

import { motion } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  Clock,
  BarChart3,
  Scale,
  TrendingUp,
} from 'lucide-react';
import { cn } from '../../../../lib/utils';

// =============================================================================
// STAT ITEM
// =============================================================================

interface StatItemProps {
  label: string;
  value: number | string;
  icon: typeof CheckCircle;
  color: string;
  bgColor: string;
  subtext?: string;
  index: number;
}

function StatItem({
  label,
  value,
  icon: Icon,
  color,
  bgColor,
  subtext,
  index,
}: StatItemProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={cn(
        'p-3 rounded-lg',
        bgColor
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
            {label}
          </p>
          <p className={cn('text-xl font-bold', color)}>
            {value}
          </p>
          {subtext && (
            <p className="text-[10px] text-slate-400 mt-0.5">
              {subtext}
            </p>
          )}
        </div>
        <div
          className={cn(
            'w-8 h-8 rounded-lg flex items-center justify-center',
            'bg-white/50 dark:bg-slate-800/50'
          )}
        >
          <Icon size={16} className={color} />
        </div>
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface ContributionStatsProps {
  totalContributions: number;
  approved: number;
  rejected: number;
  pending: number;
  voteWeight: number;
  className?: string;
}

export function ContributionStats({
  totalContributions,
  approved,
  rejected,
  pending,
  voteWeight,
  className,
}: ContributionStatsProps) {
  const successRate = totalContributions > 0
    ? Math.round((approved / totalContributions) * 100)
    : 0;

  const stats = [
    {
      label: 'Contributi Totali',
      value: totalContributions,
      icon: BarChart3,
      color: 'text-slate-700 dark:text-slate-200',
      bgColor: 'bg-slate-100 dark:bg-slate-800',
      subtext: 'Entità + Relazioni proposte',
    },
    {
      label: 'Approvati',
      value: approved,
      icon: CheckCircle,
      color: 'text-emerald-600 dark:text-emerald-400',
      bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
      subtext: 'Scritti nel Knowledge Graph',
    },
    {
      label: 'Rigettati',
      value: rejected,
      icon: XCircle,
      color: 'text-red-600 dark:text-red-400',
      bgColor: 'bg-red-50 dark:bg-red-900/20',
      subtext: 'Non approvati dalla community',
    },
    {
      label: 'In Attesa',
      value: pending,
      icon: Clock,
      color: 'text-amber-600 dark:text-amber-400',
      bgColor: 'bg-amber-50 dark:bg-amber-900/20',
      subtext: 'In coda di validazione',
    },
  ];

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-200 dark:border-slate-700',
        'bg-white dark:bg-slate-900 p-5',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 size={18} className="text-slate-500" />
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
          I Tuoi Contributi
        </h3>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {stats.map((stat, index) => (
          <StatItem
            key={stat.label}
            {...stat}
            index={index}
          />
        ))}
      </div>

      {/* Bottom row: Success Rate + Vote Weight */}
      <div className="grid grid-cols-2 gap-3">
        {/* Success Rate */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="p-3 rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800/30"
        >
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={14} className="text-blue-600 dark:text-blue-400" />
            <span className="text-xs text-slate-600 dark:text-slate-400">
              Success Rate
            </span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {successRate}%
            </span>
          </div>
          {/* Mini progress bar */}
          <div className="mt-2 h-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${successRate}%` }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="h-full bg-blue-500 rounded-full"
            />
          </div>
        </motion.div>

        {/* Vote Weight */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="p-3 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-100 dark:border-purple-800/30"
        >
          <div className="flex items-center gap-2 mb-2">
            <Scale size={14} className="text-purple-600 dark:text-purple-400" />
            <span className="text-xs text-slate-600 dark:text-slate-400">
              Peso Voto
            </span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {voteWeight.toFixed(3)}
            </span>
            <span className="text-xs text-slate-400">punti</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
            Threshold approvazione: 2.0
          </p>
        </motion.div>
      </div>
    </div>
  );
}

export default ContributionStats;
