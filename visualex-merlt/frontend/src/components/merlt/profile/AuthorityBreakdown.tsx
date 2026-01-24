/**
 * AuthorityBreakdown
 * ==================
 *
 * Visualizza il breakdown della formula RLCF come waterfall chart.
 *
 * Formula: A_u(t) = 0.3·B_u + 0.5·T_u + 0.2·P_u
 *
 * Mostra per ogni componente:
 * - Valore raw [0-1]
 * - Peso nella formula
 * - Contributo finale
 */

import { motion } from 'framer-motion';
import { GraduationCap, TrendingUp, Zap, Info, Calculator } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import { Tooltip } from '../../../ui/Tooltip';
import type { AuthorityBreakdown as AuthorityBreakdownType } from '../../../../types/merlt';

// =============================================================================
// RLCF CONSTANTS
// =============================================================================

const RLCF_WEIGHTS = {
  baseline: 0.3,
  track_record: 0.5,
  level_authority: 0.2,
} as const;

// =============================================================================
// COMPONENT CONFIG
// =============================================================================

interface ComponentConfig {
  key: keyof AuthorityBreakdownType;
  label: string;
  shortLabel: string;
  icon: typeof GraduationCap;
  color: string;
  bgColor: string;
  description: string;
  tooltip: string;
}

const COMPONENT_CONFIGS: ComponentConfig[] = [
  {
    key: 'baseline',
    label: 'Qualifiche Professionali',
    shortLabel: 'B_u',
    icon: GraduationCap,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-500',
    description: 'Credenziali e formazione giuridica',
    tooltip: 'Baseline credentials: qualifiche professionali verificate (laurea, abilitazione, specializzazioni)',
  },
  {
    key: 'track_record',
    label: 'Track Record',
    shortLabel: 'T_u',
    icon: TrendingUp,
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-500',
    description: 'Storico validazioni corrette',
    tooltip: 'Track record: storico delle performance pesato esponenzialmente (λ=0.95). I contributi recenti pesano di più.',
  },
  {
    key: 'level_authority',
    label: 'Performance Recente',
    shortLabel: 'P_u',
    icon: Zap,
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-500',
    description: 'Authority nei domini attivi',
    tooltip: 'Performance recente: authority media nei domini legali in cui hai contribuito di recente.',
  },
];

// =============================================================================
// BAR COMPONENT
// =============================================================================

interface BreakdownBarProps {
  config: ComponentConfig;
  value: number;
  weight: number;
  contribution: number;
  index: number;
  maxContribution: number;
}

function BreakdownBar({
  config,
  value,
  weight,
  contribution,
  index,
  maxContribution,
}: BreakdownBarProps) {
  const Icon = config.icon;
  const barWidth = Math.max(5, (contribution / maxContribution) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.15, duration: 0.4 }}
      className="space-y-2"
    >
      {/* Label row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              'w-7 h-7 rounded flex items-center justify-center',
              'bg-slate-100 dark:bg-slate-800'
            )}
          >
            <Icon size={14} className={config.color} />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                {config.label}
              </span>
              <Tooltip content={config.tooltip} placement="top">
                <Info size={12} className="text-slate-400 cursor-help" />
              </Tooltip>
            </div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">
              {config.description}
            </span>
          </div>
        </div>

        {/* Values */}
        <div className="text-right">
          <div className="flex items-baseline gap-1">
            <span className={cn('text-sm font-semibold', config.color)}>
              {value.toFixed(2)}
            </span>
            <span className="text-[10px] text-slate-400">
              × {weight}
            </span>
            <span className="text-xs text-slate-500">=</span>
            <span className="text-sm font-bold text-slate-700 dark:text-slate-200">
              {contribution.toFixed(3)}
            </span>
          </div>
        </div>
      </div>

      {/* Bar */}
      <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ delay: index * 0.15 + 0.2, duration: 0.6, ease: 'easeOut' }}
          className={cn('h-full rounded-full', config.bgColor)}
        />
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface AuthorityBreakdownProps {
  breakdown: AuthorityBreakdownType;
  totalScore: number;
  className?: string;
}

export function AuthorityBreakdown({
  breakdown,
  totalScore,
  className,
}: AuthorityBreakdownProps) {
  // Calcola contributi
  const contributions = COMPONENT_CONFIGS.map((config) => ({
    config,
    value: breakdown[config.key],
    weight: RLCF_WEIGHTS[config.key],
    contribution: breakdown[config.key] * RLCF_WEIGHTS[config.key],
  }));

  const maxContribution = Math.max(...contributions.map((c) => c.contribution), 0.1);
  const totalContribution = contributions.reduce((sum, c) => sum + c.contribution, 0);

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-200 dark:border-slate-700',
        'bg-white dark:bg-slate-900 p-5',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <Calculator size={18} className="text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Composizione Authority
          </h3>
        </div>
        <Tooltip
          content="Formula RLCF: A_u(t) = 0.3·B_u + 0.5·T_u + 0.2·P_u"
          placement="left"
        >
          <div className="text-[10px] font-mono text-slate-400 cursor-help">
            RLCF Formula
          </div>
        </Tooltip>
      </div>

      {/* Bars */}
      <div className="space-y-4">
        {contributions.map((item, index) => (
          <BreakdownBar
            key={item.config.key}
            config={item.config}
            value={item.value}
            weight={item.weight}
            contribution={item.contribution}
            index={index}
            maxContribution={maxContribution}
          />
        ))}
      </div>

      {/* Divider */}
      <div className="my-4 border-t border-dashed border-slate-200 dark:border-slate-700" />

      {/* Total */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="flex items-center justify-between"
      >
        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Authority Totale (A_u)
        </span>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            {totalScore.toFixed(3)}
          </span>
        </div>
      </motion.div>

      {/* Vote weight explanation */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-4 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50"
      >
        <p className="text-xs text-slate-600 dark:text-slate-400">
          <strong className="text-slate-700 dark:text-slate-300">Peso del tuo voto:</strong>{' '}
          Quando voti, il tuo voto vale{' '}
          <span className="font-semibold text-blue-600 dark:text-blue-400">
            {totalScore.toFixed(2)}
          </span>{' '}
          punti verso il threshold di approvazione (2.0 punti totali).
        </p>
      </motion.div>
    </div>
  );
}

export default AuthorityBreakdown;
