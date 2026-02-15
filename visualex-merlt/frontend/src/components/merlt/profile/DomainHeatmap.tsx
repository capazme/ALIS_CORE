/**
 * DomainHeatmap
 * =============
 *
 * Visualizza l'authority dell'utente nei diversi domini legali.
 * 8 domini del sistema giuridico italiano:
 * - Civile, Penale, Amministrativo, Costituzionale
 * - Lavoro, Commerciale, Tributario, Internazionale
 *
 * L'intensità del colore indica il livello di authority nel dominio.
 */

import { motion } from 'framer-motion';
import {
  Scale,
  Gavel,
  Building2,
  BookOpen,
  Briefcase,
  Store,
  Receipt,
  Globe,
  Info,
} from 'lucide-react';
import { cn } from '../../../lib/utils';
import { Tooltip } from '../../ui/Tooltip';
import type { LegalDomain, DomainStats } from '../../../types/merlt';

// =============================================================================
// DOMAIN CONFIG
// =============================================================================

interface DomainConfig {
  label: string;
  icon: typeof Scale;
  description: string;
}

const DOMAIN_CONFIGS: Record<LegalDomain, DomainConfig> = {
  civile: {
    label: 'Civile',
    icon: Scale,
    description: 'Diritto privato, contratti, proprietà, famiglia, successioni',
  },
  penale: {
    label: 'Penale',
    icon: Gavel,
    description: 'Reati, sanzioni, procedimento penale',
  },
  amministrativo: {
    label: 'Amministrativo',
    icon: Building2,
    description: 'PA, atti amministrativi, appalti pubblici',
  },
  costituzionale: {
    label: 'Costituzionale',
    icon: BookOpen,
    description: 'Principi fondamentali, diritti, ordinamento',
  },
  lavoro: {
    label: 'Lavoro',
    icon: Briefcase,
    description: 'Rapporti di lavoro, previdenza, sindacale',
  },
  commerciale: {
    label: 'Commerciale',
    icon: Store,
    description: 'Impresa, società, fallimento, concorrenza',
  },
  tributario: {
    label: 'Tributario',
    icon: Receipt,
    description: 'Imposte, tasse, contenzioso fiscale',
  },
  internazionale: {
    label: 'Internazionale',
    icon: Globe,
    description: 'Diritto UE, trattati, privato internazionale',
  },
};

// =============================================================================
// DOMAIN CELL
// =============================================================================

interface DomainCellProps {
  domain: LegalDomain;
  stats: DomainStats;
  index: number;
}

function DomainCell({ domain, stats, index }: DomainCellProps) {
  const config = DOMAIN_CONFIGS[domain];
  const Icon = config.icon;

  // Calcola intensità colore basata su authority (0-1)
  const intensity = Math.max(0.1, stats.authority);
  const hue = 220; // Blue base
  const saturation = 70 + intensity * 20;
  const lightness = 95 - intensity * 45; // Più scuro = più authority

  const bgStyle = {
    backgroundColor: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
  };

  const tooltipContent = (
    <div className="space-y-1.5 max-w-[200px]">
      <p className="font-medium">{config.label}</p>
      <p className="text-xs opacity-90">{config.description}</p>
      <div className="text-xs pt-1.5 border-t border-white/20 space-y-0.5">
        <p>Authority: <strong>{stats.authority.toFixed(2)}</strong></p>
        <p>Contributi: <strong>{stats.contributions}</strong></p>
        <p>Approvati: <strong>{stats.success_rate}%</strong></p>
      </div>
    </div>
  );

  return (
    <Tooltip content={tooltipContent} placement="top">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: index * 0.05, duration: 0.3 }}
        whileHover={{ scale: 1.02 }}
        className={cn(
          'relative p-3 rounded-lg cursor-pointer transition-shadow',
          'hover:shadow-md dark:hover:shadow-slate-900/50',
          'border border-transparent hover:border-blue-200 dark:hover:border-blue-800/50'
        )}
        style={bgStyle}
      >
        <div className="flex items-start gap-2">
          <Icon
            size={16}
            aria-hidden="true"
            className={cn(
              intensity > 0.5
                ? 'text-white/90'
                : 'text-slate-600 dark:text-slate-700'
            )}
          />
          <div className="flex-1 min-w-0">
            <p
              className={cn(
                'text-xs font-medium truncate',
                intensity > 0.5
                  ? 'text-white'
                  : 'text-slate-700 dark:text-slate-800'
              )}
            >
              {config.label}
            </p>
            <p
              className={cn(
                'text-lg font-bold',
                intensity > 0.5
                  ? 'text-white'
                  : 'text-slate-800 dark:text-slate-900'
              )}
            >
              {stats.authority.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Activity indicator */}
        {stats.contributions > 0 && (
          <div
            className={cn(
              'absolute bottom-1.5 right-1.5 text-[9px]',
              intensity > 0.5
                ? 'text-white/70'
                : 'text-slate-500'
            )}
          >
            {stats.contributions} contributi
          </div>
        )}
      </motion.div>
    </Tooltip>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface DomainHeatmapProps {
  domains: Record<LegalDomain, DomainStats>;
  className?: string;
}

export function DomainHeatmap({ domains, className }: DomainHeatmapProps) {
  // Ordina domini per authority (decrescente)
  const sortedDomains = Object.entries(domains)
    .sort(([, a], [, b]) => b.authority - a.authority) as [LegalDomain, DomainStats][];

  // Trova dominio più forte
  const topDomain = sortedDomains[0];
  const topConfig = DOMAIN_CONFIGS[topDomain[0]];

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
          <Globe size={18} className="text-slate-500" aria-hidden="true" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Authority per Dominio
          </h3>
        </div>
        <Tooltip
          content="L'authority varia per dominio in base ai tuoi contributi specifici in quell'area"
          placement="left"
        >
          <Info size={14} className="text-slate-400 cursor-help" aria-hidden="true" />
        </Tooltip>
      </div>

      {/* Top domain highlight */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 p-3 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800/30"
      >
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
          Il tuo dominio più forte
        </p>
        <div className="flex items-center gap-2">
          <topConfig.icon size={18} className="text-blue-600 dark:text-blue-400" aria-hidden="true" />
          <span className="font-semibold text-slate-700 dark:text-slate-200">
            {topConfig.label}
          </span>
          <span className="ml-auto text-lg font-bold text-blue-600 dark:text-blue-400">
            {topDomain[1].authority.toFixed(2)}
          </span>
        </div>
      </motion.div>

      {/* Domain Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
        {sortedDomains.map(([domain, stats], index) => (
          <DomainCell
            key={domain}
            domain={domain}
            stats={stats}
            index={index}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-[10px] text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-100" />
          <span>Bassa</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-300" />
          <span>Media</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-500" />
          <span>Alta</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-700" />
          <span>Esperto</span>
        </div>
      </div>
    </div>
  );
}

export default DomainHeatmap;
